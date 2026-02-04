"""
Joint attention mechanism for multi-modal transformer (Flax implementation).
"""

from typing import Tuple, Optional, Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array
from einops import rearrange, pack, unpack

from mmdit_jax_flax.helpers import exists, default
from mmdit_jax_flax.rope import apply_rope, build_rope_frequencies


class MultiHeadRMSNorm(nn.Module):
    """
    RMS normalization for multi-head attention (SD3.5 aligned).

    Applies RMS normalization to each attention head with learnable scale
    gamma SHARED across all heads. This matches the SD3.5 implementation
    where a single (dim,) gamma vector is broadcast across heads.

    The normalization is: (x / RMS(x)) * gamma
    where RMS(x) = sqrt(mean(x²) + eps)
    """

    dim: int
    heads: int = 1  # kept for API compatibility but not used for gamma shape
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass of MultiHeadRMSNorm.

        Args:
            x: Input tensor of shape (heads, seq_len, dim)

        Returns:
            Normalized tensor of same shape as input
        """
        # Shared gamma across all heads, shape (dim,) - matches SD3.5
        gamma = self.param('gamma', nn.initializers.ones, (self.dim,))

        # Compute RMS along the last dimension (per head, per position)
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

        # Normalize and apply shared gamma (broadcasts across heads and seq)
        return (x / rms) * gamma


class JointAttention(nn.Module):
    """
    Joint attention mechanism that handles multiple modalities.

    This attention mechanism processes multiple input modalities jointly,
    allowing cross-modal attention between different types of inputs.

    Implements attention computation manually to handle multi-modal inputs.
    """

    dim_inputs: Tuple[int, ...]
    dim_head: int = 64
    heads: int = 8

    @nn.compact
    def __call__(
        self,
        inputs: Tuple[Array, ...],
        attention_mask: Optional[Array] = None,
    ) -> Tuple[Array, ...]:
        """
        Forward pass of joint attention.

        Args:
            inputs: Tuple of input tensors for each modality,
                   each of shape (seq_len, dim_input)
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                  where total_seq is sum of all modality sequence lengths.
                  True indicates the query position can attend to the key position.
                  If None, full bidirectional attention is used (all True).

        Returns:
            Tuple of output tensors for each modality,
            each of shape (seq_len, dim_input)
        """
        # Validate inputs
        assert len(self.dim_inputs) > 0, "At least one input modality is required"
        assert self.dim_head > 0, "dim_head must be positive"
        assert self.heads > 0, "heads must be positive"
        assert len(inputs) == len(self.dim_inputs), \
            f"Expected {len(self.dim_inputs)} inputs, got {len(inputs)}"

        num_inputs = len(self.dim_inputs)
        dim_inner = self.dim_head * self.heads

        # Project each modality separately for qkv
        all_qkvs = []

        for i, (x, dim_input) in enumerate(zip(inputs, self.dim_inputs)):
            # Validate input shape
            seq_len, actual_dim = x.shape
            assert actual_dim == dim_input, \
                f"Input {i} has dimension {actual_dim}, expected {dim_input}"

            # Project to QKV
            qkv = nn.Dense(dim_inner * 3, use_bias=False, name=f'to_qkv_{i}')(x)

            # Split heads: (seq_len, dim_inner * 3) -> (3, heads, seq_len, dim_head)
            qkv = rearrange(qkv, 's (qkv h d) -> qkv h s d', qkv=3, h=self.heads)

            # Apply QK RMS normalization per modality (mandatory)
            q, k, v = qkv
            q = MultiHeadRMSNorm(dim=self.dim_head, heads=self.heads, name=f'q_rmsnorm_{i}')(q)
            k = MultiHeadRMSNorm(dim=self.dim_head, heads=self.heads, name=f'k_rmsnorm_{i}')(k)

            # Apply RoPE to queries and keys (per-modality positions: 0, 1, 2, ...)
            cos, sin = build_rope_frequencies(seq_len, self.dim_head)
            q = apply_rope(q, cos, sin)  # (heads, seq_len, dim_head)
            k = apply_rope(k, cos, sin)

            qkv = jnp.stack([q, k, v])

            all_qkvs.append(qkv)

        # Combine all qkv
        all_qkvs, packed_shape = pack(all_qkvs, 'qkv h * d')  # (3, heads, total_seq, dim_head)
        total_seq = all_qkvs.shape[2]

        # Validate attention mask if provided
        if attention_mask is not None:
            assert attention_mask.shape == (total_seq, total_seq), \
                f"attention_mask has shape {attention_mask.shape}, expected ({total_seq}, {total_seq})"

        # Attention
        q, k, v = all_qkvs  # Each is (heads, total_seq, dim_head)

        # Convert to format expected by jax.nn.dot_product_attention: (seq_len, num_heads, head_dim)
        q = rearrange(q, 'h s d -> s h d')  # (total_seq, heads, dim_head)
        k = rearrange(k, 'h s d -> s h d')  # (total_seq, heads, dim_head)
        v = rearrange(v, 'h s d -> s h d')  # (total_seq, heads, dim_head)

        # Apply optimized attention using JAX's built-in function
        out = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=attention_mask,  # (total_seq, total_seq) or None
        )  # (total_seq, heads, dim_head)

        # Merge heads back to (total_seq, heads * dim_head)
        outs = rearrange(out, 's h d -> s (h d)')
        outs = unpack(outs, packed_shape, '* d')  # list of (modality 0 seq, modality 1 seq, etc...)

        # Apply output projections
        all_outs = []
        for i, (out_i, dim_input) in enumerate(zip(outs, self.dim_inputs)):
            out_proj = nn.Dense(dim_input, use_bias=False, name=f'to_out_{i}')(out_i)
            all_outs.append(out_proj)

        return tuple(all_outs)
