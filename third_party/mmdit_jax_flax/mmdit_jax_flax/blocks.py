"""
MMDiT Block implementation - the core transformer block for multi-modal diffusion (Flax).
"""

from typing import Tuple, Optional, Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn

from mmdit_jax_flax.helpers import exists, default
from mmdit_jax_flax.layers import AdaptiveLayerNorm
from mmdit_jax_flax.attention import JointAttention
from mmdit_jax_flax.feedforward import FeedForward


class MMDiTBlock(nn.Module):
    """
    Multi-Modal Diffusion Transformer Block.

    This block processes multiple modalities jointly using:
    1. Adaptive layer normalization (conditioned on time)
    2. Joint attention across modalities
    3. Feedforward layers for each modality
    4. Residual connections (ResNet-style)
    """

    dim_modalities: Tuple[int, ...]
    dim_cond: int
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4

    @nn.compact
    def __call__(
        self,
        modality_tokens: Tuple[jax.Array, ...],
        time_cond: jax.Array,
        attention_mask: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, ...]:
        """
        Forward pass of MMDiTBlock.

        Args:
            modality_tokens: Tuple of token tensors for each modality
            time_cond: Time conditioning tensor for flow matching timesteps
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                where total_seq is sum of all modality sequence lengths.
                True indicates the query position can attend to the key position.

        Returns:
            Tuple of processed token tensors for each modality
        """
        assert len(self.dim_modalities) > 1, 'must have at least two modalities'
        num_modalities = len(self.dim_modalities)
        assert len(modality_tokens) == num_modalities

        # Compute cumsum for splitting
        dim_cumsum = [self.dim_modalities[0]]
        for i in range(1, len(self.dim_modalities)):
            dim_cumsum.append(dim_cumsum[i - 1] + self.dim_modalities[i])

        # Get gate parameters from conditioning (following SD3.5 modulation)
        # Zero initialization for gates (adaLN-Zero)
        total_cond_dims = sum(self.dim_modalities) * 2  # 2 gates per modality
        gates = nn.silu(time_cond)
        gates = nn.Dense(
            total_cond_dims,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='to_cond'
        )(gates)  # (2 * sum(dim_modalities),)

        # Split into attention and feedforward gates
        attn_gates, ff_gates = jnp.split(gates, 2, axis=-1)  # each (sum(dim_modalities),)

        # Split by modality dimensions
        attn_gate_splits = jnp.split(attn_gates, dim_cumsum, axis=-1)
        ff_gate_splits = jnp.split(ff_gates, dim_cumsum, axis=-1)

        # ATTENTION BLOCK WITH RESIDUAL CONNECTION
        # Store residuals for attention
        attn_residuals = modality_tokens

        # Apply adaptive layer norm with modulation
        modality_tokens = tuple(
            AdaptiveLayerNorm(dim=dim, dim_cond=self.dim_cond, name=f'attn_layernorm_{i}')(tokens, time_cond)
            for i, (tokens, dim) in enumerate(zip(modality_tokens, self.dim_modalities))
        )

        # Apply joint attention
        modality_tokens = JointAttention(
            dim_inputs=self.dim_modalities,
            dim_head=self.dim_head,
            heads=self.heads,
            name='joint_attn'
        )(modality_tokens, attention_mask=attention_mask)

        # Apply gate and add residual: residual + gate * attn_output
        modality_tokens = tuple(
            residual + gate.reshape(1, -1) * tokens
            for tokens, residual, gate in zip(modality_tokens, attn_residuals, attn_gate_splits)
        )

        # FEEDFORWARD BLOCK WITH RESIDUAL CONNECTION
        # Store residuals for feedforward
        ff_residuals = modality_tokens

        # Apply adaptive layer norm with modulation
        modality_tokens = tuple(
            AdaptiveLayerNorm(dim=dim, dim_cond=self.dim_cond, name=f'ff_layernorm_{i}')(tokens, time_cond)
            for i, (tokens, dim) in enumerate(zip(modality_tokens, self.dim_modalities))
        )

        # Apply feedforward layers
        modality_tokens = tuple(
            FeedForward(dim=dim, mult=self.ff_mult, name=f'feedforward_{i}')(tokens)
            for i, (tokens, dim) in enumerate(zip(modality_tokens, self.dim_modalities))
        )

        # Apply gate and add residual: residual + gate * ff_output
        modality_tokens = tuple(
            residual + gate.reshape(1, -1) * tokens
            for tokens, residual, gate in zip(modality_tokens, ff_residuals, ff_gate_splits)
        )

        return modality_tokens
