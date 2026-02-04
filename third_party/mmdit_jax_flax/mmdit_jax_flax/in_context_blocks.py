"""
In-Context MMDiT Block implementation - transformer block without AdaLN modulation (Flax).

This block uses standard RMSNorm instead of Adaptive LayerNorm, and standard
residual connections without gating. Time conditioning is expected to be
provided as tokens in the input sequence (handled by the user externally).
"""

from typing import Tuple, Optional, Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn

from mmdit_jax_flax.attention import JointAttention
from mmdit_jax_flax.feedforward import FeedForward
from mmdit_jax_flax.in_context_final_layer import RMSNorm


class InContextMMDiTBlock(nn.Module):
    """
    In-Context Multi-Modal Diffusion Transformer Block.

    Unlike the standard MMDiTBlock, this block:
    - Uses standard RMSNorm instead of AdaptiveLayerNorm (no modulation)
    - Uses standard residual connections (no gating)
    - Does not take time conditioning - conditioning is expected to be
      provided as tokens in the input sequence

    This follows the modern LLM transformer pattern (pre-norm + residual).
    """

    dim_modalities: Tuple[int, ...]
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4

    @nn.compact
    def __call__(
        self,
        modality_tokens: Tuple[jax.Array, ...],
        attention_mask: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, ...]:
        """
        Forward pass of InContextMMDiTBlock.

        Args:
            modality_tokens: Tuple of token tensors for each modality,
                each of shape (seq_len, dim)
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                where total_seq is sum of all modality sequence lengths.
                True indicates the query position can attend to the key position.

        Returns:
            Tuple of processed token tensors for each modality
        """
        assert len(self.dim_modalities) > 1, 'must have at least two modalities'
        num_modalities = len(self.dim_modalities)
        assert len(modality_tokens) == num_modalities

        # ATTENTION BLOCK (Pre-norm + Residual)
        # Store residuals
        attn_residuals = modality_tokens

        # Apply RMSNorm (pre-norm)
        modality_tokens = tuple(
            jax.vmap(RMSNorm(dim=dim, name=f'attn_norm_{i}'))(tokens)
            for i, (tokens, dim) in enumerate(zip(modality_tokens, self.dim_modalities))
        )

        # Apply joint attention
        modality_tokens = JointAttention(
            dim_inputs=self.dim_modalities,
            dim_head=self.dim_head,
            heads=self.heads,
            name='joint_attn'
        )(modality_tokens, attention_mask=attention_mask)

        # Add residual (no gating)
        modality_tokens = tuple(
            residual + tokens
            for tokens, residual in zip(modality_tokens, attn_residuals)
        )

        # FEEDFORWARD BLOCK (Pre-norm + Residual)
        # Store residuals
        ff_residuals = modality_tokens

        # Apply RMSNorm (pre-norm)
        modality_tokens = tuple(
            jax.vmap(RMSNorm(dim=dim, name=f'ff_norm_{i}'))(tokens)
            for i, (tokens, dim) in enumerate(zip(modality_tokens, self.dim_modalities))
        )

        # Apply feedforward layers
        modality_tokens = tuple(
            FeedForward(dim=dim, mult=self.ff_mult, name=f'feedforward_{i}')(tokens)
            for i, (tokens, dim) in enumerate(zip(modality_tokens, self.dim_modalities))
        )

        # Add residual (no gating)
        modality_tokens = tuple(
            residual + tokens
            for tokens, residual in zip(modality_tokens, ff_residuals)
        )

        return modality_tokens
