"""
In-Context MMDiT model implementation (Flax).

This is a pure token-processing transformer without any conditioning logic.
Time conditioning should be handled externally by the user (e.g., prepending
time tokens to specific modalities). The model simply processes tokens through
joint attention and feedforward layers.
"""

from typing import Tuple, Optional, Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn

from mmdit_jax_flax.in_context_blocks import InContextMMDiTBlock
from mmdit_jax_flax.in_context_final_layer import SimpleFinalLayer


class InContextMMDiT(nn.Module):
    """
    In-Context Multi-Modal Diffusion Transformer.

    Unlike the standard MMDiT which uses AdaLN modulation for conditioning,
    this model is a pure token processor. Time conditioning should be
    handled externally by the user (e.g., by prepending time tokens to
    specific modalities before passing to this model).

    Key differences from MMDiT:
    - No timestep embedding or conditioning logic
    - Uses RMSNorm instead of AdaptiveLayerNorm
    - Uses standard residuals instead of gated residuals
    - SimpleFinalLayer instead of ModulatedFinalLayer

    Example usage for imitation learning:
        # User handles time embedding externally
        time_embed = TimestepEmbedding(dim_embed=256, dim_out=256)
        time_to_token = nn.Dense(action_dim)

        # Create time tokens
        time_cond = time_embed(t=0.5)
        time_token = time_to_token(time_cond)
        time_tokens = jnp.tile(time_token[None, :], (2, 1))  # 2 tokens

        # Prepend to actions (user's choice)
        action_with_time = jnp.concatenate([time_tokens, action_tokens], axis=0)

        # Model just processes tokens
        model = InContextMMDiT(dim_modalities=(obs_dim, action_dim), ...)
        obs_out, action_out = model((obs_tokens, action_with_time), attention_mask)

        # User strips time tokens from output
        action_out = action_out[2:]
    """

    depth: int
    dim_modalities: Tuple[int, ...]
    dim_outs: Tuple[int, ...]
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
        Forward pass through InContextMMDiT.

        Args:
            modality_tokens: Tuple of token tensors, each of shape (seq_len, dim).
                Time conditioning tokens (if any) should already be included
                in the appropriate modality sequences by the user.
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                where total_seq is sum of all modality sequence lengths.
                True indicates the query position can attend to the key position.

        Returns:
            Tuple of output tensors for each modality
        """
        # Process through transformer blocks sequentially
        for block_idx in range(self.depth):
            modality_tokens = InContextMMDiTBlock(
                dim_modalities=self.dim_modalities,
                dim_head=self.dim_head,
                heads=self.heads,
                ff_mult=self.ff_mult,
                name=f'block_{block_idx}'
            )(modality_tokens, attention_mask)

        # Apply final layers
        modality_tokens = tuple(
            SimpleFinalLayer(
                dim_in=dim,
                dim_out=dim_out,
                name=f'final_layer_{i}'
            )(tokens)
            for i, (tokens, dim, dim_out) in enumerate(zip(modality_tokens, self.dim_modalities, self.dim_outs))
        )

        return modality_tokens
