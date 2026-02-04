"""
Main MMDiT model implementation (Flax).
"""

from typing import Tuple, Optional, Sequence, Union
import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial

from mmdit_jax_flax.blocks import MMDiTBlock
from mmdit_jax_flax.layers import TimestepEmbedding
from mmdit_jax_flax.final_layer import ModulatedFinalLayer


class MMDiT(nn.Module):
    """
    Multi-Modal Diffusion Transformer (SD3.5 architecture).

    This is the main model that stacks multiple MMDiTBlocks to create
    a deep transformer for processing multiple modalities with timestep conditioning.

    Key features:
    - Sinusoidal timestep embedding with MLP
    - Modulation-based conditioning throughout
    - Mandatory QK normalization in attention
    - SwiGLU activation in feedforward
    - Modulated final output layers
    """

    depth: int
    dim_modalities: Tuple[int, ...]
    dim_outs: Tuple[int, ...]
    dim_cond: int = 1024
    timestep_embed_dim: Union[int, Tuple[int, ...]] = 256
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4

    @nn.compact
    def __call__(
        self,
        modality_tokens: Tuple[jax.Array, ...],
        timestep: Union[jax.Array, Tuple[jax.Array, ...]],
        attention_mask: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, ...]:
        """
        Forward pass through MMDiT.

        Args:
            modality_tokens: Tuple of token tensors, each of shape (seq_len, dim)
            timestep: Scalar timestep value(s) for diffusion process. Can be a single
                scalar or a tuple of scalars matching the number of conditioning variables.
            attention_mask: Optional 2D attention mask of shape (total_seq, total_seq)
                where total_seq is sum of all modality sequence lengths.
                True indicates the query position can attend to the key position.

        Returns:
            Tuple of output tensors for each modality
        """
        # Normalize timestep_embed_dim to tuple
        timestep_embed_dim = self.timestep_embed_dim
        if isinstance(timestep_embed_dim, int):
            timestep_embed_dim = (timestep_embed_dim,)

        num_conds = len(timestep_embed_dim)

        # Normalize timestep to tuple
        if not isinstance(timestep, (tuple, list)):
            timestep = (timestep,)

        assert len(timestep) == num_conds, \
            f"Expected {num_conds} conditioning variables, got {len(timestep)}"

        # Embed each timestep and sum
        time_cond = jnp.zeros((self.dim_cond,))
        for i, (dim, t) in enumerate(zip(timestep_embed_dim, timestep)):
            t_emb = TimestepEmbedding(dim_embed=dim, dim_out=self.dim_cond, name=f'timestep_embed_{i}')(t)
            time_cond = time_cond + t_emb

        # Process through transformer blocks sequentially
        for block_idx in range(self.depth):
            modality_tokens = MMDiTBlock(
                dim_modalities=self.dim_modalities,
                dim_cond=self.dim_cond,
                dim_head=self.dim_head,
                heads=self.heads,
                ff_mult=self.ff_mult,
                name=f'block_{block_idx}'
            )(modality_tokens, time_cond, attention_mask)

        # Apply modulated final layers
        modality_tokens = tuple(
            ModulatedFinalLayer(
                dim_in=dim,
                dim_out=dim_out,
                dim_cond=self.dim_cond,
                name=f'final_layer_{i}'
            )(tokens, time_cond)
            for i, (tokens, dim, dim_out) in enumerate(zip(modality_tokens, self.dim_modalities, self.dim_outs))
        )

        return modality_tokens
