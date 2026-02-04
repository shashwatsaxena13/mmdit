"""
Basic layers for the mmdit library (Flax implementation).
"""

from typing import Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange
from jaxtyping import Array
import math

from mmdit_jax_flax.helpers import exists


def timestep_embedding(t: Array, dim: int, max_period: int = 10000) -> Array:
    """
    Create sinusoidal timestep embeddings.

    Args:
        t: Timestep(s) - shape () or (,) for single timestep
        dim: Embedding dimension
        max_period: Maximum period for sinusoidal encoding

    Returns:
        Sinusoidal embedding of shape (dim,)
    """
    # Ensure t is a scalar or 0-d array
    t = jnp.atleast_1d(t).squeeze()

    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = t * freqs
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    # If dim is odd, pad with a zero
    if dim % 2 == 1:
        embedding = jnp.concatenate([embedding, jnp.zeros(1)], axis=-1)

    return embedding


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding module that converts scalar timesteps to high-dimensional embeddings.

    Uses sinusoidal encoding followed by an MLP: t -> SinEmbed -> Linear -> SiLU -> Linear
    This is the standard approach in modern diffusion transformers.
    """

    dim_embed: int = 256
    dim_out: int = 1024

    @nn.compact
    def __call__(self, t: Array) -> Array:
        """
        Embed a timestep.

        Args:
            t: Timestep value (scalar or 0-d array)

        Returns:
            Embedded timestep of shape (dim_out,)
        """
        # Get sinusoidal embedding
        t_emb = timestep_embedding(t, self.dim_embed)

        # Apply MLP: Linear -> SiLU -> Linear
        t_emb = nn.Dense(self.dim_out, name='linear1')(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(self.dim_out, name='linear2')(t_emb)

        return t_emb


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with modulation-based conditioning.

    Following SD3.5, this uses shift+scale modulation:
    output = LayerNorm(x) * (1 + scale) + shift

    where shift and scale are predicted from the conditioning signal.
    """

    dim: int
    dim_cond: int

    @nn.compact
    def __call__(
        self,
        x: Array,
        cond: Array,
    ) -> Array:
        """
        Forward pass of adaptive layer norm with modulation.

        Args:
            x: Input tensor of shape (seq, dim)
            cond: Conditioning tensor of shape (dim_cond,)

        Returns:
            Modulated normalized tensor of shape (seq, dim)
        """
        # Apply layer norm (without learnable parameters)
        x = nn.LayerNorm(use_bias=False, use_scale=False, name='ln')(x)

        # Get shift and scale from conditioning
        # Zero initialization for modulation (adaLN-Zero)
        modulation = nn.silu(cond)
        modulation = nn.Dense(
            2 * self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='to_modulation'
        )(modulation)  # (2 * dim,)

        shift, scale = jnp.split(modulation, 2, axis=-1)  # (dim,), (dim,)
        shift = shift.reshape(1, -1)  # (1, dim)
        scale = scale.reshape(1, -1)  # (1, dim)

        # Apply modulation: x * (1 + scale) + shift
        x = x * (1.0 + scale) + shift

        return x
