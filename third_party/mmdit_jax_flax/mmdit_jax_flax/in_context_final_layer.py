"""
Simple final layer for In-Context MMDiT (no modulation) - Flax implementation.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes by the RMS of the input, without mean centering.
    This is more efficient than LayerNorm and is used in modern LLMs
    (LLaMA, Mistral, etc.).

    Formula: x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x²) + eps)
    """

    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass of RMSNorm.

        Args:
            x: Input tensor of shape (dim,)

        Returns:
            Normalized tensor of shape (dim,)
        """
        gamma = self.param('gamma', nn.initializers.ones, (self.dim,))
        rms = jnp.sqrt(jnp.mean(x**2) + self.eps)
        return (x / rms) * gamma


class SimpleFinalLayer(nn.Module):
    """
    Simple final layer for In-Context MMDiT.

    Uses RMSNorm followed by a linear projection.
    Unlike ModulatedFinalLayer, this has no conditioning/modulation.
    The linear layer is zero-initialized for stable training.
    """

    dim_in: int
    dim_out: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass of simple final layer.

        Args:
            x: Input tensor of shape (seq, dim_in)

        Returns:
            Output tensor of shape (seq, dim_out)
        """
        # Apply RMSNorm to each token
        rms_norm = RMSNorm(dim=self.dim_in, name='rms_norm')
        x = jax.vmap(rms_norm)(x)

        # Apply output projection - zero initialized
        x = nn.Dense(
            self.dim_out,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='linear_out'
        )(x)

        return x
