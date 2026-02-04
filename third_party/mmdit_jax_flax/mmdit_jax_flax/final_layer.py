"""
Final output layer for MMDiT with modulation (Flax implementation).
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array


class ModulatedFinalLayer(nn.Module):
    """
    Modulated final layer for MMDiT output.

    Applies adaptive layer normalization followed by a linear projection,
    with both conditioned on the timestep embedding. This follows the SD3.5
    approach of using modulation-based conditioning throughout.

    The layer produces: LayerNorm(x) * (1 + scale) + shift -> Linear
    where shift and scale are predicted from the timestep conditioning.
    """

    dim_in: int
    dim_out: int
    dim_cond: int

    @nn.compact
    def __call__(self, x: Array, cond: Array) -> Array:
        """
        Forward pass of modulated final layer.

        Args:
            x: Input tensor of shape (seq, dim_in)
            cond: Conditioning tensor of shape (dim_cond,)

        Returns:
            Output tensor of shape (seq, dim_out)
        """
        # Apply layer norm (without learnable parameters)
        x = nn.LayerNorm(use_bias=False, use_scale=False, name='ln')(x)

        # Get shift and scale from conditioning
        # Zero initialization for modulation (adaLN-Zero)
        modulation = nn.silu(cond)
        modulation = nn.Dense(
            2 * self.dim_in,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='adaLN_modulation'
        )(modulation)  # (2 * dim_in,)

        shift, scale = jnp.split(modulation, 2, axis=-1)  # (dim_in,), (dim_in,)
        shift = shift.reshape(1, -1)  # (1, dim_in)
        scale = scale.reshape(1, -1)  # (1, dim_in)

        # Modulate: x * (1 + scale) + shift
        x = x * (1.0 + scale) + shift

        # Apply output projection - zero initialized
        x = nn.Dense(
            self.dim_out,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='linear_out'
        )(x)

        return x
