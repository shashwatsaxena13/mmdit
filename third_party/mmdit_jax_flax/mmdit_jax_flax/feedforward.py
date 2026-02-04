"""
Feedforward layers for the mmdit library (Flax implementation).
"""

from typing import Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array


class FeedForward(nn.Module):
    """
    Feedforward layer with SwiGLU activation.

    Uses SwiGLU (Swish-Gated Linear Unit) activation, which is the standard
    in modern diffusion transformers (SD3, SD3.5, etc).
    SwiGLU splits the hidden layer: out = Linear2(Linear1(x) * swish(Gate(x)))
    """

    dim: int
    mult: int = 4

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass of feedforward layer with SwiGLU.

        Args:
            x: Input tensor of shape (seq_len, dim)

        Returns:
            Output tensor of shape (seq_len, dim)
        """
        dim_hidden = self.dim * self.mult

        # SwiGLU uses two parallel projections for gating
        x_in = nn.Dense(dim_hidden, use_bias=False, name='linear_in')(x)
        x_gate = nn.Dense(dim_hidden, use_bias=False, name='linear_gate')(x)

        # SwiGLU: out = Linear_out(Linear_in(x) * swish(Linear_gate(x)))
        x_gated = x_in * nn.swish(x_gate)
        out = nn.Dense(self.dim, use_bias=False, name='linear_out')(x_gated)

        return out
