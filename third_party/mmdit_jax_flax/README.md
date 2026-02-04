# MMDIT JAX Flax

Multi-modal DiT (Diffusion Transformer) implementation in JAX/Flax.

This library provides a JAX/Flax implementation of multi-modal diffusion transformers,
migrated from the original Equinox implementation.

## Installation

```bash
pip install -e .
```

## Usage

```python
from mmdit_jax_flax import MMDiT
import jax
import jax.numpy as jnp

# Create model
model = MMDiT(
    depth=12,
    dim_modalities=(512, 256),
    dim_outs=(512, 256),
    dim_cond=1024,
    timestep_embed_dim=256,
    dim_head=64,
    heads=8,
)

# Initialize parameters
key = jax.random.PRNGKey(0)
x1 = jax.random.normal(key, (10, 512))
x2 = jax.random.normal(key, (5, 256))
timestep = jnp.array(0.5)

params = model.init(key, (x1, x2), timestep)

# Forward pass
y1, y2 = model.apply(params, (x1, x2), timestep)
```
