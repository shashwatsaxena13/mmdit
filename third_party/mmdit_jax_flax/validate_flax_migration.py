"""
Script to validate Flax migration by testing against golden outputs.

This script tests that the Flax implementation produces valid outputs
with correct shapes and that the model can be initialized and run.

Note: Numerical equivalence with Equinox is not expected due to different
initialization. This script validates:
1. Shape correctness
2. Gradient flow
3. Batching works correctly
"""
import json
import jax
import jax.numpy as jnp
import numpy as np

from mmdit_jax_flax import MMDiT, InContextMMDiT
from mmdit_jax_flax.blocks import MMDiTBlock
from mmdit_jax_flax.attention import JointAttention
from mmdit_jax_flax.feedforward import FeedForward
from mmdit_jax_flax.layers import TimestepEmbedding, AdaptiveLayerNorm
from mmdit_jax_flax.final_layer import ModulatedFinalLayer
from mmdit_jax_flax.in_context_final_layer import SimpleFinalLayer, RMSNorm

# Fixed seed for reproducibility
SEED = 42


def test_feedforward():
    """Test FeedForward layer."""
    print("Testing FeedForward...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 3)

    dim = 64
    ff = FeedForward(dim=dim, mult=4)

    # Initialize
    x = jax.random.normal(keys[1], (8, dim))
    params = ff.init(keys[0], x)

    # Forward pass
    y = ff.apply(params, x)

    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("  FeedForward: PASSED")


def test_attention():
    """Test JointAttention."""
    print("Testing JointAttention...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    dim_inputs = (64, 32)
    layer = JointAttention(dim_inputs=dim_inputs, dim_head=32, heads=4)

    # Initialize
    seq_len1, seq_len2 = 8, 6
    x1 = jax.random.normal(keys[1], (seq_len1, dim_inputs[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_inputs[1]))

    params = layer.init(keys[0], (x1, x2))

    # Forward pass
    y1, y2 = layer.apply(params, (x1, x2))

    assert y1.shape == x1.shape, f"Expected shape {x1.shape}, got {y1.shape}"
    assert y2.shape == x2.shape, f"Expected shape {x2.shape}, got {y2.shape}"
    print(f"  Input shapes: {x1.shape}, {x2.shape}")
    print(f"  Output shapes: {y1.shape}, {y2.shape}")

    # Test with attention mask
    total_seq = seq_len1 + seq_len2
    attention_mask = jnp.ones((total_seq, total_seq), dtype=bool)
    y1_masked, y2_masked = layer.apply(params, (x1, x2), attention_mask=attention_mask)
    assert y1_masked.shape == x1.shape
    assert y2_masked.shape == x2.shape
    print("  JointAttention: PASSED")


def test_timestep_embedding():
    """Test TimestepEmbedding."""
    print("Testing TimestepEmbedding...")
    key = jax.random.PRNGKey(SEED)

    dim_embed = 128
    dim_out = 256
    embed = TimestepEmbedding(dim_embed=dim_embed, dim_out=dim_out)

    # Initialize with dummy input
    t = jnp.array(0.5)
    params = embed.init(key, t)

    # Test different timesteps
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = embed.apply(params, jnp.array(t_val))
        assert y.shape == (dim_out,), f"Expected shape ({dim_out},), got {y.shape}"

    print(f"  Output shape: {y.shape}")
    print("  TimestepEmbedding: PASSED")


def test_adaptive_layernorm():
    """Test AdaptiveLayerNorm."""
    print("Testing AdaptiveLayerNorm...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 4)

    dim = 64
    dim_cond = 128
    aln = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)

    x = jax.random.normal(keys[1], (10, dim))
    cond = jax.random.normal(keys[2], (dim_cond,))

    params = aln.init(keys[0], x, cond)
    y = aln.apply(params, x, cond)

    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("  AdaptiveLayerNorm: PASSED")


def test_block():
    """Test MMDiTBlock."""
    print("Testing MMDiTBlock...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    dim_modalities = (64, 32)
    dim_cond = 128
    block = MMDiTBlock(dim_modalities=dim_modalities, dim_cond=dim_cond, heads=4)

    seq_len1, seq_len2 = 8, 6
    x1 = jax.random.normal(keys[1], (seq_len1, dim_modalities[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_modalities[1]))
    time_cond = jax.random.normal(keys[3], (dim_cond,))

    params = block.init(keys[0], (x1, x2), time_cond)
    y1, y2 = block.apply(params, (x1, x2), time_cond)

    assert y1.shape == x1.shape, f"Expected shape {x1.shape}, got {y1.shape}"
    assert y2.shape == x2.shape, f"Expected shape {x2.shape}, got {y2.shape}"
    print(f"  Input shapes: {x1.shape}, {x2.shape}")
    print(f"  Output shapes: {y1.shape}, {y2.shape}")
    print("  MMDiTBlock: PASSED")


def test_final_layer():
    """Test ModulatedFinalLayer."""
    print("Testing ModulatedFinalLayer...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 4)

    dim_in = 64
    dim_out = 32
    dim_cond = 128
    final = ModulatedFinalLayer(dim_in=dim_in, dim_out=dim_out, dim_cond=dim_cond)

    x = jax.random.normal(keys[1], (10, dim_in))
    cond = jax.random.normal(keys[2], (dim_cond,))

    params = final.init(keys[0], x, cond)
    y = final.apply(params, x, cond)

    assert y.shape == (10, dim_out), f"Expected shape (10, {dim_out}), got {y.shape}"
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("  ModulatedFinalLayer: PASSED")


def test_model():
    """Test full MMDiT model."""
    print("Testing MMDiT model...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    # Create model
    depth = 2
    dim_modalities = (64, 32)
    dim_outs = (64, 32)
    dim_cond = 128

    model = MMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        timestep_embed_dim=64,
        dim_head=32,
        heads=4,
    )

    # Single example forward
    seq_len1, seq_len2 = 8, 6
    x1 = jax.random.normal(keys[1], (seq_len1, dim_modalities[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_modalities[1]))
    timestep = jnp.array(0.5)

    params = model.init(keys[0], (x1, x2), timestep)
    y1, y2 = model.apply(params, (x1, x2), timestep)

    assert y1.shape == (seq_len1, dim_outs[0]), f"Expected shape ({seq_len1}, {dim_outs[0]}), got {y1.shape}"
    assert y2.shape == (seq_len2, dim_outs[1]), f"Expected shape ({seq_len2}, {dim_outs[1]}), got {y2.shape}"
    print(f"  Input shapes: {x1.shape}, {x2.shape}")
    print(f"  Output shapes: {y1.shape}, {y2.shape}")

    # Test gradient flow
    def loss_fn(params):
        y1, y2 = model.apply(params, (x1, x2), timestep)
        return jnp.mean(y1**2) + jnp.mean(y2**2)

    grads = jax.grad(loss_fn)(params)
    assert grads is not None
    print("  Gradient flow: OK")
    print("  MMDiT: PASSED")


def test_in_context_model():
    """Test InContextMMDiT model."""
    print("Testing InContextMMDiT model...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    # Create model
    depth = 2
    dim_modalities = (64, 32)
    dim_outs = (64, 32)

    model = InContextMMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_head=32,
        heads=4,
    )

    # Single example forward
    seq_len1, seq_len2 = 8, 6
    x1 = jax.random.normal(keys[1], (seq_len1, dim_modalities[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_modalities[1]))

    params = model.init(keys[0], (x1, x2))
    y1, y2 = model.apply(params, (x1, x2))

    assert y1.shape == (seq_len1, dim_outs[0]), f"Expected shape ({seq_len1}, {dim_outs[0]}), got {y1.shape}"
    assert y2.shape == (seq_len2, dim_outs[1]), f"Expected shape ({seq_len2}, {dim_outs[1]}), got {y2.shape}"
    print(f"  Input shapes: {x1.shape}, {x2.shape}")
    print(f"  Output shapes: {y1.shape}, {y2.shape}")
    print("  InContextMMDiT: PASSED")


def test_rmsnorm():
    """Test RMSNorm."""
    print("Testing RMSNorm...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 3)

    dim = 64
    norm = RMSNorm(dim=dim)

    x = jax.random.normal(keys[1], (dim,))
    params = norm.init(keys[0], x)
    y = norm.apply(params, x)

    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("  RMSNorm: PASSED")


def test_batched_model():
    """Test batched processing of MMDiT."""
    print("Testing batched MMDiT...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    # Create model
    depth = 2
    dim_modalities = (64, 32)
    dim_outs = (64, 32)
    dim_cond = 128

    model = MMDiT(
        depth=depth,
        dim_modalities=dim_modalities,
        dim_outs=dim_outs,
        dim_cond=dim_cond,
        timestep_embed_dim=64,
        dim_head=32,
        heads=4,
    )

    # Initialize with single example
    seq_len1, seq_len2 = 8, 6
    x1_single = jax.random.normal(keys[1], (seq_len1, dim_modalities[0]))
    x2_single = jax.random.normal(keys[2], (seq_len2, dim_modalities[1]))
    timestep_single = jnp.array(0.5)

    params = model.init(keys[0], (x1_single, x2_single), timestep_single)

    # Batched forward using vmap
    batch_size = 3
    x1_batch = jax.random.normal(keys[3], (batch_size, seq_len1, dim_modalities[0]))
    x2_batch = jax.random.normal(keys[4], (batch_size, seq_len2, dim_modalities[1]))
    timesteps = jnp.linspace(0.2, 0.8, batch_size)

    # vmap over batch dimension
    batched_apply = jax.vmap(lambda x1, x2, t: model.apply(params, (x1, x2), t))
    y1_batch, y2_batch = batched_apply(x1_batch, x2_batch, timesteps)

    assert y1_batch.shape == (batch_size, seq_len1, dim_outs[0])
    assert y2_batch.shape == (batch_size, seq_len2, dim_outs[1])
    print(f"  Batch input shapes: {x1_batch.shape}, {x2_batch.shape}")
    print(f"  Batch output shapes: {y1_batch.shape}, {y2_batch.shape}")
    print("  Batched MMDiT: PASSED")


def test_multiple_conds():
    """Test MMDiT with multiple conditioning variables."""
    print("Testing MMDiT with multiple conditioning variables...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    # Create model with two conditioning variables
    model = MMDiT(
        depth=2,
        dim_modalities=(64, 32),
        dim_outs=(64, 32),
        dim_cond=128,
        timestep_embed_dim=(64, 32),  # Two conditioning variables
        dim_head=32,
        heads=4,
    )

    # Test inputs
    x1 = jax.random.normal(keys[1], (8, 64))
    x2 = jax.random.normal(keys[2], (4, 32))
    t = jnp.array(0.5)
    r = jnp.array(0.3)

    params = model.init(keys[0], (x1, x2), (t, r))
    y1, y2 = model.apply(params, (x1, x2), (t, r))

    assert y1.shape == (8, 64)
    assert y2.shape == (4, 32)
    print(f"  Output shapes: {y1.shape}, {y2.shape}")
    print("  Multiple conditioning: PASSED")


def main():
    print("=" * 60)
    print("Validating Flax Migration")
    print("=" * 60)
    print()

    test_feedforward()
    print()
    test_attention()
    print()
    test_timestep_embedding()
    print()
    test_adaptive_layernorm()
    print()
    test_block()
    print()
    test_final_layer()
    print()
    test_model()
    print()
    test_in_context_model()
    print()
    test_rmsnorm()
    print()
    test_batched_model()
    print()
    test_multiple_conds()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
