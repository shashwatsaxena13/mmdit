"""
Script to capture golden outputs from Equinox implementation for migration validation.
"""
import json
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from mmdit_jax import MMDiT, InContextMMDiT
from mmdit_jax.blocks import MMDiTBlock
from mmdit_jax.attention import JointAttention
from mmdit_jax.feedforward import FeedForward
from mmdit_jax.layers import TimestepEmbedding, AdaptiveLayerNorm
from mmdit_jax.final_layer import ModulatedFinalLayer
from mmdit_jax.in_context_final_layer import SimpleFinalLayer, RMSNorm

# Fixed seed for reproducibility
SEED = 42


def save_array(arr, name, output_dict):
    """Save array to output dict as list."""
    output_dict[name] = {
        'shape': list(arr.shape),
        'values': arr.tolist(),
        'dtype': str(arr.dtype)
    }


def capture_feedforward_outputs():
    """Capture FeedForward layer outputs."""
    print("Capturing FeedForward outputs...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 3)

    outputs = {}

    # Test case 1: Basic forward
    dim = 64
    ff = FeedForward(dim=dim, mult=4, key=keys[0])
    x = jax.random.normal(keys[1], (8, dim))
    y = ff(x)

    save_array(x, 'feedforward_input', outputs)
    save_array(y, 'feedforward_output', outputs)

    # Save weights for initialization comparison
    save_array(ff.linear_in.weight, 'feedforward_linear_in_weight', outputs)
    save_array(ff.linear_gate.weight, 'feedforward_linear_gate_weight', outputs)
    save_array(ff.linear_out.weight, 'feedforward_linear_out_weight', outputs)

    return outputs


def capture_attention_outputs():
    """Capture JointAttention outputs."""
    print("Capturing JointAttention outputs...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    outputs = {}

    # Test case 1: Two modalities
    dim_inputs = (64, 32)
    layer = JointAttention(dim_inputs=dim_inputs, dim_head=32, heads=4, key=keys[0])

    seq_len1, seq_len2 = 8, 6
    x1 = jax.random.normal(keys[1], (seq_len1, dim_inputs[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_inputs[1]))

    y1, y2 = layer((x1, x2))

    save_array(x1, 'attention_input_0', outputs)
    save_array(x2, 'attention_input_1', outputs)
    save_array(y1, 'attention_output_0', outputs)
    save_array(y2, 'attention_output_1', outputs)

    # Test case 2: With attention mask
    total_seq = seq_len1 + seq_len2
    attention_mask = jnp.ones((total_seq, total_seq), dtype=bool)
    action_causal = jnp.tril(jnp.ones((seq_len2, seq_len2), dtype=bool))
    attention_mask = attention_mask.at[seq_len1:, seq_len1:].set(action_causal)

    y1_masked, y2_masked = layer((x1, x2), attention_mask=attention_mask)

    save_array(y1_masked, 'attention_output_masked_0', outputs)
    save_array(y2_masked, 'attention_output_masked_1', outputs)

    return outputs


def capture_timestep_embedding_outputs():
    """Capture TimestepEmbedding outputs."""
    print("Capturing TimestepEmbedding outputs...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 3)

    outputs = {}

    dim_embed = 128
    dim_out = 256
    embed = TimestepEmbedding(dim_embed=dim_embed, dim_out=dim_out, key=keys[0])

    # Test different timesteps
    timesteps = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

    for i, t in enumerate(timesteps):
        y = embed(t)
        save_array(y, f'timestep_embed_t{i}', outputs)

    return outputs


def capture_adaptive_layernorm_outputs():
    """Capture AdaptiveLayerNorm outputs."""
    print("Capturing AdaptiveLayerNorm outputs...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 4)

    outputs = {}

    dim = 64
    dim_cond = 128
    aln = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond, key=keys[0])

    x = jax.random.normal(keys[1], (10, dim))
    cond = jax.random.normal(keys[2], (dim_cond,))

    y = aln(x, cond)

    save_array(x, 'aln_input', outputs)
    save_array(cond, 'aln_condition', outputs)
    save_array(y, 'aln_output', outputs)

    return outputs


def capture_block_outputs():
    """Capture MMDiTBlock outputs."""
    print("Capturing MMDiTBlock outputs...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    outputs = {}

    dim_modalities = (64, 32)
    dim_cond = 128
    block = MMDiTBlock(dim_modalities=dim_modalities, dim_cond=dim_cond, heads=4, key=keys[0])

    seq_len1, seq_len2 = 8, 6
    x1 = jax.random.normal(keys[1], (seq_len1, dim_modalities[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_modalities[1]))
    time_cond = jax.random.normal(keys[3], (dim_cond,))

    y1, y2 = block((x1, x2), time_cond)

    save_array(x1, 'block_input_0', outputs)
    save_array(x2, 'block_input_1', outputs)
    save_array(time_cond, 'block_time_cond', outputs)
    save_array(y1, 'block_output_0', outputs)
    save_array(y2, 'block_output_1', outputs)

    return outputs


def capture_final_layer_outputs():
    """Capture ModulatedFinalLayer outputs."""
    print("Capturing ModulatedFinalLayer outputs...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 4)

    outputs = {}

    dim_in = 64
    dim_out = 32
    dim_cond = 128
    final = ModulatedFinalLayer(dim_in=dim_in, dim_out=dim_out, dim_cond=dim_cond, key=keys[0])

    x = jax.random.normal(keys[1], (10, dim_in))
    cond = jax.random.normal(keys[2], (dim_cond,))

    y = final(x, cond)

    save_array(x, 'final_layer_input', outputs)
    save_array(cond, 'final_layer_cond', outputs)
    save_array(y, 'final_layer_output', outputs)

    return outputs


def capture_model_outputs():
    """Capture full MMDiT model outputs."""
    print("Capturing MMDiT model outputs...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    outputs = {}

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
        key=keys[0]
    )

    # Single example forward
    seq_len1, seq_len2 = 8, 6
    x1 = jax.random.normal(keys[1], (seq_len1, dim_modalities[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_modalities[1]))
    timestep = jnp.array(0.5)

    y1, y2 = model((x1, x2), timestep)

    save_array(x1, 'model_input_0', outputs)
    save_array(x2, 'model_input_1', outputs)
    save_array(y1, 'model_output_0', outputs)
    save_array(y2, 'model_output_1', outputs)

    # Batched forward
    batch_size = 3
    model_batched = eqx.filter_vmap(model, in_axes=0)

    x1_batch = jax.random.normal(keys[3], (batch_size, seq_len1, dim_modalities[0]))
    x2_batch = jax.random.normal(keys[4], (batch_size, seq_len2, dim_modalities[1]))
    timesteps = jnp.linspace(0.2, 0.8, batch_size)

    y1_batch, y2_batch = model_batched((x1_batch, x2_batch), timesteps)

    save_array(x1_batch, 'model_input_batch_0', outputs)
    save_array(x2_batch, 'model_input_batch_1', outputs)
    save_array(y1_batch, 'model_output_batch_0', outputs)
    save_array(y2_batch, 'model_output_batch_1', outputs)

    return outputs


def capture_in_context_model_outputs():
    """Capture InContextMMDiT model outputs."""
    print("Capturing InContextMMDiT model outputs...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 5)

    outputs = {}

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
        key=keys[0]
    )

    # Single example forward
    seq_len1, seq_len2 = 8, 6
    x1 = jax.random.normal(keys[1], (seq_len1, dim_modalities[0]))
    x2 = jax.random.normal(keys[2], (seq_len2, dim_modalities[1]))

    y1, y2 = model((x1, x2))

    save_array(x1, 'ic_model_input_0', outputs)
    save_array(x2, 'ic_model_input_1', outputs)
    save_array(y1, 'ic_model_output_0', outputs)
    save_array(y2, 'ic_model_output_1', outputs)

    return outputs


def capture_rmsnorm_outputs():
    """Capture RMSNorm outputs."""
    print("Capturing RMSNorm outputs...")
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 3)

    outputs = {}

    dim = 64
    norm = RMSNorm(dim=dim, key=keys[0])

    x = jax.random.normal(keys[1], (10, dim))
    y = norm(x)

    save_array(x, 'rmsnorm_input', outputs)
    save_array(y, 'rmsnorm_output', outputs)

    return outputs


def main():
    print("=" * 60)
    print("Capturing Golden Outputs for Flax Migration Validation")
    print("=" * 60)

    all_outputs = {}

    all_outputs['feedforward'] = capture_feedforward_outputs()
    all_outputs['attention'] = capture_attention_outputs()
    all_outputs['timestep_embedding'] = capture_timestep_embedding_outputs()
    all_outputs['adaptive_layernorm'] = capture_adaptive_layernorm_outputs()
    all_outputs['block'] = capture_block_outputs()
    all_outputs['final_layer'] = capture_final_layer_outputs()
    all_outputs['model'] = capture_model_outputs()
    all_outputs['in_context_model'] = capture_in_context_model_outputs()
    all_outputs['rmsnorm'] = capture_rmsnorm_outputs()

    # Save to file
    output_path = '/data/user_data/ssaxena2/mmdit_test_outputs/golden_outputs.json'
    with open(output_path, 'w') as f:
        json.dump(all_outputs, f, indent=2)

    print(f"\nGolden outputs saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
