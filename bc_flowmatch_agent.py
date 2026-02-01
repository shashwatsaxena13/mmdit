"""
BCFlowMatchAgent: Flow Matching for Behavioral Cloning in JAX/Equinox.

This module implements a training and inference agent for flow matching
with multi-modal diffusion transformers.
"""

from typing import Dict, Tuple, Optional
from dataclasses import asdict
import json
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, PRNGKeyArray

from bc_mmdit_network import BC_MMDiT
from bc_flowmatch.configs.test_state_only_bc_config import AgentConfig
from device_mesh import DeviceMesh


class BCFlowMatchAgent:
    """
    Flow Matching agent for behavioral cloning.
    
    This agent handles:
    - Training with flow matching loss
    - Action padding masks for variable-length trajectories
    - EMA model updates via optax.incremental_update
    - Inference via Euler integration
    - Multi-device training (GPU/TPU) with data parallelism
    """
    
    def __init__(
        self,
        config: AgentConfig,
        key: PRNGKeyArray,
        mesh: Optional[DeviceMesh] = None,
    ):
        """
        Initialize BCFlowMatchAgent.
        
        Args:
            config: Agent configuration
            key: Random key for initialization
            mesh: Optional DeviceMesh for multi-device training.
                 If None, uses single-device training.
        """
        self.config = config
        self.mesh = mesh
        
        # Store normalization ranges as arrays
        self.state_min_max = (
            jnp.array(config.state_min_max[0]),
            jnp.array(config.state_min_max[1])
        )
        self.action_min_max = (
            jnp.array(config.action_min_max[0]),
            jnp.array(config.action_min_max[1])
        )
        
        # Store normalization and sampling functions from config
        # These are always provided (never None)
        self.normalize_fn = config.normalize_fn
        self.unnormalize_fn = config.unnormalize_fn
        self.sample_timestep_fn = config.sample_timestep_fn
        
        # Initialize model
        self.model = BC_MMDiT(
            n_obs_steps=config.n_obs_steps,
            state_dim=config.state_dim,
            horizon=config.horizon,
            act_dim=config.act_dim,
            hidden_size=config.hidden_size,
            sigma=config.sigma,
            depth=config.depth,
            dim_cond=config.dim_cond,
            timestep_embed_dim=config.timestep_embed_dim,
            dim_head=config.dim_head,
            heads=config.heads,
            ff_mult=config.ff_mult,
            key=key,
        )
        
        # Setup optimizer with weight decay
        self.optimizer = optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Initialize EMA parameters (copy of model parameters)
        model_params = eqx.filter(self.model, eqx.is_array)
        self.ema_params = jax.tree.map(lambda x: jnp.copy(x), model_params)
        
        # Training step counter
        self.step = 0
        
        # Count parameters
        param_count = sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(self.model, eqx.is_array))
        )
        print(f"Number of trainable parameters: {param_count / 1e6:.3f}M")
        
        # Replicate model and optimizer to all devices for multi-device training
        if mesh is not None:
            print(f"Replicating model to {mesh.device_count} devices...")
            self.model = mesh.replicate(self.model)
            self.ema_params = mesh.replicate(self.ema_params)
            self.opt_state = mesh.replicate(self.opt_state)
    
    def get_ema_model(self) -> BC_MMDiT:
        """
        Reconstruct EMA model from EMA parameters.
        
        Returns:
            BC_MMDiT model with EMA parameters
        """
        # Combine EMA params with model structure
        return eqx.combine(self.ema_params, eqx.filter(self.model, eqx.is_array, inverse=True))
    
    def loss_fn(
        self,
        model: BC_MMDiT,
        batch: Dict[str, Array],
        key: PRNGKeyArray,
    ) -> Array:
        """
        Compute flow matching loss with action masking.
        
        Args:
            model: BC_MMDiT model
            batch: Dictionary containing:
                - 'observation.state': (B, n_obs_steps, state_dim)
                - 'action': (B, horizon, act_dim)
                - 'action_is_pad': (B, horizon) boolean mask (True = padding)
            key: PRNG key
            
        Returns:
            Scalar loss value
        """
        keys = jax.random.split(key, 3)
        
        # Normalize observations and actions using configured function
        state_obs = self.normalize_fn(batch['observation.state'], self.state_min_max)
        actions = self.normalize_fn(batch['action'], self.action_min_max)
        action_is_pad = batch['action_is_pad']  # (B, horizon)
        
        B, horizon, act_dim = actions.shape
        
        # Sample x0 (random noise) - same shape as actions
        x0 = jax.random.normal(keys[0], actions.shape)
        
        # Sample timesteps using configured function
        t = self.sample_timestep_fn(keys[1], (B,), max_t=self.config.max_t)
        
        # Compute target velocity: u = z - x0 (where z is target actions)
        u = actions - x0  # (B, horizon, act_dim)
        
        # Forward pass through model (vmapped over batch)
        def single_forward(state, action, x0_i, t_i, key_i):
            _, pred = model(state, action, x0_i, t_i, key_i)
            return pred
        
        # Use eqx.filter_vmap for proper handling of Equinox modules
        batched_forward = eqx.filter_vmap(single_forward, in_axes=(0, 0, 0, 0, 0))
        v = batched_forward(
            state_obs,
            actions,
            x0,
            t,
            jax.random.split(keys[2], B)
        )  # (B, horizon, act_dim)
        
        # Compute MSE loss
        loss_per_element = (u - v) ** 2  # (B, horizon, act_dim)
        
        # Apply mask: zero out loss for padded actions
        # action_is_pad is True for padding, so we want (~action_is_pad)
        mask = ~action_is_pad  # (B, horizon) - True for valid actions
        mask_expanded = mask[..., None]  # (B, horizon, 1) for broadcasting
        
        masked_loss = loss_per_element * mask_expanded  # (B, horizon, act_dim)
        
        # Mean over all elements (padded actions already have zero loss)
        loss = jnp.mean(masked_loss)
        
        return loss
    
    def _make_train_step_jit(self):
        """
        Create a JIT-compiled training step function.
        
        This is created once and cached to avoid recompilation.
        Following Equinox best practices: put all numerical work in one JIT region.
        
        For multi-device training, this function automatically:
        - Applies sharding constraints to guide the compiler
        - Averages gradients across devices (via compiler-inserted collectives)
        - Returns synchronized loss values
        """
        # Capture static parts via closure
        # IMPORTANT: Capture function references, not bound methods or self
        normalize_fn = self.normalize_fn
        unnormalize_fn = self.unnormalize_fn
        sample_timestep_fn = self.sample_timestep_fn
        state_min_max = self.state_min_max
        action_min_max = self.action_min_max
        max_t = self.config.max_t
        optimizer = self.optimizer
        ema_decay = self.config.ema_decay
        mesh = self.mesh
        is_distributed = mesh is not None and mesh.is_distributed
        
        # Capture shardings for multi-device training
        if is_distributed:
            replicated_sharding = mesh.replicated_sharding
            batch_sharding = mesh.batch_sharding
        
        @eqx.filter_jit(donate="all")
        def train_step_jit(model, ema_params, opt_state, batch, key):
            """JIT-compiled training step operating on arrays only."""
            # Apply sharding constraints (tells compiler how to shard computation)
            # This is critical for gradient aggregation in multi-device training!
            if is_distributed:
                model, ema_params, opt_state = eqx.filter_shard(
                    (model, ema_params, opt_state), replicated_sharding
                )
                batch = eqx.filter_shard(batch, batch_sharding)
            
            # Define loss function inline to avoid bound method issues
            def compute_loss(model, batch, key):
                keys = jax.random.split(key, 3)
                
                # Normalize observations and actions
                state_obs = normalize_fn(batch['observation.state'], state_min_max)
                actions = normalize_fn(batch['action'], action_min_max)
                action_is_pad = batch['action_is_pad']
                
                B, horizon, act_dim = actions.shape
                
                # Sample x0 (random noise)
                x0 = jax.random.normal(keys[0], actions.shape)
                
                # Sample timesteps
                t = sample_timestep_fn(keys[1], (B,), max_t=max_t)
                
                # Compute target velocity
                u = actions - x0
                
                # Forward pass through model (vmapped over batch)
                def single_forward(state, action, x0_i, t_i, key_i):
                    _, pred = model(state, action, x0_i, t_i, key_i)
                    return pred
                
                batched_forward = eqx.filter_vmap(single_forward, in_axes=(0, 0, 0, 0, 0))
                v = batched_forward(
                    state_obs,
                    actions,
                    x0,
                    t,
                    jax.random.split(keys[2], B)
                )
                
                # Compute MSE loss with masking
                loss_per_element = (u - v) ** 2
                mask = ~action_is_pad
                mask_expanded = mask[..., None]
                masked_loss = loss_per_element * mask_expanded
                loss = jnp.mean(masked_loss)
                
                return loss
            
            # Compute loss and gradients
            loss, grads = eqx.filter_value_and_grad(compute_loss)(model, batch, key)
            
            # Update model parameters
            # Note: Gradients are automatically averaged across devices due to
            # sharding constraints above (compiler inserts collective operations)
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            
            # Update EMA parameters
            model_params = eqx.filter(new_model, eqx.is_array)
            new_ema_params = optax.incremental_update(
                new_tensors=model_params,
                old_tensors=ema_params,
                step_size=1.0 - ema_decay
            )
            
            # Apply sharding constraints to outputs (ensures they stay replicated)
            if is_distributed:
                new_model, new_ema_params, new_opt_state = eqx.filter_shard(
                    (new_model, new_ema_params, new_opt_state), replicated_sharding
                )
            
            return new_model, new_ema_params, new_opt_state, loss
        
        return train_step_jit
    
    def train_step(
        self,
        batch: Dict[str, Array],
        key: PRNGKeyArray,
    ) -> Tuple['BCFlowMatchAgent', Array]:
        """
        Perform one training step.
        
        Args:
            batch: Training batch
            key: PRNG key
            
        Returns:
            Tuple of (updated_agent, loss)
        """
        # Create JIT function if not cached
        if not hasattr(self, '_train_step_jit_cached'):
            self._train_step_jit_cached = self._make_train_step_jit()
        
        # Run JIT-compiled step
        model, ema_params, opt_state, loss = self._train_step_jit_cached(
            self.model, self.ema_params, self.opt_state, batch, key
        )
        
        # Create updated agent (fast, only copies Python object references)
        new_agent = BCFlowMatchAgent.__new__(BCFlowMatchAgent)
        new_agent.config = self.config
        new_agent.mesh = self.mesh
        new_agent.state_min_max = self.state_min_max
        new_agent.action_min_max = self.action_min_max
        new_agent.normalize_fn = self.normalize_fn
        new_agent.unnormalize_fn = self.unnormalize_fn
        new_agent.sample_timestep_fn = self.sample_timestep_fn
        new_agent.model = model
        new_agent.optimizer = self.optimizer
        new_agent.opt_state = opt_state
        new_agent.ema_params = ema_params
        new_agent.step = self.step + 1
        new_agent._train_step_jit_cached = self._train_step_jit_cached  # Carry over cached JIT
        
        return new_agent, loss
    
    def _make_sample_jit(self, sampling_steps: int):
        """
        Create a JIT-compiled sampling function for a given number of steps.
        
        Args:
            sampling_steps: Number of Euler integration steps
            
        Returns:
            JIT-compiled sampling function
        """
        # Capture static parts via closure
        normalize_fn = self.normalize_fn
        unnormalize_fn = self.unnormalize_fn
        state_min_max = self.state_min_max
        action_min_max = self.action_min_max
        horizon = self.config.horizon
        act_dim = self.config.act_dim
        max_t = self.config.max_t
        
        @eqx.filter_jit
        def sample_jit(model, state_obs_raw, key):
            """JIT-compiled sampling with Euler integration.
            
            Optimization: Process state observations only once before the integration loop,
            since they remain constant across all sampling steps.
            """
            # Normalize observations
            state_obs = normalize_fn(state_obs_raw, state_min_max)

            B = state_obs.shape[0]
            keys = jax.random.split(key, sampling_steps + 1)

            # Start with random noise
            a_hat = jax.random.normal(keys[0], (B, horizon, act_dim))

            # Time schedule
            t_schedule = jnp.linspace(0.0, max_t, sampling_steps + 1)

            # Pre-process state observations once (optimization for inference)
            # Process each state observation in the batch
            def process_single_state(state):
                state_tokens = model.process_obs(state)  # (n_obs_steps, hidden_size)
                state_tokens = state_tokens + model.state_pos_embed
                return state_tokens
            
            # Vmap over batch dimension
            state_tokens_batch = jax.vmap(process_single_state)(state_obs)  # (B, n_obs_steps, hidden_size)

            # Euler integration loop using scan
            def integration_step(a_hat, i):
                t = t_schedule[i]
                t_next = t_schedule[i + 1]
                dt = t_next - t

                # Predict velocity for each sample in batch using pre-processed state tokens
                def single_forward(state_tokens, action, key_i):
                    # Use action as both x_t and x0 (for inference)
                    _, v = model.forward_with_state_tokens(state_tokens, action, action, t, key_i)
                    return v

                # Use eqx.filter_vmap for batching
                batched_forward = eqx.filter_vmap(single_forward, in_axes=(0, 0, 0))
                v = batched_forward(
                    state_tokens_batch,
                    a_hat,
                    jax.random.split(keys[i + 1], B)
                )

                # Euler step
                a_hat_next = a_hat + v * dt
                # lax.scan requires (carry, output) tuple
                # We only need the final carry, so output is unused (hence the _ below)
                return a_hat_next, None

            # Scan over integration steps
            # Returns (final_carry, stacked_outputs), we only use final_carry
            a_hat_final, _ = jax.lax.scan(
                integration_step,
                a_hat,
                jnp.arange(sampling_steps)
            )

            # Unnormalize final result
            actions = unnormalize_fn(a_hat_final, action_min_max)

            return actions
        
        return sample_jit
    
    def sample(
        self,
        batch: Dict[str, Array],
        key: PRNGKeyArray,
        sampling_steps: int = 10,
        use_ema: bool = True,
    ) -> Array:
        """
        Sample actions using Euler integration.
        
        Args:
            batch: Batch containing observations
            key: PRNG key
            sampling_steps: Number of integration steps
            use_ema: Whether to use EMA model (default: True)
            
        Returns:
            Predicted actions of shape (B, horizon, act_dim)
        """
        # Cache JIT functions per sampling_steps value
        cache_key = f'_sample_jit_cached_{sampling_steps}'
        if not hasattr(self, cache_key):
            setattr(self, cache_key, self._make_sample_jit(sampling_steps))
        
        sample_jit = getattr(self, cache_key)
        
        # Select model (EMA or main)
        model = self.get_ema_model() if use_ema else self.model
        
        # Run JIT-compiled sampling
        actions = sample_jit(model, batch['observation.state'], key)
        
        return actions
    
    def save_checkpoint(
        self,
        filename: str,
        save_optimizer: bool = True,
    ) -> None:
        """
        Save agent checkpoint to file.
        
        Following Equinox serialization pattern:
        - Line 1: JSON with hyperparameters and metadata
        - Rest: Binary data with model weights, EMA params, and optionally optimizer state
        
        For multi-device training, this automatically extracts parameters from device 0
        to avoid saving redundant replicated copies.
        
        Args:
            filename: Path to save checkpoint
            save_optimizer: Whether to save optimizer state (default: True)
        
        Reference:
            https://docs.kidger.site/equinox/examples/serialisation/
        """
        # For multi-device: extract arrays to device 0 to avoid saving replicated copies
        # For single-device: this is a no-op
        def to_device_0(pytree):
            """Extract arrays to device 0 for saving."""
            return jax.tree_util.tree_map(
                lambda x: jax.device_get(x) if isinstance(x, jax.Array) else x,
                pytree
            )
        
        model_to_save = to_device_0(self.model)
        ema_to_save = to_device_0(self.ema_params)
        opt_state_to_save = to_device_0(self.opt_state) if save_optimizer else None
        
        # Prepare metadata and hyperparameters
        checkpoint_data = {
            'hyperparameters': self.config.to_serializable_dict(),
            'step': int(self.step),
            'save_optimizer': save_optimizer,
        }
        
        # Write to file
        with open(filename, 'wb') as f:
            # Write JSON metadata on first line
            metadata_str = json.dumps(checkpoint_data)
            f.write((metadata_str + '\n').encode())
            
            # Serialize model weights
            eqx.tree_serialise_leaves(f, model_to_save)
            
            # Serialize EMA parameters
            eqx.tree_serialise_leaves(f, ema_to_save)
            
            # Optionally serialize optimizer state
            if save_optimizer:
                eqx.tree_serialise_leaves(f, opt_state_to_save)
    
        print(f"Checkpoint saved to {filename}")

    @classmethod
    def load_checkpoint(
        cls,
        filename: str,
        key: PRNGKeyArray,
        mesh: Optional[DeviceMesh] = None,
    ) -> 'BCFlowMatchAgent':
        """
        Load agent checkpoint from file.
        
        Args:
            filename: Path to checkpoint file
            key: Random key for model initialization (only used for structure)
            mesh: Optional DeviceMesh for multi-device training.
                 If provided, replicates loaded parameters across devices.
            
        Returns:
            Loaded BCFlowMatchAgent with restored weights and state
            
        Reference:
            https://docs.kidger.site/equinox/examples/serialisation/
        """
        with open(filename, 'rb') as f:
            # Read and parse JSON metadata from first line
            checkpoint_data = json.loads(f.readline().decode())
            
            hyperparams = checkpoint_data['hyperparameters']
            step = checkpoint_data['step']
            has_optimizer = checkpoint_data.get('save_optimizer', True)
            
            # Create skeleton agent with same hyperparameters (without mesh for loading)
            config = AgentConfig.from_serializable_dict(hyperparams)
            skeleton = cls(config, key, mesh=None)
            
            # Deserialize model weights
            model = eqx.tree_deserialise_leaves(f, skeleton.model)
            
            # Deserialize EMA parameters
            ema_params = eqx.tree_deserialise_leaves(f, skeleton.ema_params)
            
            # Deserialize optimizer state if it was saved
            if has_optimizer:
                opt_state = eqx.tree_deserialise_leaves(f, skeleton.opt_state)
            else:
                # Use fresh optimizer state
                opt_state = skeleton.opt_state
        
        # Replicate to devices if mesh is provided
        if mesh is not None:
            print(f"Replicating loaded checkpoint to {mesh.device_count} devices...")
            model = mesh.replicate(model)
            ema_params = mesh.replicate(ema_params)
            opt_state = mesh.replicate(opt_state)
        
        # Create new agent instance with loaded state
        loaded_agent = cls.__new__(cls)
        loaded_agent.config = config
        loaded_agent.mesh = mesh
        loaded_agent.state_min_max = skeleton.state_min_max
        loaded_agent.action_min_max = skeleton.action_min_max
        loaded_agent.normalize_fn = skeleton.normalize_fn
        loaded_agent.unnormalize_fn = skeleton.unnormalize_fn
        loaded_agent.sample_timestep_fn = skeleton.sample_timestep_fn
        loaded_agent.model = model
        loaded_agent.optimizer = skeleton.optimizer
        loaded_agent.opt_state = opt_state
        loaded_agent.ema_params = ema_params
        loaded_agent.step = step
        
        print(f"Checkpoint loaded from {filename}")
        return loaded_agent

