"""
BC_MMDiT: Flow Matching Imitation Learning Policy with Multi-Modal DiT.

This module implements a behavioral cloning policy using flow matching and
multi-modal diffusion transformer architecture in JAX/Equinox.
"""
import jax.numpy as jnp
from jaxtyping import Array

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array
import math

from mmdit_jax import MMDiT

def create_joint_attention_mask(obs_seq_len: int, action_seq_len: int) -> Array:
    """
    Create a joint attention mask for MM-DiT with causal action attention.

    The mask is structured as:
        - Observation tokens (rows 0 to obs_seq_len-1):
            - Can attend to ALL tokens (both obs and action) - bidirectional
        - Action tokens (rows obs_seq_len to total_seq_len-1):
            - Can attend to ALL observation tokens
            - Can only attend to PREVIOUS action tokens (causal within actions)

    Example for obs_seq_len=2, action_seq_len=4:
             Obs     |   Actions
           o0 o1  | a0 a1 a2 a3
        --------------------------------
    o0  |  1  1  |  1  1  1  1    (obs sees everything)
    o1  |  1  1  |  1  1  1  1
        --------------------------------
    a0  |  1  1  |  1  0  0  0    (action 0 sees obs + only itself)
    a1  |  1  1  |  1  1  0  0    (action 1 sees obs + a0, a1)
    a2  |  1  1  |  1  1  1  0
    a3  |  1  1  |  1  1  1  1

    Args:
        obs_seq_len: Number of observation tokens (state tokens for state-based,
                     or n_obs_steps * (num_cameras + 1) for image-based)
        action_seq_len: Number of action tokens (horizon)

    Returns:
        Boolean mask of shape (total_seq, total_seq) where total_seq = obs_seq_len + action_seq_len.
        mask[i, j] = True means query position i can attend to key position j.
    """
    total_seq = obs_seq_len + action_seq_len

    # Initialize full mask (all True)
    mask = jnp.ones((total_seq, total_seq), dtype=bool)

    # Create causal mask for action-to-action attention
    # Action tokens start at index obs_seq_len
    action_causal = jnp.tril(jnp.ones((action_seq_len, action_seq_len), dtype=bool))

    # Place causal mask in the action-to-action block (bottom-right quadrant)
    mask = mask.at[obs_seq_len:, obs_seq_len:].set(action_causal)

    return mask



def get_1d_sincos_pos_embed(embed_dim: int, length: int, max_period: int = 10000) -> Array:
    """
    Generate 1D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        length: Sequence length
        max_period: Maximum period for sinusoidal encoding
        
    Returns:
        Positional embeddings of shape (length, embed_dim)
    """
    positions = jnp.arange(length, dtype=jnp.float32)
    half = embed_dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    
    # Outer product: (length, 1) x (1, half) -> (length, half)
    args = positions[:, None] * freqs[None, :]
    
    # Concatenate cos and sin
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    
    # If embed_dim is odd, pad with zeros
    if embed_dim % 2 == 1:
        embedding = jnp.concatenate([embedding, jnp.zeros((length, 1))], axis=-1)
    
    return embedding


class BC_MMDiT(eqx.Module):
    """
    Flow Matching Imitation Learning Policy with Multi-Modal DiT.
    
    This model uses a multi-modal diffusion transformer to predict action sequences
    from state observations using flow matching. It processes state observations and
    noisy actions jointly through the transformer and predicts the denoised actions.
    
    Architecture:
        state_obs -> state_encoder -> + pos_embed -> |
                                                      | -> MMDiT -> action_head -> actions
        actions -> action_encoder -> + pos_embed ->   |
                                      (with noise)
    """

    # Model hyperparameters
    n_obs_steps: int
    state_dim: int
    horizon: int
    act_dim: int
    hidden_size: int
    sigma: float

    # Encoders
    state_encoder: eqx.nn.Linear
    action_encoder: eqx.nn.Linear

    # Positional embeddings (frozen)
    state_pos_embed: Array
    action_pos_embed: Array

    # Core transformer
    mmdit: MMDiT

    # Output projection
    # action_head: eqx.nn.Linear
    
    def __init__(
        self,
        n_obs_steps: int = 2,
        state_dim: int = 2,
        horizon: int = 16,
        act_dim: int = 2,
        hidden_size: int = 512,
        sigma: float = 0.0,
        depth: int = 12,
        dim_cond: int = 1024,
        timestep_embed_dim: int = 256,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initialize BC_MMDiT model.
        
        Args:
            n_obs_steps: Number of observation timesteps
            state_dim: State observation dimension
            horizon: Action prediction horizon (sequence length)
            act_dim: Action dimension
            hidden_size: Model hidden dimension
            sigma: Noise level for probability path (0.0 = optimal transport)
            depth: Number of transformer blocks
            dim_cond: Conditioning dimension for timestep embedding
            timestep_embed_dim: Sinusoidal timestep embedding dimension
            dim_head: Dimension per attention head
            heads: Number of attention heads
            ff_mult: Feedforward hidden dimension multiplier
            key: Random key for initialization
        """
        keys = jax.random.split(key, 4)
        
        # Store hyperparameters
        self.n_obs_steps = n_obs_steps
        self.state_dim = state_dim
        self.horizon = horizon
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.sigma = sigma
        
        # State encoder: state_dim -> hidden_size
        self.state_encoder = eqx.nn.Linear(state_dim, hidden_size, key=keys[0])
        
        # Action encoder: act_dim -> hidden_size
        self.action_encoder = eqx.nn.Linear(act_dim, hidden_size, key=keys[1])
        
        # Generate frozen sinusoidal positional embeddings
        self.state_pos_embed = get_1d_sincos_pos_embed(hidden_size, n_obs_steps)
        self.action_pos_embed = get_1d_sincos_pos_embed(hidden_size, 100 + horizon)[100:]

        # Multi-modal diffusion transformer
        # Two modalities: state tokens and action tokens
        self.mmdit = MMDiT(
            depth=depth,
            dim_modalities=(hidden_size, hidden_size),  # (state_dim, action_dim)
            dim_cond=dim_cond,
            dim_outs=(hidden_size, act_dim),
            timestep_embed_dim=timestep_embed_dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            key=keys[2],
        )
        
        # Action output head: hidden_size -> act_dim
        # Note: MMDiT already applies ModulatedFinalLayer internally (with time_cond).
        # The action_out from MMDiT has already been modulated and processed.
        # We just need a simple projection to action dimension.
        # self.action_head = eqx.nn.Linear(hidden_size, act_dim, key=keys[3])
    
    def conditional_prob_path(
        self,
        t: Array,
        z: Array,
        x0: Array,
        key: PRNGKeyArray,
    ) -> Array:
        """
        Sample from the conditional probability path.
        
        Implements: x_t = t * z + (1 - t) * x0 + sigma * noise
        
        Args:
            t: Timestep (scalar in [0, 1])
            z: Target sample (typically random noise or future action)
            x0: Source sample (ground truth action)
            key: Random key for noise generation
            
        Returns:
            Sample on the probability path at time t
        """
        # Ensure t is broadcastable
        t = jnp.atleast_1d(t).squeeze()
        
        # Linear interpolation path
        mean = t * z + (1.0 - t) * x0
        
        # Add Gaussian noise
        if self.sigma > 0.0:
            noise = jax.random.normal(key, z.shape) * self.sigma
            mean = mean + noise
            
        return mean
    
    def conditional_vel_field(self, z: Array, x0: Array) -> Array:
        """
        Compute the conditional velocity field (target for training).
        
        For flow matching with optimal transport (sigma=0), the velocity is:
        v_t = z - x0
        
        Args:
            z: Target sample
            x0: Source sample
            
        Returns:
            Velocity field
        """
        return z - x0
    
    def process_obs(self, state_obs: Array) -> Array:
        """
        Process state observations into tokens.
        
        Args:
            state_obs: State observations of shape (n_obs_steps, state_dim)
            
        Returns:
            State tokens of shape (n_obs_steps, hidden_size)
        """
        # Encode states: (n_obs_steps, state_dim) -> (n_obs_steps, hidden_size)
        state_tokens = jax.vmap(self.state_encoder)(state_obs)
        return state_tokens

    def forward_with_state_tokens(
        self,
        state_tokens: Array,
        actions: Array,
        x0: Array,
        t: Array,
        key: PRNGKeyArray,
    ) -> Tuple[Array, Array]:
        """
        Forward pass with pre-processed state tokens.
        
        This is optimized for inference where state tokens don't change
        across multiple sampling steps.
        
        Args:
            state_tokens: Pre-processed state tokens of shape (n_obs_steps, hidden_size)
            actions: Action samples (z) of shape (horizon, act_dim)
            x0: Ground truth actions of shape (horizon, act_dim)
            t: Timestep (scalar in [0, 1])
            key: Random key for probability path sampling
            
        Returns:
            Tuple of (state_tokens, predicted_actions):
                - state_tokens: Shape (n_obs_steps, hidden_size)
                - predicted_actions: Shape (horizon, act_dim)
        """
        # 1. Sample noisy actions from probability path
        x_t = self.conditional_prob_path(t, actions, x0, key)  # (horizon, act_dim)

        # 2. Encode actions
        action_tokens = jax.vmap(self.action_encoder)(x_t)  # (horizon, hidden_size)
        action_tokens = action_tokens + jax.lax.stop_gradient(self.action_pos_embed)

        # 3. Process through MMDiT with causal masking on actions
        # State tokens can attend to everything (bidirectional)
        # Action tokens: can attend to all states + causal within actions
        state_out, action_out = self.mmdit(
            modality_tokens=(state_tokens, action_tokens),
            timestep=t,
            attention_mask=create_joint_attention_mask(self.n_obs_steps, self.horizon),
        )

        # 4. Project action tokens to action space
        # predicted_actions = jax.vmap(self.action_head)(action_out)  # (horizon, act_dim)
        predicted_actions = action_out

        return state_out, predicted_actions

    def __call__(
        self,
        state_obs: Array,
        actions: Array,
        x0: Array,
        t: Array,
        key: PRNGKeyArray,
    ) -> Tuple[Array, Array]:
        """
        Forward pass of BC_MMDiT.
        
        This mirrors the PyTorch implementation's forward method:
        1. Process observations into state tokens
        2. Sample noisy actions via probability path
        3. Encode actions into action tokens
        4. Process through MMDiT
        5. Project to action space
        
        Args:
            state_obs: State observations of shape (n_obs_steps, state_dim)
            actions: Action samples (z) of shape (horizon, act_dim)
            x0: Ground truth actions of shape (horizon, act_dim)
            t: Timestep (scalar in [0, 1])
            key: Random key for probability path sampling
            
        Returns:
            Tuple of (state_tokens, predicted_actions):
                - state_tokens: Shape (n_obs_steps, hidden_size)
                - predicted_actions: Shape (horizon, act_dim)
        """
        # 1. Process state observations
        state_tokens = self.process_obs(state_obs)  # (n_obs_steps, hidden_size)
        state_tokens = state_tokens + jax.lax.stop_gradient(self.state_pos_embed)

        # 2-5. Use forward_with_state_tokens for remaining steps
        return self.forward_with_state_tokens(state_tokens, actions, x0, t, key)

