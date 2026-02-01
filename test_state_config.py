from typing import Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from dataclasses import dataclass, field
from typing import Callable
import numpy as np

def normalize_min_max(x: Array, min_max: Tuple[Array, Array]) -> Array:
    """
    Normalize to [-1, 1] using min-max normalization.
    
    Formula: x_norm = 2 * (x - min) / (max - min) - 1
    
    Args:
        x: Input array
        min_max: Tuple of (min, max) arrays
        
    Returns:
        Normalized array in [-1, 1]
    """
    min_val, max_val = min_max
    # Normalize to [0, 1]
    x = (x - min_val) / (max_val - min_val + 1e-8)
    # Normalize to [-1, 1]
    x = 2.0 * x - 1.0
    return x


def unnormalize_min_max(x: Array, min_max: Tuple[Array, Array]) -> Array:
    """
    Unnormalize from [-1, 1] using min-max denormalization.
    
    Formula: x = (x_norm + 1) / 2 * (max - min) + min
    
    Args:
        x: Input array in [-1, 1]
        min_max: Tuple of (min, max) arrays
        
    Returns:
        Unnormalized array in [min, max]
    """
    min_val, max_val = min_max
    # Unnormalize from [-1, 1] to [0, 1]
    x = (x + 1.0) / 2.0
    # Unnormalize from [0, 1] to [min, max]
    x = x * (max_val - min_val) + min_val
    return x


def normalize_zscore(x: Array, stats: Tuple[Array, Array]) -> Array:
    """
    Normalize using Z-score normalization (standardization).
    
    Formula: x_norm = (x - mean) / (std + eps)
    
    This transforms data to have mean=0 and std=1.
    
    Args:
        x: Input array
        stats: Tuple of (mean, std) arrays
        
    Returns:
        Normalized array with mean≈0, std≈1
    """
    mean, std = stats
    return (x - mean) / (std + 1e-8)


def unnormalize_zscore(x: Array, stats: Tuple[Array, Array]) -> Array:
    """
    Unnormalize from Z-score normalization.
    
    Formula: x = x_norm * std + mean
    
    Args:
        x: Input array (Z-score normalized)
        stats: Tuple of (mean, std) arrays
        
    Returns:
        Unnormalized array in original scale
    """
    mean, std = stats
    return x * std + mean


def sample_uniform_timesteps(key: PRNGKeyArray, shape: Tuple[int, ...], max_t: float = 1.0) -> Array:
    """
    Sample timesteps uniformly from [0, max_t].
    
    This is the standard approach for flow matching, providing
    consistent training across the entire trajectory.
    
    Args:
        key: PRNG key
        shape: Output shape (e.g., (batch_size,))
        max_t: Maximum timestep value (default 1.0)
        
    Returns:
        Uniform timesteps in [0, max_t]
    """
    return jax.random.uniform(key, shape, minval=0.0, maxval=max_t)




# Function registry for serialization
FUNCTION_REGISTRY = {
    'normalize_min_max': normalize_min_max,
    'unnormalize_min_max': unnormalize_min_max,
    'normalize_zscore': normalize_zscore,
    'unnormalize_zscore': unnormalize_zscore,
    'sample_uniform_timesteps': sample_uniform_timesteps,
}

# Allow registration of additional functions at runtime
def register_function(name: str, func: Callable):
    """Register a custom function for use in configs."""
    FUNCTION_REGISTRY[name] = func

def get_function(name: str) -> Callable:
    """Get a function from the registry by name."""
    if name not in FUNCTION_REGISTRY:
        raise ValueError(f"Function '{name}' not found in registry. Available: {list(FUNCTION_REGISTRY.keys())}")
    return FUNCTION_REGISTRY[name]


@dataclass
class AgentConfig:
    """Configuration for BCFlowMatchAgent."""
    # Model architecture
    n_obs_steps: int = 2
    state_dim: int = 4
    horizon: int = 16
    act_dim: int = 2
    hidden_size: int = 256
    depth: int = 8
    dim_cond: int = 1024
    timestep_embed_dim: int = 256
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    ema_decay: float = 0.999
    sigma: float = 0.0
    max_t: float = 1.0  # Maximum timestep for uniform sampling
    
    # Normalization ranges (per-dimension min/max as numpy arrays of shape (2, dim))
    # First row is minimums, second row is maximums
    state_min_max: np.ndarray = field(default_factory=lambda: np.array([[-1.0], [1.0]], dtype=np.float32))
    action_min_max: np.ndarray = field(default_factory=lambda: np.array([[-1.0], [1.0]], dtype=np.float32))
    
    # Normalization functions (required, default to min-max normalization)
    normalize_fn: Callable = normalize_min_max
    unnormalize_fn: Callable = unnormalize_min_max
    
    # Timestep sampling function (required, default to uniform sampling)
    sample_timestep_fn: Callable = sample_uniform_timesteps
    
    def get_function_name(self, func: Callable) -> str:
        """Get the registered name of a function."""
        for name, registered_func in FUNCTION_REGISTRY.items():
            if func is registered_func:
                return name
        raise ValueError(f"Function {func} is not registered in FUNCTION_REGISTRY")
    
    def to_serializable_dict(self) -> dict:
        """Convert config to a JSON-serializable dictionary."""
        data = {}
        for key, value in self.__dict__.items():
            if callable(value):
                # Save function name instead of function object
                data[key] = self.get_function_name(value)
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays to lists
                data[key] = value.tolist()
            else:
                data[key] = value
        return data
    
    @classmethod
    def from_serializable_dict(cls, data: dict) -> 'AgentConfig':
        """Create config from a serializable dictionary."""
        # Reconstruct numpy arrays
        if 'state_min_max' in data and isinstance(data['state_min_max'], list):
            data['state_min_max'] = np.array(data['state_min_max'], dtype=np.float32)
        if 'action_min_max' in data and isinstance(data['action_min_max'], list):
            data['action_min_max'] = np.array(data['action_min_max'], dtype=np.float32)
        
        # Reconstruct functions from names
        for key in ['normalize_fn', 'unnormalize_fn', 'sample_timestep_fn']:
            if key in data and isinstance(data[key], str):
                data[key] = get_function(data[key])
        
        return cls(**data)

