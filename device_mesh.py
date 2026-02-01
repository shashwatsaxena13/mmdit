"""
Device mesh utilities for multi-GPU/TPU training with JAX.

This module provides utilities for managing device topology and sharding
strategies for data-parallel training (DDP) on multiple GPUs or TPUs.
"""

from typing import Optional, Any, Tuple
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


class DeviceMesh:
    """
    Manager for JAX device mesh and sharding strategies.
    
    This class handles:
    - Device topology setup for multi-GPU/TPU training
    - Sharding specifications for data parallelism (DDP)
    - Backward compatibility with single-device training
    
    Example:
        >>> # Multi-device training
        >>> mesh = DeviceMesh()  # Auto-detects all devices
        >>> print(f"Training on {mesh.device_count} devices")
        >>> 
        >>> # Shard batch across devices
        >>> batch = mesh.shard_batch(batch)
        >>> 
        >>> # Replicate model parameters
        >>> model = mesh.replicate(model)
    """
    
    def __init__(
        self,
        devices: Optional[list] = None,
        axis_name: str = 'batch',
    ):
        """
        Initialize device mesh.
        
        Args:
            devices: List of JAX devices to use. If None, uses all available devices.
            axis_name: Name for the mesh axis (default: 'batch' for data parallelism)
        """
        # Get devices
        if devices is None:
            devices = jax.devices()
        
        self.devices = devices
        self.device_count = len(devices)
        self.axis_name = axis_name
        
        # Create mesh for multi-device training
        if self.device_count > 1:
            self.mesh = Mesh(devices, axis_names=(axis_name,))
            self.is_distributed = True
        else:
            # Single device - no mesh needed
            self.mesh = None
            self.is_distributed = False
        
        # Create sharding specs
        self._create_sharding_specs()
        
        # Log device info
        self.device_type = devices[0].platform.lower()  # 'gpu', 'tpu', or 'cpu'
        device_type_display = self.device_type.upper()
        if self.is_distributed:
            print(f"🚀 Distributed training initialized: {self.device_count} {device_type_display}s")
            print(f"   Mesh axis: '{axis_name}'")
            print(f"   Devices: {[d.id for d in devices]}")
        else:
            print(f"📍 Single-device training: {device_type_display} {devices[0].id}")
    
    def _create_sharding_specs(self):
        """Create sharding specifications for different use cases."""
        if not self.is_distributed:
            # Single device - no sharding needed
            self.replicated_sharding = None
            self.batch_sharding = None
            return
        
        # Replicated sharding: same data on all devices (for model parameters)
        self.replicated_sharding = NamedSharding(self.mesh, P())
        
        # Batch sharding: split batch dimension across devices (for data)
        self.batch_sharding = NamedSharding(self.mesh, P(self.axis_name))
    
    def replicate(self, pytree: Any) -> Any:
        """
        Replicate a pytree (e.g., model parameters) across all devices.
        
        Uses eqx.filter_shard to properly handle Equinox modules that may
        contain non-array components (like activation functions).
        
        Args:
            pytree: Any JAX pytree (arrays, dicts, lists, etc.)
            
        Returns:
            Replicated pytree
        """
        if not self.is_distributed:
            return pytree
        
        # Use eqx.filter_shard for Equinox compatibility
        import equinox as eqx
        return eqx.filter_shard(pytree, self.replicated_sharding)
    
    def shard_batch(self, batch: Any) -> Any:
        """
        Shard a batch across devices on the batch dimension.
        
        For data parallelism, this splits the batch so each device gets
        a portion of the data.
        
        Args:
            batch: Batch pytree where first dimension is batch size
            
        Returns:
            Sharded batch
            
        Note:
            The batch size must be divisible by the number of devices.
            If using this function, ensure your dataloader produces batches
            with size = per_device_batch_size * device_count
        """
        if not self.is_distributed:
            return batch
        
        return jax.device_put(batch, self.batch_sharding)
    
    def get_global_batch_size(self, per_device_batch_size: int) -> int:
        """
        Get the global batch size from per-device batch size.
        
        Args:
            per_device_batch_size: Batch size per device
            
        Returns:
            Global batch size across all devices
        """
        return per_device_batch_size * self.device_count
    
    def get_per_device_batch_size(self, global_batch_size: int) -> int:
        """
        Get the per-device batch size from global batch size.
        
        Args:
            global_batch_size: Total batch size across all devices
            
        Returns:
            Batch size per device
            
        Raises:
            ValueError: If global_batch_size is not divisible by device_count
        """
        if global_batch_size % self.device_count != 0:
            raise ValueError(
                f"Global batch size ({global_batch_size}) must be divisible "
                f"by device count ({self.device_count})"
            )
        return global_batch_size // self.device_count
    
    def gather(self, pytree: Any) -> Any:
        """
        Gather sharded data from all devices.
        
        This is useful for collecting metrics or predictions from all devices.
        
        Args:
            pytree: Sharded pytree
            
        Returns:
            Gathered pytree on a single device
        """
        if not self.is_distributed:
            return pytree
        
        # For sharded arrays, this will concatenate along the batch dimension
        # For replicated arrays, this will just return the value
        return jax.tree.map(lambda x: jnp.asarray(x), pytree)
    
    def local_device_id(self) -> int:
        """Get the ID of the current device (for process-local operations)."""
        return jax.local_devices()[0].id
    
    def get_prefetch_size(self) -> int:
        """
        Get recommended prefetch size based on device type.
        
        GPU benefits from prefetching (overlap data transfer with compute).
        TPU handles data transfer differently and doesn't need prefetching.
        
        Returns:
            Recommended prefetch size (0 = no prefetching)
        """
        if self.device_type == 'gpu':
            return 2  # Recommended for GPUs
        else:
            # TPU and CPU don't need prefetching
            return 0
    
    def barrier(self):
        """
        Synchronization barrier across all devices.
        
        This ensures all devices have completed their current work before proceeding.
        Useful before checkpointing or evaluation.
        """
        if not self.is_distributed:
            return
        
        # Use a dummy all-reduce operation as a barrier
        dummy = jnp.array(1.0)
        jax.lax.psum(dummy, axis_name=self.axis_name)
    
    def __repr__(self) -> str:
        """String representation of the device mesh."""
        if self.is_distributed:
            return (f"DeviceMesh(devices={self.device_count}, "
                   f"type={self.devices[0].platform.upper()}, "
                   f"axis='{self.axis_name}')")
        else:
            return f"DeviceMesh(single_device={self.devices[0].platform.upper()})"


def auto_detect_mesh(
    preferred_device_count: Optional[int] = None,
    axis_name: str = 'batch',
) -> DeviceMesh:
    """
    Auto-detect and create an appropriate device mesh.
    
    Args:
        preferred_device_count: If specified, use only this many devices.
                              If None, uses all available devices.
        axis_name: Name for the mesh axis (default: 'batch')
        
    Returns:
        DeviceMesh configured for the available hardware
        
    Example:
        >>> # Use all available devices
        >>> mesh = auto_detect_mesh()
        >>> 
        >>> # Use only 4 devices
        >>> mesh = auto_detect_mesh(preferred_device_count=4)
    """
    available_devices = jax.devices()
    
    if preferred_device_count is not None:
        if preferred_device_count > len(available_devices):
            print(f"⚠️  Warning: Requested {preferred_device_count} devices, "
                  f"but only {len(available_devices)} available. Using all available devices.")
            devices = available_devices
        else:
            devices = available_devices[:preferred_device_count]
            if preferred_device_count < len(available_devices):
                print(f"ℹ️  Using {preferred_device_count} of {len(available_devices)} available devices")
    else:
        devices = available_devices
    
    return DeviceMesh(devices=devices, axis_name=axis_name)


def validate_batch_size(
    batch_size: int,
    device_count: int,
    raise_error: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that batch size is compatible with device count.
    
    Args:
        batch_size: Batch size to validate
        device_count: Number of devices
        raise_error: If True, raises ValueError on validation failure
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Raises:
        ValueError: If batch size is invalid and raise_error=True
    """
    # if batch_size % device_count != 0:
    #     msg = (f"Batch size ({batch_size}) must be divisible by device count ({device_count}). ",
    #            f"Suggested batch sizes: {[batch_size + i - (batch_size % device_count) 
    #                                       for i in range(1, device_count + 1)]}")
    #     if raise_error:
    #         raise ValueError(msg)
    #     return False, msg
    if batch_size % device_count != 0:
        suggested = [
            batch_size + i - (batch_size % device_count)
            for i in range(1, device_count + 1)
        ]
        msg = (
            f"Batch size ({batch_size}) must be divisible by device count ({device_count}). "
            f"Suggested batch sizes: {suggested}"
        )
        if raise_error:
            raise ValueError(msg)
        return False, msg

    return True, None

