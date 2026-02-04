"""
Utility functions for the mmdit library.
"""

from typing import Optional, Any


def exists(v: Any) -> bool:
    """Check if a value exists (is not None)."""
    return v is not None


def default(v: Any, d: Any) -> Any:
    """Return v if it exists, otherwise return d."""
    return v if exists(v) else d
