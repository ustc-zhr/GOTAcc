"""Policy registration and discovery for GOTAcc interfaces."""

from .builtins import POLICY_REGISTRY
from .registry import PolicyDefinition, PolicyRegistry

__all__ = [
    "POLICY_REGISTRY",
    "PolicyDefinition",
    "PolicyRegistry",
]
