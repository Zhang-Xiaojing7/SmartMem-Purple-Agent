import os

from .fifo import FIFOMemory
from .lru import LRUMemory
from .semantic import SemanticMemory
from .schema import MemoryItem, ToolInteraction
from .base import BaseMemory

__all__ = [
    "BaseMemory",
    "FIFOMemory",
    "LRUMemory",
    "SemanticMemory",
    "MemoryItem",
    "ToolInteraction"
]

MEMORY_REGISTRY = {
    "fifo": FIFOMemory,
    "lru": LRUMemory,
    "semantic": SemanticMemory,
}

def get_memory_manager(config_name: str = None, **kwargs):

    target_name = config_name or os.getenv("MEMORY_MANAGER_TYPE", "fifo")

    memory_class = MEMORY_REGISTRY.get(target_name)

    if not memory_class:
        raise ValueError(f"Unknown memory type: '{target_name}'. Available: {list(MEMORY_REGISTRY.keys())}")

    print(f"[System] Initializing Memory Manager: {target_name}")

    return memory_class(**kwargs)