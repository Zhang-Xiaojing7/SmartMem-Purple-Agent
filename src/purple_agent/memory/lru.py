"""
LRU (Least Recently Used) Memory Manager

This memory strategy tracks access patterns and prioritizes keeping
recently accessed memories. When memory limits are exceeded, it removes
the least recently accessed items first.

Key features:
- Tracks last access time for each memory item
- Prioritizes keeping recently accessed memories
- Better for scenarios where recent context is more important
"""

from typing import List, Optional, Callable, Dict
from datetime import datetime
from .base import BaseMemory
from .schema import MemoryItem


class LRUMemory(BaseMemory):
    def __init__(self,
                 max_tokens: int = 2000,
                 max_items: int = 100,
                 token_counter: Optional[Callable[[str], int]] = None):
        """
        Args:
            max_tokens: Maximum token count for all memory texts
            max_items: Maximum number of memory blocks to preserve
            token_counter: Token estimator function. If not provided, uses chars / 4
        """
        super().__init__()
        self.storage: List[MemoryItem] = []
        self.max_tokens = max_tokens
        self.max_items = max_items

        self.current_total_tokens = 0
        self.system_prompt: Optional[MemoryItem] = None

        # Track last access time for each memory item by ID
        self.access_times: Dict[str, datetime] = {}

        # Default estimator
        self.token_counter = token_counter or (lambda s: len(s) // 4)

    def set_system_prompt(self, content: str):
        self.system_prompt = MemoryItem(role="system", content=content)

    def _add_to_storage(self, item: MemoryItem) -> None:
        if item.token_count is None:
            text_to_count = item.get_display_content() or ""
            count = self.token_counter(text_to_count)
            item.token_count = count
        else:
            count = item.token_count

        self.storage.append(item)
        self.current_total_tokens += count

        # Record access time
        self.access_times[item.id] = datetime.now()

    def _manage_memory_constraints(self) -> None:
        """
        Remove least recently used items when constraints are exceeded.
        """
        while (self.current_total_tokens > self.max_tokens or
               len(self.storage) > self.max_items) and self.storage:

            # Find the least recently used item
            lru_item = min(
                self.storage,
                key=lambda x: self.access_times.get(x.id, datetime.min)
            )

            # Remove it
            self.storage.remove(lru_item)
            count = lru_item.token_count or 0
            self.current_total_tokens -= count

            # Clean up access time tracking
            if lru_item.id in self.access_times:
                del self.access_times[lru_item.id]

            print(f"[LRU] Pruned item {lru_item.id}, freed {count} tokens. Current: {self.current_total_tokens}")

    def retrieve(self, query: str = None, top_k: int = 5) -> List[MemoryItem]:
        """
        Retrieve memories and update their access times.
        LRU returns all items but updates access times for tracking.
        """
        # Update access times for all retrieved items
        current_time = datetime.now()
        for item in self.storage:
            self.access_times[item.id] = current_time

        return self.storage

    def get_chat_messages(self) -> List[Dict]:
        """
        Override to include system prompt and update access times.
        """
        items = self.retrieve()

        messages = []
        if self.system_prompt:
            messages.extend(self.system_prompt.to_openai_messages())

        for item in items:
            messages.extend(item.to_openai_messages())

        return messages

    def get_prompt_context(self, query: str = None) -> str:
        """
        Returns a plain text representation of the memory.
        """
        items = self.retrieve()
        if not items:
            return None

        context_str = ""
        for item in items:
            role = item.role.capitalize()
            content = item.get_display_content()
            context_str += f"{role}: {content}\n"
        return context_str

    def clear(self) -> None:
        """Reset state."""
        self.storage = []
        self.current_total_tokens = 0
        self.access_times = {}
