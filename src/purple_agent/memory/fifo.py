from typing import List, Optional, Callable
from .base import BaseMemory
from .schema import MemoryItem


class FIFOMemory(BaseMemory):
    def __init__(self, 
                 max_tokens: int = 2000, 
                 max_items: int = 100,
                 token_counter: Optional[Callable[[str], int]] = None):
        """
        Args:
            max_tokens: the maximum token number of all the memory texts.
            max_items: the maximum memory blocks we preserve. tokens of the system prompt are not included.
            token_counter: estimator. if not provided, use chars / 4.
        """
        super().__init__()
        self.storage: List[MemoryItem] = [] # 除了system prompt以外的记忆 - 便于FIFO操作
        self.max_tokens = max_tokens
        self.max_items = max_items
        
        self.current_total_tokens = 0
        self.system_prompt: Optional[MemoryItem] = None 
        
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

    def _manage_memory_constraints(self) -> None:
        """
        Triggered automatically after every add().
        """
        while (self.current_total_tokens > self.max_tokens or 
               len(self.storage) > self.max_items) and self.storage:
            
            removed_item = self.storage.pop(0)

            # Defensive coding: use 0 if None (though add logic guarantees it's set)
            count = removed_item.token_count or 0
            self.current_total_tokens -= count
            
            # Debug log (Optional)
            print(f"Pruned item {removed_item.id}, freed {count} tokens. Current: {self.current_total_tokens}")

    def retrieve(self, query: str = None, top_k: int = 5) -> List[MemoryItem]:
        """
        FIFO ignores query and top_k, returning the sliding window.
        """
        memories = [self.system_prompt]
        memories.extend(self.storage)
        return self.storage

    def get_prompt_context(self, query: str = None) -> str:
        """
        Returns a plain text representation of the memory.
        Useful for models that don't support the ChatML (list of dicts) format.
        """
        items = self.retrieve()
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
