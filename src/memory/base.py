from typing import List, Dict, Any
from .schema import MemoryItem
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """
    Abstract base class for memory management.
    Defines the interface for storage, retrieval, and trace compression.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def add(self, item: MemoryItem) -> None:
        """
        Public Interface: Agent calls this to store memory.
        """
        # 1. save (and calculate tokens if needed)
        self._add_to_storage(item)
        
        # 2. prune/compress if limits exceeded
        self._manage_memory_constraints()
        
    def get_chat_messages(self) -> List[Dict[str, Any]]:
        """
        Convert memory content directly into messages content that can be passed to the OpenAI API.
        """
        # Call retrieve with default args (get everything or strictly relevant context)
        items = self.retrieve() 
        
        messages = []
        for item in items:
            # Expand tool chains into flat OpenAI messages
            messages.extend(item.to_openai_messages())
            
        return messages

    @abstractmethod
    def _add_to_storage(self, item: MemoryItem) -> None:
        """Internal implementation of storage (List, VectorDB, etc.)"""
        pass

    @abstractmethod
    def _manage_memory_constraints(self) -> None:
        """Internal Logic: Decides WHEN and HOW to prune/compress memories."""
        pass

    @abstractmethod
    def retrieve(self, query: str = None, top_k: int = 5) -> List[MemoryItem]:
        """
        Retrieve relevant memories. 
        Note: FIFO memory might ignore query/top_k and just return window.
        """
        pass

    @abstractmethod
    def get_prompt_context(self, query: str = None) -> str:
        """Format retrieved memories into a single string (useful for local LLMs or RAG context)."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories (reset state)."""
        pass