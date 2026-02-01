"""
Semantic Memory Manager

This memory strategy uses semantic similarity to retrieve relevant memories.
It maintains a sliding window of recent memories but can also retrieve
older memories that are semantically relevant to the current query.

Key features:
- Uses TF-IDF or simple keyword matching for semantic similarity
- Combines recency with relevance for memory retrieval
- Better for scenarios requiring long-term memory recall
"""

from typing import List, Optional, Callable, Dict, Any
from datetime import datetime
import re
from collections import Counter
from .base import BaseMemory
from .schema import MemoryItem


class SemanticMemory(BaseMemory):
    def __init__(self,
                 max_tokens: int = 2000,
                 max_items: int = 100,
                 recent_window: int = 10,
                 semantic_top_k: int = 5,
                 token_counter: Optional[Callable[[str], int]] = None):
        """
        Args:
            max_tokens: Maximum token count for all memory texts
            max_items: Maximum number of memory blocks to preserve
            recent_window: Number of recent items to always include
            semantic_top_k: Number of semantically relevant items to retrieve
            token_counter: Token estimator function
        """
        super().__init__()
        self.storage: List[MemoryItem] = []
        self.max_tokens = max_tokens
        self.max_items = max_items
        self.recent_window = recent_window
        self.semantic_top_k = semantic_top_k

        self.current_total_tokens = 0
        self.system_prompt: Optional[MemoryItem] = None

        # Cache for term frequencies
        self.tf_cache: Dict[str, Counter] = {}

        # Document frequency for IDF calculation
        self.doc_freq: Counter = Counter()
        self.total_docs = 0

        # Default estimator
        self.token_counter = token_counter or (lambda s: len(s) // 4)

        # Track the last query for semantic retrieval
        self.last_query: Optional[str] = None

    def set_system_prompt(self, content: str):
        self.system_prompt = MemoryItem(role="system", content=content)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for semantic matching."""
        if not text:
            return []
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter out very short words
        return [w for w in words if len(w) > 2]

    def _compute_tf(self, text: str) -> Counter:
        """Compute term frequency for a text."""
        tokens = self._tokenize(text)
        return Counter(tokens)

    def _update_idf(self, item: MemoryItem) -> None:
        """Update document frequency counts."""
        content = item.get_display_content() or ""
        tokens = set(self._tokenize(content))
        for token in tokens:
            self.doc_freq[token] += 1
        self.total_docs += 1

    def _compute_similarity(self, query_tf: Counter, doc_tf: Counter) -> float:
        """
        Compute TF-IDF based similarity between query and document.
        Uses a simplified cosine similarity.
        """
        if not query_tf or not doc_tf:
            return 0.0

        # Get common terms
        common_terms = set(query_tf.keys()) & set(doc_tf.keys())
        if not common_terms:
            return 0.0

        # Compute weighted score
        score = 0.0
        for term in common_terms:
            # Simple IDF: log(total_docs / doc_freq)
            df = self.doc_freq.get(term, 1)
            idf = 1.0 + (self.total_docs / df) if df > 0 else 1.0
            score += query_tf[term] * doc_tf[term] * idf

        return score

    def _add_to_storage(self, item: MemoryItem) -> None:
        if item.token_count is None:
            text_to_count = item.get_display_content() or ""
            count = self.token_counter(text_to_count)
            item.token_count = count
        else:
            count = item.token_count

        self.storage.append(item)
        self.current_total_tokens += count

        # Update TF cache and IDF
        content = item.get_display_content() or ""
        self.tf_cache[item.id] = self._compute_tf(content)
        self._update_idf(item)

        # Track last user message as query for semantic retrieval
        if item.role == "user" and item.content:
            self.last_query = item.content

    def _manage_memory_constraints(self) -> None:
        """
        Remove oldest items when constraints are exceeded.
        Keeps recent window intact.
        """
        while (self.current_total_tokens > self.max_tokens or
               len(self.storage) > self.max_items) and len(self.storage) > self.recent_window:

            # Remove the oldest item (outside recent window)
            removed_item = self.storage.pop(0)
            count = removed_item.token_count or 0
            self.current_total_tokens -= count

            # Clean up caches
            if removed_item.id in self.tf_cache:
                del self.tf_cache[removed_item.id]

            print(f"[Semantic] Pruned item {removed_item.id}, freed {count} tokens. Current: {self.current_total_tokens}")

    def retrieve(self, query: str = None, top_k: int = None) -> List[MemoryItem]:
        """
        Retrieve memories combining recency and semantic relevance.

        Returns:
        - All items from recent window
        - Top-k semantically relevant items from older memories
        """
        if top_k is None:
            top_k = self.semantic_top_k

        # Use provided query or last user message
        search_query = query or self.last_query

        if len(self.storage) <= self.recent_window:
            # All items fit in recent window
            return self.storage

        # Split into recent and older memories
        recent_items = self.storage[-self.recent_window:]
        older_items = self.storage[:-self.recent_window]

        if not search_query or not older_items:
            return self.storage

        # Compute query TF
        query_tf = self._compute_tf(search_query)

        # Score older items by semantic similarity
        scored_items = []
        for item in older_items:
            doc_tf = self.tf_cache.get(item.id, Counter())
            score = self._compute_similarity(query_tf, doc_tf)
            scored_items.append((score, item))

        # Sort by score and get top-k
        scored_items.sort(key=lambda x: x[0], reverse=True)
        relevant_items = [item for score, item in scored_items[:top_k] if score > 0]

        # Combine: relevant older items + recent items (in chronological order)
        result = []
        relevant_ids = {item.id for item in relevant_items}

        for item in self.storage:
            if item.id in relevant_ids or item in recent_items:
                result.append(item)

        return result

    def get_chat_messages(self) -> List[Dict[str, Any]]:
        """
        Override to include system prompt and use semantic retrieval.
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
        Returns a plain text representation of retrieved memories.
        """
        items = self.retrieve(query)
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
        self.tf_cache = {}
        self.doc_freq = Counter()
        self.total_docs = 0
        self.last_query = None
