from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import faiss
import logging
import time

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, max_history: int = 10):
        self.context: List[str] = []
        self.max_history = max_history
        self.last_context_time = time.time()
        self.context_cooldown = 5.0  # Minimum time between context updates

    def add_text(self, text: str) -> None:
        """Add text to context with deduplication and timing control"""
        current_time = time.time()
        
        # Don't add duplicate or very similar recent messages
        if self.context and (
            text in self.context[-1] or  # Exact match
            any(self._similarity(text, ctx) > 0.8 for ctx in self.context[-3:])  # Similar to recent
        ):
            return
            
        # Add new context
        self.context.append(text)
        
        # Trim history if needed
        if len(self.context) > self.max_history:
            self.context = self.context[-self.max_history:]
        
        self.last_context_time = current_time

    def get_recent_context(self) -> str:
        """Get recent context as a single string"""
        # Only return last few messages to keep context focused
        return " ".join(self.context[-5:])

    def _similarity(self, text1: str, text2: str) -> float:
        """Simple similarity check between two texts"""
        # Convert to sets of words for comparison
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def clear_context(self) -> None:
        """Clear the context history"""
        self.context = []