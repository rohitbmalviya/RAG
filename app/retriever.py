"""
Simple, generic retriever - NO hardcoded logic!

Philosophy: Just retrieve relevant chunks and let LLM decide everything else.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from .config import RetrievalConfig
from .models import RetrievedChunk
from .core.base import BaseEmbedder, BaseVectorStore
from .utils import get_logger


class Retriever:
    """
    Pure retriever - retrieves chunks based on vector similarity and filters.
    
    NO hardcoded logic for:
    - Boosting/scoring
    - "Best property" detection
    - Priority ordering
    - Special query handling
    
    The LLM decides what's relevant based on context!
    """
    
    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore, config: RetrievalConfig) -> None:
        self._embedder = embedder
        self._store = store
        self._config = config
        self._logger = get_logger(__name__)
    
    def retrieve(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None, 
        top_k: Optional[int] = None
    ) -> tuple[List[RetrievedChunk], List[RetrievedChunk]]:
        """
        Retrieve chunks based on vector similarity.
        
        Simple approach:
        1. Embed query
        2. Search vector database with filters
        3. Return results sorted by similarity score
        4. LLM decides what's relevant!
        
        Args:
            query: Search query
            filters: Optional filters (applied at database level)
            top_k: Number of results
        
        Returns:
            Tuple of (chunks_to_display, all_chunks) - same list for simplicity
        """
        self._logger.debug(f"🔍 RETRIEVER: Starting retrieval for query: '{query}'")
        
        # Get top_k
        k = top_k if top_k is not None else (self._config.top_k or 12)
        
        # Calculate num_candidates for Elasticsearch kNN
        num_candidates = k * max(1, self._config.num_candidates_multiplier or 8)
        
        self._logger.debug(f"🔍 RETRIEVER: Retrieving {k} chunks (num_candidates: {num_candidates})")
        self._logger.debug(f"🔍 RETRIEVER: Query: '{query}'")
        self._logger.debug(f"🔍 RETRIEVER: Filters: {filters}")
        
        # Embed query
        self._logger.debug(f"🔍 RETRIEVER: Embedding query...")
        query_vec = self._embedder.embed([query], task_type="retrieval_query")[0]
        self._logger.debug(f"🔍 RETRIEVER: Query embedded, vector length: {len(query_vec)}")
        
        # Validate filters (only include allowed fields)
        validated_filters = self._validate_filters(filters)
        self._logger.debug(f"🔍 RETRIEVER: Validated filters: {validated_filters}")
        
        # Search vector database
        try:
            self._logger.debug(f"🔍 RETRIEVER: Searching vector store...")
            hits = self._store.search(
                query_vec, 
                top_k=k, 
                num_candidates=num_candidates, 
                filters=validated_filters
            )
            self._logger.debug(f"🔍 RETRIEVER: Vector store returned {len(hits)} hits")
        except Exception as exc:
            self._logger.error(f"🔍 RETRIEVER: Vector store search failed: {exc}")
            raise
        
        # Convert hits to RetrievedChunk objects
        chunks: List[RetrievedChunk] = []
        self._logger.debug(f"🔍 RETRIEVER: Processing {len(hits)} hits...")
        
        for i, hit in enumerate(hits):
            score = float(hit.get("_score", 0.0))
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            text = source.get("text", "")
            
            self._logger.debug(f"🔍 RETRIEVER: Hit {i+1}: Score={score:.3f}, ID={metadata.get('id', 'unknown')}")
            self._logger.debug(f"🔍 RETRIEVER: Hit {i+1} metadata: {metadata}")
            self._logger.debug(f"🔍 RETRIEVER: Hit {i+1} text preview: {text[:200]}...")
            
            chunks.append(
                RetrievedChunk(score=score, text=text, metadata=metadata)
            )
        
        # Sort by score (highest first) - that's it!
        chunks.sort(key=lambda x: x.score, reverse=True)
        
        self._logger.debug(f"🔍 RETRIEVER: Retrieved {len(chunks)} chunks")
        if chunks:
            self._logger.debug(f"🔍 RETRIEVER: Top score: {chunks[0].score:.3f}")
            self._logger.debug(f"🔍 RETRIEVER: Bottom score: {chunks[-1].score:.3f}")
            
            # Log all retrieved chunks with details
            for i, chunk in enumerate(chunks):
                self._logger.debug(f"🔍 RETRIEVER: Chunk {i+1}: {chunk.metadata.get('property_title', 'Unknown')} (ID: {chunk.metadata.get('id', 'unknown')}, Score: {chunk.score:.3f})")
        
        # Return same list for both (simplicity)
        return chunks, chunks
    
    def _validate_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate filters against allowed fields - that's the ONLY logic here"""
        if not filters:
            return None
        
        allowed = set(self._config.filter_fields or [])
        validated = {}
        
        for key, value in filters.items():
            if key in allowed and value is not None:
                validated[key] = value
            else:
                self._logger.debug(f"Ignoring filter '{key}': not in allowed fields")
        
        return validated if validated else None
