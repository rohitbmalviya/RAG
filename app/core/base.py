from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        raise NotImplementedError
    @abstractmethod
    def get_dimension(self) -> int:
        raise NotImplementedError

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

class BaseVectorStore(ABC):
    @abstractmethod
    def ensure_index(self, dims: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_index(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, documents: List[Any], embeddings: List[List[float]], refresh: Optional[bool] = None) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int,
        num_candidates: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def delete_by_source_id(self, source_id: str) -> int:
        """
        Delete all chunks/documents whose metadata.source_id (or id prefix before ':') equals the given id.
        Returns the number of deleted documents.
        """
        raise NotImplementedError