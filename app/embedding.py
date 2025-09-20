from __future__ import annotations
from typing import Callable, Dict, List, Optional, Type
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
from .config import EmbeddingConfig
from .core.base import BaseEmbedder
from .utils import get_logger

# Constants to eliminate duplication
GOOGLE_PROVIDER = "google"
OPENAI_PROVIDER = "openai"
# Provider registry and decorator
EMBEDDING_PROVIDERS: Dict[str, Type["BaseEmbeddingProvider"]] = {}
def register_embedding_provider(name: str) -> Callable[[Type["BaseEmbeddingProvider"]], Type["BaseEmbeddingProvider"]]:
    def decorator(cls: Type["BaseEmbeddingProvider"]) -> Type["BaseEmbeddingProvider"]:
        EMBEDDING_PROVIDERS[name.lower()] = cls
        return cls
    return decorator

class BaseEmbeddingProvider:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._ensure_api_key()
        self._setup_client()
    def _ensure_api_key(self) -> None:
        if not self._config.api_key:
            import os as _os
            self._config.api_key = _os.getenv("LLM_MODEL_API_KEY")
    def _setup_client(self) -> None:
        # Optional for providers to implement
        pass
    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5))
    def _embed_single(self, text: str, task_type: Optional[str] = None) -> List[float]:
        raise NotImplementedError
# Google Embedding Provider
@register_embedding_provider(GOOGLE_PROVIDER)
class GoogleEmbedding(BaseEmbeddingProvider):
    def _setup_client(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:
            raise RuntimeError("google-generativeai package not available. Add it to requirements.txt") from exc
        if not self._config.api_key:
            self._logger.warning("Embedding API key missing for Google provider")
        else:
            genai.configure(api_key=self._config.api_key)
        self._client = genai

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5))
    def _embed_single(self, text: str, task_type: Optional[str] = None) -> List[float]:
        kwargs = {"model": self._config.model, "content": text}
        if task_type:
            kwargs["task_type"] = task_type
        try:
            response = self._client.embed_content(**kwargs)
        except Exception as exc:
            self._logger.error("Google embed_content failed: %s", exc)
            raise
        vector = response["embedding"] if isinstance(response, dict) else response.embedding
        return list(vector)
# OpenAI Embedding Provider
@register_embedding_provider(OPENAI_PROVIDER)
@register_embedding_provider("azure_openai")
class OpenAIEmbedding(BaseEmbeddingProvider):
    def _setup_client(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("openai package not available. Add it to requirements.txt") from exc
        if not self._config.api_key:
            self._logger.warning("Embedding API key missing for provider")
        self._client = OpenAI(api_key=self._config.api_key)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5))
    def _embed_single(self, text: str, task_type: Optional[str] = None) -> List[float]:
        try:
            resp = self._client.embeddings.create(model=self._config.model, input=text)
        except Exception as exc:
            self._logger.error("OpenAI embeddings.create failed: %s", exc)
            raise
        return list(resp.data[0].embedding)
class EmbeddingClient(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._logger = get_logger(__name__)
        provider_name = (config.provider or GOOGLE_PROVIDER).lower()
        provider_cls = EMBEDDING_PROVIDERS.get(provider_name)
        if provider_cls is None:
            raise ValueError(f"Unsupported embedding provider: {config.provider}")
        if not config.model:
            self._logger.warning("Embedding model is not set; API calls may fail")

        self._provider_client = provider_cls(config)
    def embed(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        embeddings: List[List[float]] = []
        batch_size = max(1, self._config.batch_size or 1)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                try:
                    vector = self._provider_client._embed_single(text, task_type=task_type)
                except RetryError as exc:
                    self._logger.error("Failed to embed text after retries: %s", exc)
                    raise
                except Exception as exc:
                    self._logger.error("Failed to embed text: %s", exc)
                    raise
                embeddings.append(vector)
        return embeddings
    def get_dimension(self) -> int:
        probe = "dimension_probe"
        vector = self._provider_client._embed_single(probe, task_type="retrieval_document")
        return len(vector)