from __future__ import annotations

from typing import List, Optional

from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

from .config import EmbeddingConfig
from .utils import get_logger


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        # Fallback to env if not set in config
        if not config.api_key:
            import os as _os
            config.api_key = _os.getenv("GOOGLE_API_KEY")
        if config.api_key:
            genai.configure(api_key=config.api_key)
        self._logger = get_logger(__name__)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5))
    def _embed_single(self, text: str, task_type: Optional[str] = None) -> List[float]:
        kwargs = {"model": self._config.model, "content": text}
        if task_type:
            kwargs["task_type"] = task_type
        response = genai.embed_content(**kwargs)
        vector = response["embedding"] if isinstance(response, dict) else response.embedding
        return list(vector)

    def embed_texts(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        embeddings: List[List[float]] = []
        batch_size = max(1, self._config.batch_size)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # gemini embed_content currently lacks official batch API; perform per-text with retry
            for text in batch:
                try:
                    vector = self._embed_single(text, task_type=task_type)
                except RetryError as exc:
                    self._logger.error("Failed to embed text after retries: %s", exc)
                    raise
                embeddings.append(vector)
        return embeddings

    def get_dimension(self) -> int:
        probe = "dimension_probe"
        vec = self._embed_single(probe, task_type="retrieval_document")
        return len(vec)
