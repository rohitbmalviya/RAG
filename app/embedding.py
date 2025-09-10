from __future__ import annotations

from typing import List, Optional

from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
try:
    from openai import OpenAI
except Exception:  
    OpenAI = None  

from .config import EmbeddingConfig
from .utils import get_logger


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        
        if not config.api_key:
            import os as _os
            config.api_key = _os.getenv("GOOGLE_API_KEY")
        self._logger = get_logger(__name__)

        self._provider = (config.provider or "google").lower()
        self._client_gg = None
        self._client_oa = None
        if not config.model:
            self._logger.warning("Embedding model is not set; API calls may fail")
        if self._provider == "google":
            if config.api_key:
                genai.configure(api_key=config.api_key)
            else:
                self._logger.warning("Embedding API key missing for Google provider")
            self._client_gg = genai
        elif self._provider in {"openai", "azure_openai"}:
            
            if OpenAI is None:
                raise RuntimeError("openai package not available. Add it to requirements.txt")
            if not config.api_key:
                import os as _os
                config.api_key = _os.getenv("OPENAI_API_KEY")
            if not config.api_key:
                self._logger.warning("Embedding API key missing for OpenAI provider")
            self._client_oa = OpenAI(api_key=config.api_key)  
        else:
            raise ValueError(f"Unsupported embedding provider: {config.provider}")

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5))
    def _embed_single(self, text: str, task_type: Optional[str] = None) -> List[float]:
        if self._provider == "google":
            kwargs = {"model": self._config.model, "content": text}
            if task_type:
                kwargs["task_type"] = task_type
            try:
                response = self._client_gg.embed_content(**kwargs)  
            except Exception as exc:
                self._logger.error("Google embed_content failed: %s", exc)
                raise
            vector = response["embedding"] if isinstance(response, dict) else response.embedding
            return list(vector)
        
        assert self._client_oa is not None
        try:
            resp = self._client_oa.embeddings.create(model=self._config.model, input=text)  
        except Exception as exc:
            self._logger.error("OpenAI embeddings.create failed: %s", exc)
            raise
        vec = resp.data[0].embedding
        return list(vec)

    def embed_texts(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        embeddings: List[List[float]] = []
        batch_size = max(1, self._config.batch_size)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            for text in batch:
                try:
                    vector = self._embed_single(text, task_type=task_type)
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
        vec = self._embed_single(probe, task_type="retrieval_document")
        return len(vec)
