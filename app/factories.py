from __future__ import annotations
from .config import EmbeddingConfig, LLMConfig, VectorDBConfig
from .core.base import BaseEmbedder, BaseLLM, BaseVectorStore
from .embedding import EmbeddingClient
from .llm import LLMClient
from .vector_db import VectorStoreClient

# Constants to eliminate duplication
ELASTICSEARCH_PROVIDER = "elasticsearch"
ES_PROVIDER = "es"

def build_embedder(cfg: EmbeddingConfig, api_key: str | None = None) -> BaseEmbedder:
    if api_key and not cfg.api_key:
        cfg.api_key = api_key
    return EmbeddingClient(cfg)

def build_llm(cfg: LLMConfig, fallback_api_key: str | None) -> BaseLLM:
    return LLMClient(cfg, api_key=fallback_api_key)

def build_vector_store(cfg: VectorDBConfig) -> BaseVectorStore:
    provider = (cfg.provider or ELASTICSEARCH_PROVIDER).lower()
    if provider in {ELASTICSEARCH_PROVIDER, ES_PROVIDER}:
        return VectorStoreClient(cfg)
    # Future: add support for other providers implementing BaseVectorStore
    raise ValueError(f"Unsupported vector store provider: {cfg.provider}")