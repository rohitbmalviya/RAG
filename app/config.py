from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from .utils import get_logger

logger = get_logger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


def _expand_env_placeholders(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        var = match.group(1)
        default = match.group(2)
        val = os.getenv(var, default if default is not None else "")
        return val

    return _ENV_PATTERN.sub(repl, text)


class DatabaseConfig(BaseModel):
    driver: str = "postgres"
    host: str = "localhost"
    port: int = 5432
    name: str
    user: str
    password: str
    table: str
    id_column: str = "id"
    columns: List[str] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    unit: str = "token"  # "token" or "char"


class EmbeddingConfig(BaseModel):
    provider: str = "google"
    model: str = "gemini-embedding-001"
    api_key: Optional[str] = None
    batch_size: int = 64


class VectorDBConfig(BaseModel):
    provider: str = "elasticsearch"
    hosts: List[str] = Field(default_factory=lambda: ["http://localhost:9200"])
    username: Optional[str] = None
    password: Optional[str] = None
    index: str = "rag_properties"
    similarity: str = "cosine"
    refresh_on_write: bool = False
    dims: Optional[int] = None


class RetrievalConfig(BaseModel):
    top_k: int = 5
    num_candidates_multiplier: int = 8
    filter_fields: List[str] = Field(default_factory=list)
    reranker: Optional[Dict[str, Any]] = None  # optional reranker stub


class LLMConfig(BaseModel):
    provider: str = "google"
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.2
    max_output_tokens: int = 1024


class IngestionConfig(BaseModel):
    batch_size: int = 500


class AppConfig(BaseModel):
    name: str = "RAG Pipeline"
    host: str = "0.0.0.0"
    port: int = 8000
    eager_init: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    expanded = _expand_env_placeholders(raw)
    data = yaml.safe_load(expanded) or {}
    return data


@lru_cache(maxsize=1)
def get_settings(config_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")) -> Settings:
    # Prefer workspace root config.yaml
    root_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "config.yaml")):
        root_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    # If caller provided a path, use it
    if config_path and os.path.exists(config_path):
        root_config = config_path

    # Fallback: try project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    proj_config = os.path.join(project_root, "config.yaml")
    if os.path.exists(proj_config):
        root_config = proj_config

    cfg = load_yaml_config(root_config)

    # Allow DATABASE_URL override
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            # postgres://user:pass@host:port/db
            import urllib.parse as _url

            parsed = _url.urlparse(db_url)
            user = parsed.username or ""
            password = parsed.password or ""
            host = parsed.hostname or "localhost"
            port = parsed.port or 5432
            name = parsed.path.lstrip("/")
            cfg.setdefault("database", {})
            cfg["database"].update({
                "host": host,
                "port": port,
                "name": name,
                "user": user,
                "password": password,
            })
        except Exception as exc:
            logger.warning("Failed to parse DATABASE_URL: %s", exc)

    settings = Settings(**cfg)

    # Fallback: load API keys from environment if not provided in config
    if not settings.embedding.api_key:
        settings.embedding.api_key = os.getenv("GOOGLE_API_KEY")

    # Configure logger level correctly
    import logging as _logging
    logger.setLevel(getattr(_logging, settings.logging.level.upper(), _logging.INFO))
    return settings
