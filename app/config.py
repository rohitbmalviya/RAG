from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

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


def _try_load_dotenv() -> None:
    """Load a .env file from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            load_dotenv()
    except Exception:
        
        pass


def _coerce_value(target_example: Any, value_str: str) -> Any:
    """Coerce ENV value string to match existing config type when possible."""
    if isinstance(target_example, bool):
        return value_str.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(target_example, int):
        try:
            return int(value_str)
        except ValueError:
            return value_str
    if isinstance(target_example, float):
        try:
            return float(value_str)
        except ValueError:
            return value_str
    if isinstance(target_example, list):
        return [item.strip() for item in value_str.split(",") if item.strip()]
    lowered = value_str.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def _apply_nested_env_overrides(cfg: Dict[str, Any]) -> None:
    """Override YAML config with ENV vars using double-underscore path keys.

    Example: APP__HOST=127.0.0.1 -> cfg["app"]["host"]
    """
    for env_key, env_value in os.environ.items():
        if "__" not in env_key:
            continue
        parts = env_key.split("__")
        if not all(part.isupper() for part in parts):
            continue
        node: Any = cfg
        for part in parts[:-1]:
            key = part.lower()
            if not isinstance(node, dict):
                return
            node = node.setdefault(key, {})
        last_key = parts[-1].lower()
        if isinstance(node, dict):
            example = node.get(last_key)
            node[last_key] = _coerce_value(example, env_value)


class DatabaseConfig(BaseModel):
    driver: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    name: str
    user: str
    password: str
    table: str
    id_column: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    
    embedding_columns: List[str] = Field(default_factory=list)
    
    
    field_types: Dict[str, str] = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    unit: Optional[str] = None  


class EmbeddingConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    batch_size: Optional[int] = None


class VectorDBConfig(BaseModel):
    provider: Optional[str] = None
    hosts: List[str] = Field(default_factory=list)
    username: Optional[str] = None
    password: Optional[str] = None
    index: Optional[str] = None
    similarity: Optional[str] = None
    refresh_on_write: Optional[bool] = None
    dims: Optional[int] = None
    index_settings: Optional[Dict[str, Any]] = None


class RetrievalConfig(BaseModel):
    top_k: Optional[int] = None
    num_candidates_multiplier: Optional[int] = None
    filter_fields: List[str] = Field(default_factory=list)
    reranker: Optional[Dict[str, Any]] = None  


class LLMConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None


class IngestionConfig(BaseModel):
    batch_size: Optional[int] = None


class AppConfig(BaseModel):
    name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    eager_init: Optional[bool] = None


class LoggingConfig(BaseModel):
    level: Optional[str] = None


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
def get_settings(config_path: Optional[str] = None) -> Settings:
    
    _try_load_dotenv()

    
    if not config_path:
        env_cfg = os.getenv("CONFIG_FILE") or os.getenv("CONFIG_PATH")
        if env_cfg and os.path.exists(env_cfg):
            config_path = env_cfg
    if not config_path:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    if not os.path.exists(config_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate = os.path.join(project_root, "config.yaml")
        if os.path.exists(candidate):
            config_path = candidate

    cfg = load_yaml_config(config_path)

    
    db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
    if db_url:
        try:
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

    
    _apply_nested_env_overrides(cfg)

    settings = Settings(**cfg)

    
    if not settings.embedding.api_key:
        settings.embedding.api_key = os.getenv("GOOGLE_API_KEY")

    
    import logging as _logging
    level_name = (settings.logging.level or "INFO").upper()
    logger.setLevel(getattr(_logging, level_name, _logging.INFO))
    return settings
