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
    requirement_gathering: Optional[RequirementGatheringConfig] = None

class IngestionConfig(BaseModel):
    batch_size: Optional[int] = None
    source_type: Optional[str] = None  
    source_path: Optional[str] = None  

class AppConfig(BaseModel):
    name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    eager_init: Optional[bool] = None

class LoggingConfig(BaseModel):
    level: Optional[str] = None

class RequirementGatheringConfig(BaseModel):
    endpoint: Optional[str] = None
    enabled: Optional[bool] = None
    auto_save_on_conversation_end: Optional[bool] = None
    ask_user_confirmation: Optional[bool] = None
    priority_fields: Optional[List[str]] = None

class FallbackConfig(BaseModel):
    enable: Optional[bool] = None
    strategy: Optional[List[str]] = None
    requirement_capture: Optional[Dict[str, Any]] = None

class MemoryConfig(BaseModel):
    type: Optional[str] = None
    strategy: Optional[Dict[str, Any]] = None

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
    requirement_gathering: RequirementGatheringConfig = Field(default_factory=RequirementGatheringConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

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
    
    # Move requirement_gathering from root to llm section if it exists
    if "requirement_gathering" in cfg and "llm" in cfg:
        cfg["llm"]["requirement_gathering"] = cfg["requirement_gathering"]
        del cfg["requirement_gathering"]
    
    settings = Settings(**cfg)
    if not settings.embedding.api_key:
        settings.embedding.api_key = os.getenv("LLM_MODEL_API_KEY")
    import logging as _logging
    level_name = (settings.logging.level or "INFO").upper()
    logger.setLevel(getattr(_logging, level_name, _logging.INFO))    
    try:
        _log_missing_settings(settings)
    except Exception as _exc:
        logger.warning("Settings validation logging failed: %s", _exc)
    return settings

def _log_missing_settings(settings: Settings) -> None:
    """Log warnings for missing or suspicious configuration values.
    This does not raise; it only surfaces potential misconfigurations.
    """
    try:
        db = settings.database
        missing_db: List[str] = []
        for key in ("host", "port", "name", "user", "password", "table"):
            if getattr(db, key) in (None, ""):
                missing_db.append(key)
        if missing_db:
            logger.warning("Database config missing values: %s", ", ".join(sorted(missing_db)))
        if not db.columns:
            logger.warning("Database config 'columns' is empty; ingestion will have no fields to read")
        if db.id_column in (None, ""):
            logger.warning("Database config 'id_column' is not set; default string ids expected from data")

        if db.embedding_columns and not set(db.embedding_columns).issubset(set(db.columns)):
            logger.warning("Some 'embedding_columns' are not present in 'columns'; they will be ignored")
    except Exception as exc:
        logger.warning("Failed checking database config: %s", exc)

    try:
        ch = settings.chunking
        if ch.chunk_size in (None, 0):
            logger.warning("Chunking 'chunk_size' is not set or zero; retrieval quality may be impacted")
        if ch.chunk_overlap is None:
            logger.warning("Chunking 'chunk_overlap' is not set; defaulting behavior may cause gaps")
        if ch.chunk_size and ch.chunk_overlap and ch.chunk_overlap >= ch.chunk_size:
            logger.warning("Chunking 'chunk_overlap' (=%s) >= 'chunk_size' (=%s); expect repeated chunks", ch.chunk_overlap, ch.chunk_size)
        if ch.unit not in (None, "token", "char"):
            logger.warning("Chunking 'unit' should be 'token' or 'char', got '%s'", ch.unit)
    except Exception as exc:
        logger.warning("Failed checking chunking config: %s", exc)

    try:
        emb = settings.embedding
        if not emb.provider:
            logger.warning("Embedding 'provider' not set; default 'google' will be used")
        if not emb.model:
            logger.warning("Embedding 'model' not set; embedding may fail")
        if emb.batch_size in (None, 0):
            logger.warning("Embedding 'batch_size' is not set or zero; defaulting to 1")
        if not emb.api_key:
            logger.warning("Embedding 'api_key' not set; will try environment variable")
    except Exception as exc:
        logger.warning("Failed checking embedding config: %s", exc)

    try:
        vdb = settings.vector_db
        if not vdb.provider:
            logger.warning("Vector DB 'provider' not set; ensure Elasticsearch defaults are intended")
        if not vdb.hosts:
            logger.warning("Vector DB 'hosts' empty; client will not reach Elasticsearch")
        if not vdb.index:
            logger.warning("Vector DB 'index' not set; operations will fail until specified")
        if not vdb.similarity:
            logger.warning("Vector DB 'similarity' not set; default backend similarity will be used if any")
        if vdb.dims in (None, 0):
            logger.info("Vector DB 'dims' not set; will infer from embedding model on init")
    except Exception as exc:
        logger.warning("Failed checking vector DB config: %s", exc)

    try:
        ret = settings.retrieval
        if ret.top_k in (None, 0):
            logger.warning("Retrieval 'top_k' not set or zero; queries may return no results")
        if ret.num_candidates_multiplier in (None, 0):
            logger.warning("Retrieval 'num_candidates_multiplier' not set or zero; using top_k only")
        if ret.filter_fields is None:
            logger.warning("Retrieval 'filter_fields' is None; will be treated as empty list")
    except Exception as exc:
        logger.warning("Failed checking retrieval config: %s", exc)

    try:
        llm = settings.llm
        if not llm.provider:
            logger.warning("LLM 'provider' not set; default 'google' will be used")
        if not llm.model:
            logger.warning("LLM 'model' not set; generation may fail")
        if llm.temperature is None:
            logger.info("LLM 'temperature' not set; using provider default")
        if llm.max_output_tokens is None:
            logger.info("LLM 'max_output_tokens' not set; using provider default")
    except Exception as exc:
        logger.warning("Failed checking LLM config: %s", exc)

    try:
        ing = settings.ingestion
        if ing.batch_size in (None, 0):
            logger.warning("Ingestion 'batch_size' not set or zero; default endpoint parameter must be provided")
    except Exception as exc:
        logger.warning("Failed checking ingestion config: %s", exc)

    try:
        appcfg = settings.app
        if not appcfg.name:
            logger.info("App 'name' not set")
        if not appcfg.host:
            logger.info("App 'host' not set; FastAPI will use default host")
        if not appcfg.port:
            logger.info("App 'port' not set; FastAPI will use default port")
    except Exception as exc:
        logger.warning("Failed checking app config: %s", exc)
