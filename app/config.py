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

def _get_project_root() -> str:
    """Get the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _get_app_dir() -> str:
    """Get the app directory (parent of current file)."""
    return os.path.dirname(os.path.dirname(__file__))

def _get_env_var(*var_names: str) -> Optional[str]:
    """Get the first available environment variable from the list."""
    for var_name in var_names:
        value = os.getenv(var_name)
        if value:
            return value
    return None

def _build_config_path(directory: str) -> str:
    """Build a config.yaml path for the given directory."""
    return os.path.join(directory, "config.yaml")

def _create_provider_validation(section: str) -> tuple[str, str, str, str]:
    """Create provider validation tuple for a config section."""
    return ("provider", "is_none_or_empty", "warning", f"{section} 'provider' not set; default 'google' will be used")

def _create_model_validation(section: str, action: str) -> tuple[str, str, str, str]:
    """Create model validation tuple for a config section."""
    return ("model", "is_none_or_empty", "warning", f"{section} 'model' not set; {action} may fail")

def _try_load_dotenv() -> None:
    """Load a .env file from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        project_root = _get_project_root()
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            load_dotenv()
    except ImportError:
        logger.debug("python-dotenv not available, skipping .env file loading")
    except Exception as exc:
        logger.warning("Failed to load .env file: %s", exc)

def _coerce_value(target_example: Any, value_str: str) -> Any:
    """Coerce ENV value string to match existing config type when possible."""
    if not value_str:
        return value_str
        
    value_str = value_str.strip()
    
    if isinstance(target_example, bool):
        return value_str.lower() in {"1", "true", "yes", "on"}
    
    if isinstance(target_example, int):
        try:
            return int(value_str)
        except ValueError:
            logger.warning("Failed to convert '%s' to int, keeping as string", value_str)
            return value_str
    
    if isinstance(target_example, float):
        try:
            return float(value_str)
        except ValueError:
            logger.warning("Failed to convert '%s' to float, keeping as string", value_str)
            return value_str
    
    if isinstance(target_example, list):
        return [item.strip() for item in value_str.split(",") if item.strip()]
    
    # Fallback type detection
    lowered = value_str.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    
    # Try numeric conversion
    try:
        return int(value_str)
    except ValueError:
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

class StatusFilterConfig(BaseModel):
    """Configuration for filtering records by status field."""
    enabled: bool = True
    field: str = Field(default="property_status", description="Field to filter on")
    value: str = Field(default="listed", description="Value to filter for")

class DisplayConfig(BaseModel):
    """Simple display configuration (direct values - no conditions!)."""
    currency: str = Field(default="$", description="Currency symbol")
    measurement_unit: str = Field(default="square feet", description="Measurement unit")

class DatabaseRelation(BaseModel):
    """Configuration for database relations (both many-to-one and one-to-many).
    
    Consistent structure for all relation types:
    - many_to_one: Uses LEFT JOIN, fetches single value, stores in metadata
    - one_to_many: Uses separate query, fetches array, stores in metadata
    """
    name: str = Field(..., description="Relation identifier (e.g., 'property_type', 'media')")
    relation_type: str = Field(default="many_to_one", description="Type: 'many_to_one' or 'one_to_many'")
    foreign_key: str = Field(..., description="FK column (in main table for many-to-one, in child table for one-to-many)")
    reference_table: str = Field(..., description="Related/child table name")
    reference_column: str = Field(default="id", description="PK column in reference table")
    fields_to_fetch: List[str] = Field(..., description="Fields to fetch from related table")
    alias: str = Field(..., description="Alias in result/metadata")
    order_by: Optional[str] = Field(None, description="Order by field (for one-to-many)")
    limit: Optional[int] = Field(None, description="Limit results (for one-to-many, None = no limit)")

class BoostingFieldsConfig(BaseModel):
    """Field names for boosting features (generic - configurable per domain)."""
    premium: str = Field(default="premiumBoostingStatus", description="Premium boost field")
    carousel: str = Field(default="carouselBoostingStatus", description="Carousel boost field")
    verification: str = Field(default="bnb_verification_status", description="Verification field")

class BoostingActiveValuesConfig(BaseModel):
    """Active values for boosting fields (configurable per domain)."""
    premium: str = Field(default="Active", description="Active value for premium")
    carousel: str = Field(default="Active", description="Active value for carousel")
    verification: str = Field(default="verified", description="Active value for verification")

class BoostingWeightsConfig(BaseModel):
    """Weight values for boosting scores (tunable)."""
    premium: float = Field(default=0.15, description="Boost weight for premium")
    carousel: float = Field(default=0.1, description="Boost weight for carousel")
    verification: float = Field(default=0.15, description="Boost weight for verification")
    all_three: float = Field(default=0.2, description="Additional boost when all three active")

class BoostingConfig(BaseModel):
    """Optional boosting configuration (can be disabled for non-boosted domains)."""
    enabled: bool = Field(default=False, description="Enable/disable boosting logic")
    fields: BoostingFieldsConfig = Field(default_factory=BoostingFieldsConfig)
    active_values: BoostingActiveValuesConfig = Field(default_factory=BoostingActiveValuesConfig)
    boost_weights: BoostingWeightsConfig = Field(default_factory=BoostingWeightsConfig)

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
    relations: List[DatabaseRelation] = Field(default_factory=list)
    status_filter: Optional[StatusFilterConfig] = None
    boolean_fields: Dict[str, str] = Field(default_factory=dict, description="Boolean fields with display labels (domain-agnostic)")
    priority_features: List[str] = Field(default_factory=list, description="High priority features for filtering")
    location_hierarchy: List[str] = Field(default_factory=list, description="Location fields ordered from broad to specific")
    display: DisplayConfig = Field(default_factory=DisplayConfig, description="Display configuration (currency, units)")
    primary_display_field: str = Field(default="id", description="Primary field to display in logs (e.g., 'property_title', 'product_name')")
    pricing_field: Optional[str] = Field(None, description="Field for price aggregations (e.g., 'rent_charge', 'price')")
    filter_priority_groups: Dict[str, List[str]] = Field(default_factory=dict, description="Grouped filters by priority for retrieval optimization")
    boosting: Optional[BoostingConfig] = Field(None, description="Optional boosting configuration (disable if not needed)")
    query_patterns: Dict[str, List[str]] = Field(default_factory=dict, description="Query keyword patterns (e.g., best_property: ['best', 'top'])")

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

class QueryHandlingConfig(BaseModel):
    """LLM query handling configuration - enable/disable features without hardcoded lists."""
    use_llm_classification: bool = True
    enable_price_aggregation: bool = True
    enable_knowledge_search: bool = True
    enable_requirement_gathering: bool = True
    enable_context_validation: bool = True
    validation_tolerance: float = 0.05

class LLMConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    requirement_gathering: Optional[RequirementGatheringConfig] = None
    query_handling: Optional[QueryHandlingConfig] = Field(default_factory=QueryHandlingConfig)

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
    """Configuration for requirement gathering feature (NEW API FORMAT)."""
    endpoint: Optional[str] = None
    enabled: Optional[bool] = None
    auto_save_on_conversation_end: Optional[bool] = None
    ask_user_confirmation: Optional[bool] = None
    priority_fields: Optional[List[str]] = None  # Legacy field
    essential_fields: Optional[List[str]] = Field(default_factory=lambda: [
        'location', 'property_type_name', 'number_of_bedrooms', 
        'rent_charge', 'furnishing_status', 'amenities', 'lease_duration'
    ])

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
    """Load and parse YAML configuration file with environment variable expansion."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        expanded = _expand_env_placeholders(raw)
        data = yaml.safe_load(expanded) or {}
        return data
    except FileNotFoundError:
        logger.error("Configuration file not found: %s", path)
        raise
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML configuration: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error loading configuration: %s", exc)
        raise

def _find_config_file(config_path: Optional[str] = None) -> str:
    """Find the configuration file using multiple fallback strategies."""
    if config_path and os.path.exists(config_path):
        return config_path
    
    # Try environment variables
    env_cfg = _get_env_var("CONFIG_FILE", "CONFIG_PATH")
    if env_cfg and os.path.exists(env_cfg):
        return env_cfg
    
    # Try relative to app directory
    app_config = _build_config_path(_get_app_dir())
    if os.path.exists(app_config):
        return app_config
    
    # Try project root
    project_root = _get_project_root()
    project_config = _build_config_path(project_root)
    if os.path.exists(project_config):
        return project_config
    
    # Default fallback
    return app_config

def _parse_database_url(db_url: str) -> Dict[str, Any]:
    """Parse DATABASE_URL into individual components."""
    try:
        import urllib.parse as _url
        parsed = _url.urlparse(db_url)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "name": parsed.path.lstrip("/"),
            "user": parsed.username or "",
            "password": parsed.password or "",
        }
    except Exception as exc:
        logger.warning("Failed to parse DATABASE_URL: %s", exc)
        return {}

def _setup_logging(settings: Settings) -> None:
    """Configure logging based on settings."""
    import logging as _logging
    level_name = (settings.logging.level or "INFO").upper()
    logger.setLevel(getattr(_logging, level_name, _logging.INFO))

@lru_cache(maxsize=1)
def get_settings(config_path: Optional[str] = None) -> Settings:
    """Load and validate application settings with caching."""
    _try_load_dotenv()
    
    # Find configuration file
    config_file = _find_config_file(config_path)
    cfg = load_yaml_config(config_file)
    
    # Parse DATABASE_URL if present
    db_url = _get_env_var("DATABASE_URL", "DB_URL")
    if db_url:
        db_config = _parse_database_url(db_url)
        if db_config:
            cfg.setdefault("database", {}).update(db_config)
    
    # Apply environment variable overrides
    _apply_nested_env_overrides(cfg)
    
    # Move requirement_gathering from root to llm section if it exists
    if "requirement_gathering" in cfg and "llm" in cfg:
        cfg["llm"]["requirement_gathering"] = cfg["requirement_gathering"]
        del cfg["requirement_gathering"]
    
    # Create settings object
    settings = Settings(**cfg)
    
    # Apply additional environment variable overrides
    if not settings.embedding.api_key:
        settings.embedding.api_key = os.getenv("LLM_MODEL_API_KEY")
    
    # Setup logging
    _setup_logging(settings)
    
    # Validate settings
    try:
        _log_missing_settings(settings)
    except Exception as exc:
        logger.warning("Settings validation logging failed: %s", exc)
    
    return settings

def _log_validation_message(level: str, message: str) -> None:
    """Log a validation message at the specified level."""
    getattr(logger, level)(message)

def _log_config_error(section_name: str, exc: Exception) -> None:
    """Log a configuration validation error."""
    logger.warning("Failed checking %s config: %s", section_name, exc)

def _validate_config_section(
    section_name: str, 
    section_obj: Any, 
    validations: List[tuple[str, str, str, str]]
) -> None:
    """Generic config validation with unified error handling.
    
    Args:
        section_name: Name of the config section for error messages
        section_obj: The config object to validate
        validations: List of (field, condition, level, message) tuples
    """
    try:
        for field, condition, level, message in validations:
            value = getattr(section_obj, field, None)
            
            if condition == "is_none_or_empty":
                if value in (None, ""):
                    _log_validation_message(level, message)
            elif condition == "is_none_or_zero":
                if value in (None, 0):
                    _log_validation_message(level, message)
            elif condition == "is_empty_list":
                if not value:
                    _log_validation_message(level, message)
            elif condition == "is_none":
                if value is None:
                    _log_validation_message(level, message)
            elif condition == "not_in_values":
                valid_values = message.split("|")[1].split(",")
                if value not in valid_values:
                    _log_validation_message(level, message.split("|")[0])
            elif condition == "custom":
                # Custom validation logic
                if field == "chunk_overlap_validation":
                    ch = section_obj
                    if (ch.chunk_size and ch.chunk_overlap and 
                        ch.chunk_overlap >= ch.chunk_size):
                        _log_validation_message(level, message.format(
                            overlap=ch.chunk_overlap, size=ch.chunk_size))
                elif field == "embedding_columns_validation":
                    db = section_obj
                    if (db.embedding_columns and 
                        not set(db.embedding_columns).issubset(set(db.columns))):
                        _log_validation_message(level, message)
    except Exception as exc:
        _log_config_error(section_name, exc)

def _log_missing_settings(settings: Settings) -> None:
    """Log warnings for missing or suspicious configuration values.
    This does not raise; it only surfaces potential misconfigurations.
    """
    # Database validation
    try:
        db = settings.database
        missing_db: List[str] = []
        for key in ("host", "port", "name", "user", "password", "table"):
            if getattr(db, key) in (None, ""):
                missing_db.append(key)
        if missing_db:
            logger.warning("Database config missing values: %s", ", ".join(sorted(missing_db)))
        
        db_validations = [
            ("columns", "is_empty_list", "warning", "Database config 'columns' is empty; ingestion will have no fields to read"),
            ("id_column", "is_none_or_empty", "warning", "Database config 'id_column' is not set; default string ids expected from data"),
            ("embedding_columns_validation", "custom", "warning", "Some 'embedding_columns' are not present in 'columns'; they will be ignored"),
        ]
        _validate_config_section("database", db, db_validations)
    except Exception as exc:
        _log_config_error("database", exc)

    # Chunking validation
    ch_validations = [
        ("chunk_size", "is_none_or_zero", "warning", "Chunking 'chunk_size' is not set or zero; retrieval quality may be impacted"),
        ("chunk_overlap", "is_none", "warning", "Chunking 'chunk_overlap' is not set; defaulting behavior may cause gaps"),
        ("chunk_overlap_validation", "custom", "warning", "Chunking 'chunk_overlap' (={overlap}) >= 'chunk_size' (={size}); expect repeated chunks"),
        ("unit", "not_in_values", "warning", "Chunking 'unit' should be 'token' or 'char', got '%s'|None,token,char"),
    ]
    _validate_config_section("chunking", settings.chunking, ch_validations)

    # Embedding validation
    emb_validations = [
        _create_provider_validation("Embedding"),
        _create_model_validation("Embedding", "embedding"),
        ("batch_size", "is_none_or_zero", "warning", "Embedding 'batch_size' is not set or zero; defaulting to 1"),
        ("api_key", "is_none_or_empty", "warning", "Embedding 'api_key' not set; will try environment variable"),
    ]
    _validate_config_section("embedding", settings.embedding, emb_validations)

    # Vector DB validation
    vdb_validations = [
        ("provider", "is_none_or_empty", "warning", "Vector DB 'provider' not set; ensure Elasticsearch defaults are intended"),
        ("hosts", "is_empty_list", "warning", "Vector DB 'hosts' empty; client will not reach Elasticsearch"),
        ("index", "is_none_or_empty", "warning", "Vector DB 'index' not set; operations will fail until specified"),
        ("similarity", "is_none_or_empty", "warning", "Vector DB 'similarity' not set; default backend similarity will be used if any"),
        ("dims", "is_none_or_zero", "info", "Vector DB 'dims' not set; will infer from embedding model on init"),
    ]
    _validate_config_section("vector_db", settings.vector_db, vdb_validations)

    # Retrieval validation
    ret_validations = [
        ("top_k", "is_none_or_zero", "warning", "Retrieval 'top_k' not set or zero; queries may return no results"),
        ("num_candidates_multiplier", "is_none_or_zero", "warning", "Retrieval 'num_candidates_multiplier' not set or zero; using top_k only"),
        ("filter_fields", "is_none", "warning", "Retrieval 'filter_fields' is None; will be treated as empty list"),
    ]
    _validate_config_section("retrieval", settings.retrieval, ret_validations)

    # LLM validation
    llm_validations = [
        _create_provider_validation("LLM"),
        _create_model_validation("LLM", "generation"),
        ("temperature", "is_none", "info", "LLM 'temperature' not set; using provider default"),
        ("max_output_tokens", "is_none", "info", "LLM 'max_output_tokens' not set; using provider default"),
    ]
    _validate_config_section("llm", settings.llm, llm_validations)

    # Ingestion validation
    ing_validations = [
        ("batch_size", "is_none_or_zero", "warning", "Ingestion 'batch_size' not set or zero; default endpoint parameter must be provided"),
    ]
    _validate_config_section("ingestion", settings.ingestion, ing_validations)

    # App validation
    app_validations = [
        ("name", "is_none_or_empty", "info", "App 'name' not set"),
        ("host", "is_none_or_empty", "info", "App 'host' not set; FastAPI will use default host"),
        ("port", "is_none_or_zero", "info", "App 'port' not set; FastAPI will use default port"),
    ]
    _validate_config_section("app", settings.app, app_validations)
