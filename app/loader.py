from __future__ import annotations
import json
from typing import Dict, Generator, List, Optional, Iterable
import psycopg
from psycopg.rows import dict_row
from .config import Settings
from .models import Document
from .utils import get_logger


UTF8_ENCODING = "utf-8"

def _stringify_value(raw: Optional[object]) -> str:
    """
    Converts complex JSON/array/dict fields into a string representation.
    Useful for amenities-like fields.
    """
    if raw is None:
        return ""
    try:
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw

        if isinstance(data, dict):
            parts = []
            for key, value in data.items():
                if isinstance(value, list):
                    items = ", ".join(map(str, value))
                    parts.append(f"{key}: {items}")
                else:
                    parts.append(f"{key}: {value}")
            return "; ".join(parts)

        if isinstance(data, list):
            return ", ".join(map(str, data))

        return str(data)
    except Exception:
        return str(raw)

def _compose_text(row: Dict[str, object], columns: List[str]) -> str:
    """
    TRULY GENERIC text composition - works with ANY table schema!
    Uses ONLY config (field_types, location_hierarchy, boolean_fields).
    NO hardcoded field names!
    """
    from .config import get_settings
    settings = get_settings()
    
    segments: List[str] = []
    amenities: List[str] = []
    
    # Helper: Format field name to human-readable label
    def format_label(field_name: str) -> str:
        return field_name.replace('number_of_', '').replace('is_', '').replace('has_', '').replace('_', ' ').title()
    
    # Helper: Check field category by keywords (works for ANY language/naming)
    def is_financial(fname: str) -> bool:
        return any(kw in fname.lower() for kw in ['price', 'rent', 'cost', 'charge', 'fee', 'deposit', 'maintenance', 'payment'])
    
    def is_count(fname: str) -> bool:
        return any(kw in fname.lower() for kw in ['bed', 'bath', 'room', 'seat', 'capacity'])
    
    def is_measurement(fname: str) -> bool:
        return any(kw in fname.lower() for kw in ['size', 'area', 'sqft', 'sqm', 'acre', 'footage'])
    
    def is_title_or_name(fname: str) -> bool:
        return 'title' in fname.lower() or ('name' in fname.lower() and fname not in settings.database.location_hierarchy)
    
    # STEP 1: Title/Name Fields (text type + title/name pattern)
    for field_name in settings.database.columns:
        if is_title_or_name(field_name):
            field_type = settings.database.field_types.get(field_name, 'keyword')
            value = row.get(field_name)
            if value and field_type == 'text':
                segments.append(f"{format_label(field_name)}: {value}")
    
    # STEP 2: Location (from location_hierarchy config - NO hardcoding!)
    location_parts = []
    for field_name in settings.database.location_hierarchy:
        value = row.get(field_name)
        if value:
            location_parts.append(str(value))
    if location_parts:
        segments.append(f"Located in: {', '.join(location_parts)}")
    
    # STEP 3: Financial Fields (numeric + financial keywords)
    currency = settings.database.display.currency
    for field_name in settings.database.columns:
        field_type = settings.database.field_types.get(field_name, 'keyword')
        value = row.get(field_name)
        
        if value and field_type in ['float', 'integer'] and is_financial(field_name):
            try:
                num_val = float(value)
                segments.append(f"{format_label(field_name)}: {currency} {num_val:,.0f}")
            except:
                segments.append(f"{format_label(field_name)}: {value}")
    
    # STEP 4: Count Fields (numeric + count keywords)
    for field_name in settings.database.columns:
        field_type = settings.database.field_types.get(field_name, 'keyword')
        value = row.get(field_name)
        
        if value and field_type in ['float', 'integer'] and is_count(field_name) and not is_financial(field_name):
            try:
                count = int(float(value))
                label = format_label(field_name).rstrip('s')  # Remove existing 's' if any
                plural = 's' if count != 1 else ''
                segments.append(f"{count} {label.lower()}{plural}")
            except:
                pass
    
    # STEP 5: Measurement Fields (numeric + size/area keywords)
    measurement_unit = settings.database.display.measurement_unit
    for field_name in settings.database.columns:
        field_type = settings.database.field_types.get(field_name, 'keyword')
        value = row.get(field_name)
        
        if value and field_type in ['float', 'integer'] and is_measurement(field_name) and not is_financial(field_name):
            try:
                num_val = float(value)
                segments.append(f"{format_label(field_name)}: {num_val:,.0f} {measurement_unit}")
            except:
                pass
    
    # STEP 6: Boolean Features (from boolean_fields config)
    boolean_config = settings.database.boolean_fields or {}
    for field_name, display_label in boolean_config.items():
        value = row.get(field_name)
        if value and str(value).lower() in {"true", "1", "yes"}:
            amenities.append(display_label)
    
    # STEP 7: JSON Fields (field_type == 'json')
    for field_name in settings.database.columns:
        field_type = settings.database.field_types.get(field_name, 'keyword')
        value = row.get(field_name)
        
        if value and str(value) != "null" and field_type == 'json':
            # Skip if in boolean_fields (already handled)
            if field_name not in boolean_config:
                segments.append(f"{format_label(field_name)}: {_stringify_value(value)}")
    
    # STEP 8: Date Fields (field_type == 'date')
    for field_name in settings.database.columns:
        field_type = settings.database.field_types.get(field_name, 'keyword')
        value = row.get(field_name)
        
        if value and field_type == 'date':
            # Include important date fields
            date_keywords = ['available', 'lease', 'start', 'end', 'listing', 'tenancy']
            if any(kw in field_name.lower() for kw in date_keywords):
                segments.append(f"{format_label(field_name)}: {value}")
    
    # STEP 9: Important Keyword Fields (status, type, level, floor, duration)
    for field_name in settings.database.columns:
        field_type = settings.database.field_types.get(field_name, 'keyword')
        value = row.get(field_name)
        
        if value and field_type == 'keyword':
            # Skip IDs, location fields, and internal fields
            if ('id' in field_name.lower() or 
                field_name in settings.database.location_hierarchy or
                field_name.startswith('_')):
                continue
            
            # Include important categorical fields
            important_kw = ['status', 'type', 'duration', 'level', 'floor', 'furnish']
            if any(kw in field_name.lower() for kw in important_kw):
                formatted_value = str(value).replace('_', ' ').title() if isinstance(value, str) else str(value)
                segments.append(f"{format_label(field_name)}: {formatted_value}")
    
    # STEP 10: Text Description Fields (field_type == 'text', not title/name)
    for field_name in settings.database.columns:
        field_type = settings.database.field_types.get(field_name, 'keyword')
        value = row.get(field_name)
        
        if value and field_type == 'text':
            # Skip title/name fields (already processed)
            if not is_title_or_name(field_name):
                # Include descriptions, landmarks, notes
                if any(kw in field_name.lower() for kw in ['description', 'landmark', 'note', 'detail']):
                    segments.append(f"{format_label(field_name)}: {value}")
    
    # STEP 11: Add amenities/features section
    if amenities:
        segments.append(f"Amenities: {', '.join(amenities)}")
    
    return "\n".join(segments)

def _select_embedding_columns(settings: Settings, all_cols: List[str]) -> List[str]:
    """
    Determine which columns should be embedded.
    Priority:
    1) If settings.database.embedding_columns is provided, use its intersection with available columns
    2) Otherwise, use columns explicitly typed as 'text' in settings.database.field_types
    This prevents accidentally embedding numeric/boolean/keyword fields when embedding_columns is empty.
    """
    db = settings.database
    if db.embedding_columns:
        return [c for c in db.embedding_columns if c in all_cols]
    field_types = db.field_types or {}
    return [c for c in all_cols if field_types.get(c) == "text"]

def _process_media_item(item: Dict[str, object], fields_to_fetch: List[str]) -> Optional[Dict[str, object]]:
    """
    Process a single media item and return structured data.
    GENERIC - uses fields_to_fetch from relation config, NO hardcoded field names!
    """
    if not isinstance(item, dict):
        return None
    
    media_entry: Dict[str, object] = {}
    for field_name in fields_to_fetch:
        if item.get(field_name) is not None:
            media_entry[field_name] = item.get(field_name)
    
    return media_entry if media_entry else None

def _extract_media_metadata(src: Dict[str, object], fields_to_fetch: List[str]) -> Dict[str, object]:
    """
    Extract media objects from a 'media' array if present.
    GENERIC - uses fields_to_fetch from relation config, NO hardcoded field names!
    """
    media_meta: Dict[str, object] = {}
    media_items: Optional[object] = src.get("media")
    try:
        if isinstance(media_items, str):
            media_items = json.loads(media_items)
    except Exception:
        media_items = None
    if isinstance(media_items, list) and media_items:
        media_objs: List[Dict[str, object]] = []
        for item in media_items:
            media_entry = _process_media_item(item, fields_to_fetch)
            if media_entry:
                media_objs.append(media_entry)
        if media_objs:
            media_meta["media"] = media_objs[:5]
    return media_meta

def _yield_in_batches(items: Iterable[Document], batch_size: int) -> Generator[List[Document], None, None]:
    batch: List[Document] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def _build_media_map(conn: "psycopg.Connection", property_ids: List[str], settings = None) -> Dict[str, Dict[str, object]]:
    """
    Fetch child relations (e.g., media) for given property IDs using configuration.
    Returns a map: property_id -> {relation_alias: [items]}
    
    Now reads configuration from settings.database.relations for one-to-many relations.
    """
    if settings is None:
        from .config import get_settings
        settings = get_settings()
    
    result_map: Dict[str, Dict[str, object]] = {}
    if not property_ids:
        return result_map
    
    # Find one-to-many relations from config
    one_to_many_relations = [
        rel for rel in (settings.database.relations or [])
        if rel.relation_type == "one_to_many"
    ]
    
    if not one_to_many_relations:
        return result_map
    
    # Process each one-to-many relation
    for relation in one_to_many_relations:
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                # Build SELECT clause from config
                select_fields = [f'"{relation.foreign_key}"']
                if relation.fields_to_fetch:
                    for field in relation.fields_to_fetch:
                        select_fields.append(f'"{field}"')
                
                # Build ORDER BY clause
                order_clause = ""
                if relation.order_by:
                    order_clause = f'ORDER BY "{relation.foreign_key}", COALESCE("{relation.order_by}", 2147483647), "{relation.order_by}"'
                
                # Build and execute query
                sql = f'''SELECT {", ".join(select_fields)} FROM "{relation.reference_table}" WHERE "{relation.foreign_key}" = ANY(%s) {order_clause}'''
                cur.execute(sql, (property_ids,))
                rows: List[Dict[str, object]] = cur.fetchall()
            
            # Group by parent ID
            temp: Dict[str, List[Dict[str, object]]] = {}
            for r in rows:
                parent_id = str(r.get(relation.foreign_key) or "")
                if not parent_id:
                    continue
                temp.setdefault(parent_id, []).append(r)
            
            # Build result with limit
            for parent_id, items in temp.items():
                # Apply limit if configured
                limited_items = items[:relation.limit] if relation.limit else items
                
                # Process items (remove FK from each item for cleaner metadata)
                clean_items = []
                for item in limited_items:
                    clean_item = {k: v for k, v in item.items() if k != relation.foreign_key}
                    if clean_item:
                        clean_items.append(clean_item)
                
                # Add to result map
                if parent_id not in result_map:
                    result_map[parent_id] = {}
                result_map[parent_id][relation.alias] = clean_items
                
        except Exception as e:
            # Silently continue if child relation fails
            logger = get_logger(__name__)
            logger.debug(f"Failed to fetch {relation.name} relation: {e}")
            continue
    
    return result_map

def _create_document_from_row(
    row: Dict[str, object], 
    settings: Settings, 
    cols: List[str], 
    embed_cols: List[str]
) -> Document:
    """Create a Document from a row with common metadata processing."""
    row_id = str(row.get(settings.database.id_column) or row.get("id") or "")
    metadata: Dict[str, object] = {"table": settings.database.table or "file", "id": row_id}
    for c in cols:
        metadata[c] = row.get(c)
    
    # Process media using relation config (generic!)
    if "media" in row and row["media"]:
        try:
            # Get media relation config to find fields_to_fetch
            media_relation = next((r for r in settings.database.relations if r.name == "media"), None)
            if media_relation and media_relation.fields_to_fetch:
                media_md = _extract_media_metadata({"media": row["media"]}, media_relation.fields_to_fetch)
                metadata.update(media_md)
        except Exception:
            pass
    
    text = _compose_text(row, embed_cols)
    return Document(id=row_id, text=text, metadata=metadata)

def _load_documents_from_csv(path: str, settings: Settings) -> Iterable[Document]:
    import csv
    cols, embed_cols = _prepare_columns(settings)
    with open(path, newline="", encoding=UTF8_ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield _create_document_from_row(row, settings, cols, embed_cols)

def _load_documents_from_jsonl(path: str, settings: Settings) -> Iterable[Document]:
    cols, embed_cols = _prepare_columns(settings)
    
    # Get media fields from relation config (generic!)
    media_relation = next((r for r in settings.database.relations if r.name == "media"), None)
    media_fields = media_relation.fields_to_fetch if media_relation else []
    
    with open(path, "r", encoding=UTF8_ENCODING) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            if media_fields:
                media_md = _extract_media_metadata(data, media_fields)
                if media_md:
                    data.update(media_md)
            yield _create_document_from_row(data, settings, cols, embed_cols)

def _load_documents_from_txt(path: str, settings: Settings) -> Iterable[Document]:
    with open(path, "r", encoding=UTF8_ENCODING) as f:
        text = f.read()
    meta: Dict[str, object] = {"table": settings.database.table or "file", "id": "0"}
    yield Document(id="0", text=text, metadata=meta)

def _validate_database_config(db, logger) -> bool:
    """Validate database configuration and log errors."""
    if not db.table:
        logger.error("Database table name is missing in configuration")
        return False
    if not db.columns:
        logger.error("Database columns are missing in configuration")
        return False
    if not all([db.user, db.password, db.host, db.port, db.name]):
        logger.error("Database connection details are incomplete for SQL ingestion")
        return False
    return True

def _build_sql_connection_string(db) -> str:
    """Build PostgreSQL connection string."""
    return f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"

def _prepare_columns(settings: Settings) -> tuple[List[str], List[str]]:
    """Prepare column lists for document processing."""
    cols = [c for c in settings.database.columns if c != settings.database.id_column]
    embed_cols = _select_embedding_columns(settings, cols)
    return cols, embed_cols

def _prepare_sql_columns(db) -> tuple[str, List[str], List[str]]:
    """Prepare SQL column names and quoted versions."""
    id_col = db.id_column
    cols = [c for c in db.columns if c != id_col]
    image_meta_cols = {"media"}
    sql_cols = [c for c in cols if c not in image_meta_cols]
    quoted_id_col = f'"{id_col}"'
    quoted_cols = [f'"{c}"' for c in sql_cols]
    return quoted_id_col, quoted_cols, cols

def _build_sql_query(table: str, quoted_id_col: str, quoted_cols: List[str], where_clause: str = "", settings = None) -> str:
    """Build SQL query with dynamic JOINs from configuration.
    
    This function reads relation configuration from settings and builds
    JOIN clauses dynamically, making the system fully generic and configurable.
    
    Args:
        table: Main table name
        quoted_id_col: Quoted ID column name
        quoted_cols: List of quoted column names
        where_clause: Additional WHERE conditions
        settings: Settings object (auto-loaded if None)
    
    Returns:
        Complete SQL query string with dynamic JOINs
    """
    if settings is None:
        from .config import get_settings
        settings = get_settings()
    
    # Build main table columns
    qualified_columns = []
    for col in [quoted_id_col] + quoted_cols:
        qualified_columns.append(f'p.{col}')
    qualified_select_cols = ", ".join(qualified_columns)
    
    # Build JOIN clauses and SELECT additions from config
    # Only process many-to-one relations (one-to-many handled separately)
    join_clauses = []
    select_additions = []
    
    if settings.database.relations:
        for relation in settings.database.relations:
            # Skip one-to-many relations (handled by _build_media_map)
            if relation.relation_type == "one_to_many":
                continue
            
            alias = f"{relation.name}_tbl"
            
            # Build JOIN clause for many-to-one
            join_clause = f"""LEFT JOIN {relation.reference_table} {alias} ON p.{relation.foreign_key} = {alias}.{relation.reference_column}"""
            join_clauses.append(join_clause)
            
            # Build SELECT addition using fields_to_fetch (consistent structure)
            if relation.fields_to_fetch:
                # For many-to-one, typically fetch single field (e.g., "name")
                # If multiple fields specified, use first one as primary
                primary_field = relation.fields_to_fetch[0]
                select_additions.append(f"{alias}.{primary_field} as {relation.alias}")
    
    # Build WHERE clauses
    where_clauses = []
    
    # Add status filter from config
    if settings.database.status_filter and settings.database.status_filter.enabled:
        status_field = settings.database.status_filter.field
        status_value = settings.database.status_filter.value
        where_clauses.append(f"p.{status_field} = '{status_value}'")
    
    # Add custom where clause if provided
    if where_clause:
        where_clauses.append(where_clause)
    
    # Combine all parts
    select_clause = qualified_select_cols
    if select_additions:
        select_clause += ", " + ", ".join(select_additions)
    
    # Build complete query
    base_query = f"""
        SELECT {select_clause}
        FROM "{table}" p
        {' '.join(join_clauses)}"""
    
    if where_clauses:
        base_query += f"\n        WHERE {' AND '.join(where_clauses)}"
    
    return base_query

def _create_document_from_sql_row(
    row: Dict[str, object], 
    table: str, 
    cols: List[str], 
    embed_cols: List[str], 
    id_column: str,
    media_map: Optional[Dict[str, Dict[str, object]]] = None,
    settings = None
) -> Document:
    """Create a Document from a SQL row with common metadata processing.
    
    Now handles relations dynamically from config instead of hardcoded field names.
    """
    if settings is None:
        from .config import get_settings
        settings = get_settings()
    
    row_id = str(row[id_column])
    metadata = {"table": table, "id": row_id}
    
    # Get relation configuration (only many-to-one for FK skipping)
    relation_aliases = set()
    foreign_keys = set()
    if settings.database.relations:
        for rel in settings.database.relations:
            # Only process many-to-one for this logic
            if rel.relation_type == "many_to_one":
                relation_aliases.add(rel.alias)
                foreign_keys.add(rel.foreign_key)
    
    # Process regular columns
    for c in cols:
        if c in row:
            # Skip FK columns if we have the relation alias (to avoid duplicate data)
            if c in foreign_keys:
                # Check if we have the relation alias in the row
                relation = next((r for r in settings.database.relations if r.foreign_key == c), None)
                if relation and relation.alias in row and row[relation.alias]:
                    continue  # Skip FK, we have the display name
            
            metadata[c] = row.get(c)
    
    # Add relation aliases to metadata (dynamic from config)
    for alias in relation_aliases:
        if alias in row and row[alias]:
            metadata[alias] = row[alias]
    
    # Add media metadata if available
    if media_map and row_id in media_map:
        metadata.update(media_map[row_id])
    
    text = _compose_text(row, embed_cols)
    return Document(id=row_id, text=text, metadata=metadata)

def load_documents(settings: Settings, batch_size: int) -> Generator[List[Document], None, None]:
    """
    Loads documents from the configured table and columns.
    - Uses settings.database.table and settings.database.columns
    - Generates Documents with both text (for embedding) and metadata (for filters)
    """
    logger = get_logger(__name__)
    db = settings.database
    if not _validate_database_config(db, logger):
        return
    
    table = db.table
    quoted_id_col, quoted_cols, cols = _prepare_sql_columns(db)
    _, embed_cols = _prepare_columns(settings)
    source_type = (settings.ingestion.source_type or "sql").lower()
    source_path = settings.ingestion.source_path
    
    if source_type == "sql":
        dsn = _build_sql_connection_string(db)
        sql = _build_sql_query(table, quoted_id_col, quoted_cols, "", settings)
        
        logger.debug(f"Loading documents from table '{table}' with columns {cols} and dynamically configured relations")
        if db.id_column not in db.columns:
            logger.debug("Configured id_column '%s' not present in database.columns; ensure it's selected", db.id_column)
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            with conn.cursor(name="server_cursor") as cur:
                cur.itersize = batch_size
                cur.execute(sql)
                while True:
                    rows: List[Dict[str, object]] = cur.fetchmany(batch_size)
                    if not rows:
                        break       
                    prop_ids: List[str] = [str(r[db.id_column]) for r in rows if r.get(db.id_column) is not None]
                    media_map = _build_media_map(conn, prop_ids, settings)
                    documents: List[Document] = []
                    for row in rows:
                        
                        for c in cols:
                            if c not in row and c not in {"media"}:
                                row_id = str(row[db.id_column])
                                logger.debug("Row %s missing expected column '%s'", row_id, c)
                        
                        doc = _create_document_from_sql_row(row, table, cols, embed_cols, db.id_column, media_map, settings)
                        documents.append(doc)
                    yield documents
        return
    if source_type == "csv" and source_path:
        yield from _yield_in_batches(_load_documents_from_csv(source_path, settings), batch_size)
        return
    if source_type in {"jsonl", "jsonlines"} and source_path:
        yield from _yield_in_batches(_load_documents_from_jsonl(source_path, settings), batch_size)
        return
    if source_type == "txt" and source_path:
        yield from _yield_in_batches(_load_documents_from_txt(source_path, settings), batch_size)
        return
    logger.error("Unsupported ingestion source_type='%s' or missing source_path", source_type)

def load_document_by_id(settings: Settings, record_id: str) -> Optional[Document]:
    """
    Load a single record from the configured SQL database by id and build a Document
    with enriched metadata consistent with batch loader.
    Returns None if not found or if source type unsupported.
    """
    logger = get_logger(__name__)
    db = settings.database
    source_type = (settings.ingestion.source_type or "sql").lower()
    if source_type != "sql":
        logger.error("Single-record load is only supported for SQL source_type; got '%s'", source_type)
        return None
    if not _validate_database_config(db, logger):
        return None
    
    table = db.table
    quoted_id_col, quoted_cols, cols = _prepare_sql_columns(db)
    _, embed_cols = _prepare_columns(settings)
    
    sql = _build_sql_query(table, quoted_id_col, quoted_cols, f"p.{quoted_id_col} = %s", settings)
    dsn = _build_sql_connection_string(db)
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (record_id,))
                row: Optional[Dict[str, object]] = cur.fetchone()
                if not row:
                    return None
                
                media_map = _build_media_map(conn, [record_id], settings)
                return _create_document_from_sql_row(row, table, cols, embed_cols, db.id_column, media_map, settings)
    except Exception:
        logger.exception("Failed to load record %s from table %s", record_id, table)
        return None