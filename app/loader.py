from __future__ import annotations
import json
from typing import Dict, Generator, List, Optional, Iterable
import psycopg
from psycopg.rows import dict_row
from .config import Settings
from .models import Document
from .utils import get_logger

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
    Dynamically compose a text representation of a row based on configured columns.
    Instead of hardcoding, it uses whatever columns are listed in config.yml.
    """
    segments: List[str] = []
    amenities: List[str] = []
    
    for col in columns:
        value = row.get(col)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            value = _stringify_value(value)

        value_str = str(value).strip()
        if not value_str:
            continue
            
        # Handle boolean amenities specially
        if col in {
            "maids_room", "security_available", "concierge_available", "central_ac_heating",
            "elevators", "balcony_terrace", "storage_room", "laundry_room", "gym_fitness_center",
            "childrens_play_area", "bbq_area", "pet_friendly", "smart_home_features",
            "beach_access", "jogging_cycling_tracks", "mosque_nearby", "waste_disposal_system",
            "power_backup", "chiller_included", "sublease_allowed"
        }:
            if value_str.lower() in {"true", "1", "yes"}:
                amenity_name = col.replace('_', ' ').replace('available', '').replace('  ', ' ').strip()
                amenities.append(amenity_name.title())
            continue
            
        # Handle property/rent type value fields and legacy IDs
        if col in {"property_type_id", "property_type"}:
            property_type_name = row.get("property_type_name") or (value_str if col == "property_type" else None)
            if property_type_name:
                segments.append(f"Property Type: {property_type_name}")
                continue
        elif col == "rent_type_id":
            rent_type_name = row.get("rent_type_name")
            if rent_type_name:
                segments.append(f"Rent Type: {rent_type_name}")
                continue
        
        # Handle special formatting for certain fields
        if col == "rent_charge":
            segments.append(f"Annual Rent: AED {value_str}")
        elif col == "security_deposit":
            segments.append(f"Security Deposit: AED {value_str}")
        elif col == "maintenance_charge":
            segments.append(f"Maintenance Charge: AED {value_str}")
        elif col == "property_size":
            segments.append(f"Property Size: {value_str} sq.ft.")
        elif col == "year_built":
            segments.append(f"Year Built: {value_str}")
        elif col == "lease_duration":
            segments.append(f"Lease Duration: {value_str}")
        elif col == "available_from":
            segments.append(f"Available From: {value_str}")
        elif col == "developer_name":
            segments.append(f"Developer: {value_str}")
        elif col in {"parking", "swimming_pool", "public_transport_type", "retail_shopping_access"}:
            # Handle JSON fields for amenities
            if value_str and value_str != "null":
                segments.append(f"{col.replace('_', ' ').title()}: {value_str}")
        else:
            segments.append(f"{col.replace('_', ' ').title()}: {value_str}")
    
    # Add amenities as a single section
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

def _extract_media_metadata(src: Dict[str, object]) -> Dict[str, object]:
    """
    Extract media objects from a 'media' array if present. Produces:
    - media: top 5 objects {id, file_name, thumbnail_url}
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
            if not isinstance(item, dict):
                continue
            media_entry: Dict[str, object] = {}
            if item.get("id") is not None:
                media_entry["id"] = item.get("id")
            if item.get("file_name") is not None:
                media_entry["file_name"] = item.get("file_name")
            if item.get("thumbnail_url") is not None:
                media_entry["thumbnail_url"] = item.get("thumbnail_url")
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

def _build_media_map(conn: "psycopg.Connection", property_ids: List[str]) -> Dict[str, Dict[str, object]]:
    """
    Attempt to fetch media rows for given property_ids from table 'property_media'.
    Returns a map: propertyId -> {image_urls[:5]}
    Silently returns empty on any failure (e.g., table missing).
    """
    media_map: Dict[str, Dict[str, object]] = {}
    if not property_ids:
        return media_map
    try:
        with conn.cursor(row_factory=dict_row) as mcur:
            sql = (
                'SELECT "propertyId", "id", "file_name", "thumbnail_url", "order" '
                'FROM "property_media" WHERE "propertyId" = ANY(%s) '
                'ORDER BY "propertyId", COALESCE("order", 2147483647), "order", "id"'
            )
            mcur.execute(sql, (property_ids,))
            rows: List[Dict[str, object]] = mcur.fetchall()
        tmp: Dict[str, List[Dict[str, object]]] = {}
        for r in rows:
            pid = str(r.get("propertyId") or "")
            if not pid:
                continue
            tmp.setdefault(pid, []).append(r)
        for pid, items in tmp.items():
            media_objs: List[Dict[str, object]] = []
            for item in items:
                media_entry: Dict[str, object] = {}
                if item.get("id") is not None:
                    media_entry["id"] = item.get("id")
                if item.get("file_name") is not None:
                    media_entry["file_name"] = item.get("file_name")
                if item.get("thumbnail_url") is not None:
                    media_entry["thumbnail_url"] = item.get("thumbnail_url")
                if media_entry:
                    media_objs.append(media_entry)
            entry: Dict[str, object] = {}
            if media_objs:
                entry["media"] = media_objs[:5]
            if entry:
                media_map[pid] = entry
    except Exception:
        # Swallow errors (table missing or permission issues)
        return {}
    return media_map

def _load_documents_from_csv(path: str, settings: Settings) -> Iterable[Document]:
    import csv
    logger = get_logger(__name__)
    cols = [c for c in settings.database.columns if c != settings.database.id_column]
    embed_cols = _select_embedding_columns(settings, cols)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = str(row.get(settings.database.id_column) or row.get("id") or "")
            metadata: Dict[str, object] = {"table": settings.database.table or "file", "id": row_id}
            for c in cols:
                metadata[c] = row.get(c)
            # Derive media metadata if a 'media' JSON column exists in CSV
            if "media" in row and row["media"]:
                try:
                    media_md = _extract_media_metadata({"media": row["media"]})
                    metadata.update(media_md)
                except Exception:
                    pass
            text = _compose_text(row, embed_cols)
            yield Document(id=row_id, text=text, metadata=metadata)

def _load_documents_from_jsonl(path: str, settings: Settings) -> Iterable[Document]:
    logger = get_logger(__name__)
    cols = [c for c in settings.database.columns if c != settings.database.id_column]
    embed_cols = _select_embedding_columns(settings, cols)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            row_id = str(data.get(settings.database.id_column) or data.get("id") or "")
            metadata: Dict[str, object] = {"table": settings.database.table or "file", "id": row_id}
            for c in cols:
                metadata[c] = data.get(c)
            # Extract media-based image fields for downstream UI usage
            media_md = _extract_media_metadata(data)
            if media_md:
                metadata.update(media_md)
            text = _compose_text(data, embed_cols)
            yield Document(id=row_id, text=text, metadata=metadata)

def _load_documents_from_txt(path: str, settings: Settings) -> Iterable[Document]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    meta: Dict[str, object] = {"table": settings.database.table or "file", "id": "0"}
    yield Document(id="0", text=text, metadata=meta)

def load_documents(settings: Settings, batch_size: int) -> Generator[List[Document], None, None]:
    """
    Loads documents from the configured table and columns.
    - Uses settings.database.table and settings.database.columns
    - Generates Documents with both text (for embedding) and metadata (for filters)
    """
    logger = get_logger(__name__)
    db = settings.database
    if not db.table:
        logger.error("Database table name is missing in configuration")
    if not db.columns:
        logger.error("Database columns are missing in configuration")
    table = db.table
    id_col = db.id_column
    cols = [c for c in db.columns if c != id_col]
    embed_cols = _select_embedding_columns(settings, cols)
    # Some fields like images are derived from related tables or JSON inputs and
    # do not exist as columns in SQL. Exclude them from SELECT to avoid errors.
    image_meta_cols = {"media"}
    sql_cols = [c for c in cols if c not in image_meta_cols]
    quoted_id_col = f'"{id_col}"'
    quoted_cols = [f'"{c}"' for c in sql_cols]
    source_type = (settings.ingestion.source_type or "sql").lower()
    source_path = settings.ingestion.source_path
    if source_type == "sql":
        if not all([db.user, db.password, db.host, db.port, db.name]):
            logger.error("Database connection details are incomplete for SQL ingestion")
        dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
        
        # Build JOIN query to get property type and rent type names
        # Qualify all column names with table alias to avoid ambiguity
        qualified_columns = []
        for col in [quoted_id_col] + quoted_cols:
            qualified_columns.append(f'p.{col}')
        qualified_select_cols = ", ".join(qualified_columns)
        
        sql = f"""
        SELECT {qualified_select_cols},
               pt.name as property_type_name,
               prt.name as rent_type_name
        FROM "{table}" p
        LEFT JOIN property_types pt ON p.property_type_id = pt.id
        LEFT JOIN property_rent_types prt ON p.rent_type_id = prt.id
        WHERE p.property_status = 'listed'
        """
        logger.info(f"Loading documents from table '{table}' with columns {cols} and joined property/rent type names")
        if id_col not in db.columns:
            logger.warning("Configured id_column '%s' not present in database.columns; ensure it's selected", id_col)
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            with conn.cursor(name="server_cursor") as cur:
                cur.itersize = batch_size
                cur.execute(sql)
                while True:
                    rows: List[Dict[str, object]] = cur.fetchmany(batch_size)
                    if not rows:
                        break
                    # Build media map for this batch (if media table exists)
                    prop_ids: List[str] = [str(r[id_col]) for r in rows if r.get(id_col) is not None]
                    media_map = _build_media_map(conn, prop_ids)
                    documents: List[Document] = []
                    for row in rows:
                        row_id = str(row[id_col])
                        metadata = {"table": table, "id": row_id}
                        for c in cols:
                            if c in row:
                                metadata[c] = row.get(c)
                            else:
                                # Skip warning for derived image metadata fields
                                if c not in image_meta_cols:
                                    logger.warning("Row %s missing expected column '%s'", row_id, c)

                        # Enrich from media_map if available
                        media_md = media_map.get(row_id)
                        if media_md:
                            metadata.update(media_md)

                        text = _compose_text(row, embed_cols)
                        documents.append(Document(id=row_id, text=text, metadata=metadata))
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
    if not all([db.user, db.password, db.host, db.port, db.name]):
        logger.error("Database connection details are incomplete for SQL ingestion")
        return None
    table = db.table
    id_col = db.id_column
    cols = [c for c in db.columns if c != id_col]
    embed_cols = _select_embedding_columns(settings, cols)
    quoted_id_col = f'"{id_col}"'
    quoted_cols = [f'"{c}"' for c in cols if c != "media"]
    # Qualify all column names with table alias to avoid ambiguity
    qualified_columns: List[str] = [f'p.{quoted_id_col}'] + [f'p.{c}' for c in quoted_cols]
    qualified_select_cols = ", ".join(qualified_columns)
    sql = f"""
        SELECT {qualified_select_cols},
               pt.name as property_type_name,
               prt.name as rent_type_name
        FROM "{table}" p
        LEFT JOIN property_types pt ON p.property_type_id = pt.id
        LEFT JOIN property_rent_types prt ON p.rent_type_id = prt.id
        WHERE p.{quoted_id_col} = %s AND p.property_status = 'listed'
    """
    dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (record_id,))
                row: Optional[Dict[str, object]] = cur.fetchone()
                if not row:
                    return None
                row_id = str(row[id_col])
                metadata: Dict[str, object] = {"table": table, "id": row_id}
                for c in cols:
                    if c in row:
                        metadata[c] = row.get(c)
                # Enrich media metadata if media table exists
                media_map = _build_media_map(conn, [row_id])
                if row_id in media_map:
                    metadata.update(media_map[row_id])
                text = _compose_text(row, embed_cols)
                return Document(id=row_id, text=text, metadata=metadata)
    except Exception:
        logger.exception("Failed to load record %s from table %s", record_id, table)
        return None