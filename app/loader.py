from __future__ import annotations

import json
from typing import Dict, Generator, List, Optional

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
    for col in columns:
        value = row.get(col)
        if value is None:
            continue

        # Handle JSON/amenities-style fields nicely
        if isinstance(value, (dict, list)):
            value = _stringify_value(value)

        value_str = str(value).strip()
        if not value_str:
            continue

        # Make text more readable (capitalize field names)
        segments.append(f"{col.replace('_', ' ').title()}: {value_str}")

    return "\n".join(segments)


def load_documents(settings: Settings, batch_size: int) -> Generator[List[Document], None, None]:
    """
    Loads documents from the configured table and columns.
    - Uses settings.database.table and settings.database.columns
    - Generates Documents with both text (for embedding) and metadata (for filters)
    """
    logger = get_logger(__name__)

    db = settings.database
    dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
    table = db.table
    id_col = db.id_column
    cols = [c for c in db.columns if c != id_col]
    # Quote identifiers to preserve camelCase and special names
    quoted_id_col = f'"{id_col}"'
    quoted_cols = [f'"{c}"' for c in cols]
    select_cols = ", ".join([quoted_id_col] + quoted_cols)

    sql = f"SELECT {select_cols} FROM \"{table}\""

    logger.info(f"Loading documents from table '{table}' with columns {cols}")

    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor(name="server_cursor") as cur:
            cur.itersize = batch_size
            cur.execute(sql)

            while True:
                rows: List[Dict[str, object]] = cur.fetchmany(batch_size)
                if not rows:
                    break

                documents: List[Document] = []
                for row in rows:
                    row_id = str(row[id_col])

                    # metadata = full row
                    metadata = {"table": table, "id": row_id}
                    for c in cols:
                        metadata[c] = row.get(c)

                    # text = composed embedding string
                    text = _compose_text(row, cols)

                    documents.append(Document(id=row_id, text=text, metadata=metadata))

                yield documents
