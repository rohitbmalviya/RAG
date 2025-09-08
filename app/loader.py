from __future__ import annotations

import json
from typing import Dict, Generator, Iterable, List, Optional

import psycopg
from psycopg.rows import dict_row

from .config import Settings
from .models import Document
from .utils import get_logger


def _stringify_amenities(raw: Optional[object]) -> str:
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
                    items = ", ".join([str(v) for v in value])
                    parts.append(f"{key}: {items}")
                else:
                    parts.append(f"{key}: {value}")
            return "; ".join(parts)
        if isinstance(data, list):
            return ", ".join([str(v) for v in data])
        return str(data)
    except Exception:
        return str(raw)


def _compose_text(row: Dict[str, object], table: str) -> str:
    title = str(row.get("property_title") or "").strip()
    description = str(row.get("property_description") or "").strip()
    emirate = str(row.get("emirate") or "").strip()
    city = str(row.get("city") or "").strip()
    district = str(row.get("district") or "").strip()
    property_type_id = str(row.get("property_type_id") or "").strip()
    rent_charge = str(row.get("rent_charge") or "").strip()
    lease_duration = str(row.get("lease_duration") or "").strip()
    amenities = _stringify_amenities(row.get("amenities"))

    segments: List[str] = []
    if title:
        segments.append(f"Title: {title}")
    if description:
        segments.append(f"Description: {description}")
    loc_parts = [p for p in [emirate, city, district] if p]
    if loc_parts:
        segments.append(f"Location: {', '.join(loc_parts)}")
    type_parts = []
    if property_type_id:
        type_parts.append(f"TypeId: {property_type_id}")
    if rent_charge:
        type_parts.append(f"Rent: {rent_charge}")
    if lease_duration:
        type_parts.append(f"Lease: {lease_duration}")
    if type_parts:
        segments.append(" | ".join(type_parts))
    if amenities:
        segments.append(f"Amenities: {amenities}")

    return "\n".join(segments)


def load_documents(settings: Settings, batch_size: int) -> Generator[List[Document], None, None]:
    logger = get_logger(__name__)

    db = settings.database
    dsn = (
        f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
    )
    table = db.table
    id_col = db.id_column
    cols = [c for c in db.columns if c != id_col]
    select_cols = ", ".join([id_col] + cols)

    sql = f"SELECT {select_cols} FROM {table}"

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
                    metadata = {
                        "table": table,
                        "id": row_id,
                    }
                    for c in cols:
                        metadata[c] = row.get(c)
                    text = _compose_text(row, table)
                    documents.append(
                        Document(id=row_id, text=text, metadata=metadata)
                    )
                yield documents
