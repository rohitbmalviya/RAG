from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    score: float
    text: str
    metadata: Dict[str, Any]
