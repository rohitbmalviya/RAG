from __future__ import annotations
from typing import Iterable, List
from .utils import get_logger
from .models import Document

def _split_text_to_tokens(text: str) -> List[str]:
    return text.split()

def _join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)

def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int,
    chunk_overlap: int,
    unit: str = "token",
) -> List[Document]:
    logger = get_logger(__name__)
    if chunk_size is None or chunk_size <= 0:
        logger.warning("chunk_documents called with invalid chunk_size=%s", chunk_size)
    if chunk_overlap is None or chunk_overlap < 0:
        logger.warning("chunk_documents called with invalid chunk_overlap=%s", chunk_overlap)
    if unit not in {"token", "char"}:
        logger.warning("chunk_documents received unknown unit='%s'; defaulting to 'token' logic", unit)
    chunked: List[Document] = []
    for doc in documents:
        index = 0
        if unit == "char":
            text = doc.text
            if not text:
                logger.warning("Document %s has empty text; skipping chunking", getattr(doc, "id", "<unknown>"))
                continue
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                text_chunk = text[start:end]
                chunk_id = f"{doc.id}:{index}"
                metadata = dict(doc.metadata)
                metadata.update({
                    "source_id": doc.id,
                    "chunk_index": index,
                    "chunk_offset": start,
                    "chunk_unit": unit,
                    "table": doc.metadata.get("table"),
                })
                chunked.append(Document(id=chunk_id, text=text_chunk, metadata=metadata))
                if end == len(text):
                    break
                start = end - chunk_overlap if chunk_overlap > 0 else end
                if start < 0:
                    start = 0
                index += 1
        else:
            tokens = _split_text_to_tokens(doc.text)
            if not tokens:
                logger.warning("Document %s has no tokens; skipping chunking", getattr(doc, "id", "<unknown>"))
                continue
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                text_chunk = _join_tokens(tokens[start:end])
                chunk_id = f"{doc.id}:{index}"
                metadata = dict(doc.metadata)
                metadata.update({
                    "source_id": doc.id,
                    "chunk_index": index,
                    "chunk_offset": start,
                    "chunk_unit": unit,
                    "table": doc.metadata.get("table"),
                })
                chunked.append(Document(id=chunk_id, text=text_chunk, metadata=metadata))
                if end == len(tokens):
                    break
                start = end - chunk_overlap if chunk_overlap > 0 else end
                if start < 0:
                    start = 0
                index += 1
    return chunked
