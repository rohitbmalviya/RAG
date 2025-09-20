from __future__ import annotations
from typing import Iterable, List, Union
from .utils import get_logger
from .models import Document

# Constants to eliminate duplication
TOKEN_UNIT = "token"
CHAR_UNIT = "char"

def _split_text_to_tokens(text: str) -> List[str]:
    """Split text into tokens (words)."""
    return text.split()

def _join_tokens(tokens: List[str]) -> str:
    """Join tokens back into text."""
    return " ".join(tokens)

def _create_chunk_metadata(
    doc: Document, 
    chunk_index: int, 
    chunk_offset: int, 
    unit: str
) -> dict:
    """Create metadata for a chunk."""
    metadata = dict(doc.metadata)
    metadata.update({
        "source_id": doc.id,
        "chunk_index": chunk_index,
        "chunk_offset": chunk_offset,
        "chunk_unit": unit,
        "table": doc.metadata.get("table"),
    })
    return metadata

def _chunk_sequence(
    sequence: Union[str, List[str]], 
    chunk_size: int, 
    chunk_overlap: int,
    is_text: bool = False
) -> List[tuple[str, int]]:
    """Generic chunking function that works with both text and token sequences.
    Returns:
        List of tuples containing (chunk_text, chunk_offset)
    """
    if not sequence:
        return []
    chunks = []
    start = 0
    while start < len(sequence):
        end = min(start + chunk_size, len(sequence))
        if is_text:
            chunk = sequence[start:end]
        else:
            chunk = _join_tokens(sequence[start:end])
        chunks.append((chunk, start))
        if end == len(sequence):
            break  
        start = end - chunk_overlap if chunk_overlap > 0 else end
        if start < 0:
            start = 0
    return chunks

def _validate_document_content(doc: Document, unit: str, logger) -> bool:
    """Validate document content and log warnings if invalid."""
    doc_id = getattr(doc, "id", "<unknown>")
    if unit == CHAR_UNIT:
        if not doc.text:
            logger.warning("Document %s has empty text; skipping chunking", doc_id)
            return False
    else:
        tokens = _split_text_to_tokens(doc.text)
        if not tokens:
            logger.warning("Document %s has no tokens; skipping chunking", doc_id)
            return False
    return True

def _process_document_chunks(
    doc: Document, 
    chunks: List[tuple[str, int]], 
    unit: str
) -> List[Document]:
    """Process chunks into Document objects."""
    chunked_docs = []
    for index, (chunk_text, chunk_offset) in enumerate(chunks):
        chunk_id = f"{doc.id}:{index}"
        metadata = _create_chunk_metadata(doc, index, chunk_offset, unit)
        chunked_docs.append(Document(id=chunk_id, text=chunk_text, metadata=metadata))
    return chunked_docs

def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int,
    chunk_overlap: int,
    unit: str = TOKEN_UNIT,
) -> List[Document]:
    """Chunk documents into smaller pieces with optional overlap."""
    logger = get_logger(__name__)
    # Validate parameters
    if chunk_size is None or chunk_size <= 0:
        logger.warning("chunk_documents called with invalid chunk_size=%s", chunk_size)
    if chunk_overlap is None or chunk_overlap < 0:
        logger.warning("chunk_documents called with invalid chunk_overlap=%s", chunk_overlap)
    if unit not in {TOKEN_UNIT, CHAR_UNIT}:
        logger.warning("chunk_documents received unknown unit='%s'; defaulting to 'token' logic", unit)
    chunked: List[Document] = []
    for doc in documents:
        # Validate document content
        if not _validate_document_content(doc, unit, logger):
            continue
        # Get chunks based on unit type
        if unit == CHAR_UNIT:
            chunks = _chunk_sequence(doc.text, chunk_size, chunk_overlap, is_text=True)
        else:
            tokens = _split_text_to_tokens(doc.text)
            chunks = _chunk_sequence(tokens, chunk_size, chunk_overlap, is_text=False)
        # Process chunks into Document objects
        chunked.extend(_process_document_chunks(doc, chunks, unit))
    return chunked
