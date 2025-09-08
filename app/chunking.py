from __future__ import annotations

from typing import Iterable, List

from .models import Document


def _split_text_to_tokens(text: str) -> List[str]:
    return text.split()


def _join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)


def chunk_documents(documents: Iterable[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    chunked: List[Document] = []
    for doc in documents:
        tokens = _split_text_to_tokens(doc.text)
        if not tokens:
            continue
        start = 0
        index = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            text_chunk = _join_tokens(tokens[start:end])
            chunk_id = f"{doc.id}:{index}"
            metadata = dict(doc.metadata)
            metadata.update({
                "source_id": doc.id,
                "chunk_index": index,
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
