from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from .chunking import chunk_documents
from .config import Settings, get_settings
from .embedding import EmbeddingClient
from .loader import load_documents
from .llm import LLMClient, build_prompt
from .models import RetrievedChunk
from .retriever import Retriever
from .utils import get_logger
from .vector_db import ElasticsearchVectorStore

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = get_logger(__name__)

app = FastAPI(title="RAG Pipeline")


class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None


class SourceItem(BaseModel):
    score: float
    text: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class IngestRequest(BaseModel):
    rebuild_index: bool = False
    batch_size: Optional[int] = None


class State:
    settings: Optional[Settings] = None
    embedder: Optional[EmbeddingClient] = None
    vector_store: Optional[ElasticsearchVectorStore] = None
    retriever: Optional[Retriever] = None
    llm: Optional[LLMClient] = None
    index_ready: bool = False


state = State()


def initialize_components() -> None:
    settings = get_settings(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"))
    state.settings = settings

    embedder = EmbeddingClient(settings.embedding)
    store = ElasticsearchVectorStore(settings.vector_db)
    retriever = Retriever(embedder, store, settings.retrieval)
    llm = LLMClient(settings.llm, api_key=settings.embedding.api_key)

    state.embedder = embedder
    state.vector_store = store
    state.retriever = retriever
    state.llm = llm


@app.on_event("startup")
async def on_startup() -> None:
    initialize_components()
    assert state.settings is not None
    if state.settings.app.eager_init:
        try:
            dims = state.vector_store._config.dims if state.vector_store and state.vector_store._config.dims else state.embedder.get_dimension()  # type: ignore[arg-type]
            state.vector_store.ensure_index(int(dims))  # type: ignore[arg-type]
            state.index_ready = True
        except Exception as exc:
            logger.warning("Failed eager index init: %s", exc)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    if not (state.retriever and state.llm):
        raise HTTPException(status_code=500, detail="Server not initialized")
    chunks: List[RetrievedChunk] = state.retriever.retrieve(request.query, filters=request.filters, top_k=request.top_k)
    prompt = build_prompt(request.query, chunks)
    answer = state.llm.generate(prompt)
    sources: List[SourceItem] = [SourceItem(score=c.score, text=c.text, metadata=c.metadata) for c in chunks]
    return QueryResponse(answer=answer, sources=sources)


def _run_ingestion(batch_size: int, rebuild_index: bool) -> None:
    if not (state.settings and state.embedder and state.vector_store):
        raise RuntimeError("Components not initialized")
    settings = state.settings
    if rebuild_index:
        try:
            dims = state.embedder.get_dimension()
            state.vector_store.ensure_index(dims)
            state.index_ready = True
        except Exception as exc:
            logger.error("Failed to prepare index: %s", exc)
            raise

    total_rows = 0
    total_chunks = 0
    total_indexed = 0

    for docs_batch in load_documents(settings, batch_size=batch_size):
        chunks = chunk_documents(
            docs_batch,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            unit=settings.chunking.unit,
        )
        if not chunks:
            continue
        embeddings = state.embedder.embed_texts([c.text for c in chunks], task_type="retrieval_document")
        success, errors = state.vector_store.upsert(chunks, embeddings)
        total_rows += len(docs_batch)
        total_chunks += len(chunks)
        total_indexed += success
        logger.info("Ingested rows=%s chunks=%s indexed=%s errors=%s", total_rows, total_chunks, total_indexed, errors)


@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    if not state.settings:
        raise HTTPException(status_code=500, detail="Server not initialized")
    batch_size = request.batch_size or state.settings.ingestion.batch_size
    background_tasks.add_task(_run_ingestion, batch_size=batch_size, rebuild_index=request.rebuild_index)
    return {"status": "started", "batch_size": batch_size, "rebuild_index": request.rebuild_index}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "index_ready": state.index_ready}
