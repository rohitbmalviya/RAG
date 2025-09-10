from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
from .chunking import chunk_documents
from .config import Settings, get_settings
from .core.base import BaseEmbedder, BaseVectorStore, BaseLLM
from .loader import load_documents
from .llm import build_prompt
from .models import RetrievedChunk
from .retriever import Retriever
from .utils import get_logger
from .factories import build_embedder, build_llm, build_vector_store
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
logger = get_logger(__name__)
app = FastAPI(title="RAG Pipeline")
def main() -> None:
    initialize_components()
    if not (pipeline_state.embedder_client and pipeline_state.retriever_client and pipeline_state.llm_client):
        raise RuntimeError("Pipeline not initialized")
    sample_query = "What does the dataset say about pricing?"
    chunks: List[RetrievedChunk] = pipeline_state.retriever_client.retrieve(sample_query)
    prompt = build_prompt(sample_query, chunks)
    answer = pipeline_state.llm_client.generate(prompt)
    logger.info("Answer: %s", answer)
    for idx, c in enumerate(chunks[:5], start=1):
        logger.info("Source %s: score=%.3f id=%s", idx, c.score, c.metadata.get("id"))

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
    parallel: Optional[bool] = None
    source_type: Optional[str] = None
    source_path: Optional[str] = None

class PipelineState:
    settings: Optional[Settings] = None
    embedder_client: Optional[BaseEmbedder] = None
    vector_store_client: Optional[BaseVectorStore] = None
    retriever_client: Optional[Retriever] = None
    llm_client: Optional[BaseLLM] = None
    index_ready: bool = False

pipeline_state = PipelineState()

def initialize_components() -> None:
    settings = get_settings(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"))
    pipeline_state.settings = settings
    if not settings.database.columns:
        logger.error("database.columns must be configured in config.yaml")
        raise RuntimeError("database.columns must be configured")
    if settings.retrieval.filter_fields is None:
        logger.warning("retrieval.filter_fields is None; treating as empty list")
        settings.retrieval.filter_fields = []  
    try:
        embedder = build_embedder(settings.embedding, api_key=settings.embedding.api_key)
    except Exception as exc:
        logger.error("Failed to initialize embedder: %s", exc)
        raise
    try:
        store = build_vector_store(settings.vector_db)
    except Exception as exc:
        logger.error("Failed to initialize vector store: %s", exc)
        raise
    try:
        retriever = Retriever(embedder, store, settings.retrieval)
    except Exception as exc:
        logger.error("Failed to initialize retriever: %s", exc)
        raise
    try:
        llm = build_llm(settings.llm, fallback_api_key=settings.embedding.api_key)
    except Exception as exc:
        logger.error("Failed to initialize LLM: %s", exc)
        raise

    pipeline_state.embedder_client = embedder
    pipeline_state.vector_store_client = store
    pipeline_state.retriever_client = retriever
    pipeline_state.llm_client = llm


@app.on_event("startup")
async def on_startup() -> None:
    initialize_components()
    assert pipeline_state.settings is not None
    if pipeline_state.settings.app.eager_init:
        try:
            dims = pipeline_state.vector_store_client._config.dims if pipeline_state.vector_store_client and pipeline_state.vector_store_client._config.dims else pipeline_state.embedder_client.get_dimension()  
            pipeline_state.vector_store_client.ensure_index(int(dims))  
            pipeline_state.index_ready = True
        except Exception as exc:
            logger.warning("Failed eager index init: %s", exc)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    if not (pipeline_state.retriever_client and pipeline_state.llm_client):
        raise HTTPException(status_code=500, detail="Server not initialized")
    chunks: List[RetrievedChunk] = pipeline_state.retriever_client.retrieve(request.query, filters=request.filters, top_k=request.top_k)
    prompt = build_prompt(request.query, chunks)
    answer = pipeline_state.llm_client.generate(prompt)
    sources: List[SourceItem] = [SourceItem(score=c.score, text=c.text, metadata=c.metadata) for c in chunks]
    return QueryResponse(answer=answer, sources=sources)


def _run_ingestion(batch_size: int, rebuild_index: bool) -> None:
    if not (pipeline_state.settings and pipeline_state.embedder_client and pipeline_state.vector_store_client):
        raise RuntimeError("Components not initialized")
    settings = pipeline_state.settings
    try:
        dims = pipeline_state.vector_store_client._config.dims if pipeline_state.vector_store_client and pipeline_state.vector_store_client._config.dims else pipeline_state.embedder_client.get_dimension()  
        if rebuild_index:
            logger.info("Rebuilding index '%s'", pipeline_state.vector_store_client._config.index)
            pipeline_state.vector_store_client.delete_index()
        pipeline_state.vector_store_client.ensure_index(int(dims))  
        pipeline_state.index_ready = True
    except Exception as exc:
        logger.error("Failed to prepare index: %s", exc)
        raise

    total_rows = 0
    total_chunks = 0
    total_indexed = 0
    total_errors = 0

    for docs_batch in load_documents(settings, batch_size=batch_size):
        logger.info("Fetched %s rows from DB", len(docs_batch))
        chunks = chunk_documents(
            docs_batch,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            unit=settings.chunking.unit,
        )
        if not chunks:
            continue
        logger.info("Created %s chunks", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = pipeline_state.embedder_client.embed(texts, task_type="retrieval_document")
        logger.info("Generated %s embeddings", len(embeddings))
        success, errors = pipeline_state.vector_store_client.upsert(chunks, embeddings)
        if success > 0 and not pipeline_state.index_ready:
            pipeline_state.index_ready = True
        total_rows += len(docs_batch)
        total_chunks += len(chunks)
        total_indexed += success
        total_errors += int(errors)
        logger.info("Upserted %s chunks (errors=%s)", success, errors)
        logger.info("Ingested totals rows=%s chunks=%s indexed=%s errors=%s", total_rows, total_chunks, total_indexed, total_errors)


@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    if not pipeline_state.settings:
        raise HTTPException(status_code=500, detail="Server not initialized")
    if request.source_type:
        pipeline_state.settings.ingestion.source_type = request.source_type
    if request.source_path:
        pipeline_state.settings.ingestion.source_path = request.source_path
    batch_size = request.batch_size or pipeline_state.settings.ingestion.batch_size
    background_tasks.add_task(_run_ingestion, batch_size=batch_size, rebuild_index=request.rebuild_index)
    return {"status": "started", "batch_size": batch_size, "rebuild_index": request.rebuild_index}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "index_ready": pipeline_state.index_ready}
