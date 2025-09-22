from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
from .query_processor import preprocess_query
from .chunking import chunk_documents
from .config import Settings, get_settings
from .core.base import BaseEmbedder, BaseVectorStore, BaseLLM
from .loader import load_documents, load_document_by_id
from .llm import LLMClient
from .models import RetrievedChunk
from .retriever import Retriever
from .utils import get_logger
from .factories import build_embedder, build_llm, build_vector_store
from fastapi.middleware.cors import CORSMiddleware

# Constants to eliminate duplication
SERVER_NOT_INITIALIZED = "Server not initialized"
COMPONENTS_NOT_INITIALIZED = "Components not initialized"
RETRIEVAL_DOCUMENT_TASK = "retrieval_document"
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
logger = get_logger(__name__)
app = FastAPI(title="RAG Pipeline")

# Enable CORS for frontend running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    session_id: Optional[str] = None

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
    source_type: Optional[str] = None
    source_path: Optional[str] = None

class UpsertOneResponse(BaseModel):
    status: str
    indexed_chunks: int
    errors: int

class DeleteResponse(BaseModel):
    status: str
    deleted: int

class PipelineState:
    settings: Optional[Settings] = None
    embedder_client: Optional[BaseEmbedder] = None
    vector_store_client: Optional[BaseVectorStore] = None
    retriever_client: Optional[Retriever] = None
    llm_client: Optional[BaseLLM] = None
    index_ready: bool = False

pipeline_state = PipelineState()

# Session management for conversations
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self, ttl_minutes: int = 30):
        self.sessions = {}
        self.last_activity = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def get_or_create_session(self, session_id: str):
        self._cleanup_expired()
        if session_id not in self.sessions:
            from .factories import build_llm
            self.sessions[session_id] = build_llm(
                pipeline_state.settings.llm,
                fallback_api_key=pipeline_state.settings.embedding.api_key
            )
        self.last_activity[session_id] = datetime.now()
        return self.sessions[session_id]
    
    def _cleanup_expired(self):
        now = datetime.now()
        expired = [
            sid for sid, last in self.last_activity.items()
            if now - last > self.ttl
        ]
        for sid in expired:
            del self.sessions[sid]
            del self.last_activity[sid]
            logger.info(f"Cleaned up expired session: {sid}")

# Replace global conversation_sessions with:
session_manager = SessionManager(ttl_minutes=30)

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
            dims = _get_vector_dimensions()
            pipeline_state.vector_store_client.ensure_index(dims)  
            pipeline_state.index_ready = True
        except Exception as exc:
            logger.warning("Failed eager index init: %s", exc)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    if not (pipeline_state.retriever_client and pipeline_state.llm_client):
        raise HTTPException(status_code=500, detail=SERVER_NOT_INITIALIZED)
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    # Get or create session with proper LLM client initialization
    session_id = request.session_id or "default"
    llm_client = session_manager.get_or_create_session(session_id)
    # Process query using dynamic LLM approach
    normalized_query, filters = preprocess_query(request.query, llm_client)
    # Retrieve chunks for property queries
    chunks: List[RetrievedChunk] = pipeline_state.retriever_client.retrieve(
        normalized_query,
        filters=filters,
        top_k=request.top_k,
    )
    # Use dynamic LLM chat method which handles classification internally
    answer = llm_client.chat(normalized_query, retrieved_chunks=chunks)
    
    # Determine if sources should be shown based on query type
    show_sources = True
    query_lower = request.query.lower()
    
    # Don't show sources for certain query types
    if any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
        show_sources = False
    elif "what is" in query_lower or "define" in query_lower:
        show_sources = False
    elif "average" in query_lower and "price" in query_lower:
        show_sources = False
    
    # Filter sources based on query type
    if show_sources:
        filtered_chunks = [c for c in chunks if getattr(c, "score", 1.0) >= 0.7]
        sources: List[SourceItem] = [
            SourceItem(score=c.score, text=c.text, metadata=c.metadata)
            for c in filtered_chunks
        ]
    else:
        sources = []
    
    return QueryResponse(answer=answer, sources=sources)

def _validate_components() -> None:
    """Validate that all required components are initialized"""
    if not (pipeline_state.settings and pipeline_state.embedder_client and pipeline_state.vector_store_client):
        raise RuntimeError(COMPONENTS_NOT_INITIALIZED)

def _get_vector_dimensions() -> int:
    """Get vector dimensions from config or embedder"""
    if pipeline_state.vector_store_client and pipeline_state.vector_store_client._config.dims:
        return int(pipeline_state.vector_store_client._config.dims)
    return pipeline_state.embedder_client.get_dimension()

def _process_documents_to_vectors(documents: List[Any], settings: Settings) -> tuple[List[Any], List[List[float]]]:
    """Process documents through chunking and embedding pipeline"""
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        unit=settings.chunking.unit,
    )
    if not chunks:
        return [], []
    
    texts = [c.text for c in chunks]
    embeddings = pipeline_state.embedder_client.embed(texts, task_type=RETRIEVAL_DOCUMENT_TASK)
    return chunks, embeddings

def _upsert_chunks_to_vector_store(chunks: List[Any], embeddings: List[List[float]], refresh: bool = False) -> tuple[int, int]:
    """Upsert chunks and embeddings to vector store"""
    return pipeline_state.vector_store_client.upsert(chunks, embeddings, refresh=refresh)

def _prepare_index(rebuild_index: bool = False) -> None:
    """Prepare the vector index with proper error handling"""
    try:
        dims = _get_vector_dimensions()
        if rebuild_index:
            logger.info("Rebuilding index '%s'", pipeline_state.vector_store_client._config.index)
            pipeline_state.vector_store_client.delete_index()
        pipeline_state.vector_store_client.ensure_index(dims)
        pipeline_state.index_ready = True
    except Exception as exc:
        logger.error("Failed to prepare index: %s", exc)
        raise

def _run_ingestion(batch_size: int, rebuild_index: bool) -> None:
    _validate_components()
    settings = pipeline_state.settings
    _prepare_index(rebuild_index)
    total_rows = 0
    total_chunks = 0
    total_indexed = 0
    total_errors = 0
    for docs_batch in load_documents(settings, batch_size=batch_size):
        logger.info("Fetched %s rows from DB", len(docs_batch))
        chunks, embeddings = _process_documents_to_vectors(docs_batch, settings)
        if not chunks:
            continue
        logger.info("Created %s chunks", len(chunks))
        logger.info("Generated %s embeddings", len(embeddings))
        success, errors = _upsert_chunks_to_vector_store(chunks, embeddings)
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
        raise HTTPException(status_code=500, detail=SERVER_NOT_INITIALIZED)
    if request.source_type:
        pipeline_state.settings.ingestion.source_type = request.source_type
    if request.source_path:
        pipeline_state.settings.ingestion.source_path = request.source_path
    batch_size = request.batch_size or pipeline_state.settings.ingestion.batch_size
    background_tasks.add_task(_run_ingestion, batch_size=batch_size, rebuild_index=request.rebuild_index)
    return {"status": "started", "batch_size": batch_size, "rebuild_index": request.rebuild_index}

@app.post("/ingest/{property_id}", response_model=UpsertOneResponse)
async def ingest_single_property(property_id: str) -> UpsertOneResponse:
    try:
        _validate_components()
        settings = pipeline_state.settings
        _prepare_index()
    except Exception as exc:
        logger.error("Failed to prepare index: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to prepare index")

    # Load single document by id
    doc = load_document_by_id(settings, property_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Property not found")

    # Chunk and embed
    chunks, embeddings = _process_documents_to_vectors([doc], settings)
    if not chunks:
        return UpsertOneResponse(status="no_content", indexed_chunks=0, errors=0)
    success, errors = _upsert_chunks_to_vector_store(chunks, embeddings, refresh=True)
    return UpsertOneResponse(status="ok", indexed_chunks=int(success), errors=int(errors))

@app.delete("/vectors/{property_id}", response_model=DeleteResponse)
async def delete_vectors_by_id(property_id: str) -> DeleteResponse:
    if not pipeline_state.vector_store_client:
        raise HTTPException(status_code=500, detail=SERVER_NOT_INITIALIZED)
    try:
        deleted = pipeline_state.vector_store_client.delete_by_source_id(property_id)
    except Exception as exc:
        logger.error("Failed to delete vectors for %s: %s", property_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete vectors")
    return DeleteResponse(status="ok", deleted=deleted)

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "index_ready": pipeline_state.index_ready}
