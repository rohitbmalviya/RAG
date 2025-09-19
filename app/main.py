from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from .query_processor import preprocess_query
from .chunking import chunk_documents
from .config import Settings, get_settings
from .core.base import BaseEmbedder, BaseVectorStore, BaseLLM
from .loader import load_documents
from .llm import build_prompt, filter_retrieved_chunks, is_property_query
from .models import RetrievedChunk
from .retriever import Retriever
from .utils import get_logger
from .factories import build_embedder, build_llm, build_vector_store
from fastapi.middleware.cors import CORSMiddleware
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
    parallel: Optional[bool] = None
    source_type: Optional[str] = None
    source_path: Optional[str] = None

class RequirementRequest(BaseModel):
    user_query: str
    preferences: Dict[str, Any]
    conversation_summary: str
    session_id: Optional[str] = None

class RequirementResponse(BaseModel):
    status: str
    message: str

class PipelineState:
    settings: Optional[Settings] = None
    embedder_client: Optional[BaseEmbedder] = None
    vector_store_client: Optional[BaseVectorStore] = None
    retriever_client: Optional[Retriever] = None
    llm_client: Optional[BaseLLM] = None
    index_ready: bool = False

pipeline_state = PipelineState()

# Session management for conversations
conversation_sessions: Dict[str, Any] = {}

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
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    
    # Get or create session
    session_id = request.session_id or "default"
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = pipeline_state.llm_client
    
    llm_client = conversation_sessions[session_id]
    
    normalized_query, filters = preprocess_query(request.query, llm_client)
    
    # Check if it's a property query
    if is_property_query(normalized_query):
        chunks: List[RetrievedChunk] = pipeline_state.retriever_client.retrieve(
            normalized_query,
            filters=filters,
            top_k=request.top_k,
        )
        answer = llm_client.chat(normalized_query, retrieved_chunks=chunks)
        filtered_chunks = filter_retrieved_chunks(chunks, min_score=0.7)
        sources: List[SourceItem] = [
            SourceItem(score=c.score, text=c.text, metadata=c.metadata)
            for c in filtered_chunks
        ]
        return QueryResponse(answer=answer, sources=sources)
    else:
        answer = llm_client.chat(normalized_query)
        return QueryResponse(answer=answer, sources=[])

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


@app.post("/requirements", response_model=RequirementResponse)
async def requirements_endpoint(request: RequirementRequest) -> RequirementResponse:
    """Endpoint to save user requirements"""
    try:
        # In a real implementation, you would save this to a database
        # For now, we'll just log it
        logger.info(f"User requirement saved: {request.user_query}")
        logger.info(f"Preferences: {request.preferences}")
        logger.info(f"Conversation summary: {request.conversation_summary}")
        
        return RequirementResponse(
            status="success",
            message="Requirements saved successfully. Our team will work with agencies to find matching properties."
        )
    except Exception as exc:
        logger.error(f"Failed to save requirements: {exc}")
        return RequirementResponse(
            status="error",
            message="Failed to save requirements. Please try again later."
        )

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "index_ready": pipeline_state.index_ready}
