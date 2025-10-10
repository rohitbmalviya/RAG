from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from .chunking import chunk_documents
from .config import Settings, get_settings
from .core.base import BaseEmbedder, BaseVectorStore, BaseLLM
from .loader import load_documents, load_document_by_id
from .monitoring import get_metrics_collector, get_error_handler, get_performance_monitor, QueryMetrics
from .retriever import Retriever
from .utils import get_logger
from .factories import build_embedder, build_llm, build_vector_store
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from collections import defaultdict

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
app = FastAPI(
    title="RAG Pipeline - LeaseOasis",
    description="Intelligent property search assistant for UAE real estate leasing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Simple rate limiter implementation (compatible with all FastAPI versions)
class SimpleRateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request from this IP is allowed."""
        now = datetime.now()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < timedelta(seconds=self.window_seconds)
        ]
        
        # Check if under limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Add this request
        self.requests[client_ip].append(now)
        return True
    
    def cleanup_old_entries(self):
        """Cleanup old IP entries to prevent memory growth."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds * 2)
        
        ips_to_remove = []
        for ip, requests in self.requests.items():
            if not requests or all(req_time < cutoff for req_time in requests):
                ips_to_remove.append(ip)
        
        for ip in ips_to_remove:
            del self.requests[ip]

# Initialize simple rate limiter
rate_limiter = SimpleRateLimiter(max_requests=60, window_seconds=60)

# Enable CORS for frontend running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language query about UAE properties",
        example="Find me 2-bedroom apartments in Dubai Marina under 150k"
    )
    top_k: Optional[int] = Field(
        None,
        description="Number of top results to return (default: 12)",
        example=5,
        ge=1,
        le=50
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity",
        example="user-session-abc123"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "query": "Show me 2-bedroom apartments in Dubai Marina",
                    "top_k": 5,
                    "session_id": "user-123"
                },
                {
                    "query": "What is the average rent in Dubai?",
                    "session_id": "user-456"
                },
                {
                    "query": "Find villas with pool in Palm Jumeirah under 200k",
                    "top_k": 10,
                    "session_id": "user-789"
                }
            ]
        }

class SourceItem(BaseModel):
    score: float = Field(
        ...,
        description="Relevance score (0.0 - 1.0)",
        example=0.95,
        ge=0.0,
        le=1.0
    )
    text: str = Field(
        ...,
        description="Property description from database",
        example="Property: Luxury Apartment\nLocated in: Dubai, Dubai Marina\nRent: AED 120,000/year\n2 bedrooms, 2 bathrooms\nFurnishing: Furnished\nAmenities: Pool, Gym, Parking"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Property metadata with all fields",
        example={
            "id": "prop-uuid-123",
            "property_title": "Luxury Apartment in Marina",
            "property_type_name": "apartment",
            "emirate": "dubai",
            "community": "dubai marina",
            "rent_charge": 120000,
            "number_of_bedrooms": 2,
            "number_of_bathrooms": 2,
            "furnishing_status": "furnished",
            "swimming_pool": True,
            "gym_fitness_center": True,
            "parking": True
        }
    )

class QueryResponse(BaseModel):
    answer: str = Field(
        ...,
        description="AI-generated response to user query",
        example="I found 3 great apartments in Dubai Marina that match your criteria! Here are the top options:\n\n1. **Luxury Apartment in Marina** - AED 120,000/year\n   - 2 bedrooms, 2 bathrooms\n   - Furnished with pool, gym, and parking\n   - Available immediately\n\n2. **Marina Heights Residence** - AED 135,000/year\n   - 2 bedrooms, 2 bathrooms\n   - Sea view, balcony, and concierge service\n\nWould you like more details about any of these properties?"
    )
    sources: List[SourceItem] = Field(
        ...,
        description="Property sources with metadata (empty for greetings/general knowledge)",
        example=[]
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "answer": "I found 3 apartments matching your criteria in Dubai Marina...",
                    "sources": [
                        {
                            "score": 0.95,
                            "text": "Property: Luxury Apartment\nRent: AED 120,000/year",
                            "metadata": {
                                "property_title": "Luxury Apartment",
                                "rent_charge": 120000,
                                "emirate": "dubai"
                            }
                        }
                    ]
                },
                {
                    "answer": "Hello! I'm LeaseOasis. How can I help you find the perfect property in UAE?",
                    "sources": []
                }
            ]
        }

class IngestRequest(BaseModel):
    rebuild_index: bool = Field(
        False,
        description="Whether to rebuild the Elasticsearch index (deletes existing data)",
        example=False
    )
    batch_size: Optional[int] = Field(
        None,
        description="Number of documents to process per batch",
        example=500,
        ge=1,
        le=1000
    )
    source_type: Optional[str] = Field(
        None,
        description="Source type: sql, csv, jsonl, txt",
        example="sql"
    )
    source_path: Optional[str] = Field(
        None,
        description="Path to source file (for csv/jsonl/txt)",
        example="/path/to/data.csv"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "rebuild_index": True,
                    "batch_size": 500
                }
            ]
        }

class UpsertOneResponse(BaseModel):
    status: str = Field(
        ...,
        description="Operation status",
        example="ok"
    )
    indexed_chunks: int = Field(
        ...,
        description="Number of chunks successfully indexed",
        example=1
    )
    errors: int = Field(
        ...,
        description="Number of errors encountered",
        example=0
    )

class DeleteResponse(BaseModel):
    status: str = Field(
        ...,
        description="Operation status",
        example="ok"
    )
    deleted: int = Field(
        ...,
        description="Number of documents deleted",
        example=5
    )

class PipelineState:
    settings: Optional[Settings] = None
    embedder_client: Optional[BaseEmbedder] = None
    vector_store_client: Optional[BaseVectorStore] = None
    retriever_client: Optional[Retriever] = None
    llm_client: Optional[BaseLLM] = None
    index_ready: bool = False

pipeline_state = PipelineState()

# Session management for conversations

class SessionManager:
    """Manages LLM client sessions for multi-user conversations"""
    
    def __init__(self, ttl_minutes: int = 30):
        self.sessions: Dict[str, Any] = {}
        self.last_activity: Dict[str, datetime] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def get_or_create_session(self, session_id: str):
        """Get or create an LLM client for a session"""
        self._cleanup_expired()
        
        if session_id not in self.sessions:
            from .factories import build_llm
            # Create new LLM client for this session
            self.sessions[session_id] = build_llm(
                pipeline_state.settings.llm,
                fallback_api_key=pipeline_state.settings.embedding.api_key
            )
            logger.debug(f"Created new session: {session_id}")
        
        self.last_activity[session_id] = datetime.now()
        return self.sessions[session_id]
    
    def _cleanup_expired(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired = [
            sid for sid, last in self.last_activity.items()
            if now - last > self.ttl
        ]
        for sid in expired:
            del self.sessions[sid]
            del self.last_activity[sid]
            logger.debug(f"Cleaned up expired session: {sid}")
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            del self.last_activity[session_id]
            logger.debug(f"Cleared session: {session_id}")

# Replace global conversation_sessions with:
session_manager = SessionManager(ttl_minutes=30)

def initialize_components() -> None:
    settings = get_settings(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"))
    pipeline_state.settings = settings
    if not settings.database.columns:
        logger.error("database.columns must be configured in config.yaml")
        raise RuntimeError("database.columns must be configured")
    if settings.retrieval.filter_fields is None:
        logger.debug("retrieval.filter_fields is None; treating as empty list")
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
            logger.debug("Failed eager index init: %s", exc)

def _get_search_plan_from_llm(llm_client: Any, user_query: str) -> Dict[str, Any]:
    """
    Ask LLM to plan the search strategy.
    LLM decides: search query, filters, and strategy.
    
    Returns:
        Dict with search_query, filters, strategy
    """
    from .instructions import SEARCH_PLANNER_PROMPT
    import json
    import re
    
    # Get conversation context
    conversation_summary = llm_client.conversation.get_conversation_summary()
    
    # Build prompt from instructions.py
    prompt = SEARCH_PLANNER_PROMPT.format(
        conversation_summary=conversation_summary,
        user_query=user_query
    )
    
    try:
        # Call LLM to get search plan
        response = llm_client.generate(prompt)
        logger.debug(f"🔍 PLANNER: LLM response: {response}")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            search_plan = json.loads(json_match.group(0))
            logger.debug(f"🔍 PLANNER: Successfully parsed search plan")
            return search_plan
        else:
            logger.debug(f"🔍 PLANNER: No JSON found in LLM response")
            
    except Exception as e:
        logger.debug(f"🔍 PLANNER: Failed to get search plan from LLM: {e}")
    
    # Fallback: use query as-is with no filters
    return {
        "search_query": user_query,
        "filters": {},
        "strategy": "semantic_only"
    }

@app.post("/query",response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, http_request: Request) -> QueryResponse:
    """
    Simplified query endpoint using generic LLM intelligence.
    
    Flow:
    1. Validate input
    2. Get/create session LLM client
    3. Extract filters from query (optional)
    4. Retrieve relevant chunks from database
    5. Let LLM handle everything (greetings, searches, alternatives, etc.)
    6. Return response with sources
    """
    # Rate limiting check
    client_ip = http_request.client.host if http_request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        logger.debug(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 60 requests per minute. Please try again later."
        )
    
    # Initialize monitoring
    metrics_collector = get_metrics_collector()
    error_handler = get_error_handler()
    performance_monitor = get_performance_monitor()
    
    # Start timing
    query_id = f"query_{int(time.time() * 1000)}"
    performance_monitor.start_timer(query_id)
    
    try:
        # Validation
        if not (pipeline_state.retriever_client and pipeline_state.llm_client):
            logger.error("Pipeline not initialized")
            raise HTTPException(status_code=500, detail=SERVER_NOT_INITIALIZED)
        
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")
        
        session_id = request.session_id or "default"
        logger.debug(f"Query: '{request.query[:100]}' | Session: {session_id}")
        
        # Get/create session-specific LLM client
        try:
            llm_client = session_manager.get_or_create_session(session_id)
        except Exception as exc:
            logger.error(f"Failed to create session: {exc}")
            raise HTTPException(status_code=500, detail="Failed to initialize session")
        
        # LLM-DRIVEN SEARCH: LLM plans the search strategy
        chunks = []
        try:
            top_k = request.top_k or pipeline_state.settings.retrieval.top_k or 40
            logger.info(f"🔍 MAIN: Starting LLM-driven search with top_k={top_k}...")
            
            # Step 1: Ask LLM to plan the search (LLM decides filters + query)
            search_plan = _get_search_plan_from_llm(llm_client, request.query)
            logger.debug(f"🔍 MAIN: LLM search plan: {search_plan}")
            logger.debug(f"🔍 MAIN: Search query: '{search_plan.get('search_query')}'")
            logger.debug(f"🔍 MAIN: Filters: {search_plan.get('filters')}")
            logger.debug(f"🔍 MAIN: Strategy: {search_plan.get('strategy')}")
            
            # Step 2: Execute LLM's search plan
            chunks_display, _ = pipeline_state.retriever_client.retrieve(
                search_plan.get("search_query", request.query),
                filters=search_plan.get("filters", {}),
                top_k=top_k,
                retrieve_for_refinement=False
            )
            chunks = chunks_display
            logger.info(f"🔍 MAIN: LLM-planned search retrieved {len(chunks)} chunks")
            
            # Log chunk details
            for i, chunk in enumerate(chunks):
                logger.debug(f"🔍 MAIN: Chunk {i+1}: {chunk.metadata.get('property_title', 'Unknown')} "
                           f"(ID: {chunk.metadata.get('id', 'unknown')}, Score: {chunk.score:.3f})")
                
        except Exception as exc:
            logger.error(f"🔍 MAIN: Retrieval failed: {exc}")
            # Continue with empty chunks - LLM can still handle greetings, etc.
        
        # Let LLM handle everything - it decides how to respond AND which properties to show
        logger.debug(f"🔍 MAIN: Calling LLM chat...")
        llm_response = ""
        selected_property_ids = []
        
        try:
            llm_response = llm_client.chat(request.query, retrieved_chunks=chunks)
            logger.debug(f"🔍 MAIN: LLM response: {len(llm_response)} chars")
            logger.debug(f"🔍 MAIN: LLM response content: {llm_response}")
            
            # Extract answer and property IDs from LLM response
            answer, selected_property_ids = _extract_answer_and_ids(llm_response)
            logger.debug(f"🔍 MAIN: Extracted answer: {answer}")
            logger.debug(f"🔍 MAIN: Selected property IDs: {selected_property_ids}")
            
            # Check if LLM triggered requirement collection
            if "COLLECT_REQUIREMENTS" in answer:
                logger.debug("Requirement collection triggered by LLM")
                
                # Extract and send requirements (LLM handles everything)
                result = llm_client.collect_and_send_requirements()
                
                # Let LLM decide the response message based on success/failure
                if result.get("success"):
                    logger.debug(f"Requirements sent successfully to API")
                    # LLM will provide the success message in its response
                    # Just remove the trigger from the answer
                    answer = answer[:answer.find("COLLECT_REQUIREMENTS")].strip()
                    if not answer:
                        answer = "Perfect! I've saved your requirements. We'll reach out as soon as we find matching properties. Thank you!"
                else:
                    logger.error(f"Failed to send requirements: {result.get('error')}")
                    # LLM will handle the error message
                    answer = answer[:answer.find("COLLECT_REQUIREMENTS")].strip()
                    if not answer:
                        answer = "I've noted your requirements. However, there was a technical issue. Please try again or contact support."
                
        except Exception as exc:
            logger.error(f"LLM chat failed: {exc}")
            answer = (
                "I apologize, but I encountered an issue processing your request. "
                "Please try again or rephrase your question."
            )
        
        # Store search results for progressive filtering
        if chunks:
            property_ids = [chunk.metadata.get("id") for chunk in chunks if chunk.metadata.get("id")]
            llm_client.conversation.store_search_results(property_ids)
            logger.debug(f"🔍 MAIN: Stored {len(property_ids)} property IDs for progressive filtering")
        
        # Prepare sources - Show only when answer is showing actual properties
        sources: List[SourceItem] = []
        
        # Check if answer is a greeting/intro (don't show sources for these)
        greeting_patterns = ["what are you looking for", "how can i help", "tell me what", "i'm leaseoasis", "i'm here to help"]
        is_greeting = any(pattern in answer.lower() for pattern in greeting_patterns)
        
        # Check if answer mentions specific property search results
        result_indicators = ["found", "here are", "take a look", "check them out", "check which", "several", "options"]
        has_results = any(indicator in answer.lower() for indicator in result_indicators)
        
        if chunks and has_results and not is_greeting:
            # Use LLM-selected property IDs to filter sources (LLM-driven filtering!)
            if selected_property_ids:
                # Filter chunks to only include properties the LLM selected
                filtered_chunks = [
                    c for c in chunks 
                    if c.metadata.get("id") in selected_property_ids
                ]
                logger.debug(f"🔍 MAIN: LLM selected {len(selected_property_ids)} properties")
                logger.debug(f"🔍 MAIN: Filtered {len(chunks)} chunks to {len(filtered_chunks)} LLM-selected sources")
            else:
                # Fallback: if LLM didn't provide IDs, show top chunks
                filtered_chunks = chunks[:10]
                logger.debug(f"🔍 MAIN: No IDs from LLM, showing top {len(filtered_chunks)} chunks")
            
            sources = [
                SourceItem(score=c.score, text=c.text, metadata=c.metadata)
                for c in filtered_chunks
            ]
            logger.debug(f"Showing {len(sources)} sources")
        
        # Calculate processing time
        processing_time_ms = performance_monitor.end_timer(query_id)
        
        # Record metrics
        metrics = QueryMetrics(
            query_id=query_id,
            session_id=session_id,
            query_text=request.query,
            query_category="auto",  # LLM handles categorization
            processing_time_ms=processing_time_ms,
            chunks_retrieved=len(chunks),
            chunks_filtered=len(sources),
            sources_shown=len(sources) > 0
        )
        metrics_collector.record_query(metrics)
        
        logger.debug(f"Response ready ({processing_time_ms:.0f}ms)")
        
        return QueryResponse(answer=answer, sources=sources)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Handle unexpected errors gracefully
        processing_time_ms = performance_monitor.end_timer(query_id)
        
        logger.error(f"Query endpoint error: {e}", exc_info=True)
        
        # Record error metrics
        metrics = QueryMetrics(
            query_id=query_id,
            session_id=request.session_id or "default",
            query_text=request.query,
            query_category="error",
            processing_time_ms=processing_time_ms,
            chunks_retrieved=0,
            chunks_filtered=0,
            sources_shown=False,
            error_occurred=True,
            error_message=str(e)
        )
        metrics_collector.record_query(metrics)
        
        # Return user-friendly error
        error_message = error_handler.handle_query_error(e, request.query, request.session_id or "default")
        return QueryResponse(answer=error_message, sources=[])

def _extract_answer_and_ids(llm_response: str) -> tuple[str, List[str]]:
    """
    Extract answer text and property IDs from LLM response.
    
    LLM response format:
    ```
    Great news! I found some amazing properties for you!
    
    RELEVANT_IDS: ["id1", "id2", "id3"]
    ```
    
    Returns:
        Tuple of (answer_text, property_ids_list)
    """
    import re
    import json
    
    # Split by RELEVANT_IDS marker
    if "RELEVANT_IDS:" in llm_response:
        parts = llm_response.split("RELEVANT_IDS:")
        answer = parts[0].strip()
        
        # Extract JSON array from the second part
        try:
            ids_section = parts[1].strip()
            # Find JSON array
            json_match = re.search(r'\[(.*?)\]', ids_section, re.DOTALL)
            if json_match:
                ids_json = f"[{json_match.group(1)}]"
                property_ids = json.loads(ids_json)
                logger.debug(f"🔍 EXTRACT: Successfully extracted {len(property_ids)} property IDs")
                return answer, property_ids
        except (json.JSONDecodeError, IndexError) as e:
            logger.debug(f"🔍 EXTRACT: Failed to parse property IDs: {e}")
    
    # Fallback: no IDs found, return whole response as answer
    return llm_response.strip(), []

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
            logger.debug("Rebuilding index '%s'", pipeline_state.vector_store_client._config.index)
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
        logger.debug("Fetched %s rows from DB", len(docs_batch))
        chunks, embeddings = _process_documents_to_vectors(docs_batch, settings)
        if not chunks:
            continue
        logger.debug("Created %s chunks", len(chunks))
        logger.debug("Generated %s embeddings", len(embeddings))
        success, errors = _upsert_chunks_to_vector_store(chunks, embeddings)
        if success > 0 and not pipeline_state.index_ready:
            pipeline_state.index_ready = True
        total_rows += len(docs_batch)
        total_chunks += len(chunks)
        total_indexed += success
        total_errors += int(errors)
        logger.debug("Upserted %s chunks (errors=%s)", success, errors)
        logger.debug("Ingested totals rows=%s chunks=%s indexed=%s errors=%s", total_rows, total_chunks, total_indexed, total_errors)

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

@app.post("/ingest/{property_id}",response_model=UpsertOneResponse)
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

@app.delete("/vectors/{property_id}",response_model=DeleteResponse)
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
    """Basic health check endpoint."""
    return {"status": "ok", "index_ready": pipeline_state.index_ready}
