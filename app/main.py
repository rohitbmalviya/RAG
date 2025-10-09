from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from .query_processor import preprocess_query
from .chunking import chunk_documents
from .config import Settings, get_settings
from .core.base import BaseEmbedder, BaseVectorStore, BaseLLM
from .loader import load_documents, load_document_by_id
from .monitoring import get_metrics_collector, get_error_handler, get_performance_monitor, QueryMetrics
from .models import RetrievedChunk
from .retriever import Retriever
from .utils import get_logger
from .factories import build_embedder, build_llm, build_vector_store
from .monitoring import get_metrics_collector, get_error_handler, get_performance_monitor, QueryMetrics
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

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query UAE Properties",
    description="""
    Process natural language queries to find properties in UAE.
    
    **Features:**
    - Conversational AI with memory (tracks session context)
    - Intelligent filter extraction from natural language
    - 10 query categories supported (property search, greetings, best properties, average prices, etc.)
    - UAE-only property scope (Dubai, Abu Dhabi, Sharjah, etc.)
    - Strict filter compliance (only shows matching properties)
    - No hallucinations (context-bound responses)
    
    **Query Categories:**
    1. **Property Search** - "Find 2-bedroom apartments in Dubai"
    2. **Best Properties** - "Show me the best properties in Dubai Marina"
    3. **Average Prices** - "What's the average rent in Dubai?"
    4. **General Knowledge** - "What is Ejari?"
    5. **Greetings** - "Hi, how are you?"
    6. **Location-Based** - "Properties in Palm Jumeirah"
    7. **Amenity-Focused** - "Properties with pool and gym"
    8. **Combination Queries** - "Furnished 2BR in Marina under 150k with pool"
    9. **Requirement Gathering** - "Save my requirements for later"
    10. **Conversation Flow** - Multi-turn conversations with context
    
    **Rate Limiting:** 60 requests per minute per IP
    """,
    response_description="AI-generated answer with property sources (if applicable)",
    responses={
        200: {
            "description": "Successful query - property search results",
            "content": {
                "application/json": {
                    "examples": {
                        "property_search": {
                            "summary": "Property Search Query",
                            "description": "User searches for specific properties",
                            "value": {
                                "answer": "I found 3 great apartments in Dubai Marina matching your criteria:\n\n1. **Luxury Apartment** - AED 120,000/year\n   - 2 bedrooms, 2 bathrooms, 850 sq.ft\n   - Furnished, pool, gym, parking\n\n2. **Marina Heights** - AED 135,000/year\n   - 2 bedrooms, 2 bathrooms, 900 sq.ft\n   - Sea view, balcony, concierge\n\nWould you like more details?",
                                "sources": [
                                    {
                                        "score": 0.95,
                                        "text": "Property: Luxury Apartment\nLocated in: Dubai, Dubai Marina\nRent: AED 120,000/year",
                                        "metadata": {
                                            "id": "prop-123",
                                            "property_title": "Luxury Apartment",
                                            "rent_charge": 120000,
                                            "emirate": "dubai"
                                        }
                                    }
                                ]
                            }
                        },
                        "greeting": {
                            "summary": "Greeting Query",
                            "description": "User greets the assistant",
                            "value": {
                                "answer": "Hello! I'm LeaseOasis, your friendly UAE property assistant. I'm here to help you find the perfect property to lease in Dubai, Abu Dhabi, or other UAE cities. What type of property are you looking for?",
                                "sources": []
                            }
                        },
                        "average_price": {
                            "summary": "Average Price Query",
                            "description": "User asks for average rent statistics",
                            "value": {
                                "answer": "Based on 45 properties in Dubai Marina:\n\n**Price Statistics:**\n• **Average Annual Rent:** AED 125,000\n• **Median Annual Rent:** AED 120,000\n• **Price Range:** AED 85,000 - AED 180,000\n\nWould you like to see specific properties in this range?",
                                "sources": []
                            }
                        },
                        "general_knowledge": {
                            "summary": "General Knowledge Query",
                            "description": "User asks for property term definitions",
                            "value": {
                                "answer": "**Ejari in UAE Context:**\n\nEjari is a mandatory online registration system for all rental contracts in Dubai, managed by the Dubai Land Department (DLD). It's a legal requirement that protects both landlords and tenants by officially registering the tenancy contract.\n\nWould you like to search for properties to lease?",
                                "sources": []
                            }
                        }
                    }
                }
            }
        },
        400: {
            "description": "Bad Request - Invalid input",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Query must not be empty"
                    }
                }
            }
        },
        429: {
            "description": "Too Many Requests - Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Rate limit exceeded: 60 per minute"
                    }
                }
            }
        },
        500: {
            "description": "Internal Server Error - System issue",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Server not initialized"
                    }
                }
            }
        }
    },
    tags=["Query"]
)
async def query_endpoint(request: QueryRequest, http_request: Request) -> QueryResponse:
    """Main query endpoint with comprehensive error handling and monitoring.
    
    This endpoint:
    1. Validates input and system state
    2. Preprocesses query to extract filters
    3. Retrieves relevant property chunks from vector database
    4. Generates LLM response with intelligent source decision
    5. Applies strict filter compliance
    6. Returns answer with sources (if applicable)
    
    Error handling:
    - Returns user-friendly messages for all error types
    - Logs all errors for debugging
    - Tracks metrics for monitoring
    """
    # Rate limiting check
    client_ip = http_request.client.host if http_request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 60 requests per minute. Please try again later."
        )
    
    # Initialize monitoring
    metrics_collector = get_metrics_collector()
    error_handler = get_error_handler()
    performance_monitor = get_performance_monitor()
    
    # Start timing the entire query
    query_id = f"query_{int(time.time() * 1000)}"
    performance_monitor.start_timer(query_id)
    
    try:
        # Validation checks
        if not (pipeline_state.retriever_client and pipeline_state.llm_client):
            logger.error("Query endpoint called but pipeline not initialized")
            raise HTTPException(status_code=500, detail=SERVER_NOT_INITIALIZED)
        
        if not request.query or not request.query.strip():
            logger.warning(f"Empty query received from session {request.session_id}")
            raise HTTPException(status_code=400, detail="Query must not be empty")
        
        logger.info(f"Processing query: {request.query[:100]} | Session: {request.session_id} | Query ID: {query_id}")
        
        logger.debug(f"\n QUERY ENDPOINT HIT:")
        logger.debug(f"   Query ID: {query_id}")
        logger.debug(f"   Session ID: {request.session_id}")
        logger.debug(f"   Raw Query: '{request.query}'")
        logger.debug(f"   Top K: {request.top_k}")
        
        # FIXED: Use session_id properly
        session_id = request.session_id or "default"
        
        # Create/get session-specific LLM client with error handling
        try:
            llm_client = session_manager.get_or_create_session(session_id)
            logger.debug(f"   LLM Client: {type(llm_client).__name__}")
        except Exception as exc:
            logger.error(f"Failed to create/get session {session_id}: {exc}")
            raise HTTPException(status_code=500, detail="Failed to initialize conversation session")
        
        # CRITICAL: Use the session LLM client, not pipeline_state.llm_client
        logger.debug(f"\n PREPROCESSING QUERY:")
        try:
            normalized_query, filters = preprocess_query(request.query, llm_client)
            logger.debug(f"   Normalized Query: '{normalized_query}'")
            logger.debug(f"   Extracted Filters: {filters}")
            logger.debug(f"Query preprocessing successful - Filters: {filters}")
        except Exception as exc:
            logger.error(f"Query preprocessing failed: {exc}")
            # Continue with original query if preprocessing fails
            normalized_query = request.query
            filters = {}
        
        # Store extracted filters in LLM client for context
        if hasattr(llm_client, '_last_extracted_filters'):
            llm_client._last_extracted_filters = filters
        
        # Retrieve chunks for property queries with error handling
        logger.debug(f"\n RETRIEVING CHUNKS:")
        try:
            chunks: List[RetrievedChunk] = pipeline_state.retriever_client.retrieve(
                normalized_query,
                filters=filters,
                top_k=request.top_k,
            )
            logger.debug(f"   Retrieved {len(chunks)} chunks")
            logger.debug(f"Retrieval successful - {len(chunks)} chunks retrieved")
        except Exception as exc:
            logger.error(f"Retrieval failed: {exc}")
            # Continue with empty chunks - LLM will handle no results case
            chunks = []
            logger.warning("Continuing with empty chunks due to retrieval failure")
        
        # Use session-specific LLM client for chat - LLM determines if sources should be shown
        logger.debug(f"\n LLM PROCESSING:")
        try:
            answer, should_show_sources = llm_client.chat_with_source_decision(normalized_query, retrieved_chunks=chunks)
            logger.debug(f"   Answer Length: {len(answer)} characters")
            logger.debug(f"   Should Show Sources: {should_show_sources}")
            logger.debug(f"LLM processing successful - Sources shown: {should_show_sources}")
        except Exception as exc:
            logger.error(f"LLM processing failed: {exc}")
            # Provide fallback response
            answer = (
                "I apologize, but I encountered an issue processing your query. "
                "Please try rephrasing your question or contact support if the issue persists."
            )
            should_show_sources = False
            chunks = []  # Clear chunks to avoid showing potentially incorrect data
        
        # Check if this is a requirement gathering request
        if "REQUIREMENT_GATHERING_DETECTED" in answer or "gather requirement" in normalized_query.lower() or "save my requirements" in normalized_query.lower():
            logger.debug(f"    REQUIREMENT GATHERING DETECTED IN QUERY!")
            logger.debug(f"   Query: '{normalized_query}'")
            logger.debug(f"   Answer contains requirement gathering: {'REQUIREMENT_GATHERING_DETECTED' in answer}")
        
        # Filter sources based on LLM decision and extracted filters
        if should_show_sources:
            filtered_chunks = [c for c in chunks if getattr(c, "score", 1.0) >= 0.7]
            
            # Apply additional filtering based on extracted filters
            if filters:
                filtered_chunks = _apply_filter_compliance(filtered_chunks, filters)
                logger.debug(f"   Applied filter compliance, remaining chunks: {len(filtered_chunks)}")
            
            sources: List[SourceItem] = [
                SourceItem(score=c.score, text=c.text, metadata=c.metadata)
                for c in filtered_chunks
            ]
            logger.debug(f"   Final filtered sources: {len(sources)}")
        else:
            sources = []
            logger.debug(f"   No sources shown (LLM decision)")
        
        # Calculate processing time
        processing_time_ms = performance_monitor.end_timer(query_id)
        
        # Record successful query metrics
        metrics = QueryMetrics(
            query_id=query_id,
            session_id=session_id,
            query_text=request.query,
            query_category="property_search",  # Could be enhanced to detect category
            processing_time_ms=processing_time_ms,
            chunks_retrieved=len(chunks),
            chunks_filtered=len(sources),
            sources_shown=should_show_sources
        )
        metrics_collector.record_query(metrics)
        
        logger.debug(f"\n RESPONSE READY:")
        logger.debug(f"   Answer: {answer[:100]}...")
        logger.debug(f"   Sources Count: {len(sources)}")
        logger.debug(f"   Processing Time: {processing_time_ms:.0f}ms")
        
        return QueryResponse(answer=answer, sources=sources)
        
    except Exception as e:
        # Handle errors gracefully
        processing_time_ms = performance_monitor.end_timer(query_id)
        
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
        
        # Return user-friendly error response
        error_message = error_handler.handle_query_error(e, request.query, request.session_id or "default")
        return QueryResponse(answer=error_message, sources=[])

def _validate_components() -> None:
    """Validate that all required components are initialized"""
    if not (pipeline_state.settings and pipeline_state.embedder_client and pipeline_state.vector_store_client):
        raise RuntimeError(COMPONENTS_NOT_INITIALIZED)

def _get_vector_dimensions() -> int:
    """Get vector dimensions from config or embedder"""
    if pipeline_state.vector_store_client and pipeline_state.vector_store_client._config.dims:
        return int(pipeline_state.vector_store_client._config.dims)
    return pipeline_state.embedder_client.get_dimension()

def _apply_filter_compliance(chunks: List[RetrievedChunk], filters: Dict[str, Any]) -> List[RetrievedChunk]:
    """Apply STRICT filter compliance to ensure sources match ALL extracted filters.
    
    This validates:
    - Exact matches for keywords (property_type, emirate, city, community, furnishing_status)
    - Numeric ranges for rent_charge, bedrooms, bathrooms, property_size
    - Boolean matches for amenities
    - Date comparisons for availability
    """
    from .config import get_settings
    
    if not filters:
        return chunks
    
    settings = get_settings()
    field_types = settings.database.field_types or {}
    
    logger.debug(f"\n STRICT FILTER COMPLIANCE CHECK:")
    logger.debug(f"   Input chunks: {len(chunks)}")
    logger.debug(f"   Filters to validate: {filters}")
    
    filtered_chunks = []
    
    for chunk in chunks:
        metadata = chunk.metadata
        include_chunk = True
        property_title = metadata.get('property_title', 'Unknown')
        
        for filter_key, filter_value in filters.items():
            if not include_chunk:
                break  # Skip remaining checks if already excluded
            
            if filter_value is None:
                continue  # Skip None filters
            
            field_type = field_types.get(filter_key, "keyword")
            actual_value = metadata.get(filter_key)
            
            # KEYWORD/TEXT FILTERS (Exact match)
            if field_type in ("keyword", "text"):
                if isinstance(filter_value, list):
                    # Multiple acceptable values (OR logic)
                    if actual_value not in filter_value:
                        logger.debug(f"    {property_title} - {filter_key} mismatch: {actual_value} not in {filter_value}")
                        include_chunk = False
                else:
                    # Single value (exact match, case-insensitive for keywords)
                    expected = str(filter_value).lower()
                    actual = str(actual_value).lower() if actual_value else ""
                    if actual != expected:
                        logger.debug(f"    {property_title} - {filter_key} mismatch: {actual} != {expected}")
                        include_chunk = False
            
            # NUMERIC FILTERS (Integer/Float with range support)
            elif field_type in ("integer", "float"):
                if isinstance(filter_value, dict):
                    # Range query (gte, lte, gt, lt)
                    actual_num = float(actual_value) if actual_value is not None else None
                    
                    if actual_num is None:
                        logger.debug(f"    {property_title} - {filter_key} missing or None")
                        include_chunk = False
                    else:
                        # Check each range bound
                        if "gte" in filter_value and actual_num < filter_value["gte"]:
                            logger.debug(f"    {property_title} - {filter_key} too low: {actual_num} < {filter_value['gte']}")
                            include_chunk = False
                        if "lte" in filter_value and actual_num > filter_value["lte"]:
                            logger.debug(f"    {property_title} - {filter_key} too high: {actual_num} > {filter_value['lte']}")
                            include_chunk = False
                        if "gt" in filter_value and actual_num <= filter_value["gt"]:
                            logger.debug(f"    {property_title} - {filter_key} not greater: {actual_num} <= {filter_value['gt']}")
                            include_chunk = False
                        if "lt" in filter_value and actual_num >= filter_value["lt"]:
                            logger.debug(f"    {property_title} - {filter_key} not less: {actual_num} >= {filter_value['lt']}")
                            include_chunk = False
                else:
                    # Exact numeric match
                    expected_num = float(filter_value)
                    actual_num = float(actual_value) if actual_value is not None else None
                    if actual_num != expected_num:
                        logger.debug(f"    {property_title} - {filter_key} mismatch: {actual_num} != {expected_num}")
                        include_chunk = False
            
            # BOOLEAN FILTERS (Amenities)
            elif field_type == "boolean":
                if filter_value is True:
                    # If filter requires True, actual must be True
                    if actual_value is not True:
                        logger.debug(f"    {property_title} - {filter_key} not available: {actual_value}")
                        include_chunk = False
                # Note: We don't filter on False (user wants it, doesn't matter if property has it or not)
            
            # DATE FILTERS (String comparison for ISO dates)
            elif filter_key in ["available_from", "lease_start_date", "lease_end_date", "listing_date"]:
                if isinstance(filter_value, dict):
                    # Range query for dates
                    if "lte" in filter_value:
                        # available_from should be on or before filter date
                        if actual_value and actual_value > filter_value["lte"]:
                            logger.debug(f"    {property_title} - {filter_key} too late: {actual_value} > {filter_value['lte']}")
                            include_chunk = False
                    if "gte" in filter_value:
                        # available_from should be on or after filter date
                        if actual_value and actual_value < filter_value["gte"]:
                            logger.debug(f"    {property_title} - {filter_key} too early: {actual_value} < {filter_value['gte']}")
                            include_chunk = False
                else:
                    # Exact date match
                    if actual_value != filter_value:
                        logger.debug(f"    {property_title} - {filter_key} mismatch: {actual_value} != {filter_value}")
                        include_chunk = False
        
        if include_chunk:
            filtered_chunks.append(chunk)
            logger.debug(f"    {property_title} - PASSES all filters")
    
    logger.debug(f"   Final compliant chunks: {len(filtered_chunks)}/{len(chunks)}")
    return filtered_chunks

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

@app.post(
    "/ingest",
    summary="Ingest Properties into Vector Database",
    description="""
    Load properties from PostgreSQL and index them in Elasticsearch for vector search.
    
    **Process:**
    1. Loads properties with `property_status = 'listed'` from database
    2. Chunks property descriptions into overlapping segments
    3. Generates embeddings using Gemini embedding model
    4. Stores vectors in Elasticsearch with metadata
    
    **Options:**
    - `rebuild_index`: Deletes existing index and rebuilds (use for fresh start)
    - `batch_size`: Number of properties to process at once (default: 500)
    
    **Note:** This operation runs in the background. Check logs for progress.
    
    **Rate Limiting:** No limit (admin operation)
    """,
    responses={
        200: {
            "description": "Ingestion started successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "started",
                        "batch_size": 500,
                        "rebuild_index": True
                    }
                }
            }
        },
        500: {
            "description": "Server not initialized",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Server not initialized"
                    }
                }
            }
        }
    },
    tags=["Ingestion"]
)
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

@app.post(
    "/ingest/{property_id}",
    response_model=UpsertOneResponse,
    summary="Ingest Single Property",
    description="""
    Index or update a single property by ID.
    
    **Use Cases:**
    - Property was just created/updated in database
    - Need to refresh specific property in search index
    - Property details changed and need immediate update
    
    **Process:**
    1. Loads property from PostgreSQL by ID
    2. Chunks and embeds the property data
    3. Updates Elasticsearch index
    
    **Note:** Uses immediate refresh for real-time updates.
    """,
    responses={
        200: {
            "description": "Property indexed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "indexed_chunks": 1,
                        "errors": 0
                    }
                }
            }
        },
        404: {
            "description": "Property not found in database",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Property not found"
                    }
                }
            }
        }
    },
    tags=["Ingestion"]
)
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

@app.delete(
    "/vectors/{property_id}",
    response_model=DeleteResponse,
    summary="Delete Property Vectors",
    description="""
    Remove all vector embeddings for a specific property.
    
    **Use Cases:**
    - Property was deleted from database
    - Property status changed to unlisted
    - Need to remove property from search results
    
    **Note:** This only deletes from Elasticsearch, not from PostgreSQL.
    """,
    responses={
        200: {
            "description": "Vectors deleted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "deleted": 5
                    }
                }
            }
        },
        500: {
            "description": "Failed to delete vectors",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Failed to delete vectors"
                    }
                }
            }
        }
    },
    tags=["Ingestion"]
)
async def delete_vectors_by_id(property_id: str) -> DeleteResponse:
    if not pipeline_state.vector_store_client:
        raise HTTPException(status_code=500, detail=SERVER_NOT_INITIALIZED)
    try:
        deleted = pipeline_state.vector_store_client.delete_by_source_id(property_id)
    except Exception as exc:
        logger.error("Failed to delete vectors for %s: %s", property_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete vectors")
    return DeleteResponse(status="ok", deleted=deleted)

@app.get(
    "/health",
    summary="Health Check",
    description="""
    Check if the RAG pipeline is ready to process queries.
    
    **Returns:**
    - `status`: Overall system status ("ok" or "error")
    - `index_ready`: Whether Elasticsearch index is ready for queries
    
    **Use this to:**
    - Monitor system availability
    - Check before making query requests
    - Verify successful startup
    """,
    responses={
        200: {
            "description": "System health status",
            "content": {
                "application/json": {
                    "examples": {
                        "ready": {
                            "summary": "System Ready",
                            "value": {
                                "status": "ok",
                                "index_ready": True
                            }
                        },
                        "not_ready": {
                            "summary": "System Not Ready",
                            "value": {
                                "status": "ok",
                                "index_ready": False
                            }
                        }
                    }
                }
            }
        }
    },
    tags=["System"]
)
async def health() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {"status": "ok", "index_ready": pipeline_state.index_ready}
