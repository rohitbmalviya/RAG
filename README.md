## RAG Pipeline

A **generic, LLM-powered** Retrieval-Augmented Generation (RAG) service built with FastAPI. It loads tabular data from PostgreSQL, chunks and embeds it, stores embeddings in Elasticsearch, and uses **intelligent LLM conversations** to answer queries naturally.

### 🎯 Key Features

- **✨ LLM Intelligence Over Hardcoded Logic**: Single comprehensive system prompt replaces 20+ specific handlers
- **💬 Natural Conversations**: Context-aware chat with memory, like ChatGPT/Gemini
- **🔧 Config-First**: Everything driven by `config.yaml` - works for ANY domain (properties, cars, jobs, products)
- **🚀 Minimal Code**: ~500 lines vs 3000+ before (84% reduction)
- **📊 PostgreSQL Loader**: Dynamic SQL with configurable relations
- **🧩 Smart Chunking**: Token or character-based with overlap
- **🔍 Vector Search**: Elasticsearch kNN with metadata filtering
- **🎨 Clean API**: `/ingest`, `/query`, `/health` endpoints
- **🔐 Rate Limiting**: 60 requests/min per IP
- **📈 Monitoring**: Built-in metrics and error tracking

### 🏗️ Architecture (New - Simplified)

**Core Philosophy**: Let the LLM handle logic, not the code.

```
User Query → Retrieve Context → LLM Decides Everything → Response
                                      ↓
                    (greetings, searches, alternatives,
                     requirement gathering, follow-ups)
```

**Components**:

- **`loader.py`**: Generic PostgreSQL loader with dynamic relations
- **`chunking.py`**: Splits documents into overlapping chunks
- **`embedding.py`**: Generates embeddings (Google Gemini or OpenAI)
- **`vector_db.py`**: Elasticsearch vector store with kNN search
- **`retriever.py`**: Retrieves and ranks relevant chunks
- **`instructions.py`**: **ONE** comprehensive system prompt (200 lines)
- **`llm.py`**: Minimal LLM client with conversation memory (300 lines)
- **`main.py`**: FastAPI app that wires everything together

**What Changed**:

- ✅ Removed 50+ specific handler functions
- ✅ Removed hardcoded query classification
- ✅ Removed hardcoded alternatives generation
- ✅ Removed hardcoded greeting/knowledge handlers
- ✅ One `chat()` method handles everything
- ✅ LLM naturally decides when to ask questions, suggest alternatives, collect requirements

## Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Elasticsearch 8.17+ (matching Docker image)
- An LLM API key (Google or OpenAI)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

All runtime configuration is in `config.yaml`, with many values pulled from environment variables. Important keys:

- `app`: host/port and `eager_init` (pre-create index on startup)
- `database`: connection info and column schema used to compose embeddings and filters
- `chunking`: `chunk_size`, `chunk_overlap`, `unit`
- `embedding`: provider/model/api key/batch size
- `vector_db`: Elasticsearch hosts, credentials, index, dims, similarity
- `retrieval`: `top_k`, `filter_fields` accepted by `/query`
- `llm`: provider/model/temperature/max tokens
- `ingestion`: batch sizing

Environment variables referenced by `config.yaml`:

```bash
# LLM
LLM_MODEL_API_KEY=your_google_or_openai_key
LLM_EMBED_MODEL_NAME=gemini-embedding-001
LLM_MODEL_NAME=gemini-2.5-flash-lite

# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=leasebnb
DB_USER=postgres
DB_PASSWORD=postgres

# Elasticsearch
VECTOR_DB_HOST=http://localhost:9200
VECTOR_DB_USERNAME=elastic
VECTOR_DB_PASSWORD=changeme
```

Tip: Create a `.env` file in the project root with these values. The app will load it automatically if `python-dotenv` is present.

## Run the API

Start required services (see Docker section below), then run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger UI at `http://localhost:8000/docs`.

## Endpoints

- `GET /health`

  - Returns `{ status: "ok", index_ready: boolean }`.

- `POST /ingest`

  - Body: `{ rebuild_index?: boolean, batch_size?: number, source_type?: string, source_path?: string }`
  - Runs ingestion in the background: loads rows from PostgreSQL, chunks, embeds, and upserts to Elasticsearch. If `rebuild_index` is true, the index is dropped and recreated.

- `POST /query`
  - Body: `{ query: string, session_id?: string, top_k?: number }`
  - **NEW**: Uses LLM intelligence to handle all query types naturally
  - Supports conversation continuity with `session_id`
  - LLM decides response format (greetings, searches, alternatives, etc.)

### Examples

```bash
# Health
curl http://localhost:8000/health

# Ingest (rebuild index and ingest in batches)
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"rebuild_index": true, "batch_size": 500}'

# Simple query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "2-bedroom apartment in Dubai Marina"}'

# Conversation query (with session)
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "I want something cheaper",
    "session_id": "user-123",
    "top_k": 5
  }'
```

## Docker Compose (Elasticsearch, Kibana, Redis)

The provided `docker-compose.yaml` starts:

- Elasticsearch 8.17.1 on `localhost:9200`
- Kibana 8.17.1 on `localhost:5601`
- Redis on `localhost:6379`

Start services:

```bash
docker-compose up -d
```

Notes:

- Set `VECTOR_DB_PASSWORD`/`ELASTICSEARCH_PASSWORD` securely in your environment, and keep it consistent with the compose file. Do not commit real passwords.
- The API service itself is not included in compose; run it locally with Uvicorn as shown above.

## 🤖 LLM System Design

### Single Instruction Approach

The system uses **ONE comprehensive system prompt** (`instructions.py`) that teaches the LLM everything:

- How to search properties
- When to ask clarifying questions
- How to suggest alternatives
- When to collect requirements
- Response formatting rules
- Conversation memory usage

### Conversation Flow

**Progressive Assistance**:

1. **Initial Query**: LLM searches context and shows 2-3 best matches
2. **Refinement**: User adds criteria → LLM refines search
3. **No Results**: LLM asks 1-2 clarifying questions
4. **Still Nothing**: LLM suggests verified alternatives from context
5. **Not Satisfied**: LLM offers to collect requirements (name, email, phone)

### Example Conversation

```
User: "Show me 2 bedroom apartments in Dubai Marina"

LLM: Found 8 apartments in Dubai Marina! Here are the top matches:

🏠 **Marina Heights Luxury Apartment** - AED 120,000/year
   📍 Dubai, Dubai Marina | 2BR • 2BA • 1,200 sqft
   ✨ Pool, Gym, Parking, Sea View
   🆔 abc-123

🏠 **Elite Marina Residence** - AED 135,000/year
   📍 Dubai, Dubai Marina | 2BR • 2BA • 1,350 sqft
   ✨ Furnished, Balcony, Concierge
   🆔 def-456

💬 Would you prefer furnished or unfurnished options?

User: "I want something cheaper"

LLM: Looking for more affordable options in Dubai Marina:

🏠 **Marina Bay Apartment** - AED 95,000/year
   📍 Dubai, Dubai Marina | 2BR • 2BA • 950 sqft
   ✨ Pool, Parking, Security
   🆔 ghi-789

💬 This is under 100k. Need any specific amenities?

User: "What about 1 bedroom?"

LLM: Great! Here are 1-bedroom options in Dubai Marina under 100k:

🏠 **Marina Tower Studio Plus** - AED 75,000/year
   📍 Dubai, Dubai Marina | 1BR • 1BA • 750 sqft
   ✨ Gym, Pool, Balcony
   🆔 jkl-111

💬 Would you like to see more options or properties in nearby areas?
```

### LLM Providers

- **Google** (`provider: google`): Uses `google-generativeai` with `LLM_MODEL_API_KEY`
- **OpenAI** (`provider: openai` or `azure_openai`): Uses `openai` package with `LLM_MODEL_API_KEY`

Configure via the `llm` block in `config.yaml`. The embedding provider is configured separately in the `embedding` block and may reuse the same key.

## Troubleshooting

- **Dimension mismatch**: Ensure `vector_db.dims` matches the embedding model output (e.g., 3072 for `gemini-embedding-001`)
- **Index not created**: Enable `app.eager_init: true` or trigger ingestion once to create the index
- **Empty answers**: Check that ingestion ran successfully and `index_ready` is true; verify filters are valid
- **Auth errors to Elasticsearch**: Verify `VECTOR_DB_HOST`, `VECTOR_DB_USERNAME`, `VECTOR_DB_PASSWORD`
- **LLM not responding**: Check `LLM_MODEL_API_KEY` is set and valid

## Testing

After starting the system, test with:

```bash
# Test health
curl http://localhost:8000/health

# Test ingestion
curl -X POST http://localhost:8000/ingest -H 'Content-Type: application/json' -d '{"rebuild_index": false, "batch_size": 100}'

# Test query
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"query": "Hello"}'

# Test property search
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"query": "2 bedroom apartment in Dubai", "session_id": "test-1"}'
```

## License

Proprietary/Private (adjust as needed).
