## RAG Pipeline

A config-driven Retrieval-Augmented Generation (RAG) service built with FastAPI. It loads tabular data from PostgreSQL, chunks and embeds it, stores embeddings in Elasticsearch, and answers queries using an LLM with grounded citations.

### Key Features

- Config-first via `config.yaml`
- PostgreSQL loader (default table: `properties`)
- Chunking (`token` or `char`) with overlap
- Embeddings via Google models; OpenAI is also supported
- Elasticsearch vector store with kNN over `dense_vector`
- Metadata filtering and `top_k` retrieval
- Source-aware prompt that cites table/id
- FastAPI endpoints: `/ingest`, `/query`, `/health`
- CORS enabled for `http://localhost:3000` by default

## Architecture

- Loader reads rows from PostgreSQL using credentials provided via environment variables consumed by `config.yaml`.
- `chunking.py` splits rows into overlapping chunks.
- `embedding.py` embedder batches texts and returns vectors.
- `vector_db.py` manages the Elasticsearch index and upserts vectors.
- `retriever.py` performs ANN search with optional filters.
- `llm.py` builds prompts from retrieved chunks and calls the selected LLM provider.
- `app/main.py` wires everything and exposes the API.

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
LLM_EMBED_MODEL_NAME=text-embedding-004            # example
LLM_MODEL_NAME=gemini-1.5-flash-latest             # example

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
  - Body: `{ query: string, filters?: object, top_k?: number }`
  - `filters` keys must be among those listed in `retrieval.filter_fields` in `config.yaml` (e.g., `emirate`, `city`, `number_of_bedrooms`, `rent_charge_min`/`max`, etc.).

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
  -d '{"query": "2-bedroom apartment in Dubai Marina with sea view"}'

# Filtered query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "family apartment",
    "filters": {"city": "Dubai", "number_of_bedrooms": 2, "rent_charge_max": 120000},
    "top_k": 8
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

## LLM Providers

- Google (`provider: google`): uses `google-generativeai` with `LLM_MODEL_API_KEY`.
- OpenAI (`provider: openai` or `azure_openai`): uses `openai` package with `LLM_MODEL_API_KEY`.

Configure via the `llm` block in `config.yaml`. The embedding provider is configured separately in the `embedding` block and may reuse the same key.

## Troubleshooting

- Dimension mismatch: Ensure `vector_db.dims` matches the embedding model output (e.g., 3072 for `text-embedding-004`).
- Index not created: enable `app.eager_init: true` or trigger ingestion once to create the index.
- Empty answers: check that ingestion ran successfully and `index_ready` is true; verify filters are valid.
- Auth errors to Elasticsearch: verify `VECTOR_DB_HOST`, `VECTOR_DB_USERNAME`, `VECTOR_DB_PASSWORD`.

## License

Proprietary/Private (adjust as needed).
