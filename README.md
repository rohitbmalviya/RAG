# RAG Pipeline

A config-driven, generic Retrieval-Augmented Generation (RAG) FastAPI service for querying PostgreSQL-backed real-estate data with embeddings stored in Elasticsearch, using Google Gemini for embeddings and generation.

## Features

- Config-driven pipeline via `config.yaml`
- Load from PostgreSQL table (default: `properties`)
- Chunking with overlap; configurable unit (`token` or `char`) and offsets preserved
- Embeddings with `gemini-embedding-001` with retry/backoff and batching
- Vector store in Elasticsearch with kNN search (`dense_vector`), configurable dims and index
- Retrieval with optional metadata filters and configurable `top_k`
- Prompt construction that cites sources
- LLM answers with `gemini-2.5-flash-lite`
- FastAPI endpoints: `/ingest`, `/query`, `/health`
- Modular design, extendable to multiple tables/sources

## Requirements

- Python 3.11+
- PostgreSQL 14+
- Elasticsearch 8.13+
- Google API key (`GOOGLE_API_KEY`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Edit `config.yaml` or set environment variables referenced in it. Key items:

- Database connection via `DB_URL` (or legacy `DATABASE_URL`) or individual `DB_*` vars
- `database.table` and `database.columns`
- `chunking.unit` (`token` or `char`), `chunk_size`, `chunk_overlap`
- `embedding.model` and `GOOGLE_API_KEY`
- Elasticsearch `ELASTIC_HOST`, `ELASTIC_USERNAME`, `ELASTIC_PASSWORD`, `vector_db.index`, `vector_db.dims`
- Retrieval `top_k`

### .env example

Create a `.env` file (or copy from `.env.example` if present):

```bash
GOOGLE_API_KEY=your_google_api_key
DB_URL=postgresql://postgres:postgres@localhost:5432/leasebnb
# or set DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD individually
ELASTIC_HOST=http://localhost:9200
ELASTIC_USERNAME=elastic
ELASTIC_PASSWORD=changeme
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs`.

## Ingest

Rebuild index and ingest (runs in background):

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"rebuild_index": true, "batch_size": 500}'
```

## Query

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "2-bedroom in Dubai Marina with gym", "filters": {"city": "Dubai"}, "top_k": 5}'
```

## Notes

- The pipeline is generic; adjust `config.yaml` to switch DB/table/columns, embeddings, vector DB, LLM, chunk sizes, and retrieval parameters.
- Ensure your Elasticsearch has vector support (dense_vector with the configured similarity) and credentials are valid.
- For multiple tables, extend the loader and config to list multiple sources and iterate.
