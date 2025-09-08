# RAG Pipeline

A config-driven, generic Retrieval-Augmented Generation (RAG) FastAPI service for querying PostgreSQL-backed real-estate data with embeddings stored in Elasticsearch, using Google Gemini for embeddings and generation.

## Features

- Config-driven pipeline via `config.yaml`
- Load from PostgreSQL table (default: `properties`)
- Chunking with overlap
- Embeddings with `gemini-embedding-001`
- Vector store in Elasticsearch with kNN search
- Retrieval with optional metadata filters
- Prompt construction that cites sources
- LLM answers with `gemini-2.5-flash-lite`
- FastAPI endpoints: `/query`, `/ingest`, `/health`

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

- Database connection (`DATABASE_URL` or DB_HOST/DB_NAME/DB_USER/DB_PASSWORD)
- `database.table` and `database.columns`
- `embedding.model` and `GOOGLE_API_KEY`
- Elasticsearch `hosts`, `index`
- Retrieval `top_k`

Example environment:

```bash
export GOOGLE_API_KEY=your_key
export DB_HOST=localhost
export DB_NAME=properties
export DB_USER=postgres
export DB_PASSWORD=postgres
export ELASTIC_HOST=http://localhost:9200
export ELASTIC_USERNAME=elastic
export ELASTIC_PASSWORD=changeme
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs`.

## Ingest

Start ingestion (runs in background):

```bash
curl -X POST http://localhost:8000/ingest -H 'Content-Type: application/json' -d '{"rebuild_index": true, "batch_size": 500}'
```

## Query

```bash
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"query": "2-bedroom in Dubai Marina with gym", "filters": {"city": "Dubai"}, "top_k": 5}'
```

## Notes

- The pipeline is generic; adjust `config.yaml` to switch DB/table/columns, embeddings, vector DB, LLM, chunk sizes, and retrieval parameters.
- Ensure your Elasticsearch has vector support (dense_vector with the configured similarity).
- For multiple tables, extend the loader and config to list multiple sources and loop over them.
