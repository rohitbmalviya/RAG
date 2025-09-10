# RAG Pipeline

A config-driven, generic Retrieval-Augmented Generation (RAG) FastAPI service for querying PostgreSQL-backed real-estate data with embeddings stored in Elasticsearch, using Google Gemini for embeddings and generation.

## Features

- Config-driven pipeline via `config.yaml`
- Load from PostgreSQL table (default: `properties`)
- Chunking with overlap; configurable unit (`token` or `char`) and offsets preserved
- Embeddings with `your_embed_model_name` with retry/backoff and batching
- Vector store in Elasticsearch with kNN search (`dense_vector`), configurable dims and index
- Retrieval with optional metadata filters and configurable `top_k`
- Prompt construction that cites sources
- LLM answers with `your_model_name`
- FastAPI endpoints: `/ingest`, `/query`, `/health`
- Modular design, extendable to multiple tables/sources

## Requirements

- Python 3.11+
- PostgreSQL 14+
- Elasticsearch 8.13+
- Google API key (`LLM_MODEL_API_KEY`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Edit `config.yaml` or set environment variables referenced in it. Key items:

- Database connection `DATABASE_URL`
- `database.table` and `database.columns`
- `chunking.unit` (`token` or `char`), `chunk_size`, `chunk_overlap`
- `embedding.model` and `LLM_MODEL_API_KEY`
- Elasticsearch `ELASTIC_HOST`, `ELASTIC_USERNAME`, `ELASTIC_PASSWORD`, `vector_db.index`, `vector_db.dims`
- Retrieval `top_k`

### .env example

Create a `.env` file (or copy from `.env.example` if present):

```bash
LLM_MODEL_API_KEY=your_google_api_key
DB_URL=postgresql://postgres:postgres@localhost:5432/leasebnb
ELASTIC_HOST=http://localhost:9200
ELASTIC_USERNAME=elastic
ELASTIC_PASSWORD=changeme
```

## Start the Application

### Prerequisites

1. Ensure PostgreSQL is running and accessible
2. Ensure Elasticsearch is running and accessible
3. Set up your environment variables (see Configure section above)
4. Install dependencies: `pip install -r requirements.txt`

### Step-by-Step Local Setup

#### 1. Create Environment File

Create a `.env` file in the project root with the following content:

```bash
# Google API Configuration
LLM_MODEL_API_KEY=your_google_api_key_here
LLM_EMBED_MODEL=your_embed_model_name
LLM_MODEL_MODEL=your_model_name

# Database Configuration
DB_URL=postgresql://postgres:postgres@localhost:5432/leasebnb
# Alternative: Individual DB settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=leasebnb
DB_USER=postgres
DB_PASSWORD=postgres

# Elasticsearch Configuration
ELASTIC_HOST=http://localhost:9200
ELASTIC_USERNAME=elastic
ELASTIC_PASSWORD=DkIedPPSCb
```

**Important:** Replace `your_google_api_key_here` with your actual Google API key.

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Start Required Services

**Option A: Using Docker Compose (Recommended)**

```bash
docker-compose up -d
```

**Option B: Manual Setup**

- Start PostgreSQL on localhost:5432
- Start Elasticsearch on localhost:9200

#### 4. Start the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`. Visit `http://localhost:8000/docs` to access the interactive API documentation.

### Testing the Application

Once running, you can test the endpoints:

- **Health Check**: `curl http://localhost:8000/health`
- **Ingest Data**: `curl -X POST http://localhost:8000/ingest -H 'Content-Type: application/json' -d '{"rebuild_index": true}'`
- **Query Data**: `curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"query": "2-bedroom apartment in Dubai"}'`

## Notes

- The pipeline is generic; adjust `config.yaml` to switch DB/table/columns, embeddings, vector DB, LLM, chunk sizes, and retrieval parameters.
- Ensure your Elasticsearch has vector support (dense_vector with the configured similarity) and credentials are valid.
- For multiple tables, extend the loader and config to list multiple sources and iterate.
