# Veterinary Learning Content Search Platform

A modular Docker Compose-based test platform for veterinary learning content search using Vespa.ai as the unified backend.

## Overview

This platform supports:
- BM25, sparse, dense, and multi-vector search over veterinary learning content
- Vespa.ai as the unified backend for indexing and search
- Hugging Face models to generate dense, multi-vector, and sparse embeddings
- Flexible metadata indexing (course ID, activity ID, strand, etc.)
- FastAPI API endpoints for query testing and comparison

## Architecture

The platform consists of the following components:

1. **Vespa.ai Search Engine**: Core search backend supporting multiple retrieval methods
2. **FastAPI Service**: REST API for search and indexing operations
3. **Embedding Service**: Generates embeddings for documents using the BGE-M3 model

## Directory Structure

```
vet-search-platform/
│
├── docker-compose.yml         # Docker orchestration
├── data/                      # Content & embeddings
│   └── vet_moodle_dataset.jsonl
├── vespa/                     # Vespa application package
│   ├── services.xml           # Vespa app config
│   └── schema.sd              # Vespa schema (fields + ranking)
├── api/                       # FastAPI interface
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                # Search and indexing endpoints
├── embeddings/                # Embedding generation logic
│   ├── Dockerfile
│   ├── requirements.txt
│   └── generate_embeddings.py # Hugging Face pipelines
├── ingestion/                 # Vespa ingestion client
│   └── vespa_ingest.py        # Push documents via REST
└── README.md                  # Instructions and documentation
```

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Data in JSONL format with the following structure:

```json
{
  "id": "doc1",
  "contents": "Full textual content.",
  "course_id": "VET101",
  "activity_id": "ACT205",
  "course_name": "Small Animal Medicine",
  "activity_name": "Renal Diseases",
  "activity_type": "Moodle_Book",
  "strand": "Internal Medicine"
}
```

### Starting the Platform

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd vet-search-platform
   ```

2. Place your data file in the `data` directory:
   ```bash
   cp path/to/your/data.jsonl data/vet_moodle_dataset.jsonl
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Generate embeddings:
   ```bash
   docker-compose exec embedding-service python generate_embeddings.py
   ```

5. Index the documents:
   ```bash
   docker-compose exec api python -c "from ingestion.vespa_ingest import batch_index_from_embeddings_file; print(batch_index_from_embeddings_file('/app/data/embeddings/embeddings.json'))"
   ```

## API Usage

### Search Endpoints

The API provides the following search endpoints:

#### BM25 Search
```bash
curl -X POST http://localhost:8000/search/bm25 \
  -H "Content-Type: application/json" \
  -d '{"query": "renal disease", "limit": 10}'
```

#### Dense Vector Search
```bash
curl -X POST http://localhost:8000/search/dense \
  -H "Content-Type: application/json" \
  -d '{"query": "renal disease", "limit": 10}'
```

#### Sparse Vector Search
```bash
curl -X POST http://localhost:8000/search/sparse \
  -H "Content-Type: application/json" \
  -d '{"query": "renal disease", "limit": 10}'
```

#### Multi-Vector Search
```bash
curl -X POST http://localhost:8000/search/multivector \
  -H "Content-Type: application/json" \
  -d '{"query": "renal disease", "limit": 10}'
```

#### Combined Search
```bash
curl -X POST http://localhost:8000/search/all \
  -H "Content-Type: application/json" \
  -d '{"query": "renal disease", "limit": 10}'
```

### Indexing Documents

#### Index a Single Document
```bash
curl -X POST http://