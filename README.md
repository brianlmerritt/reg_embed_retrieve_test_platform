# Medical / Technical Learning Content Search Test Platform

This platform demonstrates a Vespa-based unified search engine for veterinary learning content with support for BM25, sparse, dense, and multi-vector search.

## Components

- **Data Preparation**: Expects a JSONL file (`data/vet_moodle_dataset.jsonl`) containing veterinary Moodle content.
- **Embedding Generation**: Uses Hugging Face models (or dummy embeddings for testing) in `embeddings/generate_embeddings.py`.
- **Vespa Index & Search**: Vespa configuration and schema are in the `vespa/` folder.
- **API Endpoints**: A FastAPI application in `api/main.py` exposes endpoints for:
  - `/index`: Index documents (automatically generating embeddings)
  - `/search/bm25`: BM25 search
  - `/search/dense`: Dense search
  - `/search/multivector`: Multi-vector search
  - `/search/sparse`: Sparse search
  - `/search/all`: Run all search types for comparison

## How to Run

1. **Start the platform:**
   ```bash
   docker-compose up

