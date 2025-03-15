# Prompt (try o3-mini-high and sonnet 3.7)
Finalized Spec: Veterinary Learning Content Search - Test Platform (Docker Compose Vespa-based)
🎯 Goal:
Create a modular, Docker Compose-based test platform that:

Supports BM25, sparse, dense, and multi-vector search over veterinary learning content.
Uses Vespa.ai as the unified backend for indexing and search.
Uses Hugging Face models in Python to generate dense, multi-vector, and sparse embeddings.
Supports flexible metadata indexing (course ID, activity ID, strand, etc.).
Provides FastAPI/Flask API endpoints for query testing and comparison.
✅ Key Components and Flow
1. Data Preparation Pipeline (Python)
Input: Veterinary Moodle content in .jsonl format:
json
Copy
Edit
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
Output: JSON-ready for indexing, including embeddings.
2. Embedding Generation Pipeline (Python + HF)
For each document:

Dense embedding (BGE-M3 CLS).
Multi-vector embedding (BGE-M3 token embeddings).
Sparse vector (simulate SPLADE/uniCOIL for testing, using term-weight pairs).
➡️ For simplicity in testing, sparse vectors will be small (~5-10 terms/doc) with synthetic or real weights.

Example sparse vector:

json
Copy
Edit
[
  {"term": "renal", "weight": 0.8},
  {"term": "disease", "weight": 0.6}
]
3. Vespa-Based Unified Index & Search System (Dockerized)
Retrieval Type	Vespa Feature	Field Name
BM25	Text indexing	contents
Sparse (uniCOIL/SPLADE)	Tensor of term-weight pairs	sparse_vector
Dense (BGE-M3)	Tensor for ANN	dense_embedding
Multi-vector (BGE-M3)	Tensor for late interaction (MaxSim)	multi_vector
➡️ Indexed with associated metadata.

4. FastAPI/Flask API for Search and Indexing
API Endpoint	Purpose
/index	Index content with embeddings
/search/bm25	BM25 search
/search/sparse	Sparse search
/search/dense	Dense embedding search
/search/multivector	Multi-vector search
/search/all	Combined search for comparison
5. Docker Compose Structure
graphql
Copy
Edit
vet-search-platform/
│
├── docker-compose.yml         # Docker orchestration
├── data/                      # Content & embeddings
│   └── vet_moodle_dataset.jsonl
├── vespa/                     # Vespa application package
│   ├── services.xml            # Vespa app config
│   └── schema.sd               # Vespa schema (fields + ranking)
├── api/                       # FastAPI/Flask interface
│   └── main.py                 # Search and indexing endpoints
├── embeddings/                # Embedding generation logic
│   └── generate_embeddings.py  # Hugging Face pipelines
├── ingestion/                 # Vespa ingestion client
│   └── vespa_ingest.py         # Push documents via REST
└── README.md                  # Instructions and documentation
✅ 6. Docker Compose Platform
Example docker-compose.yml
yaml
Copy
Edit
version: '3.8'

services:
  vespa:
    image: vespaengine/vespa
    ports:
      - "8080:8080"
    volumes:
      - ./vespa:/vespa-app
    command: >
      sh -c "
        /opt/vespa/bin/vespa-deploy prepare /vespa-app &&
        /opt/vespa/bin/vespa-deploy activate &&
        /opt/vespa/bin/vespa-start-services &&
        tail -f /dev/null
      "

  api:
    build: ./api
    ports:
      - "8000:8000"
    depends_on:
      - vespa
    volumes:
      - ./data:/app/data
✅ 7. Vespa Schema (schema.sd)
xml
Copy
Edit
schema vetsearch {
  document vetsearch {
    field id type string { indexing: attribute | summary }
    field contents type string { indexing: index | summary }
    field dense_embedding type tensor<float>(d0[768]) { indexing: attribute }
    field multi_vector type tensor<float>(x{}, d0[768]) { indexing: attribute }
    field sparse_vector type tensor<float>(x{}) { indexing: attribute }
    field course_id type string { indexing: attribute | summary }
    field activity_id type string { indexing: attribute | summary }
    field strand type string { indexing: attribute | summary }
  }

  rank-profile bm25 inherits default { first-phase { expression: bm25(contents) } }
  rank-profile dense inherits default { first-phase { expression: closeness(dense_embedding, query(dense_embedding)) } }
  rank-profile multi_vector inherits default { first-phase { expression: max(dot(multi_vector, query(multi_vector))) } }
  rank-profile sparse inherits default { first-phase { expression: sum(dot(sparse_vector, query(sparse_vector))) } }
}
✅ 8. Indexing Flow (Python)
python
Copy
Edit
# Example ingestion via Vespa REST
import requests

def index_document(doc, dense, multi, sparse):
    url = "http://localhost:8080/document/v1/vetsearch/vetsearch/docid/{}".format(doc["id"])
    vespa_doc = {
        "fields": {
            "id": doc["id"],
            "contents": doc["contents"],
            "dense_embedding": dense,
            "multi_vector": {"cells": [{"address": {"x": str(i)}, "value": vec} for i, vec in enumerate(multi)]},
            "sparse_vector": {"cells": [{"address": {"x": item['term']}, "value": item['weight']} for item in sparse]},
            "course_id": doc["course_id"],
            "activity_id": doc["activity_id"],
            "strand": doc["strand"]
        }
    }
    return requests.post(url, json=vespa_doc)
✅ 9. Search Flow Example (Python)
python
Copy
Edit
def search_bm25(query):
    data = {"yql": f'select * from sources * where userQuery();', "query": query, "ranking": "bm25", "hits": 10}
    return requests.post("http://localhost:8080/search/", json=data).json()
➡️ Similar functions for dense, sparse, multi-vector using ranking profiles.

✅ 10. Future Extensions
Add rerankers (MonoBERT, T5) via ONNX inside Vespa.
Explore hybrid rank profiles (e.g., sum of BM25, dense, multi-vector).
Add evaluation scripts to benchmark models on veterinary queries.
✅ 11. Final Step: Testing
docker-compose up to bring up Vespa and API.
Generate embeddings via embeddings/generate_embeddings.py.
Index via ingestion/vespa_ingest.py.
Query via API or search scripts.