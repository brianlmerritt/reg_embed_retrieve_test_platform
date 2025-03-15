from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import List, Optional

# Import our embedding generator and ingestion functions
from embeddings.generate_embeddings import generate_embeddings
from ingestion.vespa_ingest import index_document

app = FastAPI(title="Veterinary Learning Content Search API")

# Pydantic model for incoming documents (you can extend this as needed)
class Document(BaseModel):
    id: str
    contents: str
    course_id: str
    activity_id: str
    strand: str

@app.post("/index")
def index(doc: Document):
    # Generate embeddings (dense, multi-vector, sparse) based on the contents.
    dense, multi, sparse = generate_embeddings(doc.contents)
    vespa_response = index_document(doc.dict(), dense, multi, sparse)
    if vespa_response.status_code != 200:
        raise HTTPException(status_code=vespa_response.status_code, detail="Indexing failed")
    return {"status": "indexed", "vespa_response": vespa_response.json()}

# Generic search function to call Vespa with a given ranking profile
def search(query: str, ranking: str):
    data = {
        "yql": "select * from sources * where userQuery();",
        "query": query,
        "ranking": ranking,
        "hits": 10
    }
    # Using the service name "vespa" to refer to the Vespa container on the Docker network
    response = requests.post("http://vespa:8080/search/", json=data)
    return response

@app.post("/search/bm25")
def search_bm25(query: str):
    response = search(query, "bm25")
    return response.json()

@app.post("/search/dense")
def search_dense(query: str):
    response = search(query, "dense")
    return response.json()

@app.post("/search/multivector")
def search_multivector(query: str):
    response = search(query, "multi_vector")
    return response.json()

@app.post("/search/sparse")
def search_sparse(query: str):
    response = search(query, "sparse")
    return response.json()

@app.post("/search/all")
def search_all(query: str):
    # Run all searches and return their results for comparison
    results = {
        "bm25": search(query, "bm25").json(),
        "dense": search(query, "dense").json(),
        "multi_vector": search(query, "multi_vector").json(),
        "sparse": search(query, "sparse").json()
    }
    return results
