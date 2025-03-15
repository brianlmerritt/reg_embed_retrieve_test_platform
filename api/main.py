document.dict(),
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import json
import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI(title="Veterinary Learning Content Search API")

# Configuration
VESPA_ENDPOINT = os.getenv("VESPA_ENDPOINT", "http://vespa:8080")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/bge-m3")

# Load models (lazy loading on first request)
dense_model = None
tokenizer = None

def load_models():
    global dense_model, tokenizer
    if dense_model is None:
        try:
            dense_model = SentenceTransformer(MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise


class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    filter: Optional[Dict[str, str]] = None


class Document(BaseModel):
    id: str
    contents: str
    course_id: str
    activity_id: str
    course_name: str
    activity_name: str
    activity_type: str
    strand: str


@app.get("/")
def read_root():
    return {"status": "Veterinary Learning Content Search API is running"}


@app.post("/search/bm25")
async def search_bm25(query: SearchQuery):
    yql_query = f'select * from sources * where userQuery();'
    
    if query.filter:
        filter_conditions = []
        for field, value in query.filter.items():
            filter_conditions.append(f'{field} contains "{value}"')
        
        if filter_conditions:
            conditions = " and ".join(filter_conditions)
            yql_query = f'select * from sources * where {conditions} and userQuery();'
    
    data = {
        "yql": yql_query,
        "query": query.query,
        "ranking": "bm25",
        "hits": query.limit
    }
    
    try:
        response = requests.post(f"{VESPA_ENDPOINT}/search/", json=data)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/search/dense")
async def search_dense(query: SearchQuery):
    # Load models if not already loaded
    if dense_model is None:
        load_models()
    
    # Generate embedding for query
    query_embedding = dense_model.encode(query.query).tolist()
    
    yql_query = f'select * from sources * where userQuery();'
    if query.filter:
        filter_conditions = []
        for field, value in query.filter.items():
            filter_conditions.append(f'{field} contains "{value}"')
        
        if filter_conditions:
            conditions = " and ".join(filter_conditions)
            yql_query = f'select * from sources * where {conditions};'
    
    data = {
        "yql": yql_query,
        "ranking": "dense",
        "input.query(dense_embedding)": query_embedding,
        "hits": query.limit
    }
    
    try:
        response = requests.post(f"{VESPA_ENDPOINT}/search/", json=data)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/search/sparse")
async def search_sparse(query: SearchQuery):
    # For simplicity, we'll simulate sparse vector creation here
    # In production, you'd use a proper sparse encoder like SPLADE
    
    # Create a simple sparse vector using tokenization
    words = query.query.lower().split()
    sparse_vector = {}
    
    for word in words:
        if len(word) > 2:  # Skip very short words
            sparse_vector[word] = 1.0
    
    yql_query = f'select * from sources * where userQuery();'
    if query.filter:
        filter_conditions = []
        for field, value in query.filter.items():
            filter_conditions.append(f'{field} contains "{value}"')
        
        if filter_conditions:
            conditions = " and ".join(filter_conditions)
            yql_query = f'select * from sources * where {conditions};'
    
    data = {
        "yql": yql_query,
        "ranking": "sparse",
        "input.query(q)": {"cells": [{"address": {"x": term}, "value": weight} for term, weight in sparse_vector.items()]},
        "hits": query.limit
    }
    
    try:
        response = requests.post(f"{VESPA_ENDPOINT}/search/", json=data)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/search/multivector")
async def search_multivector(query: SearchQuery):
    # Load models if not already loaded
    if dense_model is None or tokenizer is None:
        load_models()
    
    # Tokenize query and generate token-level embeddings
    tokens = tokenizer(query.query, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = dense_model(**tokens)
    
    token_embeddings = outputs.last_hidden_state.squeeze().tolist()
    
    yql_query = f'select * from sources * where userQuery();'
    if query.filter:
        filter_conditions = []
        for field, value in query.filter.items():
            filter_conditions.append(f'{field} contains "{value}"')
        
        if filter_conditions:
            conditions = " and ".join(filter_conditions)
            yql_query = f'select * from sources * where {conditions};'
    
    data = {
        "yql": yql_query,
        "ranking": "multi_vector",
        "input.query(q)": token_embeddings[0],  # Just use the first token for simplicity
        "hits": query.limit
    }
    
    try:
        response = requests.post(f"{VESPA_ENDPOINT}/search/", json=data)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/search/all")
async def search_all(query: SearchQuery):
    results = {}
    
    # Call each search endpoint
    try:
        results["bm25"] = (await search_bm25(query)).get("root", {}).get("children", [])
        results["dense"] = (await search_dense(query)).get("root", {}).get("children", [])
        results["sparse"] = (await search_sparse(query)).get("root", {}).get("children", [])
        results["multivector"] = (await search_multivector(query)).get("root", {}).get("children", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combined search error: {str(e)}")
    
    return results


@app.post("/index")
async def index_document(document: Document):
    from ingestion.vespa_ingest import index_document
    
    # First, generate embeddings
    if dense_model is None:
        load_models()
    
    # Generate dense embedding
    dense_embedding = dense_model.encode(document.contents).tolist()
    
    # Generate multi-vector embedding
    tokens = tokenizer(document.contents, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = dense_model(**tokens)
    
    multi_vector_embeddings = outputs.last_hidden_state.squeeze().tolist()
    
    # Generate simple sparse vector
    words = document.contents.lower().split()
    sparse_vector = []
    for word in set(words):
        if len(word) > 2:
            sparse_vector.append({"term": word, "weight": words.count(word) / len(words)})
    
    # Index the document with embeddings
    try:
        result = index_document(
            document.dict(),
            dense_embedding,
            multi_vector_embeddings,
            sparse_vector
        )
        return {"status": "Document indexed successfully", "document_id": document.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")


@app.post("/index/batch")
async def index_batch(documents: List[Document]):
    results = []
    for doc in documents:
        try:
            result = await index_document(doc)
            results.append({"id": doc.id, "status": "success"})
        except Exception as e:
            results.append({"id": doc.id, "status": "error", "message": str(e)})
    
    return {"indexed": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)