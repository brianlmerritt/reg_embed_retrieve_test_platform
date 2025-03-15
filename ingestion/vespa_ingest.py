import requests
import json
import os

VESPA_ENDPOINT = os.environ.get("VESPA_ENDPOINT", "http://vespa:8080")


def index_document(doc, dense_embedding, multi_vector, sparse_vector):
    """
    Index a document into Vespa with all embedding types.
    
    Args:
        doc: Document metadata and content
        dense_embedding: Dense vector embedding (list of floats)
        multi_vector: List of token embeddings
        sparse_vector: List of term-weight pairs
    """
    url = f"{VESPA_ENDPOINT}/document/v1/vetsearch/vetsearch/docid/{doc['id']}"
    
    # Format multi-vector for Vespa
    formatted_multi_vector = {
        "cells": [
            {"address": {"x": str(i)}, "value": vec}
            for i, vec in enumerate(multi_vector)
        ]
    }
    
    # Format sparse vector for Vespa
    formatted_sparse_vector = {
        "cells": [
            {"address": {"x": item['term']}, "value": item['weight']} 
            for item in sparse_vector
        ]
    }
    
    # Create Vespa document
    vespa_doc = {
        "fields": {
            "id": doc["id"],
            "contents": doc["contents"],
            "dense_embedding": dense_embedding,
            "multi_vector": formatted_multi_vector,
            "sparse_vector": formatted_sparse_vector,
            "course_id": doc["course_id"],
            "activity_id": doc["activity_id"],
            "course_name": doc["course_name"],
            "activity_name": doc["activity_name"],
            "activity_type": doc["activity_type"],
            "strand": doc["strand"]
        }
    }
    
    response = requests.post(url, json=vespa_doc)
    
    if response.status_code not in (200, 201):
        raise Exception(f"Failed to index document: {response.text}")
    
    return response.json()


def batch_index_from_embeddings_file(embeddings_file):
    """
    Index all documents from a pre-generated embeddings file.
    
    Args:
        embeddings_file: Path to JSON file with embeddings
    """
    with open(embeddings_file, 'r') as f:
        documents = json.load(f)
    
    results = []
    
    for doc in documents:
        try:
            # Reconstruct the original document structure
            original_doc = {
                "id": doc["id"],
                "contents": doc.get("contents", ""),  # The embeddings file might not have the content
                **doc["metadata"]
            }
            
            result = index_document(
                original_doc,
                doc["dense_embedding"],
                doc["multi_vector"],
                doc["sparse_vector"]
            )
            
            results.append({"id": doc["id"], "status": "success"})
            
        except Exception as e:
            results.append({"id": doc["id"], "status": "error", "message": str(e)})
    
    return {
        "indexed": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "results": results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index documents into Vespa")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings file")
    
    args = parser.parse_args()
    
    results = batch_index_from_embeddings_file(args.embeddings)
    print(json.dumps(results, indent=2))