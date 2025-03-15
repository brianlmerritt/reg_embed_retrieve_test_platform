import os
import json
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# Configuration
DATA_PATH = os.environ.get("DATA_PATH", "/app/data/vet_moodle_dataset.jsonl")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/app/data/embeddings")
MODEL_PATH = os.environ.get("MODEL_PATH", "BAAI/bge-m3")
BATCH_SIZE = 8

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_data(data_path):
    """Load data from JSONL file."""
    if data_path.endswith('.jsonl'):
        return pd.read_json(data_path, lines=True)
    else:
        raise ValueError("Currently only supporting JSONL format")


def generate_sparse_vector(text, max_terms=10):
    """Generate a simple sparse vector representation.
    
    For production, you would use a model like SPLADE or uniCOIL.
    This is a simplified version for testing.
    """
    words = text.lower().split()
    word_counts = {}
    
    for word in words:
        if len(word) > 2:  # Skip very short words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by count and take top terms
    sorted_terms = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_terms = sorted_terms[:max_terms]
    
    # Create sparse vector
    total_count = sum(count for _, count in top_terms)
    sparse_vector = [{"term": term, "weight": count / total_count} for term, count in top_terms]
    
    return sparse_vector


def process_batch(model, tokenizer, texts):
    """Generate dense and multi-vector embeddings for a batch of texts."""
    
    # Dense embeddings
    dense_embeddings = model.encode(texts)
    
    # Multi-vector embeddings
    multi_vectors = []
    
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True, 
                           truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(tokens["input_ids"])
            
        # Get token embeddings
        token_embeddings = outputs.last_hidden_state.squeeze().tolist()
        
        # Limit to a reasonable number of vectors (e.g., 20)
        if isinstance(token_embeddings[0], list):  # Check if it's a 2D list
            multi_vectors.append(token_embeddings[:20])
        else:
            # Handle single token case
            multi_vectors.append([token_embeddings])
    
    return dense_embeddings, multi_vectors
    

def main():
    print("Loading data...")
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Loaded {len(df)} documents")
    
    print("Loading models...")
    try:
        # Load model and tokenizer
        model = SentenceTransformer(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    print("Processing documents...")
    results = []
    
    # Process in batches
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]
        texts = batch['contents'].tolist()
        
        # Generate embeddings
        dense_embeddings, multi_vectors = process_batch(model, tokenizer, texts)
        
        # Generate sparse vectors and combine results
        for j, row in enumerate(batch.itertuples()):
            sparse_vector = generate_sparse_vector(row.contents)
            
            doc_data = {
                "id": row.id,
                "dense_embedding": dense_embeddings[j].tolist(),
                "multi_vector": multi_vectors[j],
                "sparse_vector": sparse_vector,
                "metadata": {
                    "course_id": row.course_id,
                    "activity_id": row.activity_id,
                    "course_name": row.course_name,
                    "activity_name": row.activity_name,
                    "activity_type": row.activity_type,
                    "strand": row.strand
                }
            }
            results.append(doc_data)
    
    # Save results
    output_file = os.path.join(OUTPUT_PATH, "embeddings.json")
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    print(f"Saved embeddings for {len(results)} documents to {output_file}")


if __name__ == "__main__":
    main()