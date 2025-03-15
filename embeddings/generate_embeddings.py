import torch
from transformers import AutoTokenizer, AutoModel
import re
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModel
import re
from collections import Counter

# Use the actual BGE-M3 model from Hugging Face.
MODEL_NAME = "BAAI/bge-m3"

# Load tokenizer and model (this will cache the model locally on first run)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # Set the model to evaluation mode


def get_dense_and_multivector(text: str):
    """
    Generate dense and multi-vector embeddings using BGE-M3.
    
    Dense vector:
      - Uses the pooled output if available, or the [CLS] token representation as fallback.
    
    Multi-vector:
      - Extracts up to 5 token embeddings from the last hidden state, ignoring special tokens.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Dense embedding: prefer pooler_output if available, else fallback to first token's embedding
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        dense_embedding = outputs.pooler_output[0].tolist()
    else:
        dense_embedding = outputs.last_hidden_state[0, 0].tolist()
    
    # Multi-vector: select up to 5 non-special token embeddings
    input_ids = inputs["input_ids"][0]
    special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}
    token_embeddings = outputs.last_hidden_state[0]
    
    # Collect embeddings for tokens that are not special tokens
    multi_vector = [
        token_embeddings[i].tolist()
        for i in range(len(input_ids))
        if input_ids[i].item() not in special_token_ids
    ][:5]
    
    return dense_embedding, multi_vector


def get_sparse_vector(text: str, top_k: int = 5):
    """
    Simulates sparse embeddings by tokenizing the text, counting word frequencies,
    and returning the top_k most common words with normalized weights.
    """
    words = re.findall(r'\w+', text.lower())
    if not words:
        return []
    
    counter = Counter(words)
    most_common = counter.most_common(top_k)
    max_freq = most_common[0][1] if most_common else 1
    
    sparse_vector = [
        {"term": word, "weight": round(freq / max_freq, 2)}
        for word, freq in most_common
    ]
    
    return sparse_vector


def generate_embeddings(text: str):
    """
    Generate embeddings for a given text using BGE-M3.
    
    Returns:
      - dense_embedding: A list of floats (dense vector)
      - multi_vector: A list of token embeddings (each a list of floats)
      - sparse_vector: A list of dictionaries with "term" and "weight" keys
    """
    dense, multi = get_dense_and_multivector(text)
    sparse = get_sparse_vector(text)
    return dense, multi, sparse


if __name__ == "__main__":
    # Sample text for testing the embeddings
    sample_text = (
        "Renal disease is common in small animal medicine and requires immediate attention."
    )
    dense, multi, sparse = generate_embeddings(sample_text)
    print("Dense embedding (first 5 values):", dense[:5])
    print("Multi-vector shape:", (len(multi), len(multi[0]) if multi else 0))
    print("Sparse vector:", sparse)



def get_sparse_vector(text: str, top_k: int = 5):
    """
    Simulates sparse embeddings by:
      - Tokenizing the text (simple regex-based approach).
      - Counting word frequencies.
      - Returning the top_k most frequent words with normalized weights.
    """
    # Lowercase and extract word tokens (ignores punctuation)
    words = re.findall(r'\w+', text.lower())
    if not words:
        return []

    # Count word frequencies
    counter = Counter(words)
    most_common = counter.most_common(top_k)
    max_freq = most_common[0][1] if most_common else 1

    # Normalize frequencies to generate weights in [0, 1]
    sparse_vector = []
    for word, freq in most_common:
        weight = freq / max_freq
        sparse_vector.append({"term": word, "weight": round(weight, 2)})

    return sparse_vector


def generate_embeddings(text: str):
    """
    Generate embeddings for a given text.
    
    Returns:
      - dense_embedding: A list of floats (dense vector)
      - multi_vector: A list of token embeddings (list of float lists)
      - sparse_vector: A list of dictionaries with "term" and "weight" keys
    """
    dense, multi = get_dense_and_multivector(text)
    sparse = get_sparse_vector(text)
    return dense, multi, sparse


if __name__ == "__main__":
    # Sample text for testing the embeddings
    sample_text = (
        "Renal disease is common in small animal medicine and requires immediate attention."
    )
    dense, multi, sparse = generate_embeddings(sample_text)
    print("Dense embedding (first 5 values):", dense[:5])
    print("Multi-vector shape:", (len(multi), len(multi[0]) if multi else 0))
    print("Sparse vector:", sparse)
