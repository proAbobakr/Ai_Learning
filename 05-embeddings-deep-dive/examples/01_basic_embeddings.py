#!/usr/bin/env python3
"""
Example 1: Basic Embeddings
Learn the fundamentals of creating and using embeddings.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

print("=" * 60)
print("EXAMPLE 1: BASIC EMBEDDINGS")
print("=" * 60)

# Load a pre-trained model
print("\n1. Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"   Model loaded: {model}")
print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Create embeddings for single text
print("\n2. Creating embedding for single text...")
text = "Machine learning is fascinating"
embedding = model.encode(text)
print(f"   Text: '{text}'")
print(f"   Embedding shape: {embedding.shape}")
print(f"   First 5 values: {embedding[:5]}")

# Create embeddings for multiple texts
print("\n3. Creating embeddings for multiple texts...")
texts = [
    "I love programming in Python",
    "Python coding is my favorite hobby",
    "The weather is nice today",
]
embeddings = model.encode(texts)
print(f"   Number of texts: {len(texts)}")
print(f"   Embeddings shape: {embeddings.shape}")

# Calculate similarity
print("\n4. Calculating similarity...")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"   '{texts[i][:30]}...' <-> '{texts[j][:30]}...'")
        print(f"   Similarity: {sim:.4f}\n")

# Normalize embeddings
print("\n5. Normalized embeddings...")
normalized = model.encode(texts, normalize_embeddings=True)
print(f"   First embedding norm: {np.linalg.norm(normalized[0]):.4f}")  # Should be ~1.0

# For normalized vectors, dot product = cosine similarity
print("\n6. Using dot product for normalized vectors...")
sim_dot = np.dot(normalized[0], normalized[1])
sim_cos = cosine_similarity(embeddings[0], embeddings[1])
print(f"   Dot product: {sim_dot:.4f}")
print(f"   Cosine sim:  {sim_cos:.4f}")
print(f"   They're equal: {np.isclose(sim_dot, sim_cos)}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
