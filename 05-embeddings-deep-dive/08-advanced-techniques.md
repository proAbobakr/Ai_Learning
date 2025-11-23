# Part 8: Advanced Techniques

Master advanced embedding techniques for production systems.

## 1. Fine-tuning Embeddings

Train embeddings for your specific domain.

### Why Fine-tune?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GENERIC vs FINE-TUNED                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Generic Model:                   Fine-tuned Model:                     │
│  ───────────────                  ─────────────────                     │
│  "bank" = financial institution   "bank" = your specific meaning        │
│  General vocabulary               Domain-specific vocabulary            │
│  ~80% accuracy                    ~95% accuracy (on your data)          │
│                                                                         │
│  Best for:                        Best for:                             │
│  - General purpose                - Medical/Legal/Finance               │
│  - Quick prototypes               - Technical documentation             │
│  - Low volume                     - Custom terminology                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Contrastive Learning Fine-tuning

```python
"""
Fine-tune embedding model with contrastive learning.
Similar pairs should be close, different pairs should be far.
"""

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare training data
# Format: (sentence1, sentence2, similarity_score)
# similarity_score: 0.0 = completely different, 1.0 = identical meaning
train_examples = [
    # Similar pairs (high score)
    InputExample(texts=["What is machine learning?",
                       "Explain ML to me"], label=0.9),
    InputExample(texts=["How do neural networks work?",
                       "Explain deep learning architecture"], label=0.85),
    InputExample(texts=["Python programming basics",
                       "Introduction to Python coding"], label=0.95),

    # Different pairs (low score)
    InputExample(texts=["Machine learning algorithms",
                       "Recipe for chocolate cake"], label=0.1),
    InputExample(texts=["Neural network training",
                       "Car maintenance tips"], label=0.05),
]

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100,
    output_path="./fine_tuned_model"
)

# Load and use fine-tuned model
fine_tuned = SentenceTransformer('./fine_tuned_model')
```

### Triplet Loss Fine-tuning

```python
"""
Triplet loss: anchor, positive (similar), negative (different)
"""

from sentence_transformers import InputExample, losses

# Triplet examples
triplet_examples = [
    # (anchor, positive, negative)
    InputExample(texts=[
        "What is Python?",                    # Anchor
        "Tell me about Python programming",   # Positive (similar)
        "Weather forecast for tomorrow"       # Negative (different)
    ]),
    InputExample(texts=[
        "Machine learning tutorial",
        "ML beginner guide",
        "Cooking recipes collection"
    ]),
]

train_dataloader = DataLoader(triplet_examples, shuffle=True, batch_size=16)

# Triplet loss
train_loss = losses.TripletLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    output_path="./triplet_fine_tuned"
)
```

### Mining Hard Negatives

```python
"""
Hard negatives: similar-looking but different meaning sentences.
Much more effective for training!
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Your queries and their correct documents
queries = [
    "How to install Python?",
    "What is machine learning?",
    "Best programming languages",
]

# All documents (correct ones + many others)
documents = [
    "Python installation guide",        # Correct for query 0
    "Machine learning explained",       # Correct for query 1
    "Top programming languages 2024",   # Correct for query 2
    "Python snake species",             # Hard negative for query 0!
    "Machine tools and equipment",      # Hard negative for query 1!
    "Programming TV remote",            # Hard negative for query 2!
    "Weather forecast",                 # Easy negative
]

# Query-document mappings (query_idx -> correct_doc_idx)
qd_mappings = {0: [0], 1: [1], 2: [2]}

# Mine hard negatives
query_embeddings = model.encode(queries)
doc_embeddings = model.encode(documents)

def find_hard_negatives(query_idx, correct_docs, all_embeddings, top_k=5):
    """Find hard negatives - similar but incorrect documents."""
    query_emb = query_embeddings[query_idx]
    similarities = np.dot(all_embeddings, query_emb)

    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]

    # Get hard negatives (high similarity but not correct)
    hard_negatives = []
    for idx in sorted_indices:
        if idx not in correct_docs and len(hard_negatives) < top_k:
            hard_negatives.append({
                'index': idx,
                'text': documents[idx],
                'similarity': similarities[idx]
            })

    return hard_negatives

# Find hard negatives for first query
hard_negs = find_hard_negatives(0, qd_mappings[0], doc_embeddings)
print("Hard negatives for 'How to install Python?':")
for neg in hard_negs:
    print(f"  [{neg['similarity']:.4f}] {neg['text']}")
```

---

## 2. Dimensionality Reduction

Reduce embedding size while preserving quality.

### Why Reduce Dimensions?

```
Benefits:
- Faster search (less computation)
- Less storage (smaller vectors)
- Reduced noise (sometimes improves quality!)

Methods:
- PCA (Principal Component Analysis)
- Random Projection
- Matryoshka embeddings (train with multiple sizes)
```

### PCA Reduction

```python
from sklearn.decomposition import PCA
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = ["Example text " + str(i) for i in range(1000)]
embeddings = model.encode(texts)

print(f"Original shape: {embeddings.shape}")  # (1000, 384)

# Fit PCA
pca = PCA(n_components=128)
reduced = pca.fit_transform(embeddings)

print(f"Reduced shape: {reduced.shape}")  # (1000, 128)
print(f"Variance retained: {sum(pca.explained_variance_ratio_):.2%}")

# For new embeddings, use transform (not fit_transform)
new_text = "New example"
new_embedding = model.encode([new_text])
new_reduced = pca.transform(new_embedding)
```

### Quantization (Smaller Storage)

```python
"""
Quantization reduces memory by using smaller data types.
float32 -> int8 = 4x smaller!
"""

import numpy as np

def quantize_embeddings(embeddings: np.ndarray, bits: int = 8):
    """Quantize embeddings to reduce storage."""
    # Find min and max for scaling
    min_val = embeddings.min()
    max_val = embeddings.max()

    # Scale to 0-255 range (for 8-bit)
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = ((embeddings - min_val) * scale).astype(np.uint8)

    # Store metadata for dequantization
    metadata = {'min': min_val, 'max': max_val, 'scale': scale}

    return quantized, metadata

def dequantize_embeddings(quantized: np.ndarray, metadata: dict):
    """Restore embeddings from quantized form."""
    return quantized.astype(np.float32) / metadata['scale'] + metadata['min']

# Example
embeddings = np.random.rand(1000, 384).astype(np.float32)

print(f"Original size: {embeddings.nbytes / 1024:.2f} KB")

quantized, meta = quantize_embeddings(embeddings)
print(f"Quantized size: {quantized.nbytes / 1024:.2f} KB")
print(f"Compression: {embeddings.nbytes / quantized.nbytes:.1f}x")

# Verify quality
restored = dequantize_embeddings(quantized, meta)
error = np.mean(np.abs(embeddings - restored))
print(f"Mean absolute error: {error:.6f}")
```

### Binary Embeddings

```python
"""
Binary embeddings: Convert to 0/1 for ultra-fast search.
Uses Hamming distance instead of cosine similarity.
"""

import numpy as np

def binarize_embeddings(embeddings: np.ndarray):
    """Convert to binary (0/1) based on sign."""
    return (embeddings > 0).astype(np.uint8)

def hamming_distance(a: np.ndarray, b: np.ndarray):
    """Count differing bits."""
    return np.sum(a != b)

def hamming_similarity(a: np.ndarray, b: np.ndarray):
    """Higher = more similar."""
    return 1 - (hamming_distance(a, b) / len(a))

# Example
emb1 = np.array([0.5, -0.3, 0.8, -0.2, 0.1])
emb2 = np.array([0.4, -0.1, 0.7, -0.4, 0.2])
emb3 = np.array([-0.5, 0.3, -0.8, 0.2, -0.1])

bin1 = binarize_embeddings(emb1)
bin2 = binarize_embeddings(emb2)
bin3 = binarize_embeddings(emb3)

print(f"emb1 vs emb2 (similar): {hamming_similarity(bin1, bin2):.2f}")
print(f"emb1 vs emb3 (opposite): {hamming_similarity(bin1, bin3):.2f}")
```

---

## 3. Hybrid Search

Combine vector search with keyword search for best results.

```python
"""
Hybrid Search: Vector + Keyword (BM25)
Best of both worlds!
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict

class HybridSearch:
    """Combine semantic and keyword search."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.bm25 = None

    def index(self, documents: List[str]):
        """Index documents for both search methods."""
        self.documents = documents

        # Vector embeddings
        self.embeddings = self.model.encode(documents, normalize_embeddings=True)

        # BM25 keyword index
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        print(f"Indexed {len(documents)} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        Hybrid search.

        Args:
            query: Search query
            top_k: Number of results
            alpha: Balance factor (0 = keyword only, 1 = vector only)

        Returns:
            List of results with scores
        """
        # Vector search
        query_emb = self.model.encode([query], normalize_embeddings=True)
        vector_scores = np.dot(self.embeddings, query_emb.T).flatten()

        # Normalize to 0-1 range
        vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-6)

        # Keyword search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # Normalize to 0-1 range
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # Combine scores
        hybrid_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

        # Get top results
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'hybrid_score': hybrid_scores[idx],
                'vector_score': vector_scores[idx],
                'keyword_score': bm25_scores[idx],
                'index': idx
            })

        return results


# Example
search = HybridSearch()

documents = [
    "Python programming language tutorial for beginners",
    "Machine learning with Python: Complete guide",
    "JavaScript web development fundamentals",
    "Python snake habitat and behavior",
    "Data science using Python and pandas",
    "Introduction to deep learning neural networks",
]

search.index(documents)

# Test different alpha values
query = "Python programming"

print("="*70)
print(f"Query: '{query}'")
print("="*70)

for alpha in [0.0, 0.5, 1.0]:
    print(f"\nAlpha = {alpha} ({'keyword' if alpha == 0 else 'vector' if alpha == 1 else 'hybrid'}):")
    print("-" * 50)
    results = search.search(query, top_k=3, alpha=alpha)
    for r in results:
        print(f"  [{r['hybrid_score']:.4f}] {r['document']}")
```

---

## 4. Re-ranking

Improve initial search results with a second-stage model.

```python
"""
Re-ranking: Use a more powerful model to re-score top results.
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple

class ReRanker:
    """Re-rank search results for better accuracy."""

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents by relevance to query.

        Args:
            query: The search query
            documents: List of documents to re-rank
            top_k: Return only top k results (None = all)

        Returns:
            List of (document, score) tuples, sorted by score
        """
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            scored_docs = scored_docs[:top_k]

        return scored_docs


# Example: Two-stage retrieval
from sentence_transformers import SentenceTransformer
import numpy as np

# Stage 1: Fast retrieval with bi-encoder
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Python is a programming language created by Guido van Rossum",
    "Pythons are large snakes found in tropical regions",
    "Python programming is popular for data science and machine learning",
    "The Monty Python comedy group influenced Python's name",
    "Learn Python basics: variables, loops, and functions",
    "Python web frameworks include Django and Flask",
    "Python snakes can grow over 20 feet long",
    "Python has a simple syntax that's easy to learn",
]

doc_embeddings = bi_encoder.encode(documents, normalize_embeddings=True)

query = "Python programming tutorial"
query_embedding = bi_encoder.encode([query], normalize_embeddings=True)

# Get top 5 candidates quickly
similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
top_5_indices = np.argsort(similarities)[::-1][:5]
candidates = [documents[i] for i in top_5_indices]

print("Stage 1 - Bi-encoder candidates:")
for i, idx in enumerate(top_5_indices):
    print(f"  {i+1}. [{similarities[idx]:.4f}] {documents[idx][:50]}...")

# Stage 2: Re-rank with cross-encoder
reranker = ReRanker()
reranked = reranker.rerank(query, candidates)

print("\nStage 2 - Cross-encoder re-ranked:")
for i, (doc, score) in enumerate(reranked):
    print(f"  {i+1}. [{score:.4f}] {doc[:50]}...")
```

---

## 5. Query Expansion

Improve search by expanding the query.

```python
"""
Query Expansion: Add synonyms and related terms.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

class QueryExpander:
    """Expand queries for better recall."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

        # Pre-defined expansion terms (or use an LLM to generate)
        self.expansions = {
            "ML": ["machine learning", "AI", "artificial intelligence"],
            "DL": ["deep learning", "neural networks"],
            "NLP": ["natural language processing", "text analysis"],
            "python": ["Python programming", "Python language"],
        }

    def expand_query(self, query: str) -> str:
        """Add expansion terms to query."""
        expanded = query
        for term, expansions in self.expansions.items():
            if term.lower() in query.lower():
                expanded += " " + " ".join(expansions)
        return expanded

    def multi_query_search(
        self,
        query: str,
        documents: list,
        doc_embeddings: np.ndarray,
        top_k: int = 5
    ) -> list:
        """
        Search with multiple query variations.
        Combine results from original and expanded queries.
        """
        # Original query
        queries = [query]

        # Expanded query
        expanded = self.expand_query(query)
        if expanded != query:
            queries.append(expanded)

        # Encode all queries
        query_embeddings = self.model.encode(queries, normalize_embeddings=True)

        # Search with each query
        all_scores = []
        for q_emb in query_embeddings:
            scores = np.dot(doc_embeddings, q_emb)
            all_scores.append(scores)

        # Max fusion: take maximum score across queries
        combined_scores = np.max(all_scores, axis=0)

        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        return [(documents[i], combined_scores[i]) for i in top_indices]


# Example
expander = QueryExpander()

query = "ML tutorial"
expanded = expander.expand_query(query)
print(f"Original: {query}")
print(f"Expanded: {expanded}")
```

---

## 6. Embedding Caching Strategies

```python
"""
Smart caching for production systems.
"""

import hashlib
import json
import redis
import numpy as np
from typing import Optional

class EmbeddingCache:
    """Redis-based embedding cache."""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 86400):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl  # Cache TTL in seconds

    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return f"emb:{hashlib.sha256(content.encode()).hexdigest()}"

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self._get_key(text, model)
        cached = self.redis.get(key)

        if cached:
            return np.array(json.loads(cached))
        return None

    def set(self, text: str, model: str, embedding: np.ndarray):
        """Cache embedding."""
        key = self._get_key(text, model)
        self.redis.setex(key, self.ttl, json.dumps(embedding.tolist()))

    def get_or_compute(
        self,
        text: str,
        model_name: str,
        compute_fn
    ) -> np.ndarray:
        """Get from cache or compute and cache."""
        # Try cache first
        cached = self.get(text, model_name)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = compute_fn(text)

        # Cache it
        self.set(text, model_name, embedding)

        return embedding


# Usage with batching
class BatchEmbedder:
    """Batch embeddings with caching."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        # self.cache = EmbeddingCache()  # Uncomment if using Redis

    def encode_batch(self, texts: list) -> np.ndarray:
        """Encode with caching."""
        results = []
        texts_to_encode = []
        text_indices = []

        for i, text in enumerate(texts):
            # cached = self.cache.get(text, self.model_name)
            # if cached is not None:
            #     results.append(cached)
            # else:
            texts_to_encode.append(text)
            text_indices.append(i)
            results.append(None)

        # Batch encode uncached texts
        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode)
            for idx, emb, text in zip(text_indices, new_embeddings, texts_to_encode):
                results[idx] = emb
                # self.cache.set(text, self.model_name, emb)

        return np.array(results)
```

---

## 7. Production Best Practices

```python
"""
Production-ready embedding service.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

class EmbeddingService:
    """Production embedding service with batching and async support."""

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        max_batch_size: int = 32,
        max_workers: int = 4
    ):
        self.model = SentenceTransformer(model_name)
        self.max_batch_size = max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def encode_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous encoding with automatic batching."""
        all_embeddings = []

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    async def encode_async(self, texts: List[str]) -> np.ndarray:
        """Asynchronous encoding."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.encode_sync,
            texts
        )

    def health_check(self) -> Dict:
        """Check if service is healthy."""
        try:
            test_embedding = self.model.encode(["test"])
            return {
                "status": "healthy",
                "model": self.model._model_config.get('name', 'unknown'),
                "dimension": len(test_embedding[0])
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# FastAPI example
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
service = EmbeddingService()

class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
async def embed(request: EmbedRequest):
    embeddings = await service.encode_async(request.texts)
    return {"embeddings": embeddings.tolist()}

@app.get("/health")
def health():
    return service.health_check()
"""
```

---

## Summary Checklist

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED TECHNIQUES CHECKLIST                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [ ] Fine-tuning                                                        │
│      - Train on domain-specific data                                    │
│      - Use contrastive or triplet loss                                  │
│      - Mine hard negatives                                              │
│                                                                         │
│  [ ] Dimensionality Reduction                                           │
│      - PCA for moderate reduction                                       │
│      - Quantization for storage savings                                 │
│      - Binary embeddings for speed                                      │
│                                                                         │
│  [ ] Hybrid Search                                                      │
│      - Combine vector + keyword (BM25)                                  │
│      - Tune alpha parameter                                             │
│                                                                         │
│  [ ] Re-ranking                                                         │
│      - Use cross-encoder for top results                                │
│      - Two-stage retrieval pipeline                                     │
│                                                                         │
│  [ ] Query Expansion                                                    │
│      - Add synonyms and related terms                                   │
│      - Multi-query fusion                                               │
│                                                                         │
│  [ ] Production                                                         │
│      - Implement caching                                                │
│      - Batch processing                                                 │
│      - Async support                                                    │
│      - Health checks                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What's Next?

Apply everything you've learned in a complete project:
- [Real-World Project: Semantic Search Engine](./project/README.md)
