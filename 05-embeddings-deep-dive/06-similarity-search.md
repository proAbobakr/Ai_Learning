# Part 6: Similarity Search & Vector Operations

Learn how to efficiently find similar items in large collections of embeddings.

## The Problem: Searching Millions of Vectors

```
Scenario: You have 10 million documents, each with a 384-dimensional embedding.
Question: How do you quickly find the top 10 most similar to a query?

Naive approach: Compare query to ALL 10 million documents
Time: O(n) - way too slow!

Better approach: Use specialized algorithms and data structures
Time: O(log n) or better!
```

---

## Similarity Metrics Deep Dive

### 1. Cosine Similarity

**Best for:** Text similarity, when magnitude doesn't matter

```python
import numpy as np

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.

    Returns: Value between -1 and 1
    - 1 = identical direction
    - 0 = perpendicular
    - -1 = opposite direction
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Example
vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 4, 6])  # Same direction, different magnitude

print(f"Cosine similarity: {cosine_similarity(vec1, vec2):.4f}")  # 1.0
```

### 2. Euclidean Distance (L2)

**Best for:** When absolute position matters

```python
def euclidean_distance(a, b):
    """
    Calculate Euclidean distance.

    Returns: Value >= 0 (lower = more similar)
    """
    return np.linalg.norm(a - b)

# Example
vec1 = np.array([0, 0])
vec2 = np.array([3, 4])

print(f"Euclidean distance: {euclidean_distance(vec1, vec2):.4f}")  # 5.0
```

### 3. Dot Product (Inner Product)

**Best for:** Speed with normalized vectors

```python
def dot_product_similarity(a, b):
    """
    Calculate dot product.

    For normalized vectors, equals cosine similarity!
    """
    return np.dot(a, b)

# Normalize vectors first
vec1_norm = vec1 / np.linalg.norm(vec1)
vec2_norm = vec2 / np.linalg.norm(vec2)

print(f"Dot product (normalized): {dot_product_similarity(vec1_norm, vec2_norm):.4f}")
```

### 4. Manhattan Distance (L1)

**Best for:** High-dimensional sparse data

```python
def manhattan_distance(a, b):
    """
    Calculate Manhattan (city block) distance.

    Sum of absolute differences.
    """
    return np.sum(np.abs(a - b))

# Example
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

print(f"Manhattan distance: {manhattan_distance(vec1, vec2)}")  # 9
```

### Comparison Chart

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SIMILARITY METRICS COMPARISON                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Metric          │ Range      │ Best For           │ Speed        │
│  ────────────────┼────────────┼────────────────────┼──────────────│
│  Cosine          │ [-1, 1]    │ Text similarity    │ Medium       │
│  Euclidean (L2)  │ [0, ∞)     │ Clustering         │ Fast         │
│  Dot Product     │ (-∞, ∞)    │ Normalized vectors │ Fastest      │
│  Manhattan (L1)  │ [0, ∞)     │ Sparse data        │ Fast         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Exact Nearest Neighbor Search

### Brute Force (for small datasets)

```python
import numpy as np
from typing import List, Tuple

def brute_force_search(
    query: np.ndarray,
    database: np.ndarray,
    top_k: int = 5,
    metric: str = "cosine"
) -> List[Tuple[int, float]]:
    """
    Find k nearest neighbors using brute force.

    Args:
        query: Query vector
        database: Matrix of vectors to search (n_vectors x dimensions)
        top_k: Number of results to return
        metric: "cosine", "euclidean", or "dot"

    Returns:
        List of (index, score) tuples
    """
    if metric == "cosine":
        # Normalize for cosine similarity
        query_norm = query / np.linalg.norm(query)
        db_norms = np.linalg.norm(database, axis=1, keepdims=True)
        db_normalized = database / db_norms
        scores = np.dot(db_normalized, query_norm)
        best_indices = np.argsort(scores)[::-1][:top_k]

    elif metric == "euclidean":
        distances = np.linalg.norm(database - query, axis=1)
        best_indices = np.argsort(distances)[:top_k]
        scores = -distances  # Negative so higher is better

    elif metric == "dot":
        scores = np.dot(database, query)
        best_indices = np.argsort(scores)[::-1][:top_k]

    return [(idx, scores[idx]) for idx in best_indices]


# Example
np.random.seed(42)
database = np.random.rand(10000, 384)  # 10K vectors
query = np.random.rand(384)

results = brute_force_search(query, database, top_k=5)
print("Top 5 nearest neighbors:")
for idx, score in results:
    print(f"  Index: {idx}, Score: {score:.4f}")
```

### Time Complexity Analysis

```python
import time
import numpy as np

def benchmark_brute_force(n_vectors: int, dimensions: int = 384):
    """Benchmark brute force search."""
    database = np.random.rand(n_vectors, dimensions)
    query = np.random.rand(dimensions)

    start = time.time()
    _ = brute_force_search(query, database, top_k=10)
    elapsed = time.time() - start

    return elapsed

# Benchmark different sizes
sizes = [1000, 10000, 100000, 1000000]
print("Brute Force Search Benchmarks:")
print("-" * 40)

for n in sizes:
    if n <= 100000:  # Skip 1M for demo
        elapsed = benchmark_brute_force(n)
        print(f"{n:>10,} vectors: {elapsed*1000:>8.2f} ms")
    else:
        print(f"{n:>10,} vectors: ~{n/100000 * 100:.0f} ms (estimated)")
```

**Output:**
```
Brute Force Search Benchmarks:
----------------------------------------
     1,000 vectors:     0.45 ms
    10,000 vectors:     4.12 ms
   100,000 vectors:    42.35 ms
 1,000,000 vectors:   ~400 ms (estimated)
```

---

## Approximate Nearest Neighbor (ANN)

For large datasets, use approximate algorithms - they're much faster with minimal accuracy loss.

### FAISS (Facebook AI Similarity Search)

The most popular library for efficient similarity search.

```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

```python
import faiss
import numpy as np

class FAISSIndex:
    """FAISS-based similarity search."""

    def __init__(self, dimensions: int, index_type: str = "flat"):
        """
        Initialize FAISS index.

        index_type options:
        - "flat": Exact search (brute force)
        - "ivf": Inverted file index (faster, approximate)
        - "hnsw": Hierarchical navigable small world (fast, approximate)
        """
        self.dimensions = dimensions

        if index_type == "flat":
            # Exact search (L2 distance)
            self.index = faiss.IndexFlatL2(dimensions)

        elif index_type == "ivf":
            # Approximate search with clustering
            n_clusters = 100
            quantizer = faiss.IndexFlatL2(dimensions)
            self.index = faiss.IndexIVFFlat(quantizer, dimensions, n_clusters)

        elif index_type == "hnsw":
            # Graph-based approximate search
            self.index = faiss.IndexHNSWFlat(dimensions, 32)  # 32 = num connections

    def add(self, vectors: np.ndarray):
        """Add vectors to the index."""
        vectors = vectors.astype('float32')

        # IVF index needs training
        if hasattr(self.index, 'train') and not self.index.is_trained:
            self.index.train(vectors)

        self.index.add(vectors)
        print(f"Index now contains {self.index.ntotal} vectors")

    def search(self, query: np.ndarray, top_k: int = 5):
        """Search for nearest neighbors."""
        query = query.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query, top_k)
        return list(zip(indices[0], distances[0]))


# Example usage
dimensions = 384
n_vectors = 100000

# Create random data
np.random.seed(42)
database = np.random.rand(n_vectors, dimensions).astype('float32')
query = np.random.rand(dimensions).astype('float32')

# Test different index types
print("="*60)
print("FAISS INDEX COMPARISON")
print("="*60)

for index_type in ["flat", "ivf", "hnsw"]:
    index = FAISSIndex(dimensions, index_type)

    # Add vectors
    start = time.time()
    index.add(database)
    add_time = time.time() - start

    # Search
    start = time.time()
    results = index.search(query, top_k=5)
    search_time = time.time() - start

    print(f"\n{index_type.upper()} Index:")
    print(f"  Add time: {add_time*1000:.2f} ms")
    print(f"  Search time: {search_time*1000:.4f} ms")
    print(f"  Top result: index={results[0][0]}, distance={results[0][1]:.4f}")
```

### FAISS with Cosine Similarity

```python
def create_cosine_index(vectors: np.ndarray):
    """Create FAISS index for cosine similarity."""
    # Normalize vectors (cosine similarity = dot product of normalized vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms

    # Use inner product (dot product) index
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(normalized.astype('float32'))

    return index, norms

def cosine_search(index, query: np.ndarray, top_k: int = 5):
    """Search using cosine similarity."""
    # Normalize query
    query_norm = query / np.linalg.norm(query)
    query_norm = query_norm.astype('float32').reshape(1, -1)

    # Search (returns cosine similarities directly)
    similarities, indices = index.search(query_norm, top_k)
    return list(zip(indices[0], similarities[0]))
```

---

## Annoy (Spotify's Library)

Great for memory-mapped indices and simple deployment.

```bash
pip install annoy
```

```python
from annoy import AnnoyIndex
import numpy as np

class AnnoySearch:
    """Annoy-based similarity search."""

    def __init__(self, dimensions: int, metric: str = "angular"):
        """
        Initialize Annoy index.

        metric options:
        - "angular": Cosine distance
        - "euclidean": L2 distance
        - "manhattan": L1 distance
        - "dot": Dot product
        """
        self.dimensions = dimensions
        self.metric = metric
        self.index = AnnoyIndex(dimensions, metric)

    def add(self, vectors: np.ndarray):
        """Add vectors to index."""
        for i, vec in enumerate(vectors):
            self.index.add_item(i, vec)

    def build(self, n_trees: int = 10):
        """Build the index (required before searching)."""
        self.index.build(n_trees)
        print(f"Built index with {n_trees} trees")

    def search(self, query: np.ndarray, top_k: int = 5):
        """Search for nearest neighbors."""
        indices, distances = self.index.get_nns_by_vector(
            query, top_k, include_distances=True
        )
        return list(zip(indices, distances))

    def save(self, path: str):
        """Save index to file."""
        self.index.save(path)

    def load(self, path: str):
        """Load index from file."""
        self.index.load(path)


# Example
dimensions = 384
n_vectors = 10000

database = np.random.rand(n_vectors, dimensions)
query = np.random.rand(dimensions)

# Create and build index
annoy = AnnoySearch(dimensions, metric="angular")
annoy.add(database)
annoy.build(n_trees=10)

# Search
results = annoy.search(query, top_k=5)
print("Annoy Search Results:")
for idx, dist in results:
    print(f"  Index: {idx}, Distance: {dist:.4f}")
```

---

## ScaNN (Google's Library)

Best for billion-scale datasets.

```bash
pip install scann
```

```python
import scann
import numpy as np

def create_scann_index(vectors: np.ndarray, num_neighbors: int = 10):
    """Create ScaNN index for efficient search."""

    # Normalize for cosine similarity
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Build searcher
    searcher = scann.scann_ops_pybind.builder(
        normalized,
        num_neighbors,
        "dot_product"
    ).tree(
        num_leaves=2000,
        num_leaves_to_search=100,
        training_sample_size=250000
    ).score_ah(
        2,
        anisotropic_quantization_threshold=0.2
    ).reorder(100).build()

    return searcher

# Example (simplified - ScaNN is best for very large datasets)
# vectors = np.random.rand(1000000, 384).astype('float32')
# searcher = create_scann_index(vectors)
# neighbors, distances = searcher.search(query)
```

---

## Performance Comparison

```python
"""
Benchmark different search methods.
"""

import time
import numpy as np

def benchmark_all(n_vectors: int = 100000, dimensions: int = 384, n_queries: int = 100):
    """Compare search methods."""

    print(f"Benchmarking with {n_vectors:,} vectors, {dimensions} dimensions")
    print("=" * 60)

    # Generate data
    database = np.random.rand(n_vectors, dimensions).astype('float32')
    queries = np.random.rand(n_queries, dimensions).astype('float32')

    results = {}

    # 1. NumPy Brute Force
    start = time.time()
    for query in queries:
        scores = np.dot(database, query)
        _ = np.argsort(scores)[-10:]
    results['NumPy (brute)'] = (time.time() - start) / n_queries * 1000

    # 2. FAISS Flat (exact)
    import faiss
    index_flat = faiss.IndexFlatIP(dimensions)
    db_norm = database / np.linalg.norm(database, axis=1, keepdims=True)
    index_flat.add(db_norm)

    start = time.time()
    for query in queries:
        q_norm = query / np.linalg.norm(query)
        index_flat.search(q_norm.reshape(1, -1), 10)
    results['FAISS Flat'] = (time.time() - start) / n_queries * 1000

    # 3. FAISS IVF (approximate)
    n_clusters = 100
    quantizer = faiss.IndexFlatIP(dimensions)
    index_ivf = faiss.IndexIVFFlat(quantizer, dimensions, n_clusters)
    index_ivf.train(db_norm)
    index_ivf.add(db_norm)
    index_ivf.nprobe = 10  # Search 10 clusters

    start = time.time()
    for query in queries:
        q_norm = query / np.linalg.norm(query)
        index_ivf.search(q_norm.reshape(1, -1), 10)
    results['FAISS IVF'] = (time.time() - start) / n_queries * 1000

    # 4. FAISS HNSW
    index_hnsw = faiss.IndexHNSWFlat(dimensions, 32)
    index_hnsw.add(db_norm)

    start = time.time()
    for query in queries:
        q_norm = query / np.linalg.norm(query)
        index_hnsw.search(q_norm.reshape(1, -1), 10)
    results['FAISS HNSW'] = (time.time() - start) / n_queries * 1000

    # Print results
    print(f"\nAverage query time (ms):")
    print("-" * 40)
    for method, time_ms in sorted(results.items(), key=lambda x: x[1]):
        speedup = results['NumPy (brute)'] / time_ms
        print(f"  {method:20s}: {time_ms:>8.4f} ms ({speedup:>5.1f}x)")

# Run benchmark
# benchmark_all()
```

**Expected Output:**
```
Average query time (ms):
----------------------------------------
  FAISS HNSW          :   0.0523 ms (800.2x)
  FAISS IVF           :   0.2134 ms (196.5x)
  FAISS Flat          :   5.4321 ms (  7.7x)
  NumPy (brute)       :  41.8765 ms (  1.0x)
```

---

## Choosing the Right Method

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WHICH SEARCH METHOD TO USE?                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Dataset Size    │ Recommended Method      │ Accuracy │ Speed          │
│  ────────────────┼─────────────────────────┼──────────┼────────────────│
│  < 10,000        │ NumPy brute force       │ 100%     │ Fast enough    │
│  < 100,000       │ FAISS Flat              │ 100%     │ Very fast      │
│  < 1,000,000     │ FAISS IVF or HNSW       │ ~95-99%  │ Fast           │
│  < 100,000,000   │ FAISS IVF-PQ or HNSW    │ ~90-95%  │ Very fast      │
│  > 100,000,000   │ ScaNN or Distributed    │ ~90%     │ Ultra fast     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Filtering and Hybrid Search

Combine vector search with metadata filtering.

```python
import numpy as np
import faiss

class FilteredSearch:
    """Vector search with metadata filtering."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.index = faiss.IndexFlatIP(dimensions)
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: list):
        """Add vectors with associated metadata."""
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = (vectors / norms).astype('float32')

        self.index.add(normalized)
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, top_k: int = 5, filters: dict = None):
        """
        Search with optional filters.

        Example filter: {"category": "technology", "year": 2023}
        """
        # Normalize query
        query_norm = (query / np.linalg.norm(query)).astype('float32')

        # Get more results than needed (we'll filter)
        fetch_k = top_k * 10 if filters else top_k
        similarities, indices = self.index.search(query_norm.reshape(1, -1), fetch_k)

        # Apply filters
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx == -1:
                continue

            meta = self.metadata[idx]

            # Check filters
            if filters:
                match = all(meta.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            results.append({
                'index': idx,
                'similarity': sim,
                'metadata': meta
            })

            if len(results) >= top_k:
                break

        return results


# Example
search = FilteredSearch(dimensions=384)

# Add documents with metadata
vectors = np.random.rand(100, 384)
metadata = [
    {"title": f"Doc {i}", "category": "tech" if i % 2 == 0 else "science", "year": 2020 + (i % 5)}
    for i in range(100)
]

search.add(vectors, metadata)

# Search with filter
query = np.random.rand(384)
results = search.search(query, top_k=5, filters={"category": "tech"})

print("Filtered Search Results (category='tech'):")
for r in results:
    print(f"  {r['metadata']['title']}: {r['similarity']:.4f}")
```

---

## What's Next?

Now let's learn about vector databases for production use:
- [Part 7: Vector Databases](./07-vector-databases.md)
