# Part 7: Vector Databases

Vector databases are purpose-built for storing and searching embeddings at scale.

## Why Use a Vector Database?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      WHY VECTOR DATABASES?                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Traditional Database          │  Vector Database                       │
│  ─────────────────────         │  ────────────────                      │
│  - Exact match queries         │  - Similarity search                   │
│  - SQL: WHERE name = 'John'    │  - Find similar to [0.1, 0.5, ...]    │
│  - B-tree, hash indexes        │  - HNSW, IVF indexes                   │
│  - Row/column storage          │  - Vector-optimized storage            │
│                                                                         │
│  Use Cases for Vector DBs:                                              │
│  - Semantic search             - RAG (Retrieval Augmented Generation)   │
│  - Recommendations             - Duplicate detection                    │
│  - Image similarity            - Anomaly detection                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Vector Database Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE COMPARISON                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Database    │ Type      │ Best For          │ Pricing                  │
│  ────────────┼───────────┼───────────────────┼─────────────────────────│
│  ChromaDB    │ Embedded  │ Prototyping       │ Free (open source)       │
│  Pinecone    │ Managed   │ Production        │ Free tier + paid         │
│  Weaviate    │ Both      │ Hybrid search     │ Free tier + paid         │
│  Qdrant      │ Both      │ Filtering         │ Free (open source)       │
│  Milvus      │ Self-host │ Large scale       │ Free (open source)       │
│  pgvector    │ Extension │ Existing Postgres │ Free (extension)         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. ChromaDB (Best for Starting)

Simple, embedded vector database. Perfect for learning and prototypes.

### Installation

```bash
pip install chromadb
```

### Basic Usage

```python
import chromadb
from chromadb.utils import embedding_functions

# Create client (in-memory)
client = chromadb.Client()

# Or persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection with default embedding function
collection = client.create_collection(
    name="my_documents",
    metadata={"description": "My document collection"}
)

# Add documents (ChromaDB auto-generates embeddings!)
collection.add(
    documents=[
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Python is a popular programming language",
        "JavaScript runs in web browsers",
    ],
    ids=["doc1", "doc2", "doc3", "doc4"],
    metadatas=[
        {"category": "AI", "year": 2023},
        {"category": "AI", "year": 2023},
        {"category": "programming", "year": 2023},
        {"category": "programming", "year": 2023},
    ]
)

print(f"Collection has {collection.count()} documents")
```

### Search Operations

```python
# Basic search
results = collection.query(
    query_texts=["What is artificial intelligence?"],
    n_results=3
)

print("Search Results:")
for doc, distance, metadata in zip(
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0]
):
    print(f"  [{distance:.4f}] {doc}")
    print(f"           Category: {metadata['category']}")

# Search with metadata filter
results = collection.query(
    query_texts=["programming languages"],
    n_results=5,
    where={"category": "programming"}  # Filter!
)

# Search with combined filters
results = collection.query(
    query_texts=["technology"],
    n_results=5,
    where={
        "$and": [
            {"category": "AI"},
            {"year": {"$gte": 2022}}
        ]
    }
)
```

### Custom Embedding Function

```python
from sentence_transformers import SentenceTransformer

# Use custom embedding model
class CustomEmbeddingFunction:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        return self.model.encode(input).tolist()

# Create collection with custom embeddings
custom_ef = CustomEmbeddingFunction('all-mpnet-base-v2')

collection = client.create_collection(
    name="custom_embeddings",
    embedding_function=custom_ef
)
```

### Complete ChromaDB Example

```python
"""
Complete ChromaDB Example: Document Q&A System
"""

import chromadb
from sentence_transformers import SentenceTransformer

class DocumentQA:
    def __init__(self, persist_dir: str = "./qa_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: list, metadatas: list = None):
        """Add documents to the collection."""
        # Generate embeddings
        embeddings = self.model.encode(documents).tolist()

        # Generate IDs
        start_id = self.collection.count()
        ids = [f"doc_{start_id + i}" for i in range(len(documents))]

        # Default metadata
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Added {len(documents)} documents. Total: {self.collection.count()}")

    def search(self, query: str, top_k: int = 3, filters: dict = None):
        """Search for relevant documents."""
        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=filters
        )

        return [
            {
                'document': doc,
                'distance': dist,
                'metadata': meta,
                'id': id_
            }
            for doc, dist, meta, id_ in zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0],
                results['ids'][0]
            )
        ]

    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection("documents")


# Usage
qa = DocumentQA()

# Add knowledge base
documents = [
    "Python was created by Guido van Rossum in 1991.",
    "JavaScript was invented by Brendan Eich in 1995.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "The Earth is approximately 4.5 billion years old.",
    "Water boils at 100 degrees Celsius at sea level.",
]

metadatas = [
    {"category": "programming", "topic": "python"},
    {"category": "programming", "topic": "javascript"},
    {"category": "AI", "topic": "ml"},
    {"category": "AI", "topic": "dl"},
    {"category": "science", "topic": "geology"},
    {"category": "science", "topic": "physics"},
]

qa.add_documents(documents, metadatas)

# Search
print("\n" + "="*60)
print("SEARCH RESULTS")
print("="*60)

queries = [
    "Who created Python?",
    "What is deep learning?",
    "Tell me about the Earth",
]

for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 40)
    results = qa.search(query, top_k=2)
    for r in results:
        print(f"  [{r['distance']:.4f}] {r['document']}")
```

---

## 2. Pinecone (Production-Ready)

Fully managed vector database. Great for production without ops overhead.

### Setup

```bash
pip install pinecone-client
```

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="my-index",
    dimension=384,  # Must match your embedding dimension!
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Get index
index = pc.Index("my-index")
```

### Operations

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Upsert vectors
documents = [
    "Machine learning transforms data into insights",
    "Neural networks mimic brain structure",
    "Python is great for data science",
]

embeddings = model.encode(documents)

# Prepare vectors with metadata
vectors = [
    {
        "id": f"vec_{i}",
        "values": emb.tolist(),
        "metadata": {
            "text": doc,
            "category": "tech"
        }
    }
    for i, (emb, doc) in enumerate(zip(embeddings, documents))
]

# Upsert (insert/update)
index.upsert(vectors=vectors)

# Query
query = "artificial intelligence"
query_embedding = model.encode([query])[0].tolist()

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

print("Pinecone Results:")
for match in results['matches']:
    print(f"  [{match['score']:.4f}] {match['metadata']['text']}")

# Query with filter
results = index.query(
    vector=query_embedding,
    top_k=3,
    filter={"category": {"$eq": "tech"}},
    include_metadata=True
)

# Delete vectors
index.delete(ids=["vec_0", "vec_1"])

# Delete by filter
index.delete(filter={"category": {"$eq": "old"}})
```

### Namespaces (Multi-tenancy)

```python
# Different namespaces for different users/projects
index.upsert(vectors=vectors, namespace="user_123")

# Query specific namespace
results = index.query(
    vector=query_embedding,
    top_k=5,
    namespace="user_123"
)
```

---

## 3. Weaviate (Hybrid Search)

Combines vector search with keyword search.

### Setup

```bash
pip install weaviate-client
```

```python
import weaviate
from weaviate.classes.init import Auth

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="your-cluster-url",
    auth_credentials=Auth.api_key("your-api-key")
)

# Or local Docker instance
client = weaviate.connect_to_local()
```

### Schema and Data

```python
from weaviate.classes.config import Property, DataType, Configure

# Create collection (class)
client.collections.create(
    name="Document",
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="category", data_type=DataType.TEXT),
        Property(name="year", data_type=DataType.INT),
    ],
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),  # Auto-vectorize!
)

# Get collection
documents = client.collections.get("Document")

# Add data
documents.data.insert({
    "content": "Machine learning is revolutionizing technology",
    "category": "AI",
    "year": 2024
})

# Batch insert
with documents.batch.dynamic() as batch:
    for doc in your_documents:
        batch.add_object(properties=doc)
```

### Search Operations

```python
# Vector search
response = documents.query.near_text(
    query="artificial intelligence applications",
    limit=5
)

for obj in response.objects:
    print(f"  {obj.properties['content']}")

# Hybrid search (vector + keyword)
response = documents.query.hybrid(
    query="machine learning python",
    limit=5,
    alpha=0.5  # 0 = keyword only, 1 = vector only
)

# Filtered search
from weaviate.classes.query import Filter

response = documents.query.near_text(
    query="technology trends",
    limit=5,
    filters=Filter.by_property("year").greater_than(2022)
)
```

---

## 4. Qdrant (Best Filtering)

Excellent filtering capabilities and local-first approach.

### Setup

```bash
pip install qdrant-client
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# In-memory (for development)
client = QdrantClient(":memory:")

# Or local file
client = QdrantClient(path="./qdrant_db")

# Or Qdrant Cloud
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key"
)
```

### Collection and Data

```python
from qdrant_client.models import Distance, VectorParams, PointStruct

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Add vectors
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    {"text": "Python programming basics", "category": "programming", "level": 1},
    {"text": "Advanced machine learning", "category": "AI", "level": 3},
    {"text": "Web development with JavaScript", "category": "programming", "level": 2},
]

points = []
for i, doc in enumerate(documents):
    embedding = model.encode(doc["text"]).tolist()
    points.append(PointStruct(
        id=i,
        vector=embedding,
        payload=doc
    ))

client.upsert(collection_name="documents", points=points)
```

### Search with Filters

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Basic search
query_embedding = model.encode("machine learning").tolist()

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5
)

# Search with exact match filter
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="AI")
            )
        ]
    ),
    limit=5
)

# Search with range filter
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="level",
                range=Range(gte=2, lte=3)
            )
        ]
    ),
    limit=5
)

# Complex filters
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="programming"))
        ],
        should=[  # OR condition
            FieldCondition(key="level", match=MatchValue(value=1)),
            FieldCondition(key="level", match=MatchValue(value=2)),
        ]
    ),
    limit=5
)

for result in results:
    print(f"  [{result.score:.4f}] {result.payload['text']}")
```

---

## 5. pgvector (PostgreSQL Extension)

Use vectors in your existing PostgreSQL database.

### Setup

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    category VARCHAR(50),
    embedding vector(384)  -- 384 dimensions
);

-- Create index for faster search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

### Python Usage

```python
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

# Connect
conn = psycopg2.connect("postgresql://user:pass@localhost/mydb")
cur = conn.cursor()

model = SentenceTransformer('all-MiniLM-L6-v2')

# Insert
text = "Machine learning is fascinating"
embedding = model.encode(text)

cur.execute(
    "INSERT INTO documents (content, category, embedding) VALUES (%s, %s, %s)",
    (text, "AI", embedding.tolist())
)
conn.commit()

# Search
query = "artificial intelligence"
query_embedding = model.encode(query)

cur.execute("""
    SELECT content, category, 1 - (embedding <=> %s) as similarity
    FROM documents
    ORDER BY embedding <=> %s
    LIMIT 5
""", (query_embedding.tolist(), query_embedding.tolist()))

results = cur.fetchall()
for content, category, similarity in results:
    print(f"  [{similarity:.4f}] {content} ({category})")
```

---

## Best Practices

### 1. Batch Operations

```python
# Instead of this (slow):
for doc in documents:
    collection.add(doc)

# Do this (fast):
collection.add_batch(documents)
```

### 2. Index Configuration

```python
# ChromaDB - configure HNSW
collection = client.create_collection(
    name="optimized",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,  # Higher = better recall
        "hnsw:search_ef": 100,        # Higher = better recall
        "hnsw:M": 16                  # Connections per node
    }
)
```

### 3. Metadata Design

```python
# Good metadata design
metadata = {
    "source": "website",           # Filterable
    "category": "technology",      # Filterable
    "date": "2024-01-15",         # Filterable + sortable
    "author_id": 123,             # For joins
    "chunk_index": 5,             # For reconstruction
    "total_chunks": 10,           # For reconstruction
}
```

### 4. Chunking Strategy

```python
def smart_chunk(text: str, chunk_size: int = 500, overlap: int = 50):
    """Smart chunking with overlap."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            # Keep last sentence for overlap
            current_chunk = current_chunk[-1:] if overlap else []
            current_length = len(current_chunk[0]) if current_chunk else 0

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append('. '.join(current_chunk))

    return chunks
```

---

## Quick Reference

```python
"""
Vector Database Quick Reference
"""

# ChromaDB
import chromadb
client = chromadb.Client()
collection = client.create_collection("name")
collection.add(documents=docs, ids=ids, metadatas=metas)
results = collection.query(query_texts=["query"], n_results=5)

# Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key="key")
index = pc.Index("name")
index.upsert(vectors=[{"id": "1", "values": [...], "metadata": {...}}])
results = index.query(vector=[...], top_k=5)

# Qdrant
from qdrant_client import QdrantClient
client = QdrantClient(":memory:")
client.create_collection("name", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.upsert("name", points=[PointStruct(id=1, vector=[...], payload={...})])
results = client.search("name", query_vector=[...], limit=5)

# Weaviate
import weaviate
client = weaviate.connect_to_local()
collection = client.collections.get("Name")
collection.data.insert({"content": "...", "category": "..."})
results = collection.query.near_text(query="...", limit=5)
```

## What's Next?

Learn advanced embedding techniques:
- [Part 8: Advanced Techniques](./08-advanced-techniques.md)
