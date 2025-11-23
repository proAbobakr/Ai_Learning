# Part 4: Popular Embedding Models

This guide covers all major embedding models, when to use each, and how to implement them.

## Model Comparison at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EMBEDDING MODELS COMPARISON                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Model                    │ Dims  │ Speed   │ Quality │ Cost    │ Best For  │
├───────────────────────────┼───────┼─────────┼─────────┼─────────┼───────────┤
│ all-MiniLM-L6-v2          │ 384   │ Fast    │ Good    │ Free    │ General   │
│ all-mpnet-base-v2         │ 768   │ Medium  │ Great   │ Free    │ Quality   │
│ OpenAI text-embedding-3-s │ 1536  │ API     │ Great   │ $0.02/M │ Production│
│ OpenAI text-embedding-3-l │ 3072  │ API     │ Best    │ $0.13/M │ Premium   │
│ Cohere embed-v3           │ 1024  │ API     │ Great   │ $0.10/M │ Enterprise│
│ BGE-large-en              │ 1024  │ Medium  │ Great   │ Free    │ RAG       │
│ E5-large-v2               │ 1024  │ Medium  │ Great   │ Free    │ Search    │
│ GTE-large                 │ 1024  │ Medium  │ Great   │ Free    │ General   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Sentence Transformers (Free & Local)

The most popular open-source embedding library.

### Installation

```bash
pip install sentence-transformers
```

### all-MiniLM-L6-v2 (Recommended Starter)

**Best for:** Fast, general-purpose embeddings

```python
from sentence_transformers import SentenceTransformer

# Load model (downloads automatically first time)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Single text
text = "Machine learning is fascinating"
embedding = model.encode(text)
print(f"Shape: {embedding.shape}")  # (384,)

# Multiple texts (batched - faster!)
texts = [
    "I love programming",
    "Coding is my passion",
    "The weather is sunny"
]
embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (3, 384)

# With options
embeddings = model.encode(
    texts,
    batch_size=32,           # Process 32 at a time
    show_progress_bar=True,  # Show progress
    normalize_embeddings=True # L2 normalize (for cosine similarity)
)
```

### all-mpnet-base-v2 (Higher Quality)

**Best for:** When quality matters more than speed

```python
model = SentenceTransformer('all-mpnet-base-v2')

texts = ["This is a test sentence", "Another test"]
embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (2, 768)
```

### Model Variants Comparison

```python
# Speed vs Quality trade-off

models = {
    # Fastest (lower quality)
    'all-MiniLM-L6-v2': {'dims': 384, 'speed': '14000 sent/sec'},

    # Balanced
    'all-MiniLM-L12-v2': {'dims': 384, 'speed': '7500 sent/sec'},

    # Higher quality
    'all-mpnet-base-v2': {'dims': 768, 'speed': '2800 sent/sec'},

    # Multilingual
    'paraphrase-multilingual-MiniLM-L12-v2': {'dims': 384, 'speed': '7500 sent/sec'},
}

# Choose based on your needs:
# - High throughput: MiniLM-L6
# - Balance: MiniLM-L12
# - Best quality: mpnet-base
# - Multiple languages: multilingual
```

---

## 2. OpenAI Embeddings (API)

Production-ready embeddings with excellent quality.

### Installation

```bash
pip install openai
```

### text-embedding-3-small

**Best for:** Production use with good cost/quality balance

```python
from openai import OpenAI
import numpy as np

client = OpenAI()  # Uses OPENAI_API_KEY environment variable

def get_openai_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI API."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)

# Single text
embedding = get_openai_embedding("Hello, world!")
print(f"Shape: {embedding.shape}")  # (1536,)

# Multiple texts
def get_openai_embeddings(texts, model="text-embedding-3-small"):
    """Get embeddings for multiple texts."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return np.array([item.embedding for item in response.data])

texts = [
    "Machine learning is great",
    "I love AI",
    "The cat sat on the mat"
]
embeddings = get_openai_embeddings(texts)
print(f"Shape: {embeddings.shape}")  # (3, 1536)
```

### text-embedding-3-large

**Best for:** Highest quality, when cost is not a concern

```python
# Same API, different model
embedding = get_openai_embedding(
    "Important document text",
    model="text-embedding-3-large"
)
print(f"Shape: {embedding.shape}")  # (3072,)
```

### Dimension Reduction (New Feature!)

OpenAI's v3 models support dimension reduction without quality loss!

```python
def get_openai_embedding_reduced(text, model="text-embedding-3-small", dimensions=256):
    """Get dimensionality-reduced embedding."""
    response = client.embeddings.create(
        input=text,
        model=model,
        dimensions=dimensions  # Reduce dimensions!
    )
    return np.array(response.data[0].embedding)

# Full dimensions
full_emb = get_openai_embedding("Hello", model="text-embedding-3-large")
print(f"Full: {full_emb.shape}")  # (3072,)

# Reduced dimensions (saves storage!)
reduced_emb = get_openai_embedding_reduced("Hello", model="text-embedding-3-large", dimensions=256)
print(f"Reduced: {reduced_emb.shape}")  # (256,)
```

### Cost Calculation

```python
def calculate_openai_cost(num_tokens, model="text-embedding-3-small"):
    """Calculate API cost."""
    prices = {
        "text-embedding-3-small": 0.00002,   # $0.02 per 1M tokens
        "text-embedding-3-large": 0.00013,   # $0.13 per 1M tokens
        "text-embedding-ada-002": 0.0001,    # Legacy: $0.10 per 1M tokens
    }
    cost = (num_tokens / 1_000_000) * prices.get(model, 0)
    return cost

# Example: 1 million tokens
print(f"Cost for 1M tokens (small): ${calculate_openai_cost(1_000_000, 'text-embedding-3-small'):.2f}")
print(f"Cost for 1M tokens (large): ${calculate_openai_cost(1_000_000, 'text-embedding-3-large'):.2f}")
```

---

## 3. Cohere Embeddings (API)

Enterprise-focused with excellent multilingual support.

### Installation

```bash
pip install cohere
```

### embed-english-v3.0

```python
import cohere
import numpy as np

co = cohere.Client('your-api-key')

def get_cohere_embedding(texts, input_type="search_document"):
    """
    Get Cohere embeddings.

    input_type options:
    - "search_document": For documents to be searched
    - "search_query": For search queries
    - "classification": For classification tasks
    - "clustering": For clustering tasks
    """
    response = co.embed(
        texts=texts if isinstance(texts, list) else [texts],
        model="embed-english-v3.0",
        input_type=input_type
    )
    return np.array(response.embeddings)

# Document embeddings
docs = ["First document", "Second document"]
doc_embeddings = get_cohere_embedding(docs, input_type="search_document")

# Query embedding (note different input_type!)
query = "search query"
query_embedding = get_cohere_embedding(query, input_type="search_query")

print(f"Doc shape: {doc_embeddings.shape}")    # (2, 1024)
print(f"Query shape: {query_embedding.shape}") # (1, 1024)
```

### Multilingual Support

```python
# embed-multilingual-v3.0 supports 100+ languages
def get_cohere_multilingual(texts):
    response = co.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )
    return np.array(response.embeddings)

multilingual_texts = [
    "Hello world",        # English
    "Bonjour le monde",   # French
    "Hola mundo",         # Spanish
    "",            # Chinese
]

embeddings = get_cohere_multilingual(multilingual_texts)
```

---

## 4. BGE Models (Free, State-of-the-Art)

Beijing Academy of AI's models - excellent for RAG systems.

### Installation

```bash
pip install sentence-transformers
# or
pip install FlagEmbedding
```

### BAAI/bge-large-en-v1.5

```python
from sentence_transformers import SentenceTransformer

# Load BGE model
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# IMPORTANT: BGE requires a special instruction prefix for queries
def get_bge_embeddings(texts, is_query=False):
    """
    Get BGE embeddings.

    For queries, add instruction prefix for better results.
    """
    if is_query:
        # Add instruction prefix for queries
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

    return model.encode(texts, normalize_embeddings=True)

# Document embeddings (no prefix)
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
]
doc_embeddings = get_bge_embeddings(documents, is_query=False)

# Query embedding (with prefix)
query = "What is machine learning?"
query_embedding = get_bge_embeddings([query], is_query=True)

print(f"Doc shape: {doc_embeddings.shape}")    # (2, 1024)
print(f"Query shape: {query_embedding.shape}") # (1, 1024)
```

### BGE-M3 (Multi-Functional)

Supports dense, sparse, and multi-vector retrieval!

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences = ["What is deep learning?", "Deep learning is a subset of ML"]

# Get all three types of embeddings
embeddings = model.encode(
    sentences,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True
)

print(f"Dense shape: {embeddings['dense_vecs'].shape}")
print(f"Sparse keys: {list(embeddings['lexical_weights'][0].keys())[:5]}")
```

---

## 5. E5 Models (Microsoft)

Excellent instruction-following embeddings.

### E5-large-v2

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-large-v2')

# E5 uses prefixes: "query: " and "passage: "
def get_e5_embeddings(texts, is_query=False):
    """Get E5 embeddings with appropriate prefix."""
    prefix = "query: " if is_query else "passage: "
    texts = [prefix + t for t in texts]
    return model.encode(texts, normalize_embeddings=True)

# Documents
docs = ["The capital of France is Paris", "Python is a programming language"]
doc_embs = get_e5_embeddings(docs, is_query=False)

# Query
query_emb = get_e5_embeddings(["What is the capital of France?"], is_query=True)
```

### Multilingual E5

```python
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Works across 100+ languages with same prefix pattern
texts = [
    "query: What is AI?",
    "passage: Artificial intelligence is...",
]
embeddings = model.encode(texts)
```

---

## 6. GTE Models (Alibaba)

General Text Embeddings - excellent all-around performance.

```python
from sentence_transformers import SentenceTransformer

# Load GTE model
model = SentenceTransformer('thenlper/gte-large')

texts = [
    "Machine learning applications",
    "Deep learning for image recognition",
]

embeddings = model.encode(texts, normalize_embeddings=True)
print(f"Shape: {embeddings.shape}")  # (2, 1024)
```

---

## 7. Instructor Models (Customizable)

Embeddings with custom instructions for any task!

```python
from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR('hkunlp/instructor-large')

# Custom instruction for your use case!
instruction = "Represent the technical document for retrieval:"

texts = [
    [instruction, "Python is a high-level programming language"],
    [instruction, "JavaScript runs in web browsers"],
]

embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (2, 768)

# Different instruction for different tasks
classification_instruction = "Represent the text for classification:"
texts_classify = [
    [classification_instruction, "This product is amazing!"],
    [classification_instruction, "Terrible experience, would not recommend"],
]
```

---

## 8. Voyage AI (Specialized)

Domain-specific embeddings for code, law, finance.

```python
import voyageai

vo = voyageai.Client()  # Uses VOYAGE_API_KEY

# Code embeddings
code_texts = [
    "def hello(): print('Hello, World!')",
    "function hello() { console.log('Hello'); }",
]
code_embeddings = vo.embed(code_texts, model="voyage-code-2")

# Legal embeddings
legal_texts = ["The defendant shall pay...", "Pursuant to section 4..."]
legal_embeddings = vo.embed(legal_texts, model="voyage-law-2")

# Finance embeddings
finance_texts = ["Q3 earnings exceeded expectations", "Revenue grew 15% YoY"]
finance_embeddings = vo.embed(finance_texts, model="voyage-finance-2")
```

---

## Complete Model Selection Guide

```python
def recommend_embedding_model(
    use_case: str,
    budget: str = "any",
    speed_priority: bool = False,
    multilingual: bool = False,
    local_only: bool = False
) -> dict:
    """
    Recommend the best embedding model for your use case.

    Args:
        use_case: "general", "rag", "search", "classification", "code", "clustering"
        budget: "free", "low", "any"
        speed_priority: If True, prioritize faster models
        multilingual: If True, need multi-language support
        local_only: If True, only suggest local models (no API)

    Returns:
        Dictionary with model recommendation and reasoning
    """

    # Free, local models
    free_models = {
        "fast_general": {
            "model": "all-MiniLM-L6-v2",
            "dims": 384,
            "reason": "Fastest open-source, good quality"
        },
        "quality_general": {
            "model": "all-mpnet-base-v2",
            "dims": 768,
            "reason": "Best quality among smaller models"
        },
        "rag_optimized": {
            "model": "BAAI/bge-large-en-v1.5",
            "dims": 1024,
            "reason": "State-of-the-art for RAG, requires query prefix"
        },
        "search_optimized": {
            "model": "intfloat/e5-large-v2",
            "dims": 1024,
            "reason": "Excellent for semantic search"
        },
        "multilingual_free": {
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "dims": 384,
            "reason": "50+ languages, good speed"
        },
    }

    # API models
    api_models = {
        "production_balanced": {
            "model": "text-embedding-3-small",
            "dims": 1536,
            "provider": "OpenAI",
            "cost": "$0.02/1M tokens",
            "reason": "Best cost/quality for production"
        },
        "production_best": {
            "model": "text-embedding-3-large",
            "dims": 3072,
            "provider": "OpenAI",
            "cost": "$0.13/1M tokens",
            "reason": "Highest quality available"
        },
        "enterprise_multilingual": {
            "model": "embed-multilingual-v3.0",
            "dims": 1024,
            "provider": "Cohere",
            "cost": "$0.10/1M tokens",
            "reason": "100+ languages, enterprise features"
        },
    }

    # Selection logic
    if local_only or budget == "free":
        if multilingual:
            return free_models["multilingual_free"]
        elif speed_priority:
            return free_models["fast_general"]
        elif use_case == "rag":
            return free_models["rag_optimized"]
        elif use_case == "search":
            return free_models["search_optimized"]
        else:
            return free_models["quality_general"]
    else:
        if multilingual:
            return api_models["enterprise_multilingual"]
        elif use_case in ["production", "high_quality"]:
            return api_models["production_best"]
        else:
            return api_models["production_balanced"]

# Examples
print(recommend_embedding_model("rag", budget="free"))
print(recommend_embedding_model("general", budget="any", multilingual=True))
print(recommend_embedding_model("search", speed_priority=True, local_only=True))
```

---

## Performance Benchmarks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MTEB BENCHMARK SCORES (Higher = Better)              │
├─────────────────────────────────────────────────────────────────────────┤
│  Model                         │ Avg Score │ Retrieval │ Classification│
├────────────────────────────────┼───────────┼───────────┼───────────────┤
│ OpenAI text-embedding-3-large  │   64.6    │   59.2    │     79.3      │
│ Cohere embed-v3                │   64.5    │   58.4    │     78.7      │
│ BGE-large-en-v1.5              │   64.2    │   58.8    │     77.8      │
│ E5-large-v2                    │   62.3    │   56.9    │     76.2      │
│ OpenAI text-embedding-3-small  │   62.3    │   56.1    │     77.1      │
│ GTE-large                      │   63.1    │   57.4    │     76.9      │
│ all-mpnet-base-v2              │   57.8    │   50.3    │     73.5      │
│ all-MiniLM-L6-v2               │   56.3    │   49.0    │     71.8      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Template

```python
"""
Embedding Model Quick Start Template
Copy and modify for your project!
"""

import numpy as np
from typing import List, Union

class EmbeddingModel:
    """Unified interface for different embedding models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", provider: str = "sentence-transformers"):
        self.model_name = model_name
        self.provider = provider
        self._load_model()

    def _load_model(self):
        if self.provider == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()

        elif self.provider == "cohere":
            import cohere
            self.client = cohere.Client()

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        if self.provider == "sentence-transformers":
            return self.model.encode(texts, **kwargs)

        elif self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name,
                **kwargs
            )
            return np.array([item.embedding for item in response.data])

        elif self.provider == "cohere":
            response = self.client.embed(
                texts=texts,
                model=self.model_name,
                input_type=kwargs.get('input_type', 'search_document')
            )
            return np.array(response.embeddings)


# Usage
model = EmbeddingModel("all-MiniLM-L6-v2", "sentence-transformers")
embeddings = model.encode(["Hello world", "How are you?"])
print(f"Shape: {embeddings.shape}")
```

## What's Next?

Let's put these models into practice:
- [Part 5: Hands-on Examples](./05-hands-on-examples.md)
