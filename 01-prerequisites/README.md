# Module 1: Prerequisites

## Overview

Before diving into RAG systems, you need a solid foundation in several areas. This module covers all the prerequisite knowledge and tools you'll need.

## Time Required: 1 Week (10-15 hours)

---

## 1. Python Programming (Required Level: Intermediate)

### Essential Python Concepts

#### 1.1 Core Python
- **Variables & Data Types**: strings, lists, dictionaries, sets
- **Functions**: defining, parameters, return values, lambda functions
- **Classes & Objects**: OOP basics, inheritance, methods
- **Error Handling**: try/except blocks
- **File I/O**: reading/writing files
- **List Comprehensions**: `[x for x in items if condition]`

#### 1.2 Important Libraries
```python
# Standard library imports you'll use frequently
import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
```

### Quick Python Refresher

#### Example 1: Class Definition with Data Classes

```python
# Example: Class definition (you'll create many like this)
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class Document:
    """Represents a document in our RAG system"""
    content: str
    metadata: Dict[str, any]
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __len__(self):
        return len(self.content)

    def __repr__(self):
        return f"Document(content_length={len(self)}, has_embedding={self.embedding is not None})"

    def get_preview(self, length: int = 100) -> str:
        """Get a preview of the document content"""
        if len(self.content) <= length:
            return self.content
        return self.content[:length] + "..."

# Usage example
doc = Document(
    content="This is a sample document about RAG systems.",
    metadata={"source": "tutorial.pdf", "page": 1}
)
print(doc.get_preview(20))  # "This is a sample do..."
print(f"Document created at: {doc.created_at}")
```

#### Example 2: Text Chunking with Overlap

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks with overlap for better context preservation.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space

        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))

            # Create overlap by keeping last few words
            overlap_words = []
            overlap_size = 0
            for w in reversed(current_chunk):
                if overlap_size + len(w) < overlap:
                    overlap_words.insert(0, w)
                    overlap_size += len(w) + 1
                else:
                    break

            current_chunk = overlap_words
            current_size = overlap_size

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Usage
text = """Retrieval-Augmented Generation (RAG) is a technique that enhances
large language models by retrieving relevant information from external
knowledge bases. This allows the model to provide more accurate and
up-to-date responses without requiring retraining."""

chunks = chunk_text(text, chunk_size=100, overlap=20)
print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(chunk)
```

#### Example 3: Working with Context Managers

```python
from typing import TextIO
import json

class DocumentProcessor:
    """Context manager for processing documents"""

    def __init__(self, filename: str):
        self.filename = filename
        self.file: Optional[TextIO] = None
        self.processed_count = 0

    def __enter__(self):
        self.file = open(self.filename, 'r', encoding='utf-8')
        print(f"Opened {self.filename}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            print(f"Processed {self.processed_count} documents")
        return False

    def process_line(self, line: str) -> Dict:
        """Process a single line/document"""
        self.processed_count += 1
        return {
            'content': line.strip(),
            'length': len(line),
            'word_count': len(line.split())
        }

# Usage
with DocumentProcessor('documents.txt') as processor:
    for line in processor.file:
        doc_data = processor.process_line(line)
        # Do something with doc_data
```

#### Example 4: List and Dictionary Comprehensions for RAG

```python
# Filter and transform documents
documents = [
    {"id": 1, "content": "Python programming", "tokens": 150},
    {"id": 2, "content": "RAG systems", "tokens": 89},
    {"id": 3, "content": "Machine learning basics", "tokens": 250},
    {"id": 4, "content": "Vector databases", "tokens": 120},
]

# Filter documents by token count
short_docs = [doc for doc in documents if doc['tokens'] < 100]
print(f"Short documents: {len(short_docs)}")

# Create a lookup dictionary
doc_lookup = {doc['id']: doc for doc in documents}
print(f"Document 2: {doc_lookup[2]['content']}")

# Extract specific fields
contents = [doc['content'] for doc in documents]
print(f"All contents: {contents}")

# Nested comprehension - split into words
all_words = [word for doc in documents for word in doc['content'].split()]
print(f"Total words across documents: {len(all_words)}")

# Dictionary comprehension with transformation
uppercase_lookup = {
    doc['id']: doc['content'].upper()
    for doc in documents
    if doc['tokens'] > 100
}
```

#### Example 5: Error Handling in RAG Systems

```python
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGError(Exception):
    """Base exception for RAG system errors"""
    pass

class EmbeddingError(RAGError):
    """Raised when embedding generation fails"""
    pass

class RetrievalError(RAGError):
    """Raised when document retrieval fails"""
    pass

def safe_embed_text(text: str, model_name: str = "default") -> Optional[List[float]]:
    """
    Safely generate embeddings with comprehensive error handling.

    Returns None if embedding fails instead of raising exception.
    """
    try:
        if not text or not text.strip():
            raise ValueError("Empty text provided")

        if len(text) > 8000:
            logger.warning(f"Text too long ({len(text)} chars), truncating")
            text = text[:8000]

        # Simulate embedding generation
        # In reality, this would call an embedding model
        logger.info(f"Generating embedding for {len(text)} characters")
        embedding = [0.1] * 768  # Placeholder

        return embedding

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return None
    except EmbeddingError as e:
        logger.error(f"Embedding failed: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return None
    finally:
        logger.debug("Embedding attempt completed")

# Usage
result = safe_embed_text("Sample text for embedding")
if result:
    print(f"Successfully generated embedding with {len(result)} dimensions")
else:
    print("Failed to generate embedding")
```

#### Example 6: Async/Await for Concurrent API Calls

```python
import asyncio
import aiohttp
from typing import List, Dict
import time

class AsyncRAGClient:
    """
    Asynchronous RAG client for concurrent API calls.
    Much faster than sequential calls when processing many documents!
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"

    async def generate_embedding_async(
        self,
        session: aiohttp.ClientSession,
        text: str
    ) -> Dict:
        """Generate embedding asynchronously"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "text-embedding-3-small",
            "input": text
        }

        try:
            async with session.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                return {
                    "text": text,
                    "embedding": result["data"][0]["embedding"],
                    "success": True
                }
        except Exception as e:
            return {
                "text": text,
                "embedding": None,
                "success": False,
                "error": str(e)
            }

    async def batch_embed_async(self, texts: List[str]) -> List[Dict]:
        """
        Generate embeddings for multiple texts concurrently.
        Much faster than sequential processing!
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.generate_embedding_async(session, text)
                for text in texts
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results

def compare_sync_vs_async():
    """
    Demonstrates the speed difference between
    synchronous and asynchronous processing.
    """
    texts = [f"Document {i} about RAG systems" for i in range(10)]

    # Synchronous version (slow - one at a time)
    def sync_process(texts):
        results = []
        for text in texts:
            time.sleep(0.5)  # Simulates API call
            results.append(f"Processed: {text}")
        return results

    # Asynchronous version (fast - all at once)
    async def async_process(texts):
        async def process_one(text):
            await asyncio.sleep(0.5)  # Simulates API call
            return f"Processed: {text}"

        tasks = [process_one(text) for text in texts]
        return await asyncio.gather(*tasks)

    # Time synchronous
    start = time.time()
    sync_results = sync_process(texts)
    sync_time = time.time() - start

    # Time asynchronous
    start = time.time()
    async_results = asyncio.run(async_process(texts))
    async_time = time.time() - start

    print(f"Synchronous: {sync_time:.2f}s")  # ~5 seconds
    print(f"Asynchronous: {async_time:.2f}s")  # ~0.5 seconds
    print(f"Speedup: {sync_time / async_time:.1f}x")

# Run comparison
compare_sync_vs_async()
```

#### Example 7: Async Context Managers for Resource Management

```python
import asyncio
import aiofiles
from typing import List

class AsyncDocumentProcessor:
    """
    Process documents asynchronously with proper resource management.
    Demonstrates async context managers.
    """

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_file(self, filepath: str) -> Dict:
        """Process a single file asynchronously"""
        async with self.semaphore:  # Limit concurrency
            try:
                # Async file reading
                async with aiofiles.open(filepath, 'r') as f:
                    content = await f.read()

                # Simulate processing
                await asyncio.sleep(0.1)

                return {
                    "filepath": filepath,
                    "content": content,
                    "word_count": len(content.split()),
                    "success": True
                }

            except Exception as e:
                return {
                    "filepath": filepath,
                    "success": False,
                    "error": str(e)
                }

    async def process_batch(self, filepaths: List[str]) -> List[Dict]:
        """Process multiple files concurrently with rate limiting"""
        tasks = [self.process_file(fp) for fp in filepaths]
        results = await asyncio.gather(*tasks)
        return results

# Usage
async def main():
    processor = AsyncDocumentProcessor(max_concurrent=5)
    files = ["doc1.txt", "doc2.txt", "doc3.txt"]

    results = await processor.process_batch(files)

    for result in results:
        if result["success"]:
            print(f"✓ {result['filepath']}: {result['word_count']} words")
        else:
            print(f"✗ {result['filepath']}: {result['error']}")

# Run
# asyncio.run(main())
```

### Python Skills Checklist

- [ ] Can write and use functions with type hints
- [ ] Understand classes and object-oriented programming
- [ ] Comfortable with list/dict comprehensions
- [ ] Can handle exceptions properly
- [ ] Know how to work with external libraries
- [ ] Familiar with async/await (helpful for concurrent processing)
- [ ] Understand decorators and context managers
- [ ] Can use dataclasses for structured data

---

## 2. Machine Learning & NLP Basics

### 2.1 Essential ML Concepts

#### What You Need to Know

1. **Embeddings / Vector Representations**
   - Text converted to numerical vectors (arrays of numbers)
   - Similar meanings = similar vectors
   - Example: "king" - "man" + "woman" ≈ "queen"

```python
# Conceptual example of embeddings
sentence1 = "I love pizza"
sentence2 = "I enjoy pizza"
sentence3 = "The weather is cold"

# After embedding (simplified):
# embedding1 = [0.8, 0.3, 0.1, ...]  # 768 dimensions typically
# embedding2 = [0.75, 0.32, 0.09, ...]  # Very similar to embedding1
# embedding3 = [0.1, -0.5, 0.8, ...]  # Different from 1 and 2
```

2. **Transformers & Language Models**
   - Neural networks that process text
   - Examples: GPT-4, Claude, BERT, LLaMA
   - Pre-trained on massive text corpora
   - Understand context and generate human-like text

3. **Similarity Measures**
   - **Cosine Similarity**: Measures angle between vectors
   - Used to find similar documents
   - Range: -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)

```python
import numpy as np
from typing import List, Tuple

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if magnitude == 0:
        return 0.0

    return dot_product / magnitude

# Example 1: Basic similarity
vec_a = [1, 2, 3]
vec_b = [1, 2, 3.1]  # Very similar
vec_c = [10, -5, 2]  # Different

print(f"Similarity A-B: {cosine_similarity(vec_a, vec_b):.3f}")  # ~0.999
print(f"Similarity A-C: {cosine_similarity(vec_a, vec_c):.3f}")  # Much lower

# Example 2: Simulating document embeddings
# In reality, these would come from an embedding model
documents = {
    "doc1": "Python is a programming language",
    "doc2": "Python is great for coding",
    "doc3": "I love eating pizza",
    "doc4": "Pizza is my favorite food"
}

# Simulated embeddings (normally from sentence-transformers or OpenAI)
embeddings = {
    "doc1": [0.8, 0.6, 0.1, 0.2],   # Programming-related
    "doc2": [0.75, 0.65, 0.15, 0.1], # Also programming-related
    "doc3": [0.1, 0.1, 0.9, 0.8],   # Food-related
    "doc4": [0.05, 0.15, 0.85, 0.9]  # Also food-related
}

def find_most_similar(query_doc: str, k: int = 2) -> List[Tuple[str, float]]:
    """Find k most similar documents to query"""
    query_embedding = embeddings[query_doc]
    similarities = []

    for doc_id, embedding in embeddings.items():
        if doc_id != query_doc:
            sim = cosine_similarity(query_embedding, embedding)
            similarities.append((doc_id, sim))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

# Find documents similar to doc1
print("\nDocuments similar to 'Python is a programming language':")
for doc_id, similarity in find_most_similar("doc1"):
    print(f"  {doc_id}: {documents[doc_id]} (similarity: {similarity:.3f})")

# Output:
#   doc2: Python is great for coding (similarity: 0.992)
#   doc3: I love eating pizza (similarity: 0.234)
```

#### Advanced Similarity Example

```python
import numpy as np
from typing import List, Dict

class VectorStore:
    """Simple in-memory vector store for understanding RAG retrieval"""

    def __init__(self):
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}

    def add(self, doc_id: str, vector: List[float], metadata: Dict = None):
        """Add a document vector to the store"""
        self.vectors[doc_id] = np.array(vector)
        self.metadata[doc_id] = metadata or {}

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict]:
        """
        Search for most similar vectors using cosine similarity.

        Returns list of dicts with doc_id, similarity score, and metadata.
        """
        query_vec = np.array(query_vector)
        results = []

        for doc_id, doc_vector in self.vectors.items():
            # Calculate cosine similarity
            similarity = np.dot(query_vec, doc_vector) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vector)
            )

            results.append({
                'doc_id': doc_id,
                'similarity': float(similarity),
                'metadata': self.metadata[doc_id]
            })

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:top_k]

# Example usage
store = VectorStore()

# Add documents (with simulated embeddings)
store.add('doc1', [0.8, 0.2, 0.1], {'title': 'Python Basics', 'author': 'Jane'})
store.add('doc2', [0.75, 0.25, 0.15], {'title': 'Python Advanced', 'author': 'John'})
store.add('doc3', [0.1, 0.8, 0.7], {'title': 'Cooking Guide', 'author': 'Chef'})

# Search
query = [0.78, 0.22, 0.12]  # Query about Python
results = store.search(query, top_k=2)

print("Search results:")
for result in results:
    print(f"  {result['doc_id']}: {result['metadata']['title']}")
    print(f"    Similarity: {result['similarity']:.3f}")
```

### 2.2 NLP Concepts

#### 1. Tokenization: Breaking text into pieces (tokens)

```python
# Simple word tokenization
text = "Hello, world! How are you?"
simple_tokens = text.split()  # Basic splitting
print(simple_tokens)  # ['Hello,', 'world!', 'How', 'are', 'you?']

# Better tokenization (handling punctuation)
import re

def tokenize(text: str) -> List[str]:
    """Simple tokenizer that handles punctuation"""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

tokens = tokenize(text)
print(tokens)  # ['Hello', ',', 'world', '!', 'How', 'are', 'you', '?']

# Using a real tokenizer (like those used in transformers)
# This is conceptual - actual implementation uses libraries
def subword_tokenize(text: str) -> List[str]:
    """
    Simulates subword tokenization (like BPE or WordPiece).
    Real models use this to handle rare words better.
    """
    # Example: "unhappiness" might become ["un", "happi", "ness"]
    # This allows the model to understand the meaning from parts
    return text.split()  # Simplified for demonstration

# Token counts matter for API limits!
def count_tokens(text: str) -> int:
    """
    Estimate token count (important for API rate limits).
    Rule of thumb: ~4 characters = 1 token for English
    """
    return len(text) // 4

text = "This is a sample document for RAG systems"
estimated_tokens = count_tokens(text)
print(f"Text: {text}")
print(f"Estimated tokens: {estimated_tokens}")
```

#### 2. Semantic Search: Finding meaning-based matches

```python
from typing import List, Tuple

class SemanticSearchDemo:
    """Demonstrates difference between keyword and semantic search"""

    def __init__(self):
        # Sample documents
        self.documents = {
            "doc1": "How to bake bread at home",
            "doc2": "Bread recipe for beginners",
            "doc3": "Best baking tutorial for homemade bread",
            "doc4": "Python programming guide",
            "doc5": "Make delicious sourdough at home"
        }

    def keyword_search(self, query: str) -> List[str]:
        """Traditional keyword-based search"""
        query_words = set(query.lower().split())
        results = []

        for doc_id, text in self.documents.items():
            text_words = set(text.lower().split())
            # Find documents with overlapping words
            if query_words & text_words:  # Set intersection
                results.append(doc_id)

        return results

    def semantic_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Semantic search using embeddings (simulated).
        In reality, this would use real embedding models.
        """
        # Simulated embeddings based on semantic meaning
        embeddings = {
            "doc1": [0.9, 0.8, 0.1],  # Bread/baking related
            "doc2": [0.85, 0.9, 0.05], # Bread/recipe related
            "doc3": [0.88, 0.85, 0.08], # Baking/tutorial related
            "doc4": [0.1, 0.1, 0.95],  # Programming related
            "doc5": [0.87, 0.82, 0.06]  # Bread/homemade related
        }

        # Query about bread baking
        query_embedding = [0.9, 0.85, 0.05]

        # Calculate similarities
        results = []
        for doc_id, doc_emb in embeddings.items():
            # Simplified similarity calculation
            similarity = sum(q * d for q, d in zip(query_embedding, doc_emb))
            results.append((doc_id, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# Example usage
search_demo = SemanticSearchDemo()

query = "bread recipe"

print("Keyword Search Results:")
keyword_results = search_demo.keyword_search(query)
print(f"  Found {len(keyword_results)} documents: {keyword_results}")

print("\nSemantic Search Results:")
semantic_results = search_demo.semantic_search(query)
for doc_id, score in semantic_results[:3]:
    print(f"  {doc_id}: {search_demo.documents[doc_id]} (score: {score:.2f})")

# Output demonstrates that semantic search finds more relevant results
# even without exact keyword matches!
```

#### 3. Real-World Tokenization Example

```python
def prepare_text_for_rag(text: str, max_length: int = 512) -> Dict:
    """
    Prepare text for RAG system ingestion.
    Demonstrates tokenization, truncation, and metadata extraction.
    """
    # Basic cleaning
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    # Tokenize
    tokens = text.split()

    # Truncate if necessary (important for API limits)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        truncated = True
    else:
        truncated = False

    # Reconstruct text
    processed_text = ' '.join(tokens)

    return {
        'text': processed_text,
        'token_count': len(tokens),
        'char_count': len(processed_text),
        'truncated': truncated,
        'original_length': len(text.split())
    }

# Example
long_text = """
Retrieval-Augmented Generation (RAG) is a powerful technique that
combines the strengths of retrieval systems and generative language
models. By retrieving relevant context before generation, RAG systems
can provide more accurate, up-to-date, and factual responses.
""" * 50  # Make it very long

result = prepare_text_for_rag(long_text, max_length=100)
print(f"Original tokens: {result['original_length']}")
print(f"After processing: {result['token_count']}")
print(f"Truncated: {result['truncated']}")
```

### ML/NLP Checklist

- [ ] Understand what embeddings are (vectors representing text)
- [ ] Know what a language model does (processes and generates text)
- [ ] Grasp similarity/distance concepts
- [ ] Familiar with tokenization basics
- [ ] Understand semantic vs. keyword search

---

## 3. Required Tools & Libraries

### 3.1 Environment Setup

#### Step-by-Step Setup Guide

```bash
# Step 1: Check Python version (should be 3.8+)
python --version  # or python3 --version

# If Python is not installed or version is too old:
# - Visit https://www.python.org/downloads/
# - Download Python 3.10 or later
# - During installation, check "Add Python to PATH"

# Step 2: Create project directory
mkdir rag_project
cd rag_project

# Step 3: Create virtual environment
python -m venv rag_env

# Step 4: Activate virtual environment
# On macOS/Linux:
source rag_env/bin/activate

# On Windows (Command Prompt):
rag_env\Scripts\activate.bat

# On Windows (PowerShell):
rag_env\Scripts\Activate.ps1

# If PowerShell gives execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Step 5: Verify activation (you should see (rag_env) in your prompt)
which python  # Should point to rag_env/bin/python
# On Windows: where python

# Step 6: Upgrade pip and essential tools
pip install --upgrade pip setuptools wheel
```

#### Troubleshooting Common Setup Issues

```bash
# Issue 1: "python: command not found"
# Solution: Try python3 instead
python3 --version
alias python=python3  # Add to ~/.bashrc or ~/.zshrc

# Issue 2: "pip: command not found"
# Solution: Use python -m pip
python -m pip install --upgrade pip

# Issue 3: "Permission denied" errors
# Solution: Never use sudo with pip! Use virtual environment instead
# WRONG: sudo pip install package
# RIGHT: pip install package (inside virtual environment)

# Issue 4: Virtual environment not activating
# Solution: Make sure you're in the right directory
pwd  # Check current directory
ls rag_env  # Should show bin/ (or Scripts/ on Windows)

# Issue 5: "externally-managed-environment" error (Python 3.11+ on some Linux)
# Solution: Always use virtual environments (which you should anyway!)
```

#### Managing Multiple Python Versions

```bash
# Install pyenv for managing Python versions
# On macOS:
brew install pyenv

# On Linux:
curl https://pyenv.run | bash

# Install specific Python version
pyenv install 3.10.12
pyenv local 3.10.12  # Use this version in current directory

# Create virtual environment with specific version
python -m venv rag_env --python=python3.10
```

### 3.2 Core Libraries

Create `requirements.txt`:

```text
# LLM & RAG Frameworks
langchain==0.1.0
langchain-community==0.0.10
langchain-openai==0.0.2

# Vector Databases
chromadb==0.4.22
pinecone-client==3.0.0
faiss-cpu==1.7.4

# Embeddings
sentence-transformers==2.3.1
openai==1.10.0

# Document Processing
pypdf2==3.0.1
python-docx==1.1.0
beautifulsoup4==4.12.3
markdown==3.5.2

# Utilities
python-dotenv==1.0.0
numpy==1.24.3
pandas==2.0.3
tqdm==4.66.1

# Web Framework (for demos)
fastapi==0.109.0
uvicorn==0.27.0
streamlit==1.30.0

# Testing
pytest==7.4.4
```

Install:
```bash
pip install -r requirements.txt
```

#### Understanding Library Categories

**1. LLM & RAG Frameworks**
- `langchain`: Main orchestration framework for RAG
- `langchain-community`: Community integrations
- `langchain-openai`: OpenAI specific integrations

**2. Vector Databases**
```python
# ChromaDB - Easy to use, great for beginners
from chromadb import Client
client = Client()  # In-memory database

# FAISS - Facebook's similarity search, very fast
import faiss
index = faiss.IndexFlatL2(768)  # 768-dimensional vectors

# Pinecone - Managed cloud vector database
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
```

**3. Embedding Models**
- `sentence-transformers`: Local embedding models (free!)
- `openai`: OpenAI's embedding API (paid, very good quality)

**4. Document Processing**
- `pypdf2`: Extract text from PDFs
- `python-docx`: Handle Word documents
- `beautifulsoup4`: Parse HTML
- `markdown`: Process Markdown files

#### Installation Troubleshooting

```bash
# Issue 1: "ERROR: Could not build wheels for X"
# Solution: Install build dependencies
# On Ubuntu/Debian:
sudo apt-get install python3-dev build-essential

# On macOS:
xcode-select --install

# On Windows:
# Install Visual Studio Build Tools from:
# https://visualstudio.microsoft.com/downloads/

# Issue 2: "SSL Certificate Error"
# Solution: Upgrade pip and try again
pip install --upgrade pip certifi
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Issue 3: Specific package fails (e.g., faiss-cpu)
# Solution: Install problematic packages separately
pip install faiss-cpu
# If that fails, try conda:
conda install -c pytorch faiss-cpu

# Issue 4: Version conflicts
# Solution: Use compatible versions
pip install langchain==0.1.0 --no-deps
pip install -r requirements.txt

# Issue 5: Out of memory during installation
# Solution: Install packages one at a time
pip install langchain
pip install chromadb
# ... etc
```

#### Verifying Installation

Create `test_installation.py`:

```python
"""Test script to verify all dependencies are installed correctly"""

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import langchain
        print("✓ LangChain installed")
    except ImportError as e:
        print(f"✗ LangChain not installed: {e}")

    try:
        import chromadb
        print("✓ ChromaDB installed")
    except ImportError as e:
        print(f"✗ ChromaDB not installed: {e}")

    try:
        import numpy
        print("✓ NumPy installed")
    except ImportError as e:
        print(f"✗ NumPy not installed: {e}")

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence Transformers installed")
    except ImportError as e:
        print(f"✗ Sentence Transformers not installed: {e}")

    try:
        import openai
        print("✓ OpenAI installed")
    except ImportError as e:
        print(f"✗ OpenAI not installed: {e}")

def test_basic_functionality():
    """Test basic functionality of key libraries"""
    print("\nTesting basic functionality:")

    # Test numpy
    import numpy as np
    vec = np.array([1, 2, 3])
    print(f"✓ NumPy array creation works: {vec}")

    # Test sentence transformers (if installed)
    try:
        from sentence_transformers import SentenceTransformer
        # This will download a small model on first run
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode("Hello world")
        print(f"✓ Sentence Transformers encoding works: {len(embedding)} dimensions")
    except Exception as e:
        print(f"✗ Sentence Transformers test failed: {e}")

    # Test ChromaDB
    try:
        import chromadb
        client = chromadb.Client()
        print("✓ ChromaDB client creation works")
    except Exception as e:
        print(f"✗ ChromaDB test failed: {e}")

if __name__ == "__main__":
    print("=== Testing RAG Environment Setup ===\n")
    test_imports()
    test_basic_functionality()
    print("\n=== Testing Complete ===")
```

Run the test:
```bash
python test_installation.py
```

### 3.3 API Keys Required

Create `.env` file (NEVER commit this to git):

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key
HUGGINGFACE_TOKEN=hf_your-token  # Optional
```

Get your keys:
- **OpenAI**: https://platform.openai.com/api-keys
- **Pinecone**: https://www.pinecone.io/ (free tier available)
- **Hugging Face**: https://huggingface.co/settings/tokens (optional)

### 3.4 Loading Environment Variables

#### Basic Configuration Loading

```python
# config.py
import os
from dotenv import load_dotenv
from typing import Dict, Optional

def load_api_keys() -> Dict[str, Optional[str]]:
    """Load API keys from .env file"""
    load_dotenv()

    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    return {
        'openai': openai_key,
        'pinecone': os.getenv('PINECONE_API_KEY'),
        'huggingface': os.getenv('HUGGINGFACE_TOKEN'),
    }

# Usage
config = load_api_keys()
```

#### Advanced Configuration with Validation

```python
# advanced_config.py
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration class for RAG application"""
    openai_api_key: str
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-3.5-turbo"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    temperature: float = 0.7

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        logger.info(f"Configuration loaded: model={self.llm_model}")

    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables"""
        load_dotenv()

        return cls(
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            pinecone_environment=os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            llm_model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            chunk_size=int(os.getenv('CHUNK_SIZE', '500')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '50')),
            top_k_results=int(os.getenv('TOP_K_RESULTS', '3')),
            temperature=float(os.getenv('TEMPERATURE', '0.7'))
        )

# Usage
try:
    config = RAGConfig.from_env()
    print(f"Loaded config: {config.llm_model}")
except ValueError as e:
    print(f"Configuration error: {e}")
    exit(1)
```

#### Environment File Template

Create a `.env.example` file to share with others (this is safe to commit):

```bash
# .env.example - Template for environment variables
# Copy this to .env and fill in your actual keys

# OpenAI Configuration (Required)
OPENAI_API_KEY=sk-your-key-here
OPENAI_ORG_ID=org-your-org-id  # Optional

# Vector Database Configuration
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=rag-index

# Hugging Face (Optional - for local models)
HUGGINGFACE_TOKEN=hf_your-token

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7

# RAG Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=3

# Application Settings
LOG_LEVEL=INFO
DEBUG=false
```

Then in your `.gitignore`:
```bash
# .gitignore
.env
*.env
!.env.example
```

---

## 4. Understanding APIs

### 4.1 REST API Basics

RAG systems frequently interact with APIs:

```python
import requests

# Example API call (conceptual)
def call_llm_api(prompt: str, api_key: str) -> str:
    """Call an LLM API with a prompt"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.7
    }

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=data
    )

    return response.json()['choices'][0]['message']['content']
```

### 4.2 Rate Limiting & Error Handling

#### Basic Retry Logic with Exponential Backoff

```python
import time
from functools import wraps
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, base_delay=1):
    """
    Decorator for retrying API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be doubled each retry)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        raise

                    wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, base_delay=2)
def call_embedding_api(text: str, api_key: str) -> List[float]:
    """Call OpenAI embedding API with retry logic"""
    import openai
    client = openai.OpenAI(api_key=api_key)

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
```

#### Advanced API Client with Rate Limiting

```python
import time
from collections import deque
from threading import Lock
from typing import Optional
import requests

class RateLimitedAPIClient:
    """
    API client with built-in rate limiting.
    Prevents exceeding API rate limits by tracking request timestamps.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times = deque()
        self.lock = Lock()

    def _wait_if_needed(self):
        """Wait if we're at the rate limit"""
        with self.lock:
            now = time.time()

            # Remove requests older than 1 minute
            while self.request_times and self.request_times[0] < now - 60:
                self.request_times.popleft()

            # If at limit, wait until oldest request is >1 minute old
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached. Waiting {sleep_time:.1f}s")
                    time.sleep(sleep_time)

            # Record this request
            self.request_times.append(time.time())

    def make_request(self, url: str, **kwargs) -> requests.Response:
        """Make an API request with rate limiting"""
        self._wait_if_needed()
        return requests.post(url, **kwargs)

# Usage example
client = RateLimitedAPIClient(requests_per_minute=50)

for i in range(100):
    response = client.make_request(
        "https://api.example.com/endpoint",
        json={"data": f"request {i}"}
    )
    print(f"Request {i} completed")
```

#### Comprehensive Error Handling for RAG APIs

```python
import openai
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class RAGAPIError(Exception):
    """Base exception for RAG API errors"""
    pass

class EmbeddingAPIError(RAGAPIError):
    """Embedding generation failed"""
    pass

class LLMAPIError(RAGAPIError):
    """LLM generation failed"""
    pass

class RAGAPIClient:
    """
    Robust API client for RAG operations with comprehensive error handling.
    """

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def generate_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> Optional[List[float]]:
        """
        Generate embedding with comprehensive error handling.

        Returns None on failure instead of raising exception.
        """
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided for embedding")
                return None

            # OpenAI has a limit of ~8000 tokens
            if len(text) > 30000:  # Rough character estimate
                logger.warning(f"Text too long ({len(text)} chars), truncating")
                text = text[:30000]

            response = self.client.embeddings.create(
                model=model,
                input=text
            )

            return response.data[0].embedding

        except openai.APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise EmbeddingAPIError(f"Failed to connect to API: {e}")

        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise EmbeddingAPIError("Rate limit exceeded, please slow down requests")

        except openai.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise EmbeddingAPIError("Invalid API key")

        except openai.BadRequestError as e:
            logger.error(f"Bad request: {e}")
            return None

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise EmbeddingAPIError(f"Unexpected error: {e}")

    def generate_response(
        self,
        prompt: str,
        context: str = "",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Generate LLM response with error handling.
        """
        try:
            messages = []

            if context:
                messages.append({
                    "role": "system",
                    "content": f"Answer based on this context:\n{context}"
                })

            messages.append({
                "role": "user",
                "content": prompt
            })

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=500
            )

            return response.choices[0].message.content

        except openai.APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            return None

        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            # You might want to wait and retry here
            time.sleep(20)
            return self.generate_response(prompt, context, model, temperature)

        except openai.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise LLMAPIError("Invalid API key")

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return None

# Usage example
try:
    client = RAGAPIClient(api_key="your-api-key")

    # Generate embedding
    embedding = client.generate_embedding("Sample text")
    if embedding:
        print(f"Generated embedding with {len(embedding)} dimensions")

    # Generate response
    response = client.generate_response(
        prompt="What is RAG?",
        context="RAG stands for Retrieval-Augmented Generation..."
    )
    if response:
        print(f"Response: {response}")

except RAGAPIError as e:
    print(f"API Error: {e}")
```

#### Batch Processing with Rate Limiting

```python
from typing import List, Callable
from tqdm import tqdm  # Progress bar

def batch_process_with_rate_limit(
    items: List[Any],
    process_func: Callable,
    batch_size: int = 10,
    delay_between_batches: float = 1.0
) -> List[Any]:
    """
    Process items in batches with rate limiting.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        batch_size: Number of items per batch
        delay_between_batches: Seconds to wait between batches

    Returns:
        List of processed results
    """
    results = []

    for i in tqdm(range(0, len(items), batch_size), desc="Processing batches"):
        batch = items[i:i + batch_size]

        # Process batch
        for item in batch:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process item {item}: {e}")
                results.append(None)

        # Rate limiting delay
        if i + batch_size < len(items):
            time.sleep(delay_between_batches)

    return results

# Example usage
documents = ["doc1", "doc2", "doc3", ...] # 100 documents

def embed_document(doc: str) -> List[float]:
    client = RAGAPIClient(api_key="your-key")
    return client.generate_embedding(doc)

# Process in batches of 10, waiting 1 second between batches
embeddings = batch_process_with_rate_limit(
    items=documents,
    process_func=embed_document,
    batch_size=10,
    delay_between_batches=1.0
)

print(f"Processed {len([e for e in embeddings if e])} documents successfully")
```

---

## 5. Data Structures & Algorithms

### Key Concepts for RAG

1. **Hash Tables / Dictionaries** - Fast lookups for document metadata
2. **Arrays / Lists** - Storing embeddings and chunks
3. **Trees** (basic understanding) - Some vector databases use tree structures
4. **Big O Notation** - Understanding performance implications

#### Example 1: Efficient Document Storage

```python
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class DocumentMetadata:
    """Metadata for a document"""
    doc_id: str
    created_at: datetime = field(default_factory=datetime.now)
    source: str = ""
    tags: Set[str] = field(default_factory=set)
    token_count: int = 0

class DocumentStore:
    """
    Efficient document storage using hash maps for O(1) lookups.
    Essential for RAG systems that need fast document retrieval.
    """

    def __init__(self):
        self.documents: Dict[str, str] = {}  # O(1) lookup by ID
        self.metadata: Dict[str, DocumentMetadata] = {}  # O(1) metadata lookup
        self.index: List[str] = []  # Ordered list of doc IDs
        self.tag_index: Dict[str, Set[str]] = {}  # Tag -> set of doc IDs

    def add(self, doc_id: str, content: str, tags: Set[str] = None):
        """
        Add a document to the store.
        Time complexity: O(1) average case for hash operations
        """
        self.documents[doc_id] = content
        self.index.append(doc_id)

        # Create metadata
        metadata = DocumentMetadata(
            doc_id=doc_id,
            source="upload",
            tags=tags or set(),
            token_count=len(content.split())
        )
        self.metadata[doc_id] = metadata

        # Update tag index
        if tags:
            for tag in tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(doc_id)

    def get(self, doc_id: str) -> Optional[str]:
        """O(1) retrieval"""
        return self.documents.get(doc_id)

    def get_by_tag(self, tag: str) -> List[str]:
        """
        Retrieve all documents with a specific tag.
        Time complexity: O(1) to get set, O(k) to convert to list
        where k is number of documents with that tag
        """
        doc_ids = self.tag_index.get(tag, set())
        return [self.documents[doc_id] for doc_id in doc_ids]

    def search(self, query: str) -> List[str]:
        """
        O(n) keyword search - will be replaced by vector search!
        This demonstrates why we need vector databases for RAG.
        """
        results = []
        query_lower = query.lower()

        for doc_id in self.index:
            if query_lower in self.documents[doc_id].lower():
                results.append(doc_id)

        return results

    def get_statistics(self) -> Dict:
        """Get store statistics - demonstrates dict comprehension"""
        return {
            'total_documents': len(self.documents),
            'total_tokens': sum(m.token_count for m in self.metadata.values()),
            'average_tokens': sum(m.token_count for m in self.metadata.values()) / len(self.metadata) if self.metadata else 0,
            'tags': list(self.tag_index.keys())
        }

# Usage example
store = DocumentStore()
store.add("doc1", "Python is great for RAG systems", tags={"python", "rag"})
store.add("doc2", "Vector databases enable semantic search", tags={"database", "rag"})
store.add("doc3", "OpenAI provides embedding APIs", tags={"openai", "api"})

print(f"Statistics: {store.get_statistics()}")
print(f"RAG docs: {store.get_by_tag('rag')}")
```

#### Example 2: Priority Queue for Document Ranking

```python
import heapq
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ScoredDocument:
    """Document with relevance score"""
    doc_id: str
    content: str
    score: float

    def __lt__(self, other):
        """Enable comparison for heap operations"""
        # Note: We negate score because heapq is a min-heap
        # and we want highest scores first
        return self.score > other.score

class DocumentRanker:
    """
    Uses a priority queue (heap) to efficiently maintain top-k documents.
    Time complexity: O(log k) for insertion into top-k
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.top_k = []  # Min heap of size k

    def add_document(self, doc_id: str, content: str, score: float):
        """
        Add document and maintain top-k.
        More efficient than sorting full list for large datasets.
        """
        doc = ScoredDocument(doc_id, content, score)

        if len(self.top_k) < self.k:
            # Haven't reached k documents yet, just add
            heapq.heappush(self.top_k, doc)
        elif score > self.top_k[0].score:
            # This score is better than worst in top-k
            heapq.heapreplace(self.top_k, doc)

    def get_top_documents(self) -> List[ScoredDocument]:
        """Get top-k documents sorted by score (highest first)"""
        return sorted(self.top_k, reverse=True)

# Example: Ranking search results
ranker = DocumentRanker(k=3)

# Simulate scoring documents (in real RAG, this would be similarity scores)
search_results = [
    ("doc1", "Python RAG tutorial", 0.95),
    ("doc2", "Machine learning basics", 0.60),
    ("doc3", "Vector database guide", 0.88),
    ("doc4", "RAG architecture", 0.92),
    ("doc5", "Cooking recipes", 0.20),
]

for doc_id, content, score in search_results:
    ranker.add_document(doc_id, content, score)

print("Top 3 documents:")
for doc in ranker.get_top_documents():
    print(f"  {doc.doc_id}: {doc.content} (score: {doc.score})")
```

#### Example 3: Trie for Efficient Prefix Search

```python
from typing import Dict, List, Optional

class TrieNode:
    """Node in a trie (prefix tree)"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word = False
        self.doc_ids: List[str] = []  # Documents containing this term

class DocumentTrie:
    """
    Trie for efficient prefix-based search in documents.
    Useful for autocomplete and keyword matching in RAG systems.

    Time complexity:
    - Insert: O(m) where m is length of word
    - Search: O(m) where m is length of prefix
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, doc_id: str):
        """Insert a word and associate it with a document"""
        node = self.root
        word = word.lower()

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True
        if doc_id not in node.doc_ids:
            node.doc_ids.append(doc_id)

    def search_prefix(self, prefix: str) -> List[str]:
        """Find all words starting with prefix"""
        node = self.root
        prefix = prefix.lower()

        # Navigate to the prefix node
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]

        # Collect all words from this node
        return self._collect_words(node, prefix)

    def _collect_words(self, node: TrieNode, prefix: str) -> List[str]:
        """Recursively collect all words from a node"""
        words = []

        if node.is_end_of_word:
            words.append(prefix)

        for char, child_node in node.children.items():
            words.extend(self._collect_words(child_node, prefix + char))

        return words

# Example usage
trie = DocumentTrie()

# Index documents
documents = {
    "doc1": "retrieval augmented generation",
    "doc2": "vector database systems",
    "doc3": "generative AI models"
}

for doc_id, content in documents.items():
    for word in content.split():
        trie.insert(word, doc_id)

# Search for words starting with "gen"
results = trie.search_prefix("gen")
print(f"Words starting with 'gen': {results}")
# Output: ['generation', 'generative']
```

#### Example 4: Cache with LRU Eviction

```python
from collections import OrderedDict
from typing import Optional, Any

class LRUCache:
    """
    Least Recently Used (LRU) cache for storing embeddings or API responses.
    Useful in RAG to avoid re-computing embeddings for same text.

    Time complexity: O(1) for both get and put operations
    """

    def __init__(self, capacity: int):
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        """Get value and mark as recently used"""
        if key not in self.cache:
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        """Put value in cache, evict LRU if necessary"""
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)

            self.cache[key] = value

    def __len__(self):
        return len(self.cache)

# Example: Caching embeddings
embedding_cache = LRUCache(capacity=100)

def get_embedding_with_cache(text: str) -> List[float]:
    """Get embedding with caching to avoid redundant API calls"""
    # Check cache first
    cached = embedding_cache.get(text)
    if cached is not None:
        print(f"Cache hit for: {text[:30]}...")
        return cached

    # Not in cache, generate new embedding
    print(f"Cache miss for: {text[:30]}...")
    embedding = [0.1] * 768  # Simulated API call

    # Store in cache
    embedding_cache.put(text, embedding)

    return embedding

# Test cache
text1 = "What is RAG?"
text2 = "How does vector search work?"

get_embedding_with_cache(text1)  # Cache miss
get_embedding_with_cache(text2)  # Cache miss
get_embedding_with_cache(text1)  # Cache hit!
```

---

## 6. Version Control (Git)

### Essential Git Commands

```bash
# Initialize repo
git init

# Basic workflow
git add .
git commit -m "Add RAG implementation"
git push origin main

# Branching
git checkout -b feature/improve-chunking
git merge feature/improve-chunking

# View history
git log --oneline
```

---

## 7. Optional but Helpful

### 7.1 Docker Basics
Useful for deploying RAG applications:

```dockerfile
# Simple Dockerfile for RAG app
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "app.py"]
```

### 7.2 Cloud Platforms
- AWS (SageMaker, Lambda, S3)
- Google Cloud (Vertex AI, Cloud Run)
- Azure (Azure AI, App Service)

---

## 8. Knowledge Check

Before proceeding to Module 2, ensure you can:

### Python
- [ ] Write functions with type hints
- [ ] Create classes with methods
- [ ] Use list/dict comprehensions
- [ ] Handle errors with try/except
- [ ] Import and use external libraries

### ML/NLP
- [ ] Explain what embeddings are
- [ ] Understand cosine similarity concept
- [ ] Know what an LLM is and does
- [ ] Grasp the difference between keyword and semantic search

### Tools
- [ ] Set up a Python virtual environment
- [ ] Install packages with pip
- [ ] Load environment variables
- [ ] Make basic API calls

### Practical Test

Try this exercise:

```python
# Exercise: Build a simple document store
# Complete the missing parts

from typing import List, Dict
import numpy as np

class SimpleDocStore:
    def __init__(self):
        self.docs: Dict[str, str] = {}

    def add_document(self, doc_id: str, text: str):
        """Add a document to the store"""
        # TODO: Implement
        pass

    def simple_search(self, query: str) -> List[str]:
        """Find documents containing the query (case-insensitive)"""
        # TODO: Implement
        pass

    def get_document(self, doc_id: str) -> str:
        """Retrieve a document by ID"""
        # TODO: Implement
        pass

# Test your implementation
store = SimpleDocStore()
store.add_document("doc1", "Python is great for AI")
store.add_document("doc2", "Machine learning uses data")
store.add_document("doc3", "Python programming is fun")

results = store.simple_search("python")
print(f"Found {len(results)} documents")  # Should be 2
```

**Solution**: See `solutions/prereq_exercise.py`

---

## 9. Resources

### Learn Python
- **Official Tutorial**: https://docs.python.org/3/tutorial/
- **Real Python**: https://realpython.com/
- **Python for Data Science**: DataCamp, Coursera

### Learn ML/NLP
- **Fast.ai**: Practical Deep Learning course
- **Hugging Face Course**: https://huggingface.co/learn/nlp-course
- **Stanford CS224N**: NLP with Deep Learning

### Practice
- **LeetCode**: Algorithm practice
- **Kaggle**: ML competitions and notebooks

---

## Next Steps

Once you're comfortable with these prerequisites, proceed to:

**[Module 2: RAG Fundamentals →](../02-fundamentals/01-what-is-rag.md)**

In Module 2, you'll learn:
- What RAG is and why it matters
- RAG architecture and components
- How RAG differs from fine-tuning
- Real-world use cases
