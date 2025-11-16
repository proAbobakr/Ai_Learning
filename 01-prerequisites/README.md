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

```python
# Example: Class definition (you'll create many like this)
from dataclasses import dataclass
from typing import List

@dataclass
class Document:
    """Represents a document in our RAG system"""
    content: str
    metadata: Dict[str, any]
    embedding: Optional[List[float]] = None

    def __len__(self):
        return len(self.content)

# Function with type hints
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of specified size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space

        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Usage
text = "Your long document text here..."
chunks = chunk_text(text, chunk_size=300)
print(f"Created {len(chunks)} chunks")
```

### Python Skills Checklist

- [ ] Can write and use functions with type hints
- [ ] Understand classes and object-oriented programming
- [ ] Comfortable with list/dict comprehensions
- [ ] Can handle exceptions properly
- [ ] Know how to work with external libraries
- [ ] Familiar with async/await (helpful but not required initially)

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

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    return dot_product / magnitude

# Example
vec_a = [1, 2, 3]
vec_b = [1, 2, 3.1]  # Very similar
vec_c = [10, -5, 2]  # Different

print(f"Similarity A-B: {cosine_similarity(vec_a, vec_b):.3f}")  # ~0.999
print(f"Similarity A-C: {cosine_similarity(vec_a, vec_c):.3f}")  # Much lower
```

### 2.2 NLP Concepts

1. **Tokenization**: Breaking text into pieces (tokens)
   ```python
   text = "Hello, world!"
   tokens = ["Hello", ",", "world", "!"]  # Word tokens
   # or: ["Hel", "lo", ",", "wor", "ld", "!"]  # Subword tokens
   ```

2. **Semantic Search**: Finding meaning-based matches (not just keyword matching)
   - Traditional: "How to bake bread" matches "bake" and "bread"
   - Semantic: Also matches "bread recipe" and "baking tutorial"

### ML/NLP Checklist

- [ ] Understand what embeddings are (vectors representing text)
- [ ] Know what a language model does (processes and generates text)
- [ ] Grasp similarity/distance concepts
- [ ] Familiar with tokenization basics
- [ ] Understand semantic vs. keyword search

---

## 3. Required Tools & Libraries

### 3.1 Environment Setup

```bash
# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
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

```python
# load_config.py
import os
from dotenv import load_dotenv

def load_api_keys():
    """Load API keys from .env file"""
    load_dotenv()

    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    return {
        'openai': openai_key,
        'pinecone': os.getenv('PINECONE_API_KEY'),
    }

# Usage
config = load_api_keys()
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

```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3):
    """Decorator for retrying API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def call_api(endpoint: str):
    # Your API call here
    pass
```

---

## 5. Data Structures & Algorithms

### Key Concepts for RAG

1. **Hash Tables / Dictionaries** - Fast lookups for document metadata
2. **Arrays / Lists** - Storing embeddings and chunks
3. **Trees** (basic understanding) - Some vector databases use tree structures
4. **Big O Notation** - Understanding performance implications

```python
# Example: Efficient document storage
class DocumentStore:
    def __init__(self):
        self.documents = {}  # O(1) lookup by ID
        self.index = []      # O(n) scan, but keeps order

    def add(self, doc_id: str, content: str):
        """O(1) insertion"""
        self.documents[doc_id] = content
        self.index.append(doc_id)

    def get(self, doc_id: str) -> Optional[str]:
        """O(1) retrieval"""
        return self.documents.get(doc_id)

    def search(self, query: str) -> List[str]:
        """O(n) search - will be replaced by vector search!"""
        results = []
        for doc_id in self.index:
            if query.lower() in self.documents[doc_id].lower():
                results.append(doc_id)
        return results
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
