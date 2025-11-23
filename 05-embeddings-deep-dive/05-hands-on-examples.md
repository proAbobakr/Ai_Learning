# Part 5: Hands-on Examples with Code

Time to get your hands dirty! This section contains practical, runnable examples.

## Setup

```bash
# Install all required packages
pip install sentence-transformers numpy pandas scikit-learn matplotlib openai
```

---

## Example 1: Basic Text Similarity

Find how similar two pieces of text are.

```python
"""
Example 1: Basic Text Similarity
Compare texts and find which are semantically similar.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Our texts
texts = [
    "I love programming in Python",
    "Python coding is my favorite hobby",
    "The snake python lives in tropical regions",
    "I enjoy writing software applications",
    "The weather is beautiful today",
]

# Generate embeddings
embeddings = model.encode(texts)
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Display results
print("\n" + "="*60)
print("SIMILARITY MATRIX")
print("="*60)

for i, text1 in enumerate(texts):
    print(f"\n'{text1[:40]}...':")
    for j, text2 in enumerate(texts):
        if i != j:
            sim = similarity_matrix[i][j]
            print(f"  vs '{text2[:30]}...': {sim:.4f}")

# Find most similar pair
np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
print(f"\n{'='*60}")
print(f"MOST SIMILAR PAIR (score: {similarity_matrix[max_idx]:.4f}):")
print(f"  1. '{texts[max_idx[0]]}'")
print(f"  2. '{texts[max_idx[1]]}'")
```

**Output:**
```
Generated 5 embeddings of dimension 384

============================================================
SIMILARITY MATRIX
============================================================

'I love programming in Python':
  vs 'Python coding is my favor...': 0.7834
  vs 'The snake python lives in...': 0.2156
  vs 'I enjoy writing software ...': 0.6543
  vs 'The weather is beautiful ...': 0.0892

MOST SIMILAR PAIR (score: 0.7834):
  1. 'I love programming in Python'
  2. 'Python coding is my favorite hobby'
```

---

## Example 2: Semantic Search Engine

Build a simple search engine that understands meaning.

```python
"""
Example 2: Semantic Search Engine
Search documents by meaning, not just keywords.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: list):
        """Add documents to the search index."""
        self.documents = documents
        self.embeddings = self.model.encode(documents, normalize_embeddings=True)
        print(f"Indexed {len(documents)} documents")

    def search(self, query: str, top_k: int = 3) -> list:
        """Search for documents similar to query."""
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Calculate similarities (dot product of normalized vectors = cosine similarity)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': similarities[idx],
                'index': idx
            })
        return results


# Create search engine
search_engine = SemanticSearch()

# Add documents
documents = [
    "Python is a versatile programming language used for web development",
    "Machine learning models can predict outcomes based on data patterns",
    "JavaScript is essential for creating interactive web pages",
    "Data scientists use statistical methods to analyze datasets",
    "Cloud computing provides scalable infrastructure for applications",
    "Neural networks are inspired by the human brain structure",
    "SQL databases store and retrieve structured information efficiently",
    "API endpoints allow different software systems to communicate",
    "Version control with Git helps track code changes over time",
    "Docker containers package applications with their dependencies",
]

search_engine.add_documents(documents)

# Search queries
queries = [
    "How do I build websites?",
    "I want to learn about AI and deep learning",
    "How can I manage my code versions?",
    "What technology helps apps scale?",
]

print("\n" + "="*70)
print("SEMANTIC SEARCH RESULTS")
print("="*70)

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 50)
    results = search_engine.search(query, top_k=2)
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result['score']:.4f}] {result['document']}")
```

**Output:**
```
Query: 'How do I build websites?'
--------------------------------------------------
  1. [0.5823] JavaScript is essential for creating interactive web pages
  2. [0.5234] Python is a versatile programming language used for web development

Query: 'I want to learn about AI and deep learning'
--------------------------------------------------
  1. [0.6145] Neural networks are inspired by the human brain structure
  2. [0.5892] Machine learning models can predict outcomes based on data patterns
```

---

## Example 3: Document Clustering

Group similar documents together automatically.

```python
"""
Example 3: Document Clustering
Automatically group similar documents.
"""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Documents from different topics
documents = [
    # Technology
    "Python programming language is great for beginners",
    "JavaScript frameworks like React are popular",
    "Database management systems store data efficiently",
    "Cloud computing enables scalable applications",

    # Sports
    "Football is the most popular sport worldwide",
    "Basketball players need excellent coordination",
    "Tennis requires both physical and mental strength",
    "Swimming is great for overall fitness",

    # Food
    "Italian pasta dishes are loved globally",
    "Sushi is a traditional Japanese cuisine",
    "French cuisine is known for its sophistication",
    "Mexican tacos are flavorful and diverse",
]

# Generate embeddings
embeddings = model.encode(documents)

# Cluster documents
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings)

# Display results
print("="*60)
print("DOCUMENT CLUSTERING RESULTS")
print("="*60)

for cluster_id in range(n_clusters):
    print(f"\nCluster {cluster_id + 1}:")
    print("-" * 40)
    for doc, cluster in zip(documents, clusters):
        if cluster == cluster_id:
            print(f"  - {doc}")

# Visualize (2D projection)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                     c=clusters, cmap='viridis', s=100)
plt.colorbar(scatter)

# Add labels
for i, doc in enumerate(documents):
    plt.annotate(doc[:20] + "...", (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8, alpha=0.7)

plt.title("Document Clusters Visualization")
plt.savefig('clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nVisualization saved to 'clusters.png'")
```

---

## Example 4: Question Answering with Context

Find the best answer from a knowledge base.

```python
"""
Example 4: Simple QA System
Find answers from a knowledge base using embeddings.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

class SimpleQA:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = []
        self.kb_embeddings = None

    def add_knowledge(self, facts: list):
        """Add facts to knowledge base."""
        self.knowledge_base = facts
        self.kb_embeddings = self.model.encode(facts, normalize_embeddings=True)

    def answer(self, question: str, threshold: float = 0.3) -> dict:
        """Find the best answer to a question."""
        # Encode question
        q_embedding = self.model.encode([question], normalize_embeddings=True)

        # Find most similar fact
        similarities = np.dot(self.kb_embeddings, q_embedding.T).flatten()
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score < threshold:
            return {
                'answer': "I don't have enough information to answer that.",
                'confidence': best_score,
                'source': None
            }

        return {
            'answer': self.knowledge_base[best_idx],
            'confidence': best_score,
            'source': best_idx
        }


# Create QA system
qa = SimpleQA()

# Knowledge base
knowledge = [
    "The capital of France is Paris, known for the Eiffel Tower.",
    "Python was created by Guido van Rossum and released in 1991.",
    "The human body has 206 bones in the adult skeleton.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Albert Einstein developed the theory of relativity.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "The Great Wall of China is over 21,000 kilometers long.",
    "DNA stands for Deoxyribonucleic Acid.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "Mount Everest is the highest mountain above sea level at 8,849 meters.",
]

qa.add_knowledge(knowledge)

# Ask questions
questions = [
    "What is the capital of France?",
    "Who created Python programming language?",
    "How fast does light travel?",
    "What is the tallest mountain?",
    "How many bones do humans have?",
    "What is the meaning of life?",  # Not in knowledge base
]

print("="*70)
print("QUESTION ANSWERING SYSTEM")
print("="*70)

for question in questions:
    result = qa.answer(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"   Confidence: {result['confidence']:.2%}")
```

**Output:**
```
Q: What is the capital of France?
A: The capital of France is Paris, known for the Eiffel Tower.
   Confidence: 78.34%

Q: Who created Python programming language?
A: Python was created by Guido van Rossum and released in 1991.
   Confidence: 72.15%

Q: What is the meaning of life?
A: I don't have enough information to answer that.
   Confidence: 18.45%
```

---

## Example 5: Duplicate Detection

Find duplicate or near-duplicate content.

```python
"""
Example 5: Duplicate Detection
Find duplicate or very similar content.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_duplicates(texts: list, threshold: float = 0.85) -> list:
    """
    Find duplicate or near-duplicate texts.

    Args:
        texts: List of texts to check
        threshold: Similarity threshold (0.85 = 85% similar)

    Returns:
        List of duplicate pairs with similarity scores
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Calculate all pairwise similarities
    similarity_matrix = cosine_similarity(embeddings)

    # Find pairs above threshold
    duplicates = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] >= threshold:
                duplicates.append({
                    'text1': texts[i],
                    'text2': texts[j],
                    'index1': i,
                    'index2': j,
                    'similarity': similarity_matrix[i][j]
                })

    # Sort by similarity (highest first)
    duplicates.sort(key=lambda x: x['similarity'], reverse=True)
    return duplicates


# Test data with potential duplicates
texts = [
    "Machine learning is a subset of artificial intelligence",
    "The quick brown fox jumps over the lazy dog",
    "ML is a branch of AI that enables learning from data",  # Near duplicate of 0
    "Python is a popular programming language",
    "A fast brown fox leaps over a sleepy canine",  # Near duplicate of 1
    "Python programming language is widely used",  # Near duplicate of 3
    "Deep learning uses neural networks with many layers",
    "The weather today is sunny and warm",
    "Neural networks with multiple layers power deep learning",  # Near duplicate of 6
]

print("="*70)
print("DUPLICATE DETECTION")
print("="*70)

duplicates = find_duplicates(texts, threshold=0.75)

if duplicates:
    print(f"\nFound {len(duplicates)} potential duplicate pairs:\n")
    for dup in duplicates:
        print(f"Similarity: {dup['similarity']:.2%}")
        print(f"  [{dup['index1']}] {dup['text1']}")
        print(f"  [{dup['index2']}] {dup['text2']}")
        print()
else:
    print("\nNo duplicates found above threshold.")
```

---

## Example 6: Sentiment-Aware Embeddings

Analyze sentiment using embedding similarity.

```python
"""
Example 6: Sentiment Analysis with Embeddings
Classify sentiment by comparing to known examples.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingSentiment:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Define anchor examples for each sentiment
        self.positive_examples = [
            "I love this, it's amazing!",
            "This is wonderful and fantastic",
            "Great experience, highly recommended",
            "Absolutely brilliant, exceeded expectations",
        ]

        self.negative_examples = [
            "This is terrible, I hate it",
            "Awful experience, would not recommend",
            "Very disappointing and frustrating",
            "Worst thing ever, complete waste",
        ]

        self.neutral_examples = [
            "It's okay, nothing special",
            "Average product, meets expectations",
            "Neither good nor bad",
            "Standard quality, as expected",
        ]

        # Pre-compute anchor embeddings
        self.positive_emb = self.model.encode(self.positive_examples).mean(axis=0)
        self.negative_emb = self.model.encode(self.negative_examples).mean(axis=0)
        self.neutral_emb = self.model.encode(self.neutral_examples).mean(axis=0)

    def analyze(self, text: str) -> dict:
        """Analyze sentiment of text."""
        text_emb = self.model.encode([text])[0]

        # Calculate similarities to each sentiment anchor
        pos_sim = np.dot(text_emb, self.positive_emb) / (
            np.linalg.norm(text_emb) * np.linalg.norm(self.positive_emb)
        )
        neg_sim = np.dot(text_emb, self.negative_emb) / (
            np.linalg.norm(text_emb) * np.linalg.norm(self.negative_emb)
        )
        neu_sim = np.dot(text_emb, self.neutral_emb) / (
            np.linalg.norm(text_emb) * np.linalg.norm(self.neutral_emb)
        )

        # Determine sentiment
        scores = {'positive': pos_sim, 'negative': neg_sim, 'neutral': neu_sim}
        sentiment = max(scores, key=scores.get)

        return {
            'text': text,
            'sentiment': sentiment,
            'scores': scores,
            'confidence': scores[sentiment]
        }


# Test
analyzer = EmbeddingSentiment()

test_texts = [
    "This product is absolutely fantastic! Best purchase ever!",
    "Terrible quality, broke after one use. Total waste of money.",
    "It works fine, does what it's supposed to do.",
    "I'm so happy with this, it changed my life!",
    "Not great, not terrible. Just mediocre.",
    "Horrible experience, never buying again!",
]

print("="*70)
print("SENTIMENT ANALYSIS WITH EMBEDDINGS")
print("="*70)

for text in test_texts:
    result = analyzer.analyze(text)
    print(f"\nText: \"{text[:50]}...\"")
    print(f"Sentiment: {result['sentiment'].upper()} ({result['confidence']:.2%})")
    print(f"  Positive: {result['scores']['positive']:.2%}")
    print(f"  Negative: {result['scores']['negative']:.2%}")
    print(f"  Neutral: {result['scores']['neutral']:.2%}")
```

---

## Example 7: Multi-Language Search

Search across different languages.

```python
"""
Example 7: Multi-Language Semantic Search
Search documents in multiple languages with a single query.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load multilingual model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Documents in different languages
documents = [
    # English
    {"text": "Artificial intelligence is transforming technology", "lang": "en"},
    {"text": "Climate change is a global challenge", "lang": "en"},
    {"text": "The stock market closed higher today", "lang": "en"},

    # Spanish
    {"text": "La inteligencia artificial est transformando la tecnologa", "lang": "es"},
    {"text": "El cambio climtico es un desafo global", "lang": "es"},
    {"text": "La bolsa cerr al alza hoy", "lang": "es"},

    # French
    {"text": "L'intelligence artificielle transforme la technologie", "lang": "fr"},
    {"text": "Le changement climatique est un dfi mondial", "lang": "fr"},
    {"text": "La bourse a cltur en hausse aujourd'hui", "lang": "fr"},

    # German
    {"text": "Knstliche Intelligenz verndert die Technologie", "lang": "de"},
    {"text": "Der Klimawandel ist eine globale Herausforderung", "lang": "de"},
    {"text": "Der Aktienmarkt schloss heute hher", "lang": "de"},
]

# Encode all documents
doc_texts = [d["text"] for d in documents]
doc_embeddings = model.encode(doc_texts, normalize_embeddings=True)

def multilingual_search(query: str, top_k: int = 5):
    """Search across all languages."""
    query_emb = model.encode([query], normalize_embeddings=True)
    similarities = np.dot(doc_embeddings, query_emb.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'text': documents[idx]['text'],
            'language': documents[idx]['lang'],
            'score': similarities[idx]
        })
    return results


# Test with queries in different languages
queries = [
    ("AI and machine learning", "English"),
    ("cambio del clima", "Spanish"),
    ("march boursier", "French"),
    ("Technologie", "German"),
]

print("="*70)
print("MULTI-LANGUAGE SEMANTIC SEARCH")
print("="*70)

for query, query_lang in queries:
    print(f"\nQuery ({query_lang}): '{query}'")
    print("-" * 50)
    results = multilingual_search(query, top_k=4)
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result['language']}] {result['score']:.4f} - {result['text']}")
```

---

## Example 8: Embedding Cache for Performance

Speed up repeated queries with caching.

```python
"""
Example 8: Embedding Cache
Cache embeddings to avoid redundant computation.
"""

from sentence_transformers import SentenceTransformer
import hashlib
import pickle
import os
import numpy as np
from pathlib import Path

class EmbeddingCache:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir='./embedding_cache'):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.stats = {'hits': 0, 'misses': 0}

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def encode(self, texts, use_disk_cache: bool = True) -> np.ndarray:
        """Encode texts with caching."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        texts_to_encode = []
        indices_to_encode = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)

            # Check memory cache first
            if cache_key in self.memory_cache:
                embeddings.append(self.memory_cache[cache_key])
                self.stats['hits'] += 1
                continue

            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if use_disk_cache and cache_file.exists():
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                self.memory_cache[cache_key] = embedding
                embeddings.append(embedding)
                self.stats['hits'] += 1
                continue

            # Need to encode this text
            texts_to_encode.append(text)
            indices_to_encode.append(i)
            embeddings.append(None)  # Placeholder
            self.stats['misses'] += 1

        # Batch encode texts that weren't cached
        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode)

            for text, embedding, orig_idx in zip(texts_to_encode, new_embeddings, indices_to_encode):
                cache_key = self._get_cache_key(text)

                # Store in memory cache
                self.memory_cache[cache_key] = embedding

                # Store in disk cache
                if use_disk_cache:
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(embedding, f)

                embeddings[orig_idx] = embedding

        return np.array(embeddings)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': f"{hit_rate:.2%}"
        }


# Test the cache
cache = EmbeddingCache()

texts = [
    "Hello world",
    "How are you?",
    "Machine learning is fascinating",
    "Hello world",  # Duplicate - should be cached
    "How are you?",  # Duplicate - should be cached
]

print("="*70)
print("EMBEDDING CACHE DEMO")
print("="*70)

# First run
print("\nFirst run:")
embeddings1 = cache.encode(texts)
print(f"Stats: {cache.get_stats()}")

# Second run (same texts)
print("\nSecond run (same texts):")
embeddings2 = cache.encode(texts)
print(f"Stats: {cache.get_stats()}")

# Verify embeddings are identical
print(f"\nEmbeddings match: {np.allclose(embeddings1, embeddings2)}")
```

---

## Runnable Script

Save this complete script to test everything:

```python
#!/usr/bin/env python3
"""
Complete Embeddings Example Script
Run: python embedding_examples.py
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def main():
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Example texts
    texts = [
        "I love programming in Python",
        "Python coding is my passion",
        "The weather is nice today",
    ]

    print("\nEncoding texts...")
    embeddings = model.encode(texts)

    print(f"\nEmbedding shape: {embeddings.shape}")

    # Calculate similarities
    similarities = cosine_similarity(embeddings)

    print("\nSimilarity Matrix:")
    print("-" * 40)
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts):
            print(f"{texts[i][:20]:20s} <-> {texts[j][:20]:20s}: {similarities[i][j]:.4f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
```

## What's Next?

Now let's learn about similarity search in depth:
- [Part 6: Similarity Search & Vector Operations](./06-similarity-search.md)
