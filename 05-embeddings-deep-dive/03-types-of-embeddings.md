# Part 3: Types of Embeddings

Embeddings come in different flavors, each designed for different use cases. Let's explore them all!

## Overview: The Three Levels

```
┌─────────────────────────────────────────────────────────────────┐
│                      TYPES OF EMBEDDINGS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   WORD EMBEDDINGS        SENTENCE EMBEDDINGS    DOCUMENT EMB    │
│   ───────────────        ─────────────────────  ────────────    │
│                                                                 │
│   "king" → [...]         "I love coding" → [...] "Full doc"    │
│   "queen" → [...]        "Programming is          → [...]       │
│   "happy" → [...]         fun" → [...]                          │
│                                                                 │
│   One word = One vector  One sentence = One vec  One doc = vec  │
│                                                                 │
│   Models:                Models:                 Models:        │
│   - Word2Vec             - Sentence-BERT         - Doc2Vec      │
│   - GloVe                - OpenAI embeddings     - Longformer   │
│   - FastText             - all-MiniLM            - OpenAI       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Word Embeddings

### What Are They?

Word embeddings convert **individual words** into vectors.

```python
# Each word gets its own vector
word_embeddings = {
    "cat":    [0.2, 0.5, -0.1, ...],
    "dog":    [0.3, 0.4, -0.2, ...],
    "python": [0.1, -0.3, 0.8, ...],  # The language
    "python": [0.4, 0.6, -0.1, ...],  # The snake (PROBLEM!)
}
```

### The History: Word2Vec Revolution (2013)

Google's Word2Vec changed everything by showing words could be represented as meaningful vectors.

**Two Training Methods:**

```
┌────────────────────────────────────────────────────────────┐
│                       WORD2VEC                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   CBOW (Continuous Bag of Words)                           │
│   ─────────────────────────────                            │
│   Context words → Predict center word                      │
│                                                            │
│   "The [cat] sat on the mat"                               │
│   Input: [The, sat, on, the, mat]                          │
│   Output: Predict "cat"                                    │
│                                                            │
│   Skip-gram                                                │
│   ─────────                                                │
│   Center word → Predict context words                      │
│                                                            │
│   Input: "cat"                                             │
│   Output: Predict [The, sat, on, the, mat]                 │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Word2Vec Example

```python
from gensim.models import Word2Vec

# Training data: list of sentences (tokenized)
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    ["cats", "and", "dogs", "are", "pets"],
    ["the", "cat", "chased", "the", "dog"],
]

# Train Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=100,  # Embedding dimension
    window=5,         # Context window size
    min_count=1,      # Minimum word frequency
    workers=4         # CPU cores
)

# Get word vector
cat_vector = model.wv['cat']
print(f"'cat' embedding shape: {cat_vector.shape}")  # (100,)

# Find similar words
similar = model.wv.most_similar('cat', topn=3)
print(f"Words similar to 'cat': {similar}")
# Output: [('dog', 0.89), ('pets', 0.75), ('chased', 0.62)]
```

### GloVe: Global Vectors

GloVe (2014) learns embeddings from word co-occurrence statistics.

```python
# Using pre-trained GloVe embeddings
import numpy as np

def load_glove(file_path):
    """Load GloVe embeddings from file."""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load pre-trained GloVe (download from nlp.stanford.edu/projects/glove/)
# glove = load_glove('glove.6B.100d.txt')
# king_vec = glove['king']
```

### FastText: Handling Unknown Words

FastText (2016) breaks words into subword pieces, handling typos and rare words!

```python
from gensim.models import FastText

# FastText can handle misspellings!
model = FastText(sentences, vector_size=100, window=5, min_count=1)

# Even works for words not in training data
# Because it uses character n-grams
unknown_word = "caat"  # Typo for "cat"
vector = model.wv[unknown_word]  # Still works!
```

### Limitations of Word Embeddings

```
PROBLEM 1: No context awareness
─────────────────────────────────
"I sat by the bank"      → bank = [financial institution vector]
"The river bank was muddy" → bank = [same vector!] WRONG!

PROBLEM 2: One word = One vector
───────────────────────────────
Can't capture phrase meanings:
"not good" should be negative, but:
not = [negative vector]
good = [positive vector]
Average might = neutral? WRONG!

PROBLEM 3: Out of vocabulary (OOV)
─────────────────────────────────
Word not in training data = No embedding
(Except FastText which handles this)
```

---

## 2. Sentence Embeddings

### What Are They?

Sentence embeddings convert **entire sentences** into single vectors, capturing the full meaning.

```python
# Whole sentence = One vector
sentence_embeddings = {
    "I love programming":  [0.8, 0.2, 0.5, ...],
    "Coding is my passion": [0.75, 0.25, 0.48, ...],  # Similar!
    "The weather is nice":  [0.1, 0.9, -0.3, ...],    # Different!
}
```

### Why Better Than Word Embeddings?

```
Word Embeddings (Average):
──────────────────────────
"not good" = avg([not], [good])
           = something neutral ❌

Sentence Embeddings:
───────────────────
"not good" = [negative sentiment vector] ✓
The model understands negation!
```

### Sentence-BERT (SBERT)

The breakthrough model for sentence embeddings (2019).

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "I love machine learning",
    "AI and ML are fascinating",
    "The pizza was delicious",
]

embeddings = model.encode(sentences)

print(f"Shape: {embeddings.shape}")  # (3, 384)

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(embeddings)
print("Similarity Matrix:")
print(sim_matrix)
# [[1.0,  0.72, 0.15],   # ML ↔ AI: 0.72 (similar)
#  [0.72, 1.0,  0.12],   # ML ↔ pizza: 0.15 (different)
#  [0.15, 0.12, 1.0]]    # AI ↔ pizza: 0.12 (different)
```

### Popular Sentence Embedding Models

| Model | Dimensions | Speed | Quality | Best For |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | General use |
| all-mpnet-base-v2 | 768 | Medium | Excellent | High quality |
| paraphrase-multilingual | 768 | Medium | Good | Multi-language |
| OpenAI text-embedding-3-small | 1536 | API | Excellent | Production |
| OpenAI text-embedding-3-large | 3072 | API | Best | Highest quality |

### How Sentence Embeddings Work

```
┌─────────────────────────────────────────────────────────────────┐
│                 SENTENCE EMBEDDING PROCESS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: "I love programming"                                    │
│           ↓                                                     │
│  ┌───────────────────────────────────┐                         │
│  │         TOKENIZATION              │                         │
│  │  ["I", "love", "program", "ming"] │                         │
│  └───────────────────────────────────┘                         │
│           ↓                                                     │
│  ┌───────────────────────────────────┐                         │
│  │      TRANSFORMER ENCODER          │                         │
│  │   (BERT, RoBERTa, etc.)          │                         │
│  │                                   │                         │
│  │   Processes all tokens together   │                         │
│  │   with attention mechanism        │                         │
│  └───────────────────────────────────┘                         │
│           ↓                                                     │
│  ┌───────────────────────────────────┐                         │
│  │         POOLING                   │                         │
│  │   Combine token embeddings into   │                         │
│  │   single sentence embedding       │                         │
│  │                                   │                         │
│  │   Methods:                        │                         │
│  │   - [CLS] token                   │                         │
│  │   - Mean pooling                  │                         │
│  │   - Max pooling                   │                         │
│  └───────────────────────────────────┘                         │
│           ↓                                                     │
│  Output: [0.234, -0.156, 0.789, ...]  (384 dimensions)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pooling Strategies Explained

```python
import torch
from transformers import AutoTokenizer, AutoModel

def get_sentence_embedding(text, model, tokenizer, pooling='mean'):
    """
    Get sentence embedding with different pooling strategies.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
    hidden_states = outputs.last_hidden_state

    if pooling == 'cls':
        # Use [CLS] token (first token)
        embedding = hidden_states[:, 0, :]
    elif pooling == 'mean':
        # Average all tokens (excluding padding)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
    elif pooling == 'max':
        # Max pooling
        embedding = hidden_states.max(dim=1)[0]

    return embedding.numpy()

# Example usage
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

text = "Machine learning is fascinating"
emb = get_sentence_embedding(text, model, tokenizer, pooling='mean')
print(f"Embedding shape: {emb.shape}")  # (1, 768)
```

---

## 3. Document Embeddings

### What Are They?

Document embeddings convert **entire documents** (paragraphs, articles, books) into vectors.

```python
# Whole document = One vector
document = """
Machine learning is a subset of artificial intelligence.
It allows computers to learn from data without being explicitly programmed.
Deep learning is a subset of machine learning using neural networks.
"""
doc_embedding = model.encode(document)  # Single vector!
```

### Challenges with Long Documents

```
PROBLEM: Context window limits
────────────────────────────────
Most models have token limits:
- BERT: 512 tokens (~300 words)
- GPT-3: 4096 tokens
- OpenAI embeddings: 8191 tokens

A book might have 100,000+ tokens!

SOLUTIONS:
1. Chunking: Split into smaller pieces
2. Hierarchical: Embed chunks, then combine
3. Long-context models: Longformer, BigBird
```

### Strategy 1: Chunking

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

def get_document_embedding(document, model, strategy='mean'):
    """
    Get document embedding using chunking.

    Strategies:
    - mean: Average of chunk embeddings
    - max: Max pooling across chunks
    - first: Use first chunk only
    """
    chunks = chunk_text(document)
    chunk_embeddings = model.encode(chunks)

    if strategy == 'mean':
        return np.mean(chunk_embeddings, axis=0)
    elif strategy == 'max':
        return np.max(chunk_embeddings, axis=0)
    elif strategy == 'first':
        return chunk_embeddings[0]

# Example
model = SentenceTransformer('all-MiniLM-L6-v2')

long_document = """
[Your very long document here...]
""" * 100  # Simulate long text

doc_embedding = get_document_embedding(long_document, model)
print(f"Document embedding shape: {doc_embedding.shape}")  # (384,)
```

### Strategy 2: Hierarchical Embeddings

```python
def hierarchical_embedding(document, model, sentences_per_chunk=5):
    """
    Create hierarchical document embedding.

    Level 1: Sentence embeddings
    Level 2: Paragraph embeddings (grouped sentences)
    Level 3: Document embedding (combined paragraphs)
    """
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize

    # Split into sentences
    sentences = sent_tokenize(document)

    # Get sentence embeddings
    sentence_embeddings = model.encode(sentences)

    # Group into paragraphs
    paragraph_embeddings = []
    for i in range(0, len(sentence_embeddings), sentences_per_chunk):
        para_emb = np.mean(sentence_embeddings[i:i+sentences_per_chunk], axis=0)
        paragraph_embeddings.append(para_emb)

    # Combine into document embedding
    document_embedding = np.mean(paragraph_embeddings, axis=0)

    return {
        'sentences': sentence_embeddings,
        'paragraphs': np.array(paragraph_embeddings),
        'document': document_embedding
    }
```

### Doc2Vec (Legacy Approach)

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Prepare documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing handles text data",
]

# Create tagged documents
tagged_docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(documents)]

# Train Doc2Vec
model = Doc2Vec(tagged_docs, vector_size=100, window=5, min_count=1, epochs=100)

# Get document vector
doc_vector = model.dv[0]  # First document
print(f"Document vector shape: {doc_vector.shape}")  # (100,)

# Infer vector for new document
new_doc = "AI is transforming technology"
new_vector = model.infer_vector(new_doc.split())
```

---

## 4. Specialized Embeddings

### Code Embeddings

```python
from sentence_transformers import SentenceTransformer

# Model trained on code
model = SentenceTransformer('microsoft/codebert-base')

code_snippets = [
    "def add(a, b): return a + b",
    "function add(a, b) { return a + b; }",
    "int add(int a, int b) { return a + b; }",
]

embeddings = model.encode(code_snippets)
# These will be similar despite different languages!
```

### Image Embeddings (CLIP)

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Text and image in same embedding space!
image = Image.open("cat.jpg")
text = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Compare image to text descriptions
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(f"Probability it's a cat: {probs[0][0]:.2%}")
```

### Multi-lingual Embeddings

```python
from sentence_transformers import SentenceTransformer

# Model that works across 50+ languages
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

sentences = [
    "Hello, how are you?",           # English
    "Hola, cmo ests?",              # Spanish
    "Bonjour, comment allez-vous?",  # French
    "",                         # Chinese
    "Hallo, wie geht es dir?",       # German
]

embeddings = model.encode(sentences)

# All asking "How are you?" - should be similar!
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
print("Cross-lingual similarities:")
print(similarities)
```

---

## Comparison Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                    EMBEDDING TYPES COMPARISON                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TYPE          GRANULARITY    CONTEXT     BEST USE CASE          │
│  ────          ───────────    ───────     ─────────────          │
│                                                                  │
│  Word          Single word    None        Word similarity,       │
│  (Word2Vec)                               analogies              │
│                                                                  │
│  Sentence      Full sentence  Full        Semantic search,       │
│  (SBERT)                                  clustering, RAG        │
│                                                                  │
│  Document      Full document  Full        Document search,       │
│  (Chunked)                                categorization         │
│                                                                  │
│  Code          Code snippets  Syntax      Code search,           │
│  (CodeBERT)                               similarity             │
│                                                                  │
│  Multi-modal   Text + Images  Cross       Image search,          │
│  (CLIP)                       modal       visual Q&A             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Choosing the Right Type

```python
# Decision tree for embedding type

def choose_embedding_type(use_case):
    """
    Guide to choosing embedding type.
    """
    if use_case == "word_analogies":
        return "Word2Vec or GloVe"

    elif use_case in ["semantic_search", "qa_system", "rag"]:
        return "Sentence embeddings (all-MiniLM-L6-v2 or OpenAI)"

    elif use_case == "document_classification":
        return "Document embeddings with chunking"

    elif use_case == "code_search":
        return "CodeBERT or StarCoder"

    elif use_case == "multilingual":
        return "paraphrase-multilingual-MiniLM-L12-v2"

    elif use_case == "image_text_matching":
        return "CLIP"

    else:
        return "Start with sentence embeddings (most versatile)"
```

## What's Next?

Now let's explore the specific models in detail:
- [Part 4: Popular Embedding Models](./04-embedding-models.md)

---

## Quick Quiz

1. **What's the main limitation of Word2Vec?**
   - Answer: No context awareness - same word gets same embedding regardless of meaning

2. **Why are sentence embeddings better for semantic search?**
   - Answer: They capture the full meaning of a sentence, including negation and context

3. **How do you handle documents longer than the model's context window?**
   - Answer: Chunking - split into smaller pieces and combine embeddings

4. **What model would you use for searching images with text queries?**
   - Answer: CLIP (multi-modal embeddings)
