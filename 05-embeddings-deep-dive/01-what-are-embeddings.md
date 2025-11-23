# Part 1: What are Embeddings?

## The Problem: Computers Don't Understand Text

Imagine you ask a computer: *"Is 'happy' similar to 'joyful'?"*

The computer sees:
- "happy" = [104, 97, 112, 112, 121] (ASCII codes)
- "joyful" = [106, 111, 121, 102, 117, 108]

These numbers tell us **nothing** about meaning! The computer has no idea these words are related.

## The Solution: Embeddings!

**Embeddings** convert words, sentences, or documents into **meaningful numbers** (vectors) where:
- **Similar meanings = Similar numbers**
- **Different meanings = Different numbers**

### Simple Analogy: GPS Coordinates

Think of embeddings like GPS coordinates for meaning:

```
Paris, France  → [48.8566, 2.3522]
London, UK     → [51.5074, -0.1278]
Tokyo, Japan   → [35.6762, 139.6503]
```

Cities close on a map have similar coordinates. **Embeddings work the same way for meaning!**

```
"happy"   → [0.82, 0.15, 0.73, ...]
"joyful"  → [0.81, 0.14, 0.75, ...]   # Very similar numbers!
"sad"     → [-0.75, 0.20, -0.60, ...] # Very different numbers!
```

## Visual Example

```
                    HAPPY 😊
                      ↓
    Meaning Space:  [0.82, 0.15, 0.73]
                         ↘
                          ↘  (very close!)
                           ↘
    Meaning Space:  [0.81, 0.14, 0.75]
                      ↑
                    JOYFUL 🎉


                    SAD 😢
                      ↓
    Meaning Space:  [-0.75, 0.20, -0.60]  # Far away from happy!
```

## What Does an Embedding Look Like?

An embedding is simply a **list of numbers** (called a vector):

```python
# Real embedding example (simplified to 5 dimensions)
"king"   → [0.50,  0.80,  0.20, -0.10,  0.90]
"queen"  → [0.48,  0.82,  0.65, -0.12,  0.88]
"man"    → [0.45,  0.75, -0.30, -0.08,  0.85]
"woman"  → [0.43,  0.77,  0.15, -0.10,  0.83]
```

In reality, embeddings have **hundreds or thousands** of dimensions:
- OpenAI's `text-embedding-3-small`: 1536 dimensions
- Sentence Transformers: 384-768 dimensions
- Word2Vec: 100-300 dimensions

## Why Do Embeddings Work?

Embeddings are created by **training on massive amounts of text**. The model learns patterns like:

1. Words appearing in similar contexts get similar embeddings
2. "The **king** sat on his throne" ↔ "The **queen** sat on her throne"
3. King and queen appear in similar contexts → similar embeddings

## The Magic: Mathematical Relationships

Here's the famous example:

```
king - man + woman ≈ queen
```

In embedding space:
```python
embedding("king") - embedding("man") + embedding("woman")
    ≈ embedding("queen")
```

This means embeddings capture **actual relationships**!

```
       "royalty" direction
            ↗
    king ----→ queen
      |         |
      |         |  (parallel!)
      ↓         ↓
    man  ----→ woman
            ↗
       "female" direction
```

## Real-World Examples

### Example 1: Finding Similar Products

```python
# E-commerce search
user_query = "comfortable running shoes"
query_embedding = get_embedding(user_query)

# Find products with similar embeddings
products = [
    "Nike Air Max running sneakers",      # Similar! (athletic footwear)
    "Adidas ultraboost jogging shoes",    # Similar! (running related)
    "Formal leather dress shoes",         # Different! (not running)
    "Winter snow boots",                  # Different! (not running)
]
```

### Example 2: Customer Support

```python
# Finding similar support tickets
new_ticket = "My order hasn't arrived yet"

similar_tickets = [
    "Package not delivered",           # Similar meaning!
    "Where is my shipment?",          # Similar meaning!
    "Delivery delayed",               # Similar meaning!
    "I want to return an item",       # Different issue
]
```

### Example 3: Document Search

```python
# Academic paper search
query = "machine learning for medical diagnosis"

# Finds papers about:
# - "AI in healthcare diagnostics"    ✓ Semantically related
# - "Neural networks for X-ray analysis"    ✓ Related
# - "Deep learning in clinical settings"    ✓ Related
# - "Traditional database systems"    ✗ Not related
```

## Embeddings vs. Traditional Search

### Traditional Keyword Search (Bad)
```
Query: "How to fix a flat tire"
Results: Documents containing "flat" AND "tire"

Problem: Misses documents about:
- "Punctured wheel repair"
- "Changing a blown tire"
- "Tire replacement guide"
```

### Semantic Search with Embeddings (Good)
```
Query: "How to fix a flat tire"
Results: Documents with SIMILAR MEANING

Finds:
✓ "Punctured wheel repair"     (same meaning, different words)
✓ "Changing a blown tire"      (related topic)
✓ "Tire replacement guide"     (related topic)
```

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Embedding** | A list of numbers representing meaning |
| **Vector** | Another name for a list of numbers |
| **Dimension** | How many numbers in the embedding |
| **Semantic** | Related to meaning |
| **Similarity** | How close two embeddings are |

## Simple Python Example

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
sentences = [
    "I love programming",
    "Coding is my passion",
    "I enjoy writing software",
    "The weather is nice today"
]

embeddings = model.encode(sentences)

# Check the shape
print(f"Number of sentences: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")
# Output:
# Number of sentences: 4
# Embedding dimension: 384
```

## What's Next?

Now that you understand what embeddings are, let's dive into:
- [Part 2: The Mathematics Behind Embeddings](./02-mathematics.md)

---

## Quick Quiz

1. **What is an embedding?**
   - Answer: A list of numbers representing the meaning of text

2. **Why are embeddings useful?**
   - Answer: They capture semantic meaning, allowing computers to understand similarity

3. **What does it mean when two embeddings are "close"?**
   - Answer: The texts they represent have similar meanings

4. **How many dimensions do modern embeddings typically have?**
   - Answer: Hundreds to thousands (e.g., 384, 768, 1536)
