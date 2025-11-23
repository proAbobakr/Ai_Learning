# Part 2: Mathematics Behind Embeddings

Don't worry! We'll explain everything simply with lots of examples.

## Vectors: The Foundation

### What is a Vector?

A **vector** is just a list of numbers. That's it!

```python
# 2D vector (like a point on a map)
location = [3, 4]

# 3D vector (like a point in space)
point_3d = [1, 2, 3]

# Embedding vector (like a point in "meaning space")
embedding = [0.5, -0.2, 0.8, 0.1, ...]  # 384+ numbers
```

### Visualizing Vectors

**2D Example:**
```
    y
    ↑
  4 |     • A = [3, 4]
    |    /
  3 |   /
    |  /
  2 | /
    |/
  1 +--------→ x
    0 1 2 3 4
```

Point A is at position (3, 4) - go 3 steps right, 4 steps up.

**Higher Dimensions:**

We can't visualize 384 dimensions, but the math works the same way!

## Measuring Distance: How Far Apart?

### Euclidean Distance (Straight Line)

The most intuitive distance - like measuring with a ruler.

```
Formula: distance = √[(x₂-x₁)² + (y₂-y₁)² + ...]
```

**Example:**
```python
import numpy as np

# Two points
A = np.array([1, 2])
B = np.array([4, 6])

# Euclidean distance
distance = np.sqrt(np.sum((A - B) ** 2))
# distance = √[(4-1)² + (6-2)²]
# distance = √[9 + 16]
# distance = √25 = 5

# Or simply:
distance = np.linalg.norm(A - B)
print(f"Distance: {distance}")  # 5.0
```

**Visual:**
```
    y
    ↑
  6 |           • B [4,6]
    |          /|
  5 |         / |
    |        /  | 4 units
  4 |       /   |
    |      /    |
  3 |     /     |
    |    /      |
  2 | • A [1,2] |
    | |←--3 units--→|
  1 +---------------→ x
```

### Cosine Similarity (Direction)

This measures the **angle** between vectors, not the distance.

- **1.0** = Same direction (identical meaning)
- **0.0** = Perpendicular (unrelated)
- **-1.0** = Opposite direction (opposite meaning)

```
Formula: cos(θ) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = magnitude (length) of A
```

**Example:**
```python
import numpy as np

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)

# Example embeddings (simplified)
happy = np.array([0.8, 0.6, 0.2])
joyful = np.array([0.75, 0.65, 0.15])
sad = np.array([-0.7, -0.5, 0.3])

print(f"happy vs joyful: {cosine_similarity(happy, joyful):.4f}")
# Output: 0.9921 (very similar!)

print(f"happy vs sad: {cosine_similarity(happy, sad):.4f}")
# Output: -0.8234 (opposite!)
```

**Visual:**
```
                    ↗ happy
                   /
                  / θ (small angle = similar)
                 /
    origin →   •←------ joyful
                 \
                  \ θ (large angle = different)
                   \
                    ↘ sad
```

### When to Use Which?

| Method | Use When | Example |
|--------|----------|---------|
| **Cosine Similarity** | Comparing meaning regardless of length | Text similarity, recommendations |
| **Euclidean Distance** | Absolute position matters | Clustering, anomaly detection |
| **Dot Product** | Speed is critical | Fast similarity search |

## The Dot Product: Simple but Powerful

The dot product multiplies corresponding elements and sums them.

```
A · B = a₁×b₁ + a₂×b₂ + a₃×b₃ + ...
```

**Example:**
```python
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

dot_product = np.dot(A, B)
# = (1×4) + (2×5) + (3×6)
# = 4 + 10 + 18
# = 32
```

**Why it matters:**
- High dot product = vectors point in similar directions
- Used internally by cosine similarity
- Very fast to compute

## Vector Operations

### Addition: Combining Meanings

```python
# Adding vectors
king = np.array([0.5, 0.8, 0.2])
royal = np.array([0.3, 0.1, 0.5])

king_plus_royal = king + royal
# Result: [0.8, 0.9, 0.7]
```

### Subtraction: Finding Differences

```python
# The famous example: king - man + woman ≈ queen

king = np.array([0.5, 0.8, 0.2, 0.9])
man = np.array([0.4, 0.7, -0.3, 0.1])
woman = np.array([0.4, 0.7, 0.2, 0.1])

# What's the "royalty" component?
royalty = king - man
# royalty = [0.1, 0.1, 0.5, 0.8]

# Add it to woman
result = woman + royalty
# result = [0.5, 0.8, 0.7, 0.9]

# This should be close to "queen"!
queen = np.array([0.5, 0.8, 0.7, 0.9])
```

**Visual:**
```
        "male royalty"
              ↑
    king •----→
         |    |
         |    | "royalty" vector
         |    |
    man  •----→
              |
              ↓
        "male"

    Apply "royalty" to "woman":
    woman + (king - man) = queen
```

## Normalization: Making Vectors Comparable

**Normalization** scales a vector to have length 1 (unit vector).

```python
def normalize(vector):
    """Convert to unit vector (length = 1)."""
    magnitude = np.linalg.norm(vector)
    return vector / magnitude

# Example
original = np.array([3, 4])
print(f"Original length: {np.linalg.norm(original)}")  # 5.0

normalized = normalize(original)
print(f"Normalized: {normalized}")  # [0.6, 0.8]
print(f"Normalized length: {np.linalg.norm(normalized)}")  # 1.0
```

**Why normalize?**
1. Cosine similarity on normalized vectors = dot product (faster!)
2. Makes vectors comparable regardless of original length
3. Many embedding models output normalized vectors

## Dimensionality: More is (Usually) Better

### What Does Each Dimension Represent?

Each dimension captures some aspect of meaning:

```
Dimension 1: Maybe "royalty" vs "common"
Dimension 2: Maybe "positive" vs "negative"
Dimension 3: Maybe "animate" vs "inanimate"
...
Dimension 384: Something the model learned!
```

**Important:** We don't manually define dimensions. The model learns what each dimension should represent during training!

### Example: 2D vs 100D

```
2D embeddings (too simple):
- "happy" and "joyful" might overlap with "excited"
- Not enough space to capture nuances

100D embeddings:
- "happy" = content, peaceful
- "joyful" = enthusiastic, celebratory
- "excited" = anticipation, energy
- Each can have its own unique position
```

### Common Dimensions

| Model | Dimensions | Quality |
|-------|------------|---------|
| Word2Vec | 100-300 | Good for words |
| GloVe | 50-300 | Good for words |
| BERT base | 768 | High quality |
| OpenAI small | 1536 | Very high quality |
| OpenAI large | 3072 | Highest quality |

## Practical Example: Complete Similarity Calculator

```python
import numpy as np
from typing import List, Tuple

class EmbeddingSimilarity:
    """Calculate various similarity metrics for embeddings."""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity.
        Returns value between -1 and 1.
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Euclidean distance.
        Returns value >= 0 (lower = more similar).
        """
        return np.linalg.norm(a - b)

    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Manhattan distance (L1 norm).
        Sum of absolute differences.
        """
        return np.sum(np.abs(a - b))

    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate dot product.
        Higher = more similar (for normalized vectors).
        """
        return np.dot(a, b)

    def find_most_similar(
        self,
        query: np.ndarray,
        candidates: List[np.ndarray],
        labels: List[str],
        metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """
        Find most similar vectors to query.

        Args:
            query: The query vector
            candidates: List of candidate vectors
            labels: Labels for each candidate
            metric: "cosine", "euclidean", or "dot"

        Returns:
            List of (label, score) tuples, sorted by similarity
        """
        scores = []

        for candidate, label in zip(candidates, labels):
            if metric == "cosine":
                score = self.cosine_similarity(query, candidate)
            elif metric == "euclidean":
                # Negative so higher is better
                score = -self.euclidean_distance(query, candidate)
            else:  # dot product
                score = self.dot_product(query, candidate)

            scores.append((label, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# Demo
if __name__ == "__main__":
    calc = EmbeddingSimilarity()

    # Simulated embeddings (in reality, these come from a model)
    embeddings = {
        "happy": np.array([0.8, 0.6, 0.2, 0.1]),
        "joyful": np.array([0.75, 0.65, 0.18, 0.12]),
        "sad": np.array([-0.7, -0.5, 0.2, 0.1]),
        "angry": np.array([-0.5, -0.6, -0.3, 0.4]),
        "excited": np.array([0.7, 0.7, 0.3, 0.2]),
    }

    # Query
    query_word = "happy"
    query_vec = embeddings[query_word]

    # Find similar words
    candidates = [v for k, v in embeddings.items() if k != query_word]
    labels = [k for k in embeddings.keys() if k != query_word]

    print(f"Words most similar to '{query_word}':\n")

    results = calc.find_most_similar(query_vec, candidates, labels)
    for label, score in results:
        print(f"  {label}: {score:.4f}")

# Output:
# Words most similar to 'happy':
#   joyful: 0.9965
#   excited: 0.9847
#   sad: -0.7842
#   angry: -0.8156
```

## Key Formulas Cheat Sheet

```
┌─────────────────────────────────────────────────────────────┐
│                    EMBEDDING MATH CHEAT SHEET               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DOT PRODUCT:                                               │
│  A · B = Σ(aᵢ × bᵢ)                                        │
│                                                             │
│  MAGNITUDE (Length):                                        │
│  ||A|| = √(Σaᵢ²)                                           │
│                                                             │
│  COSINE SIMILARITY:                                         │
│  cos(θ) = (A · B) / (||A|| × ||B||)                        │
│  Range: [-1, 1]                                             │
│                                                             │
│  EUCLIDEAN DISTANCE:                                        │
│  d = √(Σ(aᵢ - bᵢ)²)                                        │
│  Range: [0, ∞)                                              │
│                                                             │
│  NORMALIZATION:                                             │
│  Â = A / ||A||                                              │
│  Result: ||Â|| = 1                                          │
│                                                             │
│  For NORMALIZED vectors:                                    │
│  cosine_similarity = dot_product                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Summary

| Concept | What It Does | When to Use |
|---------|--------------|-------------|
| **Vector** | Represents meaning as numbers | Always |
| **Dot Product** | Measures alignment | Fast similarity |
| **Cosine Similarity** | Measures direction similarity | Semantic comparison |
| **Euclidean Distance** | Measures absolute distance | Clustering |
| **Normalization** | Scales to length 1 | Before comparison |

## What's Next?

Now that you understand the math, let's explore:
- [Part 3: Types of Embeddings](./03-types-of-embeddings.md)

---

## Practice Problems

1. Calculate the cosine similarity between [1, 0] and [0, 1]
   - Answer: 0 (perpendicular vectors)

2. What's the Euclidean distance between [0, 0] and [3, 4]?
   - Answer: 5

3. Normalize the vector [3, 4]
   - Answer: [0.6, 0.8]
