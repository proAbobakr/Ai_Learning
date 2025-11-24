# Week 9: Retrieval & Retrieval-Augmented Generation (RAG)

## Overview

This week covers information retrieval techniques and how they combine with language models in retrieval-augmented generation (RAG). We'll explore both sparse (BM25) and dense (DPR) retrieval methods.

## Learning Objectives

By the end of this week, you will be able to:
- Implement BM25 retrieval from scratch
- Train and use dense passage retrieval (DPR)
- Build a retrieval-augmented QA system
- Compare dense vs sparse retrieval approaches
- Understand RAG architecture and its variants
- Handle long context challenges with retrieval

## Core Topics

### 1. Information Retrieval Fundamentals

**The Retrieval Problem**
```
Query: "What is the capital of France?"
Corpus: [doc_1, doc_2, ..., doc_n]
Goal: Return k most relevant documents
```

**Evaluation Metrics**
- **Precision@k**: Fraction of top-k that are relevant
- **Recall@k**: Fraction of relevant docs in top-k
- **Mean Reciprocal Rank (MRR)**: 1/rank of first relevant doc
- **NDCG**: Normalized Discounted Cumulative Gain

### 2. Sparse Retrieval: BM25

**TF-IDF Recap**
```
TF-IDF(t,d) = TF(t,d) × log(N/DF(t))
```

**BM25 (Best Matching 25)**
```python
score(q, d) = Σ IDF(q_i) × [f(q_i, d) × (k1 + 1)] / [f(q_i, d) + k1 × (1 - b + b × |d|/avgdl)]

where:
- f(q_i, d) = term frequency of q_i in d
- |d| = document length
- avgdl = average document length
- k1 = 1.2-2.0 (term frequency saturation)
- b = 0.75 (length normalization)
```

**Implementation**:
```python
from rank_bm25 import BM25Okapi

corpus = ["document one text", "document two text", ...]
tokenized_corpus = [doc.split() for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

query = "search query"
tokenized_query = query.split()
scores = bm25.get_scores(tokenized_query)
top_k = scores.argsort()[-k:][::-1]
```

**Strengths**:
- No training required
- Fast indexing and retrieval
- Works well for keyword matching
- Established, well-understood

**Limitations**:
- Exact term matching only
- Synonyms/paraphrases not captured
- No semantic understanding

### 3. Dense Retrieval: DPR

**Key Idea**: Learn dense vector representations for queries and documents

**Architecture**
```
Query Encoder (BERT):   "What is the capital of France?" → q_vec [768]
Document Encoder (BERT): "Paris is the capital of France." → d_vec [768]

Similarity: sim(q, d) = q_vec · d_vec
```

**Training**
```
Positive pairs: (query, relevant_doc)
Negative pairs: (query, irrelevant_doc)

Loss: -log[exp(sim(q, d+)) / (exp(sim(q, d+)) + Σ exp(sim(q, d-)))]
```

**In-Batch Negatives**
- Use other queries' documents as negatives
- Efficient: one forward pass for all
- Works surprisingly well

```python
def contrastive_loss(query_embeds, doc_embeds, temperature=0.05):
    # query_embeds: [batch, dim]
    # doc_embeds: [batch, dim]

    similarity = query_embeds @ doc_embeds.T / temperature  # [batch, batch]
    labels = torch.arange(len(query_embeds))

    return F.cross_entropy(similarity, labels)
```

**Using DPR**:
```python
from transformers import DPRQuestionEncoder, DPRContextEncoder

q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
d_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

query_embedding = q_encoder(**tokenizer(query))
doc_embeddings = d_encoder(**tokenizer(documents))

scores = query_embedding @ doc_embeddings.T
```

### 4. Indexing for Dense Retrieval

**Challenge**: Can't compare query to all documents at scale

**FAISS (Facebook AI Similarity Search)**
```python
import faiss

# Build index
dimension = 768
index = faiss.IndexFlatIP(dimension)  # Inner product
index.add(document_embeddings)

# Search
distances, indices = index.search(query_embedding, k=10)
```

**Index Types**:
- **Flat**: Exact search, O(n)
- **IVF**: Inverted file, approximate
- **HNSW**: Hierarchical navigable small world graphs
- **PQ**: Product quantization for compression

### 5. Retrieval-Augmented Generation (RAG)

**Architecture**
```
Query → Retriever → Top-k Documents → [Query + Documents] → Generator → Answer
```

**RAG Paper (Lewis et al., 2020)**
```
P(y|x) = Σ P(z|x) × P(y|x,z)
         z∈top-k

where:
- x = input query
- z = retrieved documents
- y = generated output
```

**Two Variants**:
- **RAG-Sequence**: Retrieve once, generate full sequence
- **RAG-Token**: Retrieve per generated token (more expensive)

**Implementation**:
```python
def rag_generate(query, retriever, generator, k=5):
    # Retrieve relevant documents
    docs = retriever.retrieve(query, k=k)

    # Build context
    context = "\n\n".join([doc.text for doc in docs])

    # Generate with context
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    answer = generator.generate(prompt)

    return answer, docs
```

### 6. REALM and Other RAG Models

**REALM** (Guu et al., 2020)
- Pre-train with retrieval
- End-to-end differentiable retrieval
- Updates retriever during training

**FiD (Fusion-in-Decoder)**
- Encode query + each doc separately
- Fuse in decoder attention
- Scales better with more documents

**Toolformer**
- Model learns when to call tools
- Retrieval as one possible tool
- Self-supervised tool learning

### 7. Hybrid Retrieval

**Combine Dense + Sparse**
```python
def hybrid_search(query, bm25, dense_index, alpha=0.5):
    sparse_scores = bm25.get_scores(query)
    dense_scores = dense_index.search(encode(query))

    # Normalize scores
    sparse_norm = (sparse_scores - sparse_scores.min()) / sparse_scores.ptp()
    dense_norm = (dense_scores - dense_scores.min()) / dense_scores.ptp()

    # Combine
    combined = alpha * dense_norm + (1 - alpha) * sparse_norm
    return combined.argsort()[-k:][::-1]
```

**Benefits**:
- Best of both worlds
- Dense catches semantics
- Sparse catches exact matches

### 8. Handling Long Contexts

**Chunking Strategies**:
- Fixed-size chunks (512 tokens)
- Sentence-based chunks
- Paragraph-based chunks
- Overlapping chunks

**Challenges**:
- Information split across chunks
- Context fragmentation
- Retrieval recall issues

**Solutions**:
- Hierarchical retrieval
- Parent document retrieval
- Sliding window with overlap

## Key References

### Required Reading

1. **Karpukhin et al. (2020) - "Dense Passage Retrieval for Open-Domain Question Answering"**
   - DPR paper
   - [Paper](https://arxiv.org/abs/2004.04906)

2. **Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**
   - RAG paper
   - [Paper](https://arxiv.org/abs/2005.11401)

3. **Chen et al. (2017) - "Reading Wikipedia to Answer Open-Domain Questions"**
   - DrQA system
   - [Paper](https://arxiv.org/abs/1704.00051)

### Recommended Reading

4. **Guu et al. (2020) - "REALM: Retrieval-Augmented Language Model Pre-Training"**
   - End-to-end retrieval learning
   - [Paper](https://arxiv.org/abs/2002.08909)

5. **Schick et al. (2023) - "Toolformer: Language Models Can Teach Themselves to Use Tools"**
   - Tool use in LLMs
   - [Paper](https://arxiv.org/abs/2302.04761)

6. **Robertson & Zaragoza (2009) - "The Probabilistic Relevance Framework: BM25 and Beyond"**
   - Deep dive into BM25
   - [Paper](https://www.nowpublishers.com/article/Details/INR-019)

## Practical Exercise

### Exercise 1: BM25 Implementation

**Part A: Implement BM25 from Scratch**
```python
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        # Build index

    def _compute_idf(self):
        # Compute IDF for all terms

    def score(self, query, doc_idx):
        # Score a single document

    def search(self, query, k=10):
        # Return top-k documents
```

**Part B: Compare with Library**
- Use rank_bm25
- Verify your implementation

### Exercise 2: Dense Retrieval Pipeline

**Part A: Build DPR System**
```python
from transformers import DPRQuestionEncoder, DPRContextEncoder
import faiss

# Encode all documents
# Build FAISS index
# Implement search function
```

**Part B: Compare with BM25**
| Query Type | BM25 Recall@10 | DPR Recall@10 |
|------------|---------------|---------------|
| Keyword | | |
| Paraphrase | | |
| Complex | | |

### Exercise 3: Build RAG QA System

**Objective**: End-to-end retrieval-augmented QA

```python
class RAGSystem:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, question, k=5):
        docs = self.retriever.search(question, k)
        context = self.build_context(docs)
        prompt = self.build_prompt(question, context)
        return self.generator.generate(prompt)
```

**Dataset**: Use Natural Questions or TriviaQA

**Evaluation**:
- Exact Match (EM)
- F1 Score
- Retrieval Recall

### Exercise Files
- [exercises/bm25_implementation.py](./exercises/bm25_implementation.py)
- [exercises/dense_retrieval.py](./exercises/dense_retrieval.py)
- [exercises/rag_system.py](./exercises/rag_system.py)

## Study Questions

1. Why does BM25 include length normalization?
2. How do in-batch negatives work and why are they effective?
3. When would you prefer sparse over dense retrieval?
4. What are the trade-offs in RAG vs fine-tuning for knowledge?
5. How does chunking strategy affect retrieval quality?

## Additional Resources

### Libraries
- [FAISS](https://github.com/facebookresearch/faiss)
- [rank_bm25](https://github.com/dorianbrown/rank_bm25)
- [Sentence Transformers](https://www.sbert.net/)

### Tutorials
- [Hugging Face RAG](https://huggingface.co/docs/transformers/model_doc/rag)
- [LangChain Retrieval](https://python.langchain.com/docs/modules/data_connection/)

### Datasets
- Natural Questions
- TriviaQA
- MS MARCO

## Next Week Preview

Week 10 covers model compression and RLHF:
- Knowledge distillation
- Quantization techniques
- Reinforcement learning from human feedback

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 10, ensure you can:
- [ ] Implement BM25 retrieval
- [ ] Build a dense retrieval system with FAISS
- [ ] Compare sparse vs dense retrieval
- [ ] Create a working RAG QA system
- [ ] Evaluate retrieval and generation quality
