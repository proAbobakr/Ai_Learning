# Module 4: Intermediate RAG Techniques

## Overview

This module covers intermediate-level RAG techniques including advanced vector databases, embeddings optimization, and building production-quality RAG pipelines.

## Learning Objectives

By the end of this module, you will be able to:
- Work with multiple vector database providers
- Optimize embedding strategies for your use case
- Implement similarity search with filtering
- Build complete RAG pipelines with error handling
- Compare and benchmark different approaches

## Prerequisites

- Completed Module 3 (Basic Implementation)
- Understanding of embeddings and vector search
- Comfortable with Python and APIs

## Time Required

2-3 weeks (15-20 hours)

---

## Topics Covered

### 1. Vector Embeddings Deep Dive
**File:** `01-embeddings.md`

- Different embedding models comparison
- OpenAI vs Sentence Transformers vs Cohere
- Embedding dimensions and performance
- Fine-tuning embeddings for your domain
- Embedding caching strategies

**Hands-on:**
- Compare different embedding models
- Measure embedding quality
- Implement embedding caching

---

### 2. Vector Databases
**File:** `02-vector-databases.md`

**Covered Databases:**
- **ChromaDB**: Simple, local, great for development
- **Pinecone**: Managed, scalable, production-ready
- **Weaviate**: Open-source, feature-rich
- **FAISS**: Facebook's library, extremely fast
- **Qdrant**: Modern, fast, open-source

**Topics:**
- When to use each database
- Setup and configuration
- Performance characteristics
- Cost comparison
- Migration between databases

**Hands-on:**
- Set up each vector database
- Benchmark performance
- Implement metadata filtering
- Handle updates and deletions

---

### 3. Similarity Search
**File:** `03-similarity-search.md`

**Topics:**
- ANN (Approximate Nearest Neighbor) algorithms
- Distance metrics: Cosine, Euclidean, Dot product
- Trade-offs: Speed vs Accuracy
- Metadata filtering
- Hybrid search (vector + keyword)

**Techniques:**
- Basic similarity search
- MMR (Maximal Marginal Relevance)
- Similarity score thresholding
- Diversity in results

**Hands-on:**
- Implement different search strategies
- Optimize search parameters
- Add metadata filtering
- Build hybrid search

---

### 4. Complete RAG Pipeline
**File:** `04-complete-pipeline.md`

**Pipeline Components:**
1. Document ingestion
2. Preprocessing and cleaning
3. Intelligent chunking
4. Embedding generation
5. Vector storage
6. Query processing
7. Retrieval
8. Re-ranking (preview)
9. Generation
10. Response formatting

**Production Considerations:**
- Error handling
- Retry logic
- Logging
- Monitoring
- Performance optimization
- Cost optimization

**Hands-on:**
- Build end-to-end pipeline
- Add robust error handling
- Implement logging
- Optimize for production

---

## Code Examples

All code examples are in the `examples/intermediate-rag/` directory:

```
examples/intermediate-rag/
├── README.md
├── 01_embedding_comparison.py
├── 02_vector_db_setup.py
├── 03_similarity_search.py
├── 04_complete_pipeline.py
└── data/
    └── sample_docs/
```

---

## Practical Exercises

### Exercise 1: Embedding Showdown
Compare OpenAI, Sentence Transformers, and Cohere embeddings on the same dataset. Measure:
- Retrieval quality
- Speed
- Cost

### Exercise 2: Database Migration
Build a RAG system with ChromaDB, then migrate to Pinecone. Learn:
- Export/import processes
- Handling downtime
- Data validation

### Exercise 3: Hybrid Search
Implement a hybrid search that combines:
- Vector similarity
- Keyword matching (BM25)
- Metadata filtering

### Exercise 4: Production Pipeline
Build a complete pipeline with:
- Input validation
- Error handling
- Retry logic
- Logging
- Performance monitoring

---

## Real-World Scenarios

### Scenario 1: E-commerce Product Search
**Challenge:** Search millions of products by description
**Solution:** Optimized vector search with metadata filtering

### Scenario 2: Legal Document Analysis
**Challenge:** Find relevant precedents in case law
**Solution:** Hybrid search with date and jurisdiction filters

### Scenario 3: Customer Support Knowledge Base
**Challenge:** Fast, accurate answers from documentation
**Solution:** Multi-stage retrieval with re-ranking

---

## Performance Benchmarks

### Vector Database Comparison

| Database | Index Time | Query Time | Memory | Cost |
|----------|-----------|------------|---------|------|
| ChromaDB | Medium | Medium | High | Free |
| Pinecone | Fast | Fast | Low | $$ |
| FAISS | Fast | Very Fast | Medium | Free |
| Weaviate | Medium | Fast | Medium | Free/$ |
| Qdrant | Fast | Fast | Medium | Free/$ |

*Based on 100K documents, 1536-dim embeddings*

### Embedding Model Comparison

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|--------|---------|------|
| OpenAI ada-002 | 1536 | Fast | Excellent | $ |
| Sentence-BERT | 384 | Very Fast | Good | Free |
| E5-large | 1024 | Medium | Excellent | Free |
| Cohere embed-v3 | 1024 | Fast | Excellent | $ |

---

## Best Practices

### Choosing a Vector Database

**Use ChromaDB when:**
- Prototyping/development
- Small to medium datasets (<100K docs)
- Running locally
- No budget for managed services

**Use Pinecone when:**
- Production deployment
- Large scale (>100K docs)
- Need managed service
- Budget available

**Use FAISS when:**
- Maximum speed required
- Self-hosting
- CPU/GPU optimization important
- No cloud dependencies

**Use Weaviate when:**
- Need advanced features (GraphQL, modules)
- Open-source requirement
- Flexible deployment
- Schema management important

### Embedding Selection

**Use OpenAI when:**
- Best quality needed
- Cost is not primary concern
- English language focus

**Use Sentence Transformers when:**
- Free/local solution needed
- Multilingual support
- Custom fine-tuning planned

**Use Cohere when:**
- Multilingual requirements
- High quality needed
- Prefer API over local

---

## Common Pitfalls

### 1. Wrong Chunk Size
❌ Too large → Irrelevant info retrieved
❌ Too small → Missing context
✅ Test different sizes for your domain

### 2. Not Handling Metadata
❌ Storing vectors only
✅ Include source, date, author, etc.

### 3. Ignoring Performance
❌ Not benchmarking
✅ Measure latency, cost, quality

### 4. No Error Handling
❌ Crashes on API errors
✅ Retry logic, fallbacks, logging

---

## Assessment

Before moving to Module 5, ensure you can:

- [ ] Explain different embedding models
- [ ] Set up at least 2 different vector databases
- [ ] Implement similarity search with filtering
- [ ] Build a complete RAG pipeline
- [ ] Add error handling and logging
- [ ] Compare performance of different approaches
- [ ] Choose appropriate tools for a use case

---

## Additional Resources

### Documentation
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Pinecone Docs](https://docs.pinecone.io/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)

### Papers
- "Dense Passage Retrieval" (Facebook AI)
- "Sentence-BERT" (Reimers & Gurevych)
- "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (OpenAI)

### Videos
- Pinecone's "Vector Databases Explained"
- LangChain's "Advanced RAG Techniques"

---

## Next Module

Ready for advanced techniques?

**[→ Module 5: Advanced RAG](../05-advanced/README.md)**

You'll learn:
- Re-ranking strategies
- Query optimization
- Multi-modal RAG
- Agent-based RAG
- Production deployment
