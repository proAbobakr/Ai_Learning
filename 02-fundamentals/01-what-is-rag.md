# What is RAG (Retrieval-Augmented Generation)?

## Introduction

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by combining them with external knowledge retrieval systems. Instead of relying solely on the knowledge baked into the model during training, RAG systems dynamically fetch relevant information to answer queries.

## The Problem RAG Solves

### Issues with Standard LLMs

1. **Knowledge Cutoff**: Models only know information up to their training date
   ```
   User: "What happened in the news today?"
   LLM: "I don't have access to current information..."
   ```

2. **Hallucinations**: Models can confidently generate false information
   ```
   User: "What's our company's vacation policy?"
   LLM: *Makes up a policy* ❌
   ```

3. **No Private Data Access**: Can't answer questions about your documents
   ```
   User: "Summarize our Q4 financial report"
   LLM: "I don't have access to your documents"
   ```

4. **Expensive Updates**: Retraining models is costly and time-consuming

### How RAG Fixes These Issues

RAG allows you to:
- ✅ Provide up-to-date information
- ✅ Answer questions about private documents
- ✅ Ground responses in factual sources
- ✅ Update knowledge without retraining
- ✅ Cite sources for verification

## How RAG Works (Simple Explanation)

Think of RAG like an open-book exam:

**Without RAG (Closed-book exam):**
```
Student (LLM) → Answers from memory only
                Limited to what they studied
                Might make mistakes
```

**With RAG (Open-book exam):**
```
Student (LLM) → Looks up relevant pages in textbook
                → Uses found information to answer
                → Cites sources
```

## RAG Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────┐
│                    RAG SYSTEM                           │
│                                                         │
│  1. User Query                                          │
│      ↓                                                  │
│  2. Retrieve Relevant Documents                         │
│      ↓                                                  │
│  3. Combine Query + Retrieved Context                   │
│      ↓                                                  │
│  4. Generate Answer with LLM                            │
│      ↓                                                  │
│  5. Return Response (with citations)                    │
└─────────────────────────────────────────────────────────┘
```

### Step-by-Step Example

**Scenario**: Ask about a company handbook

```python
# 1. User asks a question
query = "What is the remote work policy?"

# 2. System searches document database
# Finds relevant sections from handbook:
retrieved_docs = [
    "Remote work is allowed up to 3 days per week...",
    "Employees must maintain core hours of 10am-3pm...",
    "VPN access is required for remote work..."
]

# 3. Create enhanced prompt
prompt = f"""
Answer the question based on the following context:

Context:
{retrieved_docs}

Question: {query}

Answer:
"""

# 4. LLM generates answer using the context
answer = llm.generate(prompt)

# 5. Response with citation
# "According to the company handbook, remote work is allowed
#  up to 3 days per week. Employees must maintain core hours..."
```

## RAG vs. Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Cost** | Low (no retraining) | High (compute intensive) |
| **Update Speed** | Instant (update docs) | Slow (retrain model) |
| **Use Case** | Dynamic knowledge | Specific task behavior |
| **Data Needs** | Any documents | Labeled training data |
| **Traceability** | High (cite sources) | Low (black box) |

### When to Use What?

**Use RAG for:**
- Question answering over documents
- Keeping information up-to-date
- Private/proprietary data
- Citing sources
- Cost-effective solutions

**Use Fine-Tuning for:**
- Changing model behavior/style
- Domain-specific language
- Improving specific capabilities
- When you have large labeled datasets

**Use Both for:**
- Fine-tune for domain expertise
- RAG for current/specific information

## Real-World RAG Examples

### 1. Customer Support Bot
```
User: "How do I reset my password?"

RAG System:
1. Searches knowledge base
2. Finds "Password Reset Guide"
3. Generates friendly response with steps
4. Includes link to full article
```

### 2. Research Assistant
```
User: "Summarize recent papers on quantum computing"

RAG System:
1. Searches paper database
2. Retrieves top 5 relevant papers
3. Generates summary with key findings
4. Cites each paper
```

### 3. Code Documentation Helper
```
User: "How do I use the authenticate() function?"

RAG System:
1. Searches codebase docs
2. Finds function documentation
3. Provides usage examples
4. Shows related functions
```

### 4. Legal Document Analysis
```
User: "What are the termination clauses in contract ABC?"

RAG System:
1. Retrieves contract ABC
2. Finds termination sections
3. Summarizes clauses
4. Quotes exact text
```

## Benefits of RAG

### 1. Accuracy & Factuality
- Grounds responses in actual documents
- Reduces hallucinations
- Provides citations

### 2. Cost-Effective
- No expensive model retraining
- Use smaller, cheaper LLMs
- Update knowledge by updating documents

### 3. Transparency
- See which documents were used
- Trace answers back to sources
- Build trust with users

### 4. Flexibility
- Works with any document type
- Easy to add/remove knowledge
- Domain-agnostic

### 5. Privacy
- Keep data in your infrastructure
- Don't need to send data for training
- Control access to documents

## Limitations of RAG

### 1. Retrieval Quality
- If retrieval fails, answer will be poor
- Depends on good document chunking
- Requires quality embeddings

### 2. Context Window Limits
- LLMs have maximum input size
- Can't retrieve unlimited documents
- Must prioritize most relevant content

### 3. Latency
- Retrieval adds processing time
- Multiple API calls needed
- Trade-off between speed and accuracy

### 4. Complexity
- More moving parts than simple LLM
- Requires vector database
- Needs careful tuning

## RAG Components (Preview)

We'll dive deep into each of these:

1. **Documents**: Source of knowledge (PDFs, web pages, databases)
2. **Chunking**: Breaking documents into manageable pieces
3. **Embeddings**: Converting text to vectors
4. **Vector Store**: Database for searching similar content
5. **Retriever**: Finding relevant chunks
6. **LLM**: Generating final answer
7. **Orchestration**: Tying it all together

## Key Metrics for RAG Systems

### Retrieval Metrics
- **Recall**: Did we retrieve relevant documents?
- **Precision**: Are retrieved documents actually relevant?
- **MRR (Mean Reciprocal Rank)**: Where does first relevant doc appear?

### Generation Metrics
- **Faithfulness**: Is answer grounded in retrieved docs?
- **Answer Relevance**: Does answer address the question?
- **Context Relevance**: Are retrieved docs relevant?

We'll cover evaluation in detail in Module 6.

## RAG Variants (Advanced Preview)

As you advance, you'll encounter:

- **Naive RAG**: Simple retrieve → generate
- **Advanced RAG**: Query rewriting, re-ranking, hybrid search
- **Modular RAG**: Multiple retrievers, routing, agents
- **Self-RAG**: Model critiques and refines its own outputs
- **Multi-modal RAG**: Images, tables, code, not just text

## Simple Code Example

Here's a minimal RAG implementation to illustrate the concept:

```python
from typing import List

# Simplified RAG system
class SimpleRAG:
    def __init__(self, documents: List[str]):
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """Simple keyword-based retrieval (we'll improve this!)"""
        scored_docs = []

        for doc in self.documents:
            # Simple scoring: count query words in document
            score = sum(1 for word in query.lower().split()
                       if word in doc.lower())
            scored_docs.append((score, doc))

        # Sort by score and return top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:top_k]]

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Simulate LLM (in reality, call OpenAI/etc)"""
        # In real implementation, this calls an actual LLM
        return f"Based on the context: {' '.join(context)}, the answer to '{query}' is..."

    def answer(self, query: str) -> str:
        """Main RAG pipeline"""
        # 1. Retrieve relevant documents
        relevant_docs = self.retrieve(query, top_k=2)

        # 2. Generate answer using retrieved context
        answer = self.generate_answer(query, relevant_docs)

        return answer


# Usage
docs = [
    "The company allows remote work 3 days per week.",
    "All employees must use VPN when working remotely.",
    "Office hours are 9 AM to 5 PM.",
    "Remote workers must attend weekly team meetings.",
]

rag = SimpleRAG(docs)
response = rag.answer("What is the remote work policy?")
print(response)
```

This is overly simplified (keyword search instead of semantic search), but illustrates the RAG flow. We'll build much better versions!

## Summary

**RAG = Retrieval + Generation**

- Retrieve relevant documents for a query
- Use those documents as context for an LLM
- Generate accurate, grounded responses
- Cite sources for transparency

RAG is the bridge between your private knowledge and powerful language models.

---

## Next Up

Now that you understand what RAG is, let's dive into the architecture:

**[→ RAG Architecture](./02-architecture.md)**

You'll learn:
- Detailed system components
- Data flow
- Design patterns
- Architecture decisions
