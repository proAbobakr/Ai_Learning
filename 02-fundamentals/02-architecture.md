# RAG Architecture

## Overview

A RAG system consists of two main phases: **Indexing** (preparation) and **Retrieval & Generation** (query time). Let's explore the complete architecture.

## Two-Phase Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: INDEXING                        │
│                    (Done Once/Periodically)                 │
│                                                             │
│  Documents → Chunking → Embeddings → Vector Store          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                PHASE 2: RETRIEVAL & GENERATION              │
│                    (Every User Query)                       │
│                                                             │
│  Query → Embed → Search Vector Store → Retrieve Chunks     │
│          → Augment Prompt → LLM → Response                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Indexing (Offline)

This happens before users start asking questions. You prepare your knowledge base.

### Step 1: Document Loading

```python
from langchain.document_loaders import PyPDFLoader, TextLoader

# Load various document types
pdf_loader = PyPDFLoader("company_handbook.pdf")
text_loader = TextLoader("policies.txt")

documents = pdf_loader.load() + text_loader.load()
print(f"Loaded {len(documents)} documents")
```

**Document Object Structure:**
```python
{
    "page_content": "The actual text content...",
    "metadata": {
        "source": "handbook.pdf",
        "page": 1,
        "author": "HR Department"
    }
}
```

### Step 2: Text Chunking

Break documents into smaller pieces (chunks) that fit in LLM context windows.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # characters per chunk
    chunk_overlap=50,      # overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""]  # split hierarchy
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
```

**Why Chunking?**
- LLMs have context limits (4K-128K tokens)
- Smaller chunks = more precise retrieval
- Chunks must be self-contained enough to be useful

**Example:**
```
Original: "...John signed the contract. The payment terms are 30 days..."

Chunk 1: "...John signed the contract."
Chunk 2: "The payment terms are 30 days..." ✗ Missing context!

Better with overlap:
Chunk 1: "...John signed the contract. The payment terms..."
Chunk 2: "...signed the contract. The payment terms are 30 days..."
```

### Step 3: Embedding Generation

Convert text chunks into numerical vectors (embeddings).

```python
from langchain.embeddings import OpenAIEmbeddings

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Embed a single chunk
text = "Remote work is allowed 3 days per week"
vector = embeddings.embed_query(text)

print(f"Embedding dimensions: {len(vector)}")  # 1536 for ada-002
print(f"First 5 values: {vector[:5]}")
# Output: [0.123, -0.456, 0.789, -0.234, 0.567]
```

**What are Embeddings?**
- Numerical representation of text meaning
- Similar meanings → similar vectors
- Enables semantic search (meaning-based, not keyword-based)

**Popular Embedding Models:**
- OpenAI `text-embedding-ada-002`: 1536 dimensions, $0.0001/1K tokens
- OpenAI `text-embedding-3-small`: 1536 dimensions, cheaper
- Sentence Transformers (free, local): 384-768 dimensions
- Cohere, Anthropic, etc.

### Step 4: Vector Store

Store embeddings in a database optimized for similarity search.

```python
from langchain.vectorstores import Chroma

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Stored {vectorstore._collection.count()} vectors")
```

**Vector Database Options:**
- **Chroma**: Simple, local, good for dev
- **Pinecone**: Managed, scalable, production-ready
- **Weaviate**: Open-source, feature-rich
- **FAISS**: Facebook's library, very fast, local
- **Qdrant**: Fast, open-source
- **Milvus**: Enterprise-grade

---

## Phase 2: Retrieval & Generation (Online)

This happens every time a user asks a question.

### Step 1: Query Embedding

```python
# User asks a question
user_query = "Can I work remotely?"

# Convert query to vector
query_vector = embeddings.embed_query(user_query)
```

### Step 2: Similarity Search

Find chunks with embeddings similar to the query embedding.

```python
# Search vector store
results = vectorstore.similarity_search(
    query=user_query,
    k=4  # retrieve top 4 most similar chunks
)

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Source: {doc.metadata['source']}")
    print(f"Similarity score: {doc.metadata.get('score', 'N/A')}")
    print()
```

**How Similarity Search Works:**
```
Query: "Can I work remotely?"
Vector: [0.1, 0.8, -0.3, ...]

Database:
- Chunk A: [0.11, 0.79, -0.28, ...]  → Cosine similarity: 0.95 ✓
- Chunk B: [-0.5, 0.2, 0.6, ...]     → Cosine similarity: 0.45
- Chunk C: [0.09, 0.82, -0.31, ...]  → Cosine similarity: 0.98 ✓

Returns: Chunks C, A (top 2)
```

### Step 3: Prompt Augmentation

Create a prompt that includes retrieved context.

```python
def create_rag_prompt(query: str, context_docs: List[Document]) -> str:
    """Build prompt with retrieved context"""
    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""You are a helpful assistant. Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer: Provide a clear answer based only on the context above. If the answer is not in the context, say so."""

    return prompt
```

### Step 4: LLM Generation

Send augmented prompt to LLM.

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)  # 0 = deterministic

# Create prompt with context
prompt = create_rag_prompt(user_query, results)

# Generate answer
answer = llm(prompt)

print(f"Question: {user_query}")
print(f"Answer: {answer}")
```

### Step 5: Response

Return answer to user, optionally with citations.

```python
response = {
    "answer": answer,
    "sources": [
        {
            "content": doc.page_content,
            "source": doc.metadata["source"],
            "page": doc.metadata.get("page")
        }
        for doc in results
    ]
}
```

---

## Complete RAG Pipeline

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1. Setup (done once)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
llm = OpenAI(temperature=0)

# 2. Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" = put all context in one prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 3. Query (happens many times)
query = "What is the vacation policy?"
answer = qa_chain.run(query)
print(answer)
```

---

## Architecture Patterns

### 1. Stuff Pattern (Simple)

Put all retrieved documents directly into prompt.

```
Prompt = System Message + Context Docs + User Query
Send to LLM → Get Answer
```

**Pros**: Simple, works well
**Cons**: Limited by context window
**When**: Few documents, short chunks

### 2. Map-Reduce Pattern

Process each document separately, then combine.

```
For each doc:
    Answer = LLM(doc + query)

Final Answer = LLM(combine all answers)
```

**Pros**: Handles many documents
**Cons**: Slower (multiple LLM calls), can lose context
**When**: Many retrieved documents

### 3. Refine Pattern

Iteratively refine answer with each document.

```
Answer = LLM(doc1 + query)
Answer = LLM(doc2 + query + previous_answer)  # refine
Answer = LLM(doc3 + query + previous_answer)  # refine again
```

**Pros**: Incorporates all docs, maintains context
**Cons**: Slowest, order-dependent
**When**: Need comprehensive answer from many docs

### 4. Map-Rerank Pattern

Score each answer and pick the best.

```
For each doc:
    (answer, score) = LLM(doc + query + "rate your confidence")

Return highest-scored answer
```

**Pros**: Good when one doc has full answer
**Cons**: Wastes computation on wrong docs
**When**: Answer in one specific document

---

## Data Flow Diagram

```
USER QUESTION
     ↓
┌────────────────────┐
│ Query Understanding│
│ & Reformulation    │ (Optional: rephrase for better retrieval)
└────────────────────┘
     ↓
┌────────────────────┐
│ Embedding Model    │
│ Query → Vector     │
└────────────────────┘
     ↓
┌────────────────────┐
│ Vector Store       │
│ Similarity Search  │
│ (ANN Search)       │
└────────────────────┘
     ↓
  Retrieved Chunks
     ↓
┌────────────────────┐
│ Re-ranking         │ (Optional: reorder by relevance)
└────────────────────┘
     ↓
┌────────────────────┐
│ Prompt Construction│
│ Context + Query    │
└────────────────────┘
     ↓
┌────────────────────┐
│ LLM Generation     │
└────────────────────┘
     ↓
┌────────────────────┐
│ Post-processing    │ (Optional: fact-checking, citation formatting)
└────────────────────┘
     ↓
   ANSWER
```

---

## Key Design Decisions

### 1. Chunk Size
- **Small (100-300 tokens)**: Precise, but may lack context
- **Medium (300-500 tokens)**: Balanced (most common)
- **Large (500-1000 tokens)**: More context, less precise

### 2. Number of Retrieved Chunks (k)
- **k=1-3**: Fast, but might miss information
- **k=3-5**: Good balance (recommended start)
- **k=5-10**: Comprehensive, but slower and more expensive

### 3. Embedding Model
- **OpenAI**: Best quality, costs money
- **Sentence Transformers**: Free, good quality, run locally
- **Domain-specific**: Fine-tuned for your domain

### 4. Vector Store
- **Development**: Chroma, FAISS (local)
- **Production**: Pinecone, Weaviate (managed)
- **Scale**: Consider data size, query volume, budget

### 5. LLM Choice
- **GPT-4**: Best quality, expensive
- **GPT-3.5**: Good balance
- **Open-source** (LLaMA, Mistral): Free, self-hosted

---

## System Requirements

### Compute
- **Embedding**: Light (unless running model locally)
- **Vector Search**: Fast (ANN algorithms)
- **LLM**: Heavy if self-hosted, light if API

### Storage
- **Documents**: Original files
- **Vectors**: N chunks × embedding dimension × 4 bytes
  - Example: 10,000 chunks × 1536 dims × 4 bytes = ~60MB
- **Metadata**: Minimal

### Memory
- **Development**: 4-8GB RAM sufficient
- **Production**: Depends on vector store and scale

---

## Monitoring Points

Track these at each stage:

1. **Indexing**:
   - Documents processed
   - Chunks created
   - Embeddings generated
   - Vector store size

2. **Retrieval**:
   - Query latency
   - Retrieval quality (precision/recall)
   - Number of queries per second

3. **Generation**:
   - LLM latency
   - Token usage (cost)
   - Answer quality

4. **End-to-End**:
   - Total latency
   - User satisfaction
   - Failure rates

---

## Summary

**Indexing Flow:**
Documents → Chunks → Embeddings → Vector Store

**Query Flow:**
Query → Embedding → Search → Retrieve → Augment Prompt → LLM → Answer

**Key Components:**
- Document Loaders
- Text Splitters
- Embedding Models
- Vector Stores
- LLMs
- Orchestration (LangChain, LlamaIndex, etc.)

---

## Next Steps

Now that you understand the architecture, let's explore each component in detail:

**[→ Key Components](./03-components.md)**

You'll learn:
- Deep dive into each component
- Implementation options
- Best practices
- Common pitfalls
