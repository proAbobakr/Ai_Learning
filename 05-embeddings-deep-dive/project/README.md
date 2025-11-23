# Real-World Project: Intelligent Document Search Engine

A complete, production-ready semantic search engine that combines everything you've learned.

## Project Overview

Build a **Smart Document Search Engine** that can:
- Search documents by meaning (not just keywords)
- Handle multiple document types (PDF, TXT, Markdown)
- Support hybrid search (vector + keyword)
- Re-rank results for better accuracy
- Provide a REST API and web interface
- Scale to thousands of documents

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │   Upload    │    │   Search    │    │        Web UI               │ │
│  │  Documents  │    │   Query     │    │     (Streamlit)             │ │
│  └──────┬──────┘    └──────┬──────┘    └─────────────────────────────┘ │
│         │                  │                                            │
│         ▼                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Backend                            │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │   │
│  │  │  Document    │  │  Embedding   │  │   Search Engine      │  │   │
│  │  │  Processor   │  │  Service     │  │   (Hybrid + Rerank)  │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      ChromaDB                                   │   │
│  │              (Vector Store + Metadata)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| Semantic Search | Find documents by meaning |
| Hybrid Search | Combine vector + keyword search |
| Re-ranking | Cross-encoder for better results |
| Document Chunking | Smart splitting for long documents |
| Metadata Filtering | Filter by date, type, source |
| REST API | FastAPI with OpenAPI docs |
| Web Interface | Streamlit dashboard |
| Batch Processing | Efficient bulk document upload |

## Project Structure

```
project/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── config.py              # Configuration settings
├── main.py                # FastAPI application
├── streamlit_app.py       # Web interface
├── models/
│   ├── __init__.py
│   └── schemas.py         # Pydantic models
├── services/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── embedding_service.py
│   ├── search_engine.py
│   └── reranker.py
├── utils/
│   ├── __init__.py
│   └── chunking.py
├── data/                  # Sample documents
│   └── sample_docs/
└── tests/
    └── test_search.py
```

## Quick Start

### 1. Install Dependencies

```bash
cd project
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
python main.py
```

### 3. Open the Web Interface

```bash
streamlit run streamlit_app.py
```

### 4. Try the API

```bash
# Add a document
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{"content": "Python is a great programming language", "metadata": {"source": "manual"}}'

# Search
curl "http://localhost:8000/search?query=programming%20languages&top_k=5"
```

## Detailed Documentation

Continue reading the implementation files:
1. [requirements.txt](./requirements.txt) - Dependencies
2. [config.py](./config.py) - Configuration
3. [main.py](./main.py) - API Server
4. [services/](./services/) - Core services
5. [streamlit_app.py](./streamlit_app.py) - Web UI

## Learning Outcomes

By building this project, you will:
- Implement a complete embedding pipeline
- Build a production-ready search engine
- Create REST APIs with FastAPI
- Build web interfaces with Streamlit
- Handle document processing and chunking
- Implement hybrid search and re-ranking
- Use vector databases effectively
