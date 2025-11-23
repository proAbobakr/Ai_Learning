"""
Smart Document Search Engine - FastAPI Application

A production-ready semantic search engine with hybrid search and re-ranking.

Run with: python main.py
API docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
from typing import Optional, List

from config import settings
from models.schemas import (
    DocumentCreate,
    DocumentResponse,
    SearchQuery,
    SearchResponse,
    SearchResult,
    HealthResponse,
    StatsResponse
)
from services.embedding_service import EmbeddingService
from services.document_processor import DocumentProcessor
from services.search_engine import SearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
embedding_service: EmbeddingService = None
document_processor: DocumentProcessor = None
search_engine: SearchEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global embedding_service, document_processor, search_engine

    logger.info("Initializing services...")

    # Initialize embedding service
    embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)

    # Initialize document processor
    document_processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    # Initialize search engine
    search_engine = SearchEngine(
        embedding_service=embedding_service,
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
        use_reranking=settings.USE_RERANKING,
        rerank_model=settings.RERANK_MODEL if settings.USE_RERANKING else None
    )

    logger.info("Services initialized successfully!")

    yield

    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="A semantic search engine powered by embeddings",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint."""
    stats = search_engine.get_stats()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        embedding_model=stats["embedding_model"],
        total_documents=stats["total_documents"],
        total_chunks=stats["total_documents"]
    )


@app.get("/stats", response_model=StatsResponse, tags=["Info"])
async def get_stats():
    """Get detailed statistics."""
    stats = search_engine.get_stats()
    return StatsResponse(
        total_documents=stats["total_documents"],
        total_chunks=stats["total_documents"],
        avg_chunks_per_doc=1.0,  # Would need to track this
        embedding_model=stats["embedding_model"],
        embedding_dimension=stats["embedding_dimension"],
        index_size_mb=0.0  # Would need to calculate
    )


# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.post("/documents", tags=["Documents"])
async def add_document(document: DocumentCreate):
    """
    Add a new document to the search engine.

    The document will be automatically chunked and indexed.
    """
    try:
        # Process document into chunks
        chunks = document_processor.process_document(
            content=document.content,
            title=document.title,
            source=document.source,
            doc_type=document.doc_type.value,
            metadata=document.metadata
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Document produced no indexable chunks"
            )

        # Add to search engine
        num_added = search_engine.add_documents(chunks)

        return {
            "status": "success",
            "message": f"Added {num_added} chunks",
            "document_id": chunks[0]['id'].split('_')[0],
            "chunks": num_added
        }

    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/batch", tags=["Documents"])
async def add_documents_batch(documents: List[DocumentCreate]):
    """Add multiple documents in batch."""
    total_chunks = 0

    for doc in documents:
        chunks = document_processor.process_document(
            content=doc.content,
            title=doc.title,
            source=doc.source,
            doc_type=doc.doc_type.value,
            metadata=doc.metadata
        )
        if chunks:
            total_chunks += search_engine.add_documents(chunks)

    return {
        "status": "success",
        "documents_processed": len(documents),
        "total_chunks": total_chunks
    }


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document by ID."""
    success = search_engine.delete_document(doc_id)
    if success:
        return {"status": "success", "deleted": doc_id}
    else:
        raise HTTPException(status_code=404, detail="Document not found")


@app.delete("/documents", tags=["Documents"])
async def clear_all_documents():
    """Clear all documents from the search engine."""
    search_engine.clear()
    return {"status": "success", "message": "All documents cleared"}


# ============================================================================
# Search Endpoints
# ============================================================================

@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
    use_hybrid: bool = Query(True, description="Use hybrid search"),
    use_rerank: bool = Query(True, description="Use re-ranking"),
    alpha: float = Query(0.7, ge=0, le=1, description="Hybrid alpha (0=keyword, 1=vector)")
):
    """
    Search documents (GET method).

    Simple search endpoint for quick queries.
    """
    results = search_engine.search(
        query=query,
        top_k=top_k,
        use_hybrid=use_hybrid,
        alpha=alpha,
        use_rerank=use_rerank and settings.USE_RERANKING
    )

    return SearchResponse(
        query=results["query"],
        total_results=results["total_results"],
        results=[
            SearchResult(
                id=r["id"],
                content=r["content"],
                title=r.get("title"),
                score=r["score"],
                vector_score=r.get("vector_score"),
                keyword_score=r.get("keyword_score"),
                rerank_score=r.get("rerank_score"),
                metadata=r.get("metadata", {})
            )
            for r in results["results"]
        ],
        search_time_ms=results["search_time_ms"],
        search_type=results["search_type"]
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_post(search_query: SearchQuery):
    """
    Search documents (POST method).

    Advanced search with filters and all options.
    """
    results = search_engine.search(
        query=search_query.query,
        top_k=search_query.top_k,
        filters=search_query.filters,
        use_hybrid=search_query.use_hybrid,
        alpha=search_query.alpha,
        use_rerank=search_query.use_rerank and settings.USE_RERANKING
    )

    return SearchResponse(
        query=results["query"],
        total_results=results["total_results"],
        results=[
            SearchResult(
                id=r["id"],
                content=r["content"],
                title=r.get("title"),
                score=r["score"],
                vector_score=r.get("vector_score"),
                keyword_score=r.get("keyword_score"),
                rerank_score=r.get("rerank_score"),
                metadata=r.get("metadata", {})
            )
            for r in results["results"]
        ],
        search_time_ms=results["search_time_ms"],
        search_type=results["search_type"]
    )


@app.get("/similar", tags=["Search"])
async def find_similar(
    text: str = Query(..., description="Text to find similar documents for"),
    top_k: int = Query(5, ge=1, le=50)
):
    """Find documents similar to the given text."""
    results = search_engine.search(
        query=text,
        top_k=top_k,
        use_hybrid=False,  # Pure vector search for similarity
        use_rerank=False
    )
    return results


# ============================================================================
# Embedding Endpoints
# ============================================================================

@app.post("/embed", tags=["Embeddings"])
async def get_embedding(texts: List[str]):
    """
    Get embeddings for texts.

    Useful for debugging or external use.
    """
    if len(texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 texts per request"
        )

    embeddings = embedding_service.encode(texts)
    return {
        "embeddings": embeddings.tolist(),
        "dimension": embedding_service.dimension,
        "model": embedding_service.model_name
    }


@app.post("/similarity", tags=["Embeddings"])
async def calculate_similarity(text1: str, text2: str):
    """Calculate similarity between two texts."""
    similarity = embedding_service.similarity(text1, text2)
    return {
        "text1": text1,
        "text2": text2,
        "similarity": similarity
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
