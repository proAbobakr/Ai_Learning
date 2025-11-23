"""
Search Engine - Core search functionality with hybrid search support.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import logging
import time

from .embedding_service import EmbeddingService
from .reranker import ReRanker

logger = logging.getLogger(__name__)


class SearchEngine:
    """Hybrid search engine combining vector and keyword search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
        use_reranking: bool = True,
        rerank_model: Optional[str] = None
    ):
        """
        Initialize search engine.

        Args:
            embedding_service: Service for generating embeddings
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the collection
            use_reranking: Whether to use re-ranking
            rerank_model: Model for re-ranking (optional)
        """
        self.embedding_service = embedding_service
        self.collection_name = collection_name

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # BM25 index for keyword search
        self.bm25_index = None
        self.document_texts = []
        self.document_ids = []

        # Re-ranker
        self.use_reranking = use_reranking
        if use_reranking:
            self.reranker = ReRanker(rerank_model)
        else:
            self.reranker = None

        # Build BM25 index from existing documents
        self._rebuild_bm25_index()

        logger.info(f"Search engine initialized with {self.collection.count()} documents")

    def _rebuild_bm25_index(self):
        """Rebuild the BM25 index from ChromaDB."""
        if self.collection.count() == 0:
            self.bm25_index = None
            self.document_texts = []
            self.document_ids = []
            return

        # Get all documents
        results = self.collection.get(include=["documents", "metadatas"])

        self.document_ids = results['ids']
        self.document_texts = results['documents']

        # Tokenize for BM25
        tokenized = [doc.lower().split() for doc in self.document_texts]
        self.bm25_index = BM25Okapi(tokenized)

        logger.info(f"BM25 index rebuilt with {len(self.document_texts)} documents")

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the search engine.

        Args:
            documents: List of document dictionaries with 'id', 'content', and metadata
            batch_size: Batch size for embedding generation

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        # Prepare data
        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = []

        for doc in documents:
            meta = {
                'title': doc.get('title', ''),
                'source': doc.get('source', ''),
                'doc_type': doc.get('doc_type', 'text'),
                'chunk_index': doc.get('chunk_index', 0),
                'total_chunks': doc.get('total_chunks', 1),
                'created_at': doc.get('created_at', ''),
            }
            # Add custom metadata
            if 'metadata' in doc and isinstance(doc['metadata'], dict):
                for k, v in doc['metadata'].items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
            metadatas.append(meta)

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(contents)} documents...")
        embeddings = self.embedding_service.encode(
            contents,
            batch_size=batch_size,
            show_progress=True
        )

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        # Rebuild BM25 index
        self._rebuild_bm25_index()

        logger.info(f"Added {len(documents)} documents to search engine")
        return len(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        alpha: float = 0.7,
        use_rerank: bool = True,
        rerank_top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Search for documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (ChromaDB where clause)
            use_hybrid: Whether to use hybrid search
            alpha: Balance between vector (1.0) and keyword (0.0) search
            use_rerank: Whether to re-rank results
            rerank_top_n: Number of results to re-rank

        Returns:
            Search results dictionary
        """
        start_time = time.time()

        # Determine search type
        if not use_hybrid or self.bm25_index is None:
            search_type = "vector"
            results = self._vector_search(query, top_k * 2, filters)
        else:
            search_type = "hybrid"
            results = self._hybrid_search(query, top_k * 2, filters, alpha)

        # Re-rank if enabled
        if use_rerank and self.reranker and len(results) > 0:
            search_type += "+rerank"
            results = self._rerank_results(query, results, rerank_top_n)

        # Limit to top_k
        results = results[:top_k]

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "search_time_ms": round(elapsed_ms, 2),
            "search_type": search_type
        }

    def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        query_embedding = self.embedding_service.encode(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        for i in range(len(results['ids'][0])):
            # ChromaDB returns distances, convert to similarity
            distance = results['distances'][0][i]
            similarity = 1 - distance  # For cosine distance

            formatted_results.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "title": results['metadatas'][0][i].get('title'),
                "score": similarity,
                "vector_score": similarity,
                "keyword_score": None,
                "rerank_score": None,
                "metadata": results['metadatas'][0][i]
            })

        return formatted_results

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform hybrid vector + keyword search."""
        # Vector search
        vector_results = self._vector_search(query, top_k, filters)

        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores = bm25_scores / max_bm25

        # Create score mapping
        id_to_bm25 = {
            doc_id: bm25_scores[i]
            for i, doc_id in enumerate(self.document_ids)
        }

        # Combine scores
        for result in vector_results:
            bm25_score = id_to_bm25.get(result['id'], 0)
            vector_score = result['vector_score']

            # Hybrid score
            hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score

            result['keyword_score'] = bm25_score
            result['score'] = hybrid_score

        # Sort by hybrid score
        vector_results.sort(key=lambda x: x['score'], reverse=True)

        return vector_results

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """Re-rank results using cross-encoder."""
        if not results:
            return results

        # Take top N for re-ranking
        to_rerank = results[:top_n]
        documents = [r['content'] for r in to_rerank]

        # Re-rank
        reranked = self.reranker.rerank(query, documents)

        # Update results with re-rank scores
        for (doc, score), result in zip(reranked, to_rerank):
            result['rerank_score'] = score
            result['score'] = score  # Use re-rank score as final score

        # Sort by re-rank score
        to_rerank.sort(key=lambda x: x['score'], reverse=True)

        # Append remaining results
        remaining = results[top_n:]
        return to_rerank + remaining

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.collection.delete(ids=[doc_id])
            self._rebuild_bm25_index()
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            "total_documents": self.collection.count(),
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.dimension,
            "use_reranking": self.use_reranking,
            "bm25_indexed": self.bm25_index is not None
        }

    def clear(self):
        """Clear all documents."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._rebuild_bm25_index()
        logger.info("Search engine cleared")
