"""
Tests for the Document Search Engine.

Run with: pytest tests/test_search.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.document_processor import DocumentProcessor
from services.reranker import ReRanker
from utils.chunking import (
    split_by_sentences,
    sliding_window_chunks,
    semantic_chunks
)


class TestEmbeddingService:
    """Tests for embedding service."""

    @pytest.fixture
    def embedding_service(self):
        return EmbeddingService("all-MiniLM-L6-v2")

    def test_encode_single(self, embedding_service):
        """Test encoding a single text."""
        text = "Hello, world!"
        embedding = embedding_service.encode(text)

        assert embedding.shape == (1, 384)
        assert np.linalg.norm(embedding) > 0  # Non-zero

    def test_encode_batch(self, embedding_service):
        """Test encoding multiple texts."""
        texts = ["Hello", "World", "Test"]
        embeddings = embedding_service.encode(texts)

        assert embeddings.shape == (3, 384)

    def test_similarity(self, embedding_service):
        """Test similarity calculation."""
        sim_same = embedding_service.similarity(
            "I love programming",
            "I enjoy coding"
        )
        sim_diff = embedding_service.similarity(
            "I love programming",
            "The weather is nice"
        )

        assert sim_same > sim_diff
        assert -1 <= sim_same <= 1
        assert -1 <= sim_diff <= 1

    def test_find_similar(self, embedding_service):
        """Test finding similar texts."""
        query = "machine learning"
        candidates = [
            "artificial intelligence",
            "deep learning",
            "cooking recipes",
            "neural networks",
            "gardening tips"
        ]

        results = embedding_service.find_similar(query, candidates, top_k=3)

        assert len(results) == 3
        # AI/ML related should be more similar
        assert "cooking recipes" not in [r[1] for r in results]


class TestDocumentProcessor:
    """Tests for document processor."""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(chunk_size=200, chunk_overlap=20)

    def test_clean_text(self, processor):
        """Test text cleaning."""
        dirty = "Hello   world\n\n\ntest  "
        clean = processor.clean_text(dirty)

        assert "   " not in clean
        assert clean == "Hello world test"

    def test_chunk_short_text(self, processor):
        """Test chunking short text."""
        short_text = "This is a short text."
        chunks = processor.chunk_text(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_long_text(self, processor):
        """Test chunking long text."""
        # Create text longer than chunk_size
        long_text = "This is a sentence. " * 50
        chunks = processor.chunk_text(long_text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= processor.chunk_size * 1.2  # Allow some slack

    def test_process_document(self, processor):
        """Test full document processing."""
        content = "First paragraph with some content. " * 20
        result = processor.process_document(
            content=content,
            title="Test Doc",
            source="test.txt",
            metadata={"category": "test"}
        )

        assert len(result) > 0
        assert all('id' in chunk for chunk in result)
        assert all('content' in chunk for chunk in result)
        assert result[0]['title'] == "Test Doc"
        assert result[0]['total_chunks'] == len(result)


class TestReRanker:
    """Tests for re-ranker."""

    @pytest.fixture
    def reranker(self):
        return ReRanker()

    def test_rerank(self, reranker):
        """Test re-ranking documents."""
        query = "What is Python programming?"
        documents = [
            "Python is a programming language",
            "Snakes live in forests",
            "Python was created by Guido van Rossum",
            "The weather is nice today"
        ]

        results = reranker.rerank(query, documents)

        assert len(results) == 4
        # Python programming docs should rank higher
        top_docs = [r[0] for r in results[:2]]
        assert "Snakes live in forests" not in top_docs
        assert "The weather is nice today" not in top_docs

    def test_score_pair(self, reranker):
        """Test scoring a single pair."""
        score_high = reranker.score_pair(
            "What is machine learning?",
            "Machine learning is a type of artificial intelligence"
        )
        score_low = reranker.score_pair(
            "What is machine learning?",
            "I like pizza"
        )

        assert score_high > score_low


class TestChunkingUtils:
    """Tests for chunking utilities."""

    def test_split_by_sentences(self):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = split_by_sentences(text)

        assert len(sentences) == 3

    def test_sliding_window(self):
        """Test sliding window chunking."""
        text = "A" * 1000  # 1000 character text
        chunks = sliding_window_chunks(text, window_size=200, step_size=100)

        assert len(chunks) > 1
        # Verify overlap
        assert chunks[0][2] > chunks[1][1]  # end of first > start of second

    def test_semantic_chunks(self):
        """Test semantic chunking."""
        text = """
        # Introduction

        This is the first paragraph with some content.

        This is the second paragraph with more content.

        # Another Section

        This section has different content.
        """
        chunks = semantic_chunks(text, max_chunk_size=200)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 200


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test the full search pipeline."""
        # Initialize services
        embedding_service = EmbeddingService("all-MiniLM-L6-v2")
        processor = DocumentProcessor(chunk_size=500)

        # Process a document
        content = """
        Python is a high-level programming language known for its simplicity.
        It was created by Guido van Rossum and released in 1991.
        Python is widely used in web development, data science, and AI.
        """
        chunks = processor.process_document(
            content=content,
            title="Python Overview"
        )

        # Generate embeddings
        chunk_texts = [c['content'] for c in chunks]
        embeddings = embedding_service.encode(chunk_texts)

        # Search
        query = "programming language for beginners"
        query_emb = embedding_service.encode(query)

        # Calculate similarity
        similarities = np.dot(embeddings, query_emb.T).flatten()

        # Verify results make sense
        assert len(similarities) == len(chunks)
        assert np.max(similarities) > 0  # Some similarity expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
