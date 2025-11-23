"""
Re-ranker Service - Cross-encoder based re-ranking for improved search results.
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ReRanker:
    """Re-rank search results using a cross-encoder model."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize re-ranker.

        Args:
            model_name: Cross-encoder model name. Default: ms-marco-MiniLM-L-6-v2
        """
        if model_name is None:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        logger.info(f"Loading re-ranker model: {model_name}")
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        logger.info("Re-ranker loaded successfully")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to re-rank
            top_k: Return only top k results (None = all)

        Returns:
            List of (document, score) tuples sorted by score descending
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get scores from cross-encoder
        scores = self.model.predict(pairs)

        # Combine documents with scores
        scored_docs = list(zip(documents, scores.tolist()))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        return scored_docs

    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.

        Args:
            query: Query text
            document: Document text

        Returns:
            Relevance score
        """
        score = self.model.predict([[query, document]])
        return float(score[0])

    def get_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "type": "cross-encoder"
        }
