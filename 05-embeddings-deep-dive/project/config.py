"""
Configuration settings for the Document Search Engine.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "Smart Document Search Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Embedding Model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Re-ranking Model (optional)
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_RERANKING: bool = True

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "documents"

    # Document Processing
    CHUNK_SIZE: int = 500  # characters per chunk
    CHUNK_OVERLAP: int = 50  # overlap between chunks
    MAX_DOCUMENT_SIZE: int = 10_000_000  # 10MB

    # Search
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 100
    HYBRID_ALPHA: float = 0.7  # 0 = keyword only, 1 = vector only
    RERANK_TOP_N: int = 20  # Re-rank top N results

    # OpenAI (optional)
    OPENAI_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Model configurations
MODEL_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_seq_length": 256,
        "description": "Fast, good quality"
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "max_seq_length": 384,
        "description": "Best quality, slower"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dimension": 384,
        "max_seq_length": 128,
        "description": "Multilingual support"
    }
}


def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["all-MiniLM-L6-v2"])
