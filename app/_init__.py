"""
Kira Search Service - Microservice de recherche intelligente

Ce package fournit un service de recherche sémantique utilisant:
- Embeddings avec SentenceTransformers
- Index vectoriel en mémoire avec persistance
- Reranking avec CrossEncoder
- Intégration MongoDB avec change streams
- API REST FastAPI

Version: 0.2.0
"""

__version__ = "0.2.0"
__author__ = "Kira Team"

from .config import load_settings
from .embedding import Embedder
from .reranker import Reranker
from .indexer import BruteForceIndex
from .utils import doc_to_text, clean_query

__all__ = [
    "load_settings",
    "Embedder", 
    "Reranker",
    "BruteForceIndex",
    "doc_to_text",
    "clean_query"
]