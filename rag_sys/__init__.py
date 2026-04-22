"""
RAG System — A from-scratch Retrieval-Augmented Generation implementation.

Public API:
  from rag import RAGPipeline, RAGConfig

Components (import individually for learning/customization):
  from rag.ingestion import DocumentLoader, Document
  from rag.chunker import TextChunker, Chunk
  from rag.embedder import Embedder
  from rag.vector_store import VectorStore
  from rag.retriever import Retriever, RAGEvaluator
  from rag.generator import AnthropicGenerator, OpenAIGenerator, OllamaGenerator
  from rag.pipeline import RAGPipeline, RAGConfig
"""

from .pipeline import RAGPipeline, RAGConfig

__all__ = ["RAGPipeline", "RAGConfig"]