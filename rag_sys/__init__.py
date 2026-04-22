"""
RAG System — A from-scratch Retrieval-Augmented Generation implementation.


Public API:
  from rag_sys import RAGPipeline, RAGConfig

Components (import individually for learning/customization):
  from rag_sys.ingestion import DocumentLoader, Document
  from rag_sys.chunker import TextChunker, Chunk
  from rag_sys.embedder import Embedder
  from rag_sys.vector_store import VectorStore
  from rag_sys.retriever import Retriever, RAGEvaluator
  from rag_sys.generator import AnthropicGenerator, OpenAIGenerator, OllamaGenerator
  from rag_sys.pipeline import RAGPipeline, RAGConfig
"""

from .pipeline import RAGPipeline, RAGConfig

__all__ = ["RAGPipeline", "RAGConfig"]