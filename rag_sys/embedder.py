"""
rag/embedder.py
---------------
COMPONENT 3: Generating Embeddings

What it does:
  Converts text (chunks or queries) into dense numerical vectors
  (embeddings) that capture semantic meaning.

Why sentence-transformers?
  - Free, open-source, runs locally (no API key needed)
  - High quality: trained on hundreds of millions of sentence pairs
  - Fast inference: can embed thousands of chunks per second on CPU
  - Many model choices with different size/quality trade-offs

Model comparison (choose at init time):
  ┌──────────────────────────────────────┬──────┬────────┬──────────┐
  │ Model                                │ Dims │ Size   │ Quality  │
  ├──────────────────────────────────────┼──────┼────────┼──────────┤
  │ all-MiniLM-L6-v2 (default)          │ 384  │ 80MB   │ Good     │
  │ all-mpnet-base-v2                    │ 768  │ 420MB  │ Better   │
  │ multi-qa-MiniLM-L6-cos-v1           │ 384  │ 80MB   │ Good*    │
  │ BAAI/bge-small-en-v1.5              │ 384  │ 130MB  │ Great    │
  └──────────────────────────────────────┴──────┴────────┴──────────┘
  * multi-qa models are trained specifically for Q&A retrieval tasks.
    They often outperform general-purpose models for RAG.

Critical insight — SAME model for docs AND queries:
  The query "What is attention?" and the chunk about attention must live in
  the SAME vector space for similarity to be meaningful. Always use the
  same model to embed everything.
"""

from pydoc import text

import numpy as np
from typing import Union
from .chunker import Chunk


class Embedder:
    """
    Wraps a sentence-transformers model to provide consistent embed()
    and embed_batch() methods used by both ingestion and retrieval.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model name. First call downloads the model;
            subsequent calls load from the local cache (~/.cache/huggingface/).
        """
        self.model_name = model_name
        self._model = None  # Lazy load: don't download until first use
        print(f"Embedder initialized (model will load on first use: {model_name})")

    @property
    def model(self):
        """Lazy-load the model on first access."""
        if self._model is None:
            print(f"  Loading embedding model '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            print(f"  Model loaded. Embedding dimension: {self.embedding_dim}")
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Returns the dimensionality of the embedding vectors."""
        # Access internal config to avoid triggering a dummy embed
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single string into a dense vector.

        Returns a numpy array of shape (embedding_dim,).
        This is used for embedding the USER QUERY at retrieval time.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        # normalize_embeddings=True scales to unit length,
        # making cosine similarity equivalent to dot product (faster).
        vector = self.model.encode(text, normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of strings efficiently.

        Batching is critical for performance: embedding 1000 texts as
        individual calls is ~10-30x slower than a single batched call
        because the GPU/CPU can parallelize within a batch.

        Returns numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        # Filter out empty strings (they cause errors in some models)
        non_empty = [t if t.strip() else " " for t in texts]

        vectors = self.model.encode(
            non_empty,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,  # Show progress for large batches
        )
        return np.array(vectors, dtype=np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> tuple[list[str], np.ndarray]:
        """
        Embed a list of Chunk objects.

        Returns
        -------
        ids     : list of chunk_id strings (used as vector DB record IDs)
        vectors : numpy array of shape (len(chunks), embedding_dim)
        """
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        print(f"  Embedding {len(chunks)} chunks (model: {self.model_name})...")
        vectors = self.embed_batch(texts)
        print(f"  Done. Vector shape: {vectors.shape}")
        return ids, vectors
