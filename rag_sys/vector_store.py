"""
rag/vector_store.py
-------------------
COMPONENT 4 & 5: Vector Storage + Semantic Search / Similarity Retrieval

What it does:
  1. Stores chunk embeddings (vectors) along with their text and metadata
  2. Given a query vector, finds the most similar chunk vectors (retrieval)

How similarity search works:
  Every chunk is a point in a high-dimensional space (384 dimensions for
  all-MiniLM-L6-v2). When a query arrives, we embed it (same space) and
  measure the angle between the query vector and every chunk vector.

  Cosine similarity = (A · B) / (|A| × |B|)

  When vectors are unit-normalized (which we do in embed()), this simplifies
  to just the dot product: similarity = A · B

  Similarity ranges from -1 (opposite) to 1 (identical direction).
  In practice, similarity between semantically related texts is 0.3–0.9.

ChromaDB vs FAISS — which should you use?
  ┌──────────────┬─────────────────────────┬──────────────────────────┐
  │              │ ChromaDB                │ FAISS                    │
  ├──────────────┼─────────────────────────┼──────────────────────────┤
  │ API          │ High-level, easy        │ Low-level, manual        │
  │ Metadata     │ Built-in                │ Must manage separately   │
  │ Persistence  │ Built-in                │ Must serialize manually  │
  │ Speed        │ Good for <10M vectors   │ Best for very large scale│
  │ Use case     │ Learning, prototypes,   │ Production, research     │
  │              │ small-to-mid production │ >10M vectors             │
  └──────────────┴─────────────────────────┴──────────────────────────┘

  We use ChromaDB here for its friendlier API. Understanding the concepts
  transfers directly to FAISS or Pinecone or Weaviate.
"""

from dataclasses import dataclass
from typing import Optional
from .chunker import Chunk


@dataclass
class SearchResult:
    """A single retrieval result with similarity score and source info."""

    chunk_id: str
    text: str
    score: float  # Cosine similarity [−1, 1]; higher = more similar
    metadata: dict
    rank: int  # 1 = most similar


class VectorStore:
    """
    Wraps ChromaDB to provide a clean interface for:
      - add()    : store chunks and their embeddings
      - search() : find the k most similar chunks to a query vector
      - delete() : remove chunks by document source
    """

    def __init__(
        self, collection_name: str = "rag_collection", persist_dir: str = "./chroma_db"
    ):
        """
        Parameters
        ----------
        collection_name : str
            Name for the ChromaDB collection (like a table name in SQL).
        persist_dir : str
            Where ChromaDB saves its data to disk.
            Set to None to use an in-memory (non-persistent) store.
        """
        import chromadb

        if persist_dir:
            # PersistentClient saves data between sessions
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            # EphemeralClient exists only in memory (lost on exit)
            self._client = chromadb.EphemeralClient()

        # get_or_create_collection: idempotent — safe to call every startup
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # cosine distance = 1 - cosine_similarity
            # ChromaDB uses distances internally; we convert to similarity in search()
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"VectorStore ready: '{collection_name}' "
            f"({self._collection.count()} existing chunks)"
        )

    def add(self, chunks: list[Chunk], embeddings) -> None:
        """
        Upsert chunks and their precomputed embeddings into ChromaDB.

        ChromaDB stores three things per record:
          ids        : unique string ID for each chunk
          embeddings : the float32 vector
          documents  : the raw text (for retrieval)
          metadatas  : arbitrary metadata dict (for filtering)

        'Upsert' = insert if new, update if ID already exists.
        This means re-ingesting a document safely updates it in place.
        """
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        # Metadata values must be str/int/float/bool (ChromaDB limitation)
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks]
        emb_list = embeddings.tolist()  # ChromaDB expects Python lists, not numpy

        # Batch upsert in groups of 500 to avoid memory issues with huge corpora
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=emb_list[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        print(f"  Stored {len(chunks)} chunks in vector store")

    def search(
        self,
        query_embedding,
        k: int = 5,
        where: Optional[dict] = None,
        similarity_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Find the k most semantically similar chunks to the query vector.

        Parameters
        ----------
        query_embedding : np.ndarray
            The embedded query vector (shape: [embedding_dim]).
        k : int
            How many results to return. More = more context for the LLM,
            but also more noise. 3–7 is a good starting range.
        where : dict, optional
            Metadata filter using ChromaDB's filter syntax.
            Example: {"source": "transformers.txt"}
            Example: {"file_type": {"$in": ["txt", "pdf"]}}
            This is what makes metadata filtering possible — you can
            narrow the search space before doing similarity comparison.
        similarity_threshold : float
            Minimum cosine similarity [0, 1] a chunk must have to be
            included in the results. Defaults to 0.0 (disabled — return all k).
            When using hybrid search (BM25 + vector + RRF), keep this at 0.0
            so the full candidate pool is available for RRF fusion.
            Apply post-RRF filtering in the retriever layer instead.

        Returns
        -------
        List of SearchResult, ordered by similarity descending.
        Only results with similarity >= similarity_threshold are returned.
        """
        if self._collection.count() == 0:
            print("WARNING: Vector store is empty. Run ingestion first.")
            return []

        # Don't request more results than we have chunks
        k = min(k, self._collection.count())

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = self._collection.query(**query_params)

        # ChromaDB returns distances (lower = more similar for cosine space).
        # We convert: similarity = 1 - distance, then apply threshold filter.
        search_results = []
        for rank, (doc, meta, dist, rid) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ),
            start=1,
        ):
            similarity = round(1.0 - dist, 4)
            if similarity < similarity_threshold:
                # Skip low-quality matches — they hurt context_precision
                continue
            search_results.append(
                SearchResult(
                    chunk_id=rid,
                    text=doc,
                    score=similarity,
                    metadata=meta,
                    rank=rank,
                )
            )

        return search_results

    def delete_by_source(self, source_name: str) -> int:
        """
        Remove all chunks from a specific source document.
        Useful for re-ingesting an updated file.
        """
        results = self._collection.get(where={"source": source_name})
        ids_to_delete = results["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        print(f"  Deleted {len(ids_to_delete)} chunks from source '{source_name}'")
        return len(ids_to_delete)

    def count(self) -> int:
        """Return the total number of chunks stored."""
        return self._collection.count()

    def get_sources(self) -> list[str]:
        """Return a list of all unique source documents in the store."""
        results = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in results["metadatas"]}
        return sorted(sources)

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        """
        ChromaDB only accepts str, int, float, or bool metadata values.
        Convert anything else to a string to avoid insertion errors.
        """
        sanitized = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized
