"""
rag/hybrid_pipeline.py
----------------------
Extends RAGPipeline with hybrid search support.

Drop-in replacement: the API is identical to RAGPipeline.
Just swap the import and set use_hybrid=True.

    from rag.hybrid_pipeline import HybridRAGPipeline
    pipeline = HybridRAGPipeline()
    pipeline.ingest_directory("./docs")     # same as before
    result = pipeline.query("What is X?")   # same as before
"""

from curses import raw

from numpy.ma import count

from ..ingestion import DocumentLoader
from ..chunker import TextChunker, Chunk
from ..embedder import Embedder
from rag_sys.vector_store import VectorStore
from .hybrid_retriever import HybridRetriever, HybridResult
from ..generator import NvidiaGenerator, build_prompt, SYSTEM_PROMPT


class HybridRAGPipeline:
    """
    Full RAG pipeline with hybrid BM25 + vector retrieval.

    Key difference from RAGPipeline:
      - Maintains both a BM25 index (in-memory) and a vector store (ChromaDB)
      - After ingestion, automatically (re)builds the BM25 index
      - Retrieval uses HybridRetriever with RRF fusion
    """

    def __init__(
        self,
        # Chunking
        chunk_size: int = 300,
        chunk_overlap: int = 60,
        chunk_strategy: str = "sentence",
        # Embedding
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        # Vector store
        collection_name: str = "hybrid_rag",
        persist_dir: str = "./chroma_db_hybrid",
        # Retrieval
        bm25_fetch_k: int = 20,
        vector_fetch_k: int = 20,
        final_k: int = 5,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        vector_similarity_threshold: float = 0.30,  # post-RRF noise filter
        # Generation
        llm_model: str = "meta/llama-3.1-70b-instruct",
        max_tokens: int = 1024,
    ):
        print("Initializing Hybrid RAG Pipeline...")
        self.final_k = final_k
        self._all_chunks: list[Chunk] = []  # kept in memory for BM25

        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            strategy=chunk_strategy,
        )
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_fetch_k=bm25_fetch_k,
            vector_fetch_k=vector_fetch_k,
            final_k=final_k,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            vector_similarity_threshold=vector_similarity_threshold,
        )
        self.generator = NvidiaGenerator(
            model=llm_model, max_tokens=max_tokens
        )
        print("Pipeline ready.\n")
        self._restore_bm25_from_store()

    # ------------------------------------------------------------------ #
    # Ingestion                                                             #
    # ------------------------------------------------------------------ #

    def ingest_directory(self, directory: str) -> int:
        """Ingest all docs from a directory, then rebuild the BM25 index."""
        docs = self.loader.load_directory(directory)
        if not docs:
            return 0

        new_chunks = []
        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            _, embeddings = self.embedder.embed_chunks(chunks)
            self.vector_store.add(chunks, embeddings)
            new_chunks.extend(chunks)

        self._all_chunks.extend(new_chunks)
        self._rebuild_bm25()

        print(f"\nIngestion complete: {len(docs)} docs → {len(new_chunks)} chunks")
        return len(new_chunks)

    def ingest_file(self, path: str) -> int:
        """Ingest a single file, then rebuild the BM25 index."""
        doc = self.loader.load_file(path)
        chunks = self.chunker.chunk_document(doc)
        _, embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.add(chunks, embeddings)
        self._all_chunks.extend(chunks)
        self._rebuild_bm25()
        return len(chunks)

    def ingest_text(self, text: str, source_name: str = "user_text") -> int:
        """Ingest raw text directly."""
        doc = self.loader.load_text(text, source_name)
        chunks = self.chunker.chunk_document(doc)
        _, embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.add(chunks, embeddings)
        self._all_chunks.extend(chunks)
        self._rebuild_bm25()
        return len(chunks)

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from all accumulated chunks."""
        if self._all_chunks:
            self.retriever.build_bm25_index(self._all_chunks)

    # Restore BM25 index from vector store chunks (useful if BM25 index is lost but vector store is intact)
    def _restore_bm25_from_store(self) -> None:
        """Rebuilds the BM25 index from chunks already stored in ChromaDB.Called automatically on startup so queries work without re-ingesting."""
        count = self.vector_store.count()
        if count == 0:
            return 

        print(f"  Restoring BM25 index from {count} existing chunks in vector store...")

        # pulled stored documents and metadata from ChromaDB
        raw = self.vector_store._collection.get(
            include=["documents", "metadatas", "embeddings"]
        )

        from ..chunker import Chunk

        chunks = []
        for doc_id, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
            chunks.append(
                Chunk(
                    text=text,
                    metadata=meta,
                    chunk_id=doc_id,
                )
            )

        self._all_chunks = chunks
        self._rebuild_bm25()
        print(f"  BM25 index restored: {len(chunks)} chunks ready.")

    # ------------------------------------------------------------------ #
    # Querying                                                              #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        metadata_filter: dict = None,
        explain: bool = False,
    ) -> dict:
        """
        Answer a question using hybrid retrieval.

        Parameters
        ----------
        question : str
            The user's question.
        metadata_filter : dict, optional
            Filter applied to the vector search lane.
        explain : bool
            Print a per-result breakdown of BM25 rank, vector rank, RRF score.
            Essential for debugging and learning how fusion works.

        Returns
        -------
        dict with "answer", "sources" (list of HybridResult), "question"
        """
        print(f"\n[Hybrid Query] {question}")

        results: list[HybridResult] = self.retriever.retrieve(
            query=question,
            metadata_filter=metadata_filter,
            explain=explain,
        )

        if not results:
            return {
                "question": question,
                "answer": "No relevant information found.",
                "sources": [],
            }

        # Print retrieval summary
        print(f"  Retrieved {len(results)} chunks via hybrid search:")
        for r in results:
            in_bm25 = f"BM25#{r.bm25_rank}" if r.bm25_rank else "BM25:—"
            in_vec = f"Vec#{r.vector_rank}" if r.vector_rank else "Vec:—"
            print(
                f"    [{r.final_rank}] rrf={r.rrf_score:.5f} "
                f"| {in_bm25} {in_vec} | {r.source} | {r.text[:70]}..."
            )

        # Build context from HybridResults (same interface as SearchResult)
        from rag_sys.vector_store import SearchResult

        search_results_for_prompt = [
            SearchResult(
                chunk_id=r.chunk.chunk_id,
                text=r.text,
                score=r.rrf_score,
                metadata=r.chunk.metadata,
                rank=r.final_rank,
            )
            for r in results
        ]

        print("\n  Generating answer...")
        answer = self.generator.generate(question, search_results_for_prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": results,
        }

    def compare_retrieval_modes(self, question: str, k: int = 5) -> None:
        """
        Run BM25-only, vector-only, and hybrid retrieval on the same question
        and print a side-by-side comparison.

        Look for chunks that appear in hybrid but
        not in either individual list — that's RRF at work.
        """
        print(f"\n{'='*70}")
        print(f"Retrieval comparison for: '{question}'")
        print(f"{'='*70}")

        # BM25 only
        bm25_only = self.retriever.retrieve_bm25_only(question, k=k)
        print(f"\n[BM25 only — {len(bm25_only)} results]")
        for i, (chunk, score) in enumerate(bm25_only, 1):
            print(
                f"  [{i}] score={score:.3f} | {chunk.metadata.get('source')} | {chunk.text[:70]}..."
            )

        # Vector only
        vector_only = self.retriever.retrieve_vector_only(question, k=k)
        print(f"\n[Vector only — {len(vector_only)} results]")
        for r in vector_only:
            print(
                f"  [{r.rank}] cos={r.score:.3f} | {r.metadata.get('source')} | {r.text[:70]}..."
            )

        # Hybrid
        hybrid = self.retriever.retrieve(question, k=k, explain=True)
        print(f"\n[Hybrid RRF — {len(hybrid)} results]")
        for r in hybrid:
            in_bm25 = f"B#{r.bm25_rank}" if r.bm25_rank else "B:—"
            in_vec = f"V#{r.vector_rank}" if r.vector_rank else "V:—"
            print(
                f"  [{r.final_rank}] rrf={r.rrf_score:.5f} ({in_bm25} {in_vec}) "
                f"| {r.source} | {r.text[:60]}..."
            )

        print(f"\n{'='*70}")
