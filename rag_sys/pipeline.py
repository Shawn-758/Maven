"""
rag/pipeline.py
---------------
THE ORCHESTRATOR: Connects all components into one coherent pipeline.

This module is the "glue code" that:
  1. Wires up ingestion → chunking → embedding → storage
  2. Wires up query → embedding → retrieval → prompt → generation

Think of it as the conductor of an orchestra: it doesn't play any
instruments itself, but coordinates all the players.

Usage pattern:
  pipeline = RAGPipeline(config)
  pipeline.ingest_directory("./my_docs")   # offline: one time
  answer = pipeline.query("What is X?")    # online: every user question
"""

from .ingestion import DocumentLoader
from .chunker import TextChunker
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import AnthropicGenerator
from .vector_store import SearchResult


class RAGConfig:
    """
    Central configuration for the entire RAG pipeline.

    Keeping configuration in one place means you can tune the whole
    system by changing values here, without touching module code.
    This also makes it easy to run A/B experiments.
    """

    def __init__(
        self,
        # Chunking
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        chunk_strategy: str = "sentence",

        # Embedding
        embedding_model: str = "all-MiniLM-L6-v2",

        # Vector store
        collection_name: str = "rag_collection",
        persist_dir: str = "./chroma_db",

        # Retrieval
        top_k: int = 5,
        use_reranker: bool = False,

        # Generation
        llm_model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.llm_model = llm_model
        self.max_tokens = max_tokens


class RAGPipeline:
    """
    End-to-end RAG system: ingest documents, answer questions.
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        cfg = self.config

        print("Initializing RAG Pipeline...")
        print(f"  Chunk size: {cfg.chunk_size} chars, overlap: {cfg.chunk_overlap}")
        print(f"  Chunk strategy: {cfg.chunk_strategy}")
        print(f"  Embedding model: {cfg.embedding_model}")
        print(f"  Top-k retrieval: {cfg.top_k}")

        # Instantiate each component
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
            strategy=cfg.chunk_strategy,
        )
        self.embedder = Embedder(model_name=cfg.embedding_model)
        self.vector_store = VectorStore(
            collection_name=cfg.collection_name,
            persist_dir=cfg.persist_dir,
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            k=cfg.top_k,
            use_reranker=cfg.use_reranker,
        )
        self.generator = AnthropicGenerator(
            model=cfg.llm_model,
            max_tokens=cfg.max_tokens,
        )
        print("Pipeline ready.\n")

    # ------------------------------------------------------------------ #
    # OFFLINE PHASE: Ingestion                                             #
    # ------------------------------------------------------------------ #

    def ingest_file(self, path: str) -> int:
        """
        Ingest a single file into the vector store.
        Returns the number of chunks added.
        """
        print(f"\n[Ingestion] Processing: {path}")

        # Step 1: Load the file
        doc = self.loader.load_file(path)
        print(f"  Loaded: {doc.metadata['char_count']:,} characters")

        # Step 2: Chunk it
        chunks = self.chunker.chunk_document(doc)

        # Step 3: Embed all chunks
        _, embeddings = self.embedder.embed_chunks(chunks)

        # Step 4: Store in vector DB
        self.vector_store.add(chunks, embeddings)

        return len(chunks)

    def ingest_directory(self, directory: str) -> int:
        """
        Ingest all supported files from a directory.
        Returns total chunks added across all files.
        """
        print(f"\n[Ingestion] Scanning directory: {directory}")

        docs = self.loader.load_directory(directory)
        if not docs:
            print("  No supported files found.")
            return 0

        total_chunks = 0
        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            _, embeddings = self.embedder.embed_chunks(chunks)
            self.vector_store.add(chunks, embeddings)
            total_chunks += len(chunks)

        print(
            f"\nIngestion complete: {len(docs)} documents → "
            f"{total_chunks} chunks stored"
        )
        print(f"Total chunks in store: {self.vector_store.count()}")
        return total_chunks

    def ingest_text(self, text: str, source_name: str = "user_text") -> int:
        """Ingest raw text directly (no file needed)."""
        doc = self.loader.load_text(text, source_name)
        chunks = self.chunker.chunk_document(doc)
        _, embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.add(chunks, embeddings)
        return len(chunks)

    # ------------------------------------------------------------------ #
    # ONLINE PHASE: Querying                                               #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        metadata_filter: dict = None,
        show_sources: bool = True,
        show_scores: bool = True,
    ) -> dict:
        """
        Answer a user question using RAG.

        Parameters
        ----------
        question : str
            The user's natural language question.
        metadata_filter : dict, optional
            Only search within documents matching this filter.
            E.g.: {"source": "transformers.txt"}
        show_sources : bool
            Include retrieved source passages in the return value.
        show_scores : bool
            Print similarity scores for debugging retrieval quality.

        Returns
        -------
        dict with keys:
          "answer"   : the LLM's grounded response
          "sources"  : list of retrieved SearchResult objects
          "question" : the original question
        """
        print(f"\n[Query] {question}")

        # Step 1: Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=question,
            metadata_filter=metadata_filter,
        )

        if not results:
            return {
                "question": question,
                "answer": "No relevant information found in the knowledge base.",
                "sources": [],
            }

        # Show retrieval results for learning/debugging
        if show_scores:
            print(f"\n  Retrieved {len(results)} chunks:")
            for r in results:
                source = r.metadata.get("source", "unknown")
                print(f"    [{r.rank}] score={r.score:.3f} | {source} | {r.text[:80]}...")

        # Step 2: Generate grounded answer
        print("\n  Generating answer...")
        answer = self.generator.generate(question, results)

        return {
            "question": question,
            "answer": answer,
            "sources": results if show_sources else [],
        }

    # ------------------------------------------------------------------ #
    # Utility methods                                                       #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """Return stats about the current knowledge base."""
        return {
            "total_chunks": self.vector_store.count(),
            "sources": self.vector_store.get_sources(),
            "embedding_model": self.config.embedding_model,
            "chunk_strategy": self.config.chunk_strategy,
        }