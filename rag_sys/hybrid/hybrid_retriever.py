"""
rag/hybrid_retriever.py
-----------------------
HYBRID COMPONENT B: Fusing BM25 + Vector Rankings

The core problem: two retrieval systems → two different ranked lists.
How do you combine them into one list?

Option 1 — Linear score combination (simple, often fragile):
  combined_score = α × vector_score + (1−α) × bm25_score

  Problems:
    - BM25 scores (e.g. 3.4, 1.2) and cosine similarities (0.85, 0.71) live
      on completely different scales. You can't just add them.
    - Requires tuning α per domain, which needs labelled evaluation data.

Option 2 — Reciprocal Rank Fusion (RRF, what we use here):
  RRF_score(chunk) = Σ  1 / (k + rank_in_list_i)
                    i ∈ {bm25, vector}

  Where k is a constant (typically 60) that dampens the influence of top ranks.

  Why RRF is better:
    - Rank-based: immune to score scale differences between BM25 and cosine sim
    - No tuning required: k=60 works well across domains (from the original paper)
    - Robust: a chunk appearing in both lists gets a strong score even if it's
      not #1 in either list — exactly what you want from ensemble retrieval

  Example (k=60):
    Chunk A: rank 1 in BM25, rank 5 in vector
      RRF = 1/(60+1) + 1/(60+5) = 0.01639 + 0.01538 = 0.03177

    Chunk B: rank 2 in BM25, not in vector
      RRF = 1/(60+2) = 0.01613

    Chunk C: rank 3 in BM25, rank 2 in vector
      RRF = 1/(60+3) + 1/(60+2) = 0.01587 + 0.01613 = 0.03200 ← wins!

  Chunk C wins because it's consistently good in both systems, even though
  it's not #1 in either. That's the intuition behind RRF.

Paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
        Learning Methods" — Cormack, Clarke, Buettcher, SIGIR 2009.
"""

from dataclasses import dataclass
from typing import Optional
from .bm25_index import BM25Index
from ..embedder import Embedder
from ..vector_store import VectorStore, SearchResult
from ..chunker import Chunk


@dataclass
class HybridResult:
    """
    A retrieval result from hybrid search, with scores from both lanes
    preserved for inspection and debugging.
    """

    chunk: Chunk
    text: str
    rrf_score: float  # Final merged score (higher = more relevant)
    bm25_score: float  # Raw BM25 score (0 if not in BM25 results)
    vector_score: float  # Cosine similarity (0 if not in vector results)
    bm25_rank: Optional[int]  # Rank in BM25 list (None if absent)
    vector_rank: Optional[int]  # Rank in vector list (None if absent)
    final_rank: int  # Position in merged output list
    source: str  # Metadata shortcut


class HybridRetriever:
    """
    Combines BM25 keyword search and dense vector search using
    Reciprocal Rank Fusion to produce a single merged ranking.

    Usage:
        # Build once (after ingesting chunks)
        retriever = HybridRetriever(embedder, vector_store)
        retriever.build_bm25_index(chunks)

        # Query (per user request)
        results = retriever.retrieve("What is cosine similarity?")
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_fetch_k: int = 20,
        vector_fetch_k: int = 20,
        final_k: int = 5,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        vector_similarity_threshold: float = 0.30,
    ):
        """
        Parameters
        ----------
        embedder : Embedder
            The same model used during ingestion.
        vector_store : VectorStore
            The populated ChromaDB collection.
        bm25_fetch_k : int
            How many candidates to fetch from BM25 before fusion.
            Fetch more than final_k to give RRF enough candidates to work with.
        vector_fetch_k : int
            How many candidates to fetch from the vector store before fusion.
        final_k : int
            Number of results to return to the LLM after fusion.
        rrf_k : int
            The RRF constant. 60 is the paper's recommendation.
            Smaller values → top ranks matter more.
            Larger values → ranks matter less (more uniform weighting).
        bm25_weight : float
            Multiply BM25's RRF contribution by this factor.
            Use >1 to weight keyword matching more heavily.
            Use <1 to downweight it (e.g. for queries that are always conceptual).
        vector_weight : float
            Same for the vector lane. Default: both weighted equally.
        vector_similarity_threshold : float
            Post-RRF filter: any final result whose *vector* cosine similarity
            is below this threshold AND whose BM25 score is also zero is dropped.
            Applied AFTER fusion so RRF can still promote BM25-only good chunks.
            Default 0.30 — lower than 0.35 to avoid cutting borderline-relevant chunks.
            Set to 0.0 to disable.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_fetch_k = bm25_fetch_k
        self.vector_fetch_k = vector_fetch_k
        self.final_k = final_k
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.vector_similarity_threshold = vector_similarity_threshold
        self.bm25_index: Optional[BM25Index] = None

    def build_bm25_index(self, chunks: list[Chunk]) -> None:
        """
        Build the BM25 index from the same chunks already stored in the vector DB.

        Call this once after all documents have been ingested.
        If you add more documents later, rebuild the entire index (or use
        an external search engine that supports incremental updates).
        """
        print(f"\n[BM25] Building keyword index over {len(chunks)} chunks...")
        self.bm25_index = BM25Index()
        self.bm25_index.build(chunks)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
        explain: bool = False,
    ) -> list[HybridResult]:
        """
        Run hybrid retrieval for a query.

        Parameters
        ----------
        query : str
            The user's natural language question.
        k : int, optional
            Override the default final_k for this call.
        metadata_filter : dict, optional
            Filter to apply to the vector search lane.
            BM25 filtering would require a separate implementation.
        explain : bool
            If True, print a per-result breakdown showing BM25 rank,
            vector rank, and the RRF contribution from each lane.
            Very useful for debugging retrieval quality.

        Returns
        -------
        List of HybridResult objects, sorted by RRF score descending.
        """
        if self.bm25_index is None:
            raise RuntimeError(
                "BM25 index not built. Call build_bm25_index(chunks) first."
            )

        k = k or self.final_k

        # ── Lane 1: BM25 keyword retrieval ──────────────────────────────── #
        bm25_results = self.bm25_index.search(query, k=self.bm25_fetch_k)
        # bm25_results: list of (Chunk, raw_score)

        # ── Lane 2: Dense vector retrieval ──────────────────────────────── #
        query_vec = self.embedder.embed(query)
        vector_results = self.vector_store.search(
            query_embedding=query_vec,
            k=self.vector_fetch_k,
            where=metadata_filter,
        )
        # vector_results: list of SearchResult (with .chunk_id, .text, .score)

        # ── Reciprocal Rank Fusion ───────────────────────────────────────── #
        merged = self._rrf_merge(bm25_results, vector_results, k=k)

        if explain:
            self._explain(query, merged)

        return merged

    def _rrf_merge(
        self,
        bm25_results: list[tuple[Chunk, float]],
        vector_results: list[SearchResult],
        k: int,
    ) -> list[HybridResult]:
        """
        Core RRF fusion logic.

        Algorithm:
          1. For each ranked list, assign each chunk a rank (1 = best)
          2. Compute RRF score = Σ weight / (rrf_k + rank) across all lists
          3. Sort by RRF score descending
          4. Return top-k

        Key design: we use chunk_id as the join key. A chunk that appears
        in both the BM25 list and the vector list gets contributions from both.
        A chunk only in one list still gets a partial score.
        """
        # Track per-chunk data, keyed by chunk_id
        scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        vector_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        vector_ranks: dict[str, int] = {}
        chunk_objects: dict[str, Chunk] = {}
        texts: dict[str, str] = {}

        # Process BM25 lane
        for rank, (chunk, raw_score) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            contribution = self.bm25_weight / (self.rrf_k + rank)
            scores[cid] = scores.get(cid, 0) + contribution
            bm25_scores[cid] = raw_score
            bm25_ranks[cid] = rank
            chunk_objects[cid] = chunk
            texts[cid] = chunk.text

        # Process vector lane
        for rank, result in enumerate(vector_results, start=1):
            cid = result.chunk_id
            contribution = self.vector_weight / (self.rrf_k + rank)
            scores[cid] = scores.get(cid, 0) + contribution
            vector_scores[cid] = result.score
            vector_ranks[cid] = rank
            # If chunk only appears in vector results, use SearchResult text
            if cid not in texts:
                texts[cid] = result.text
            # Build a minimal Chunk if we only saw it in the vector lane
            if cid not in chunk_objects:
                chunk_objects[cid] = Chunk(
                    text=result.text,
                    metadata=result.metadata,
                    chunk_id=cid,
                )

        # Sort all candidates by RRF score and take top k,
        # then apply post-RRF similarity threshold to drop pure-noise results.
        # sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        merged = []
        for cid in sorted_ids:
            if len(merged) >= k:
                break
            # Post-RRF noise filter: drop chunks that have a meaningfully low
            # vector similarity AND no BM25 signal at all (pure noise).
            vec_sim = vector_scores.get(cid, 0.0)
            has_bm25 = cid in bm25_ranks
            if (
                not has_bm25
                and self.vector_similarity_threshold > 0.0
                and vec_sim < self.vector_similarity_threshold
            ):
                continue  # skip — vector-only result below quality bar

            final_rank = len(merged) + 1
            chunk = chunk_objects[cid]
            merged.append(
                HybridResult(
                    chunk=chunk,
                    text=texts[cid],
                    rrf_score=round(scores[cid], 6),
                    bm25_score=round(bm25_scores.get(cid, 0.0), 4),
                    vector_score=round(vec_sim, 4),
                    bm25_rank=bm25_ranks.get(cid),
                    vector_rank=vector_ranks.get(cid),
                    final_rank=final_rank,
                    source=chunk.metadata.get("source", "unknown"),
                )
            )

        return merged

    def _explain(self, query: str, results: list[HybridResult]) -> None:
        """Print a debug view showing how each result was composed."""
        print(f"\n  Hybrid retrieval explanation for: '{query}'")
        print(f"  {'─'*70}")
        header = f"  {'Rank':>4}  {'RRF':>8}  {'BM25r':>6}  {'Vecr':>5}  Source"
        print(header)
        print(f"  {'─'*70}")
        for r in results:
            bm25_r = str(r.bm25_rank) if r.bm25_rank else "—"
            vec_r = str(r.vector_rank) if r.vector_rank else "—"
            print(
                f"  [{r.final_rank:>2}]  {r.rrf_score:>8.5f}  "
                f"{bm25_r:>6}  {vec_r:>5}  {r.source}"
            )
            print(f"         {r.text[:80]}...")
        print()

    def retrieve_bm25_only(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Run BM25 retrieval alone (useful for comparison/ablation)."""
        if self.bm25_index is None:
            raise RuntimeError("BM25 index not built.")
        return self.bm25_index.search(query, k=k)

    def retrieve_vector_only(
        self, query: str, k: int = 5, metadata_filter: dict = None
    ) -> list[SearchResult]:
        """Run vector retrieval alone (useful for comparison/ablation)."""
        query_vec = self.embedder.embed(query)
        return self.vector_store.search(
            query_embedding=query_vec, k=k, where=metadata_filter
        )
