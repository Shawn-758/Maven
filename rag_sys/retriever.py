"""
rag/retriever.py
----------------
COMPONENT 5 (extended): Retrieval Pipeline + Re-ranking + Evaluation

What it does:
  Orchestrates the full query-side pipeline:
    query text → embed → similarity search → (optional re-rank) → results

Re-ranking explained:
  First-stage retrieval (vector similarity) is fast but imprecise —
  it finds chunks that are "directionally similar" in embedding space.
  A re-ranker applies a slower, more accurate model to the top-k candidates
  to re-order them by true relevance.

  Think of it like a two-stage hiring process:
    Stage 1 (fast): Screen 1000 résumés based on keywords → 20 candidates
    Stage 2 (slow): Carefully interview the 20 → rank by actual fit

  For RAG this means:
    Stage 1: Retrieve top-20 by cosine similarity (fast)
    Stage 2: Re-rank the 20 with a cross-encoder model (slower, more accurate)
    Final:   Use top-5 from re-ranked list

  The cross-encoder sees BOTH the query and the chunk simultaneously,
  unlike the bi-encoder (embedder) which encodes them independently.
  This joint encoding gives much better relevance judgments.

When to re-rank:
  - When retrieval precision matters more than latency
  - When your queries are complex or ambiguous
  - When first-stage retrieval quality is low
  Trade-off: Adds 100-500ms latency per query. Worth it for most use cases.
"""

import numpy as np
from .embedder import Embedder
from .vector_store import VectorStore, SearchResult
from typing import Optional


class Retriever:
    """
    Manages the complete retrieval pipeline: embed query → search → re-rank.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        k: int = 5,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Parameters
        ----------
        embedder : Embedder
            Must be the SAME model used during ingestion.
        vector_store : VectorStore
            The populated vector DB to search against.
        k : int
            Number of results to return to the LLM.
            Tip: retrieve k*4 first if using re-ranking, then re-rank to top-k.
        use_reranker : bool
            Enable cross-encoder re-ranking for better precision.
        reranker_model : str
            HuggingFace cross-encoder model name.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.k = k
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        self._reranker = None

        if use_reranker:
            print(f"  Re-ranking enabled (model: {reranker_model})")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Full retrieval pipeline for a single query.

        Parameters
        ----------
        query : str
            The user's natural language question.
        k : int, optional
            Override the default k for this query.
        metadata_filter : dict, optional
            ChromaDB filter to restrict the search space.
            E.g.: {"source": "transformers.txt"}
            E.g.: {"file_type": {"$in": ["txt", "pdf"]}}

        Returns
        -------
        Top-k SearchResult objects, sorted by relevance descending.
        """
        k = k or self.k

        # Step 1: Embed the query
        # The query MUST be embedded with the same model as the documents
        query_vector = self.embedder.embed(query)

        # Step 2: First-stage retrieval — fast approximate nearest neighbor
        # If re-ranking, over-fetch (4x) so the re-ranker has candidates to work with
        fetch_k = k * 4 if self.use_reranker else k
        results = self.vector_store.search(
            query_embedding=query_vector,
            k=fetch_k,
            where=metadata_filter,
        )

        if not results:
            return []

        # Step 3: Optional re-ranking
        if self.use_reranker and len(results) > k:
            results = self._rerank(query, results, top_k=k)
        else:
            results = results[:k]

        # Re-assign ranks after any re-ordering
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def _rerank(
        self, query: str, candidates: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """
        Re-rank candidates using a cross-encoder model.

        Cross-encoders process (query, passage) pairs jointly, giving them
        much richer signal about relevance — but they're slower than embeddings
        because you can't pre-compute the "query" side in advance.
        """
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.reranker_model_name)
                print(f"  Cross-encoder loaded: {self.reranker_model_name}")
            except ImportError:
                print("  WARNING: sentence-transformers not installed; skipping re-rank")
                return candidates[:top_k]

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c.text) for c in candidates]
        scores = self._reranker.predict(pairs)

        # Sort by cross-encoder score (descending) and return top_k
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        reranked = [r for r, _ in ranked[:top_k]]

        # Update scores to reflect cross-encoder judgment
        for (r, s) in zip(reranked, [s for _, s in ranked[:top_k]]):
            r.score = round(float(s), 4)

        return reranked


# ------------------------------------------------------------------ #
# Evaluation utilities                                                  #
# ------------------------------------------------------------------ #

class RAGEvaluator:
    """
    Simple evaluation harness for measuring retrieval quality.

    To evaluate a RAG system you need a "golden dataset":
      A list of (question, expected_source_or_answer) pairs where you
      know in advance which chunks should be retrieved.

    Example golden_dataset entry:
      {
        "question": "What is cosine similarity?",
        "relevant_sources": ["vector_databases.txt"],  # must appear in results
        "expected_answer_contains": ["angle", "direction"]  # for answer eval
      }
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def evaluate_retrieval(
        self,
        golden_dataset: list[dict],
        k: int = 5,
    ) -> dict:
        """
        Measure Precision@k and Recall@k over a golden dataset.

        Precision@k = (relevant chunks retrieved) / k
        Recall@k    = (relevant chunks retrieved) / (total relevant chunks)
        """
        precisions = []
        recalls = []

        for item in golden_dataset:
            question = item["question"]
            relevant_sources = set(item.get("relevant_sources", []))

            results = self.retriever.retrieve(question, k=k)
            retrieved_sources = {r.metadata.get("source", "") for r in results}

            # How many of the retrieved results are relevant?
            true_positives = len(retrieved_sources & relevant_sources)

            precision = true_positives / k if k > 0 else 0
            recall = true_positives / len(relevant_sources) if relevant_sources else 0

            precisions.append(precision)
            recalls.append(recall)

            print(
                f"  Q: {question[:60]}...\n"
                f"    Retrieved: {sorted(retrieved_sources)}\n"
                f"    Expected:  {sorted(relevant_sources)}\n"
                f"    P@{k}={precision:.2f}  R@{k}={recall:.2f}\n"
            )

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        f1 = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0
        )

        print(f"\n{'='*50}")
        print(f"Evaluation Results (n={len(golden_dataset)})")
        print(f"  Mean Precision@{k}: {avg_precision:.3f}")
        print(f"  Mean Recall@{k}:    {avg_recall:.3f}")
        print(f"  F1 Score:          {f1:.3f}")
        print(f"{'='*50}")

        return {
            "precision_at_k": avg_precision,
            "recall_at_k": avg_recall,
            "f1": f1,
            "k": k,
            "n": len(golden_dataset),
        }

    def compare_retrieval_configs(
        self,
        golden_dataset: list[dict],
        configs: list[dict],
        k: int = 5,
    ) -> None:
        """
        Compare different k values or retrieval strategies.

        configs example:
          [{"name": "k=3"}, {"name": "k=7", "k_override": 7}]
        """
        print("\nConfiguration comparison:")
        for cfg in configs:
            name = cfg.get("name", "config")
            k_val = cfg.get("k_override", k)
            print(f"\n--- {name} ---")
            self.evaluate_retrieval(golden_dataset, k=k_val)