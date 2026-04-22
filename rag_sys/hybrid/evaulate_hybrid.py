#!/usr/bin/env python3
"""
evaluate_hybrid.py
------------------
A self-contained evaluation script that demonstrates WHY hybrid search
outperforms either BM25 or vector search alone.

Run:
    python evaluate_hybrid.py

What it does:
  1. Ingests the sample documents
  2. Defines a golden evaluation set covering both keyword-heavy and
     conceptual queries
  3. Runs BM25-only, vector-only, and hybrid retrieval on each query
  4. Computes Precision@k for each mode
  5. Prints a detailed comparison

This script is intentionally self-contained and verbose — it's a teaching
tool, not production code.
"""

import sys
import os

# Make sure we can import from the rag package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.hybrid_pipeline import HybridRAGPipeline


# ── Golden evaluation set ──────────────────────────────────────────────── #
#
# Each entry has:
#   question        : the query to run
#   relevant_sources: which source file(s) contain the answer
#   query_type      : "keyword" (exact term matters) or "conceptual" (synonyms ok)
#   note            : why this query stresses one retrieval mode over the other

GOLDEN_DATASET = [
    # Keyword-heavy queries — BM25 should excel here
    {
        "question": "What is BM25 and what does k1 control?",
        "relevant_sources": [
            "vector_databases.txt"
        ],  # won't be found — reveals failure
        "query_type": "keyword",
        "note": "Exact term 'BM25' and 'k1' — tests exact match retrieval",
    },
    {
        "question": "What is TF-IDF?",
        "relevant_sources": ["vector_databases.txt"],
        "query_type": "keyword",
        "note": "Exact acronym — BM25 has strong advantage over vector",
    },
    {
        "question": "What does NDCG stand for in RAG evaluation?",
        "relevant_sources": ["rag_evaluation.txt"],
        "query_type": "keyword",
        "note": "Rare abbreviation — vector may miss, BM25 should catch",
    },
    {
        "question": "What is Precision@k?",
        "relevant_sources": ["rag_evaluation.txt"],
        "query_type": "keyword",
        "note": "Technical term with @ symbol — tests tokenization edge case",
    },
    # Conceptual queries — vector search should excel here
    {
        "question": "How do language models understand the meaning of words?",
        "relevant_sources": ["transformers.txt"],
        "query_type": "conceptual",
        "note": "No keyword overlap with 'embedding' or 'self-attention' — tests semantic retrieval",
    },
    {
        "question": "How does a transformer figure out which words to focus on?",
        "relevant_sources": ["transformers.txt"],
        "query_type": "conceptual",
        "note": "Paraphrase of 'attention mechanism' — BM25 likely fails",
    },
    {
        "question": "How can I find documents that are similar in meaning to a query?",
        "relevant_sources": ["vector_databases.txt"],
        "query_type": "conceptual",
        "note": "Paraphrase of semantic search — vector wins on synonyms",
    },
    # Mixed queries — where hybrid should best both
    {
        "question": "What are the trade-offs of cosine similarity versus dot product?",
        "relevant_sources": ["vector_databases.txt"],
        "query_type": "mixed",
        "note": "Has exact terms AND conceptual content — hybrid should win",
    },
    {
        "question": "How does self-attention use Query, Key and Value matrices?",
        "relevant_sources": ["transformers.txt"],
        "query_type": "mixed",
        "note": "Technical terms + conceptual explanation needed",
    },
    {
        "question": "What metrics measure whether retrieved passages are useful?",
        "relevant_sources": ["rag_evaluation.txt"],
        "query_type": "mixed",
        "note": "Conceptual phrasing of 'retrieval evaluation metrics'",
    },
]


def run_evaluation():
    print("=" * 70)
    print("Hybrid Search Evaluation")
    print("Compares: BM25-only | Vector-only | Hybrid RRF")
    print("=" * 70)

    # Build the pipeline
    pipeline = HybridRAGPipeline(
        chunk_size=300,
        chunk_overlap=60,
        chunk_strategy="sentence",
        bm25_fetch_k=15,
        vector_fetch_k=15,
        final_k=3,
        rrf_k=60,
    )

    # Ingest sample documents
    print("\n[Step 1] Ingesting sample documents...")
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    if not os.path.exists(sample_dir):
        print(f"ERROR: {sample_dir} not found. Run from the rag_system directory.")
        sys.exit(1)

    n_chunks = pipeline.ingest_directory(sample_dir)
    print(f"Indexed {n_chunks} chunks.\n")

    # Track per-mode precision
    results_by_mode = {"bm25": [], "vector": [], "hybrid": []}
    k = pipeline.final_k

    print(f"\n[Step 2] Running evaluation (k={k})...")
    print("=" * 70)

    for item in GOLDEN_DATASET:
        q = item["question"]
        relevant = set(item["relevant_sources"])
        qtype = item["query_type"]
        note = item["note"]

        print(f"\nQuery ({qtype}): {q}")
        print(f"  Note: {note}")
        print(f"  Expected source: {relevant}")

        # BM25 only
        bm25_raw = pipeline.retriever.retrieve_bm25_only(q, k=k)
        bm25_sources = {chunk.metadata.get("source", "") for chunk, _ in bm25_raw}
        bm25_hits = len(bm25_sources & relevant)
        bm25_prec = bm25_hits / k
        results_by_mode["bm25"].append(bm25_prec)

        # Vector only
        vector_raw = pipeline.retriever.retrieve_vector_only(q, k=k)
        vector_sources = {r.metadata.get("source", "") for r in vector_raw}
        vector_hits = len(vector_sources & relevant)
        vector_prec = vector_hits / k
        results_by_mode["vector"].append(vector_prec)

        # Hybrid
        hybrid_raw = pipeline.retriever.retrieve(q, k=k)
        hybrid_sources = {r.source for r in hybrid_raw}
        hybrid_hits = len(hybrid_sources & relevant)
        hybrid_prec = hybrid_hits / k
        results_by_mode["hybrid"].append(hybrid_prec)

        # Print per-query comparison
        def fmt(prec, sources):
            hit = "HIT " if prec > 0 else "miss"
            return f"{hit} ({','.join(sorted(sources)) or 'none'})"

        print(f"  BM25:   P@{k}={bm25_prec:.2f}  {fmt(bm25_prec, bm25_sources)}")
        print(f"  Vector: P@{k}={vector_prec:.2f}  {fmt(vector_prec, vector_sources)}")
        print(f"  Hybrid: P@{k}={hybrid_prec:.2f}  {fmt(hybrid_prec, hybrid_sources)}")

    # Aggregate results
    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS (n={len(GOLDEN_DATASET)}, k={k})")
    print("=" * 70)

    for mode in ("bm25", "vector", "hybrid"):
        scores = results_by_mode[mode]
        avg = sum(scores) / len(scores)
        wins = sum(1 for s in scores if s > 0)
        print(
            f"  {mode.upper():8s}: mean P@{k} = {avg:.3f}  |  queries with ≥1 hit: {wins}/{len(scores)}"
        )

    print()

    # Per-query-type breakdown
    print("Per query type breakdown:")
    for qtype in ("keyword", "conceptual", "mixed"):
        idxs = [
            i for i, item in enumerate(GOLDEN_DATASET) if item["query_type"] == qtype
        ]
        if not idxs:
            continue
        print(f"\n  [{qtype}] (n={len(idxs)})")
        for mode in ("bm25", "vector", "hybrid"):
            scores = [results_by_mode[mode][i] for i in idxs]
            avg = sum(scores) / len(scores)
            print(f"    {mode:8s}: {avg:.3f}")

    print("\n" + "=" * 70)
    print("Key takeaways to look for:")
    print("  - Keyword queries: BM25 ≥ vector (exact term matching matters)")
    print("  - Conceptual queries: vector ≥ BM25 (semantics > keywords)")
    print("  - Overall: hybrid ≥ both (RRF captures the best of each lane)")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
