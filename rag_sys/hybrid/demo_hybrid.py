#!/usr/bin/env python3
"""
demo_hybrid.py
--------------
A focused demo showing hybrid search in action.

Run:
    python demo_hybrid.py

This script:
  1. Ingests the sample docs
  2. Runs a handful of queries in "compare" mode so you can see side-by-side
     how BM25, vector, and hybrid each rank the same chunks differently
  3. Generates answers for two questions to show end-to-end quality

It's intentionally short — meant to be read alongside the source code
of hybrid_retriever.py and bm25_index.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.hybrid_pipeline import HybridRAGPipeline


def main():
    pipeline = HybridRAGPipeline(
        chunk_size=300,
        chunk_overlap=60,
        chunk_strategy="sentence",
        bm25_fetch_k=15,
        vector_fetch_k=15,
        final_k=4,
        rrf_k=60,
    )

    # Ingest
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    pipeline.ingest_directory(sample_dir)

    # ── Demo 1: keyword-heavy query ────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("DEMO 1 — Keyword-heavy query (BM25 should contribute heavily)")
    print("=" * 70)
    pipeline.compare_retrieval_modes(
        "What does NDCG measure and how is it calculated?",
        k=3,
    )

    # ── Demo 2: purely conceptual query ───────────────────────────────── #
    print("\n" + "=" * 70)
    print("DEMO 2 — Conceptual query (vector should contribute heavily)")
    print("=" * 70)
    pipeline.compare_retrieval_modes(
        "How do language models decide which parts of the input to pay attention to?",
        k=3,
    )

    # ── Demo 3: full answer generation ────────────────────────────────── #
    print("\n" + "=" * 70)
    print("DEMO 3 — Full answer generation with hybrid retrieval")
    print("=" * 70)

    result = pipeline.query(
        "What is the difference between cosine similarity and dot product for retrieval?",
        explain=True,
    )
    print(f"\nANSWER:\n{result['answer']}")

    print("\n" + "=" * 70)
    print("DEMO 4 — Metadata filtering with hybrid search")
    print("=" * 70)
    result2 = pipeline.query(
        "What are the main evaluation metrics?",
        metadata_filter={"source": "rag_evaluation.txt"},
        explain=False,
    )
    print(f"\n(filtered to rag_evaluation.txt only)")
    print(f"ANSWER:\n{result2['answer']}")


if __name__ == "__main__":
    main()
