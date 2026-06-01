#!/usr/bin/env python3
"""
cli.py
------
A command-line interface for the RAG system.

Commands:
  python cli.py ingest  --dir ./data/sample_docs
  python cli.py query   --question "What is self-attention?"
  python cli.py chat                          # interactive mode
  python cli.py stats                         # show knowledge base info
  python cli.py evaluate                      # run golden dataset eval
  python cli.py compare-chunking              # compare chunking strategies

This CLI is intentionally simple — it's a learning tool, not a production app.
For a web API, you'd wrap the same RAGPipeline with FastAPI or Flask.
"""

import argparse
import sys
import os

import rag_sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_sys import RAGPipeline, RAGConfig


# def get_pipeline() -> RAGPipeline:
#     """Create and return a configured RAGPipeline."""
#     config = RAGConfig(
#         chunk_size=300,
#         chunk_overlap=60,
#         chunk_strategy="sentence",
#         embedding_model="BAAI/bge-small-en-v1.5",
#         top_k=5,
#         use_reranker=False,
#     )
#     return RAGPipeline(config)


def get_pipeline():
    from rag_sys.hybrid.hybrid_pipeline import HybridRAGPipeline

    return HybridRAGPipeline(
        chunk_size=300,
        chunk_overlap=60,
        chunk_strategy="paragraph",
        embedding_model="BAAI/bge-small-en-v1.5",
        bm25_fetch_k=15,
        vector_fetch_k=15,
        final_k=3,   # restored: threshold filter removes noise; k=3 needed for recall coverage
        rrf_k=60,
    )


def cmd_ingest(args):
    """Ingest documents from a directory or file."""
    pipeline = get_pipeline()

    if args.file:
        count = pipeline.ingest_file(args.file)
        print(f"\nDone: {count} chunks added from {args.file}")
    elif args.dir:
        count = pipeline.ingest_directory(args.dir)
        print(f"\nDone: {count} total chunks stored")
    else:
        print("ERROR: Provide --file or --dir")
        sys.exit(1)


def cmd_query(args):
    """Answer a single question."""
    pipeline = get_pipeline()

    metadata_filter = None
    if args.source:
        # Restrict retrieval to a specific document
        metadata_filter = {"source": args.source}
        print(f"(filtering to source: {args.source})")

    result = pipeline.query(
        question=args.question,
        metadata_filter=metadata_filter,
        explain=not args.quiet,
    )

    print(f"\n{'='*60}")
    print(f"ANSWER:\n{result['answer']}")

    # if not args.quiet and result["sources"]:
    #     print(f"\nSOURCES ({len(result['sources'])} passages):")
    #     for r in result["sources"]:
    #         print(f"  [{r.rank}] {r.metadata.get('source')} (score: {r.score:.3f})")

    if not args.quiet and result["sources"]:
        print(f"\nSOURCES ({len(result['sources'])} passages):")
        for r in result["sources"]:

            # HybridResult uses .final_rank and .rrf_score; SearchResult uses .rank and .score
            rank = getattr(r, "final_rank", getattr(r, "rank", "?"))
            score = getattr(r, "rrf_score", getattr(r, "score", 0))
            source = getattr(
                r, "source", getattr(r, "metadata", {}).get("source", "unknown")
            )
            print(f"  [{rank}] {source} (score: {score:.5f})")


def cmd_chat(args):
    """Interactive chat mode — keeps asking for questions until 'exit'."""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()

    print(f"\n{'='*60}")
    print("RAG Interactive Chat")
    print(
        f"Knowledge base: {stats['total_chunks']} chunks from {len(stats['sources'])} sources"
    )
    print(f"Sources: {', '.join(stats['sources'])}")
    print("Type 'exit' or 'quit' to stop, 'stats' for info, 'sources' to list docs.")
    print(f"{'='*60}\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if question.lower() == "stats":
            print(f"  Chunks: {stats['total_chunks']}")
            print(f"  Sources: {stats['sources']}")
            continue

        if question.lower() == "sources":
            for s in stats["sources"]:
                print(f"  - {s}")
            continue

        # Run the query
        result = pipeline.query(question, show_scores=True)
        print(f"\nAssistant: {result['answer']}\n")


def cmd_stats(args):
    """Show knowledge base statistics."""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()

    print(f"\n{'='*50}")
    print("Knowledge Base Statistics")
    print(f"{'='*50}")
    print(f"  Total chunks:    {stats['total_chunks']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Chunk strategy:  {stats['chunk_strategy']}")
    print(f"  Sources ({len(stats['sources'])}):")
    for source in stats["sources"]:
        print(f"    - {source}")


def cmd_evaluate(args):
    """
    Evaluation with both source-level AND chunk-level accuracy checking.

    Three metrics measured per query:
      Precision@k  - what fraction of retrieved chunks are from the right source
      Recall@k     - did we retrieve at least one chunk from the right source
      Top-1 Exact  - is the single best chunk actually the right paragraph
    """
    from rag_sys.retriever import RAGEvaluator

    pipeline = get_pipeline()

    # Golden dataset — each entry has:
    #   relevant_sources       : which file should be retrieved (source-level check)
    #   expected_top_contains  : text that should appear in the #1 result (chunk-level check)
    #   query_type             : keyword / conceptual / mixed (for breakdown analysis)
    golden_dataset = [
        {
            "question": "What is the self-attention mechanism?",
            "relevant_sources": ["transformers.txt"],
            "expected_top_contains": "attention",
            "query_type": "conceptual",
        },
        {
            "question": "How does cosine similarity work?",
            "relevant_sources": ["vector_databases.txt"],
            "expected_top_contains": "angle between two vectors",
            "query_type": "keyword",
        },
        {
            "question": "What is Precision@k in retrieval evaluation?",
            "relevant_sources": ["rag_evaluation.txt"],
            "expected_top_contains": "fraction of the top-k retrieved",
            "query_type": "keyword",
        },
        {
            "question": "What are Query, Key, and Value in transformers?",
            "relevant_sources": ["transformers.txt"],
            "expected_top_contains": "Query (Q)",
            "query_type": "keyword",
        },
        {
            "question": "What is FAISS and when should I use it?",
            "relevant_sources": ["vector_databases.txt"],
            "expected_top_contains": "Facebook AI Similarity Search",
            "query_type": "keyword",
        },
        {
            "question": "What is faithfulness in RAG evaluation?",
            "relevant_sources": ["rag_evaluation.txt"],
            "expected_top_contains": "factually consistent with the retrieved context",
            "query_type": "conceptual",
        },
        # Conceptual / paraphrase queries — harder for keyword search
        {
            "question": "How do transformers figure out which words to focus on?",
            "relevant_sources": ["transformers.txt"],
            "expected_top_contains": "attention",
            "query_type": "conceptual",
        },
        {
            "question": "How do you find documents with similar meaning to a query?",
            "relevant_sources": ["vector_databases.txt"],
            "expected_top_contains": "similarity",
            "query_type": "conceptual",
        },
    ]

    evaluator = RAGEvaluator(pipeline.retriever)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    # ── Run source-level evaluation at k=3 and k=5 ── #
    for k in (3, 5):
        print(f"\n--- Source-level Precision/Recall @ k={k} ---")
        evaluator.evaluate_retrieval(golden_dataset, k=k)

    # ── Run chunk-level Top-1 accuracy check ── #
    print("\n--- Chunk-level Top-1 accuracy ---")
    print("(Is the single best retrieved chunk actually the right paragraph?)\n")

    top1_correct = 0
    by_type = {"keyword": [], "conceptual": [], "mixed": []}

    for item in golden_dataset:
        results = pipeline.retriever.retrieve(item["question"], k=1)
        top_text = results[0].text if results else ""
        expected = item.get("expected_top_contains", "")
        correct = expected.lower() in top_text.lower()
        top1_correct += int(correct)

        qtype = item.get("query_type", "mixed")
        by_type[qtype].append(correct)

        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] {item['question'][:55]}...")
        if not correct:
            # Show what we got vs what we expected — useful for debugging
            print(f"         Expected fragment : '{expected}'")
            print(f"         Got (first 100)   : '{top_text[:100]}...'")

    total = len(golden_dataset)
    print(f"\n  Top-1 accuracy: {top1_correct}/{total} = {top1_correct/total:.1%}")

    # ── Per query type breakdown ── #
    print("\n--- Breakdown by query type ---")
    for qtype, scores in by_type.items():
        if scores:
            acc = sum(scores) / len(scores)
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  {qtype:12s}: {bar} {acc:.0%} ({sum(scores)}/{len(scores)})")

    # ── Diagnosis ── #
    print("\n--- Diagnosis ---")
    keyword_acc = (
        sum(by_type["keyword"]) / len(by_type["keyword"]) if by_type["keyword"] else 0
    )
    conceptual_acc = (
        sum(by_type["conceptual"]) / len(by_type["conceptual"])
        if by_type["conceptual"]
        else 0
    )

    if conceptual_acc < keyword_acc - 0.2:
        print(
            "  ► Conceptual queries underperform — try BAAI/bge-small-en-v1.5 embedding model"
        )
    if keyword_acc < 0.7:
        print("  ► Keyword queries underperform — enable hybrid search (BM25 + vector)")
    if top1_correct / total < 0.6:
        print(
            "  ► Top-1 accuracy low — reduce chunk_size from 500 to 300 and re-ingest"
        )
    if top1_correct / total >= 0.8:
        print(
            "  ► Retrieval quality is good. Focus on prompt engineering for better answers."
        )

    print("=" * 60)


def cmd_compare_chunking(args):
    """
    Compare the three chunking strategies on a sample document.

    This is purely educational — run this to understand how different
    strategies split the same text into different chunk patterns.
    """
    from .chunker import TextChunker, compare_strategies
    from .ingestion import DocumentLoader

    loader = DocumentLoader()
    sample_path = "data/sample_docs/transformers.txt"

    if not os.path.exists(sample_path):
        print(f"ERROR: {sample_path} not found. Run from the ragtem directory.")
        sys.exit(1)

    doc = loader.load_file(sample_path)
    print(f"\nComparing chunking strategies on: {sample_path}")
    print(f"Document length: {len(doc.text):,} characters\n")
    print("Strategy comparison (chunk_size=500, overlap=100):")
    results = compare_strategies(doc.text, chunk_size=500, overlap=100)

    # Show first chunk from each strategy
    print("\nFirst chunk from each strategy:")
    for strategy, data in results.items():
        if data["chunks"]:
            print(f"\n[{strategy}]")
            print(f"  {data['chunks'][0].text[:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py ingest --dir data/sample_docs
  python cli.py query --question "What is cosine similarity?"
  python cli.py query --question "What is FAISS?" --source vector_databases.txt
  python cli.py chat
  python cli.py stats
  python cli.py evaluate
  python cli.py compare-chunking
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("--dir", help="Directory of documents")
    p_ingest.add_argument("--file", help="Single file path")

    # query
    p_query = subparsers.add_parser("query", help="Ask a single question")
    p_query.add_argument("--question", required=True, help="Your question")
    p_query.add_argument("--source", help="Restrict to a specific source file")
    p_query.add_argument("--quiet", action="store_true", help="Less verbose output")

    # chat
    subparsers.add_parser("chat", help="Interactive chat mode")

    # stats
    subparsers.add_parser("stats", help="Show knowledge base statistics")

    # evaluate
    subparsers.add_parser("evaluate", help="Run evaluation on golden dataset")

    # compare-chunking
    subparsers.add_parser("compare-chunking", help="Compare chunking strategies")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "chat": cmd_chat,
        "stats": cmd_stats,
        "evaluate": cmd_evaluate,
        "compare-chunking": cmd_compare_chunking,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
