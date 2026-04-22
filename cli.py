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
from rag import RAGPipeline, RAGConfig


def get_pipeline() -> RAGPipeline:
    """Create and return a configured RAGPipeline."""
    config = RAGConfig(
        chunk_size=500,
        chunk_overlap=100,
        chunk_strategy="sentence",
        embedding_model="all-MiniLM-L6-v2",
        top_k=5,
        use_reranker=False,
    )
    return RAGPipeline(config)


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
        show_scores=not args.quiet,
    )

    print(f"\n{'='*60}")
    print(f"ANSWER:\n{result['answer']}")

    if not args.quiet and result["sources"]:
        print(f"\nSOURCES ({len(result['sources'])} passages):")
        for r in result["sources"]:
            print(f"  [{r.rank}] {r.metadata.get('source')} (score: {r.score:.3f})")


def cmd_chat(args):
    """Interactive chat mode — keeps asking for questions until 'exit'."""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()

    print(f"\n{'='*60}")
    print("RAG Interactive Chat")
    print(f"Knowledge base: {stats['total_chunks']} chunks from {len(stats['sources'])} sources")
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
    Run a small evaluation suite to measure retrieval quality.

    This demonstrates how to build a golden dataset and compute
    Precision@k and Recall@k for your RAG system.
    """
    from rag.retriever import RAGEvaluator

    pipeline = get_pipeline()

    # Define a golden dataset: questions + which source should be retrieved
    golden_dataset = [
        {
            "question": "What is the self-attention mechanism?",
            "relevant_sources": ["transformers.txt"],
        },
        {
            "question": "How does cosine similarity work?",
            "relevant_sources": ["vector_databases.txt"],
        },
        {
            "question": "What is Precision@k in retrieval evaluation?",
            "relevant_sources": ["rag_evaluation.txt"],
        },
        {
            "question": "What are Query, Key, and Value in transformers?",
            "relevant_sources": ["transformers.txt"],
        },
        {
            "question": "What is FAISS and when should I use it?",
            "relevant_sources": ["vector_databases.txt"],
        },
        {
            "question": "What is faithfulness in RAG evaluation?",
            "relevant_sources": ["rag_evaluation.txt"],
        },
    ]

    evaluator = RAGEvaluator(pipeline.retriever)

    print("\nEvaluating retrieval at k=3 and k=5:")
    for k in (3, 5):
        print(f"\n--- k={k} ---")
        evaluator.evaluate_retrieval(golden_dataset, k=k)


def cmd_compare_chunking(args):
    """
    Compare the three chunking strategies on a sample document.

    This is purely educational — run this to understand how different
    strategies split the same text into different chunk patterns.
    """
    from rag.chunker import TextChunker, compare_strategies
    from rag.ingestion import DocumentLoader

    loader = DocumentLoader()
    sample_path = "data/sample_docs/transformers.txt"

    if not os.path.exists(sample_path):
        print(f"ERROR: {sample_path} not found. Run from the rag_system directory.")
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