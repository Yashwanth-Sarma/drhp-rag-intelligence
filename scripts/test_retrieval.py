"""
scripts/test_retrieval.py

Quick test script to verify Stage 1 retrieval is working.
Run after indexing: python scripts/test_retrieval.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.base_retriever import BaseRetriever
from src.vector_store.chroma_store import ChromaStore

def run_tests():
    store = ChromaStore()
    stats = store.get_collection_stats()

    print(f"Collection: {stats['total_chunks']} chunks")
    print(f"Companies: {stats.get('companies_sampled', [])}\n")

    if stats["total_chunks"] == 0:
        print("ERROR: No documents indexed. Run: python scripts/index_documents.py")
        return

    retriever = BaseRetriever()

    test_questions = [
        ("General risk query", "What are the main risk factors mentioned in this document?", None),
        ("Financial query", "What was the revenue or financial performance mentioned?", None),
        ("Business model", "How does the company generate revenue? What is their business model?", None),
    ]

    for test_name, question, metadata_filter in test_questions:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"Q: {question}")
        result = retriever.query(question, metadata_filter=metadata_filter, top_k=3)
        print(f"\nA ({result['latency_ms']}ms, {len(result['citations'])} citations):")
        print(result["answer"][:500] + "..." if len(result["answer"]) > 500 else result["answer"])
        print(f"\nTop citation: {result['citations'][0] if result['citations'] else 'None'}")

    print(f"\n{'='*60}")
    print("Stage 1 retrieval test complete.")
    print("If answers look grounded in document text, Stage 1 is working.")

if __name__ == "__main__":
    run_tests()