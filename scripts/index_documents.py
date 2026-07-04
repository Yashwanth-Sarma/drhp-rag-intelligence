"""
scripts/index_documents.py

Incremental indexing pipeline for Stage 1 (naive chunking).
Only embeds chunks not already in ChromaDB — saves Voyage tokens.

Usage:
    python scripts/index_documents.py          # index new chunks only
    python scripts/index_documents.py --reset  # clear all and re-index
    python scripts/index_documents.py --stats  # show collection stats
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.data_loader import load_all_pdfs, get_corpus_stats
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.vector_store.chroma_store import ChromaStore
from src.configuration.config import (
    validate_env,
    EMBEDDINGS_STAGE1_DIR,
    COLLECTION_STAGE1,
)
from src.shared.logger import get_logger

logger = get_logger(__name__)


def get_existing_ids(store: ChromaStore) -> set[str]:
    """Fetch all chunk_ids currently in the collection."""
    total = store.collection.count()
    if total == 0:
        return set()

    result = store.collection.get(limit=total, include=[])
    return set(result["ids"])


def index_incremental(reset: bool = False) -> None:
    validate_env()

    store = ChromaStore(
        persist_dir=EMBEDDINGS_STAGE1_DIR,
        collection_name=COLLECTION_STAGE1,
    )
    embedder = VoyageEmbedder()

    if reset:
        confirm = input(
            f"This will delete all {store.collection.count()} existing embeddings. "
            "Type 'yes' to confirm: "
        )
        if confirm.lower() != "yes":
            print("Aborted.")
            return
        store.reset_collection()
        print("Collection cleared.")

    print("\nLoading and chunking all PDFs...")
    all_chunks = load_all_pdfs()

    if not all_chunks:
        print(f"No PDFs found in {EMBEDDINGS_STAGE1_DIR.parent / 'raw_pdfs'}")
        return

    stats = get_corpus_stats(all_chunks)
    print(f"\nCorpus on disk:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  By company:   {stats['by_company']}")
    print(f"  By doc type:  {stats['by_doc_type']}")

    print("\nChecking ChromaDB for existing chunks...")
    existing_ids = get_existing_ids(store)
    print(f"  Already indexed: {len(existing_ids)} chunks")

    new_chunks = [
        c for c in all_chunks
        if c.metadata.get("chunk_id") not in existing_ids
    ]

    if not new_chunks:
        print(f"\nAll {len(all_chunks)} chunks already indexed. Nothing to do.")
        return

    saved = len(existing_ids) * 300
    print(f"  New chunks to embed: {len(new_chunks)}")
    print(f"  Voyage tokens saved: ~{saved:,} (skipping already-indexed chunks)")

    print(f"\nEmbedding {len(new_chunks)} new chunks...")
    texts = [c.page_content for c in new_chunks]
    embeddings = embedder.embed_texts(texts)

    print(f"\nStoring in ChromaDB...")
    added = store.add_chunks(new_chunks, embeddings)

    final_stats = store.get_collection_stats()
    print(f"\nIndexing complete!")
    print(f"  New chunks added: {added}")
    print(f"  Total in store:   {final_stats['total_chunks']}")
    print(f"  Companies:        {final_stats['companies_sampled']}")
    print(f"\nRun: python scripts/test_retrieval.py")


def show_stats() -> None:
    store = ChromaStore(
        persist_dir=EMBEDDINGS_STAGE1_DIR,
        collection_name=COLLECTION_STAGE1,
    )
    stats = store.get_collection_stats()
    print(f"\nChromaDB Stage 1 Collection Stats:")
    print(f"  Total chunks:    {stats['total_chunks']}")
    print(f"  Companies:       {stats.get('companies_sampled', [])}")
    print(f"  Document types:  {stats.get('doc_types_sampled', [])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight Stage 1 indexer")
    parser.add_argument("--reset", action="store_true", help="Clear and re-index")
    parser.add_argument("--stats", action="store_true", help="Show collection stats")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        index_incremental(reset=args.reset)