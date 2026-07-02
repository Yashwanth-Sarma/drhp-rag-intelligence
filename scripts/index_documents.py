"""
scripts/index_documents.py

Incremental indexing pipeline for FinSight.

TRUE incremental behavior:
1. Compute chunk_id for every chunk from every PDF
2. Ask ChromaDB which chunk_ids already exist
3. ONLY send new chunks to Voyage AI for embedding
4. ONLY store new embeddings

This means:
- Adding one new PDF only embeds that PDF's chunks
- Re-running never re-embeds already-indexed chunks
- Zero wasted Voyage tokens after first run

Usage:
    python scripts/index_documents.py              # index new chunks only
    python scripts/index_documents.py --reset      # clear all and re-index
    python scripts/index_documents.py --stats      # show what's indexed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.data_loader import load_all_pdfs, get_corpus_stats
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.vector_store.chroma_store import ChromaStore
from src.configuration.config import validate_env
from src.shared.logger import get_logger

logger = get_logger(__name__)


def get_existing_chunk_ids(store: ChromaStore) -> set[str]:
    """
    Fetch all chunk_ids currently in ChromaDB.
    Uses batched fetching to handle large collections.
    """
    total = store.collection.count()
    if total == 0:
        return set()
    
    existing_ids = set()
    batch_size = 5000
    offset = 0
    
    while offset < total:
        result = store.collection.get(
            limit=batch_size,
            offset=offset,
            include=[]  # only fetch IDs, not documents or embeddings — fastest possible
        )
        existing_ids.update(result["ids"])
        offset += batch_size
        logger.info(f"Fetched existing IDs: {len(existing_ids)}/{total}")
    
    return existing_ids


def index_incremental(reset: bool = False) -> None:
    validate_env()
    
    store = ChromaStore()
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
    
    # ── Step 1: Load all chunks from disk ─────────────────────────────────
    print("\nLoading and chunking all PDFs...")
    all_chunks = load_all_pdfs()
    
    if not all_chunks:
        print(f"No PDFs found in data/raw_pdfs/")
        print("Download DRHPs and place them there. Any filename works now.")
        return
    
    stats = get_corpus_stats(all_chunks)
    print(f"\nCorpus on disk:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  By company:   {stats['by_company']}")
    print(f"  By doc type:  {stats['by_doc_type']}")
    
    # ── Step 2: Find which chunks are NOT yet indexed ──────────────────────
    print(f"\nChecking ChromaDB for existing chunks...")
    existing_ids = get_existing_chunk_ids(store)
    print(f"  Already indexed: {len(existing_ids)} chunks")
    
    new_chunks = [
        chunk for chunk in all_chunks
        if chunk.metadata.get("chunk_id") not in existing_ids
    ]
    
    if not new_chunks:
        print(f"\nAll {len(all_chunks)} chunks are already indexed.")
        print("Nothing to do. Add new PDFs to data/raw_pdfs/ to index more.")
        return
    
    print(f"  New chunks to embed: {len(new_chunks)}")
    print(f"  Voyage AI tokens saved by skipping: {len(existing_ids)} chunks × ~300 tokens ≈ "
          f"{len(existing_ids) * 300:,} tokens saved")
    
    # ── Step 3: Embed only new chunks ─────────────────────────────────────
    print(f"\nEmbedding {len(new_chunks)} new chunks with Voyage AI voyage-finance-2...")
    texts = [chunk.page_content for chunk in new_chunks]
    embeddings = embedder.embed_texts(texts)
    
    # ── Step 4: Store new embeddings ──────────────────────────────────────
    print(f"\nStoring {len(new_chunks)} new chunks in ChromaDB...")
    added = store.add_chunks(new_chunks, embeddings)
    
    final_count = store.collection.count()
    print(f"\nIndexing complete!")
    print(f"  New chunks added: {added}")
    print(f"  Total in store:   {final_count}")
    print(f"  Companies:        {store.get_collection_stats().get('companies_sampled', [])}")
    print(f"\nRun: python scripts/test_retrieval.py")


def show_stats() -> None:
    store = ChromaStore()
    stats = store.get_collection_stats()
    print(f"\nChromaDB Collection Stats:")
    print(f"  Total chunks:     {stats['total_chunks']}")
    print(f"  Companies:        {stats.get('companies_sampled', [])}")
    print(f"  Document types:   {stats.get('doc_types_sampled', [])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight incremental document indexer")
    parser.add_argument("--reset", action="store_true", help="Clear all embeddings and re-index from scratch")
    parser.add_argument("--stats", action="store_true", help="Show current collection stats")
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
    else:
        index_incremental(reset=args.reset)