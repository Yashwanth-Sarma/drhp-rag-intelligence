"""
scripts/index_documents.py

One-time (and incremental) indexing script.
Run this after downloading new PDFs to data/raw_documents/

Usage:
    python scripts/index_documents.py
    python scripts/index_documents.py --reset   # Clears existing index first
    python scripts/index_documents.py --company zomato_drhp_2021  # Index one file

This script:
1. Discovers all PDFs in data/raw_documents/
2. Loads and chunks them using data_loader.py
3. Embeds chunks using VoyageEmbedder
4. Stores embeddings in ChromaDB
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.data_loader import load_all_pdfs, load_pdf, chunk_documents, get_corpus_stats
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.vector_store.chroma_store import ChromaStore
from src.configuration.config import RAW_PDFS_DIR, validate_env
from src.shared.logger import get_logger

logger = get_logger(__name__)


def index_all(reset: bool = False) -> None:
    """Index all PDFs found in data/raw_documents/"""
    validate_env()

    store = ChromaStore()
    embedder = VoyageEmbedder()

    if reset:
        confirm = input("This will delete all existing embeddings. Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return
        store.reset_collection()
        logger.info("Collection reset.")

    # Load and chunk all PDFs
    logger.info("Loading all PDFs...")
    all_chunks = load_all_pdfs()

    if not all_chunks:
        print(f"\nNo PDFs found in {RAW_PDFS_DIR}")
        print("Download DRHPs and save as: data/raw_documents/zomato_drhp_2021.pdf")
        print("Filename must match a key in COMPANY_METADATA in config.py")
        return

    # Print corpus stats before embedding
    stats = get_corpus_stats(all_chunks)
    print(f"\nCorpus loaded:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  By company:   {stats['by_company']}")
    print(f"  By doc type:  {stats['by_doc_type']}")

    # Embed all chunks
    print(f"\nEmbedding {len(all_chunks)} chunks with Voyage AI voyage-finance-2...")
    print("This may take a few minutes for large corpora...")

    texts = [chunk.page_content for chunk in all_chunks]
    embeddings = embedder.embed_texts(texts)

    # Store in ChromaDB
    print(f"\nStoring in ChromaDB...")
    added = store.add_chunks(all_chunks, embeddings)

    final_stats = store.get_collection_stats()
    print(f"\nIndexing complete!")
    print(f"  Chunks added this run: {added}")
    print(f"  Total in store: {final_stats['total_chunks']}")
    print(f"  Companies indexed: {final_stats['companies_sampled']}")
    print(f"\nRun 'python scripts/test_retrieval.py' to verify.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index financial documents into FinSight")
    parser.add_argument("--reset", action="store_true", help="Clear existing index before indexing")
    parser.add_argument("--company", type=str, help="Index only this company key (e.g. zomato_drhp_2021)")
    args = parser.parse_args()

    index_all(reset=args.reset)