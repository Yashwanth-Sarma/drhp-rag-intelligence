"""
scripts/index_documents_stage2.py

Stage 2 indexing — contextual chunking + hybrid embeddings.
Stores in a separate collection from Stage 1 so both are preserved
for ablation study comparison.

Usage:
    python scripts/index_documents_stage2.py          # index new chunks only
    python scripts/index_documents_stage2.py --reset  # clear and re-index
    python scripts/index_documents_stage2.py --stats  # show collection stats
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.data_loader import load_all_pdfs, get_corpus_stats
from src.chunking.contextual_chunker import ContextualChunker
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.vector_store.chroma_store import ChromaStore
from src.configuration.config import (
    validate_env,
    EMBEDDINGS_STAGE2_DIR,
    COLLECTION_STAGE2,
)
from src.shared.logger import get_logger

logger = get_logger(__name__)


def show_stats() -> None:
    store = ChromaStore(
        persist_dir=EMBEDDINGS_STAGE2_DIR,
        collection_name=COLLECTION_STAGE2,
    )
    stats = store.get_collection_stats()
    print(f"\nChromaDB Stage 2 Collection Stats:")
    print(f"  Total chunks:    {stats['total_chunks']}")
    print(f"  Companies:       {stats.get('companies_sampled', [])}")
    print(f"  Document types:  {stats.get('doc_types_sampled', [])}")


def index_stage2(reset: bool = False) -> None:
    validate_env()

    store = ChromaStore(
        persist_dir=EMBEDDINGS_STAGE2_DIR,
        collection_name=COLLECTION_STAGE2,
    )
    embedder = VoyageEmbedder()

    if reset:
        confirm = input(
            f"This will delete all Stage 2 embeddings "
            f"({store.collection.count()} chunks). Type 'yes' to confirm: "
        )
        if confirm.lower() != "yes":
            print("Aborted.")
            return
        store.reset_collection()
        print("Stage 2 collection cleared.")

    print(f"\nStage 2 collection: {store.collection.count()} existing chunks")

    print("\nLoading all PDFs...")
    chunks = load_all_pdfs()
    if not chunks:
        print("No PDFs found.")
        return

    stats = get_corpus_stats(chunks)
    print(f"Total chunks in corpus: {stats['total_chunks']}")
    print(f"By company: {stats['by_company']}")

    # ContextualChunker checks Stage 2 internally and skips existing chunks
    chunker = ContextualChunker()
    enriched = chunker.enrich_all_chunks(chunks)

    if not enriched:
        print("\nNothing new to embed and store.")
        print(f"Stage 2 collection total: {store.collection.count()} chunks")
        return

    # Embed and store enriched chunks
    print(f"\nEmbedding {len(enriched)} enriched chunks...")
    texts = [c.page_content for c in enriched]
    embeddings = embedder.embed_texts(texts)

    print(f"\nStoring in Stage 2 collection...")
    added = store.add_chunks(enriched, embeddings)

    final_stats = store.get_collection_stats()
    print(f"\nSession complete!")
    print(f"  Added this session: {added}")
    print(f"  Total in Stage 2:   {final_stats['total_chunks']}")
    print(f"  Companies:          {final_stats['companies_sampled']}")

    remaining = stats['total_chunks'] - final_stats['total_chunks']
    if remaining > 0:
        print(f"\n  Remaining chunks: {remaining} — run again tomorrow to continue.")
    else:
        print(f"\n  All chunks indexed! Run evaluation:")
        print(f"  python tests/evaluation/ragas_evaluator.py --stage 2 --quick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight Stage 2 indexer")
    parser.add_argument("--reset", action="store_true", help="Clear Stage 2 and re-index")
    parser.add_argument("--stats", action="store_true", help="Show Stage 2 collection stats")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        index_stage2(reset=args.reset)