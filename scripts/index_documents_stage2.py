"""
scripts/index_documents_stage2.py

Re-indexes all documents using Stage 2 contextual chunking.
Run AFTER index_documents.py has already indexed Stage 1.

What this does differently from Stage 1:
- Each chunk gets an LLM-generated context prefix before embedding
- Stores contextual embeddings in a SEPARATE ChromaDB collection
  so Stage 1 baseline embeddings are preserved for comparison

This separation is critical for the ablation study —
you need both Stage 1 and Stage 2 embeddings to compare RAGAs scores.

Usage:
    python scripts/index_documents_stage2.py
    python scripts/index_documents_stage2.py --reset
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from src.parsers.data_loader import load_all_pdfs, get_corpus_stats
from src.chunking.contextual_chunker import ContextualChunker
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.configuration.config import EMBEDDINGS_DIR, validate_env
from src.shared.logger import get_logger

logger = get_logger(__name__)

STAGE2_COLLECTION = "finsight_documents_stage2"


def index_stage2(reset: bool = False) -> None:
    validate_env()

    # Use a separate persistent directory for Stage 2 embeddings
    stage2_dir = EMBEDDINGS_DIR.parent / "embeddings_stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(stage2_dir),
        settings=Settings(anonymized_telemetry=False)
    )

    if reset:
        confirm = input("Reset Stage 2 index? This deletes all Stage 2 embeddings. Type 'yes': ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return
        try:
            client.delete_collection(STAGE2_COLLECTION)
            logger.info("Stage 2 collection reset.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=STAGE2_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Stage 2 collection: {collection.count()} existing chunks")

    # Load all PDFs
    print("\nLoading all PDFs...")
    chunks = load_all_pdfs()
    if not chunks:
        print(f"No PDFs found. Check data/raw_pdfs/")
        return

    stats = get_corpus_stats(chunks)
    print(f"Loaded {stats['total_chunks']} chunks from {stats['by_company']}")

    # Contextual enrichment
    print(f"\nStarting contextual enrichment ({len(chunks)} chunks)...")
    print("This calls Groq for each chunk — takes ~20-30 minutes for 3,000 chunks.")
    print("Groq free tier: 14,400 requests/day — you have enough headroom.\n")

    chunker = ContextualChunker()
    enriched_chunks = chunker.enrich_all_chunks(chunks)

    # Embed enriched chunks
    print(f"\nEmbedding {len(enriched_chunks)} enriched chunks...")
    embedder = VoyageEmbedder()
    texts = [chunk.page_content for chunk in enriched_chunks]
    embeddings = embedder.embed_texts(texts)

    # Store in Stage 2 collection
    print("\nStoring in ChromaDB Stage 2 collection...")
    

    print(f"\nStage 2 indexing complete!")
    print(f"  Chunks added: {len(ids)}")
    print(f"  Total in Stage 2 collection: {collection.count()}")
    print(f"\nRun evaluation to compare Stage 1 vs Stage 2 RAGAs scores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    index_stage2(reset=args.reset)