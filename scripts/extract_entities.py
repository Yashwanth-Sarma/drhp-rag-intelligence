"""
scripts/extract_entities.py

Run entity extraction on your corpus.
Resumable — safe to stop and restart.

Usage:
    python scripts/extract_entities.py          # extract from key sections only
    python scripts/extract_entities.py --all    # extract from all sections
    python scripts/extract_entities.py --stats  # show progress
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.data_loader import load_all_pdfs
from src.knowledge_graph.entity_extractor import EntityExtractor, PROGRESS_FILE, ENTITIES_FILE
from src.configuration.config import validate_env


def show_stats() -> None:
    if not PROGRESS_FILE.exists():
        print("No extraction started yet.")
        return

    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

    processed = len(progress.get("processed_chunk_ids", []))

    entity_count = 0
    rel_count = 0
    if ENTITIES_FILE.exists():
        with open(ENTITIES_FILE) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    entity_count += len(record.get("entities", []))
                    rel_count += len(record.get("relationships", []))
                except Exception:
                    pass

    print(f"\nEntity Extraction Progress:")
    print(f"  Chunks processed:    {processed}")
    print(f"  Entities extracted:  {entity_count}")
    print(f"  Relationships found: {rel_count}")
    print(f"  Output file:         {ENTITIES_FILE}")


def main(extract_all: bool = False) -> None:
    validate_env()

    print("Loading corpus...")
    chunks = load_all_pdfs()
    print(f"Total chunks: {len(chunks)}")

    extractor = EntityExtractor()

    if extract_all:
        target_sections = None  # all sections
    else:
        target_sections = [
            "Risk Factors",
            "Business Overview",
            "Financial Statements",
            "Capital Structure",
            "Management",
            "Objects of the Offer",
        ]
        print(f"Targeting sections: {target_sections}")
        print("Use --all to process every section (uses more tokens)")

    processed = extractor.extract_from_chunks(
    chunks=chunks,
    sections_filter=target_sections,
)
    print(f"\nSession complete. Processed: {processed} chunks.")
    show_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Extract from all sections")
    parser.add_argument("--stats", action="store_true", help="Show extraction progress")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        main(extract_all=args.all)