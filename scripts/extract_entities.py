"""
scripts/extract_entities.py

Run entity extraction on the document corpus.
Fully resumable — safe to stop and restart anytime.

Usage:
    python scripts/extract_entities.py          # key sections only (saves tokens)
    python scripts/extract_entities.py --all    # all sections
    python scripts/extract_entities.py --stats  # show current progress
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.data_loader import load_all_pdfs
from src.knowledge_graph.entity_extractor import (
    EntityExtractor,
    HIGH_VALUE_SECTIONS,
)
from src.configuration.config import validate_env


def main(extract_all: bool = False) -> None:
    validate_env()

    print("Loading corpus...")
    chunks = load_all_pdfs()
    print(f"Total chunks loaded: {len(chunks)}")

    extractor = EntityExtractor()

    if extract_all:
        target_sections = []  # empty list = all sections
        print("Processing ALL sections.")
    else:
        target_sections = HIGH_VALUE_SECTIONS
        print(f"Processing key sections only (use --all for all sections).")

    extractor.extract_from_chunks(
        chunks=chunks,
        sections_filter=target_sections,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight entity extractor")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract from all sections (more tokens, more entities)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current extraction progress",
    )
    args = parser.parse_args()

    if args.stats:
        from src.knowledge_graph.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        extractor._print_stats()
    else:
        main(extract_all=args.all)