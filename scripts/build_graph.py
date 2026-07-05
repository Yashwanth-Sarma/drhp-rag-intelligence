"""
scripts/build_graph.py

Load extracted entities into Neo4j.
Run AFTER extract_entities.py has processed chunks.

Usage:
    python scripts/build_graph.py
    python scripts/build_graph.py --clear   # clear graph first, then reload
    python scripts/build_graph.py --stats   # show current graph stats
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph.graph_builder import GraphBuilder
from src.configuration.config import validate_env


def main(clear: bool = False) -> None:
    validate_env()
    builder = GraphBuilder()

    if clear:
        confirm = input("Clear entire Neo4j graph? Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return
        builder.clear_graph()

    stats_before = builder.get_stats()
    print(f"Neo4j before: {stats_before}")

    builder.load_from_jsonl()

    stats_after = builder.get_stats()
    print(f"Neo4j after:  {stats_after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        builder = GraphBuilder()
        print(f"Graph stats: {builder.get_stats()}")
    else:
        main(clear=args.clear)