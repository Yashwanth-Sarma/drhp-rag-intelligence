"""Explicit local model acquisition and index build; never download during requests."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.runtime'), str(ROOT)]
from finsight.store import Store
from finsight.retrieval import build_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build_index(Store(ROOT / 'data'), download=args.download), indent=2))
