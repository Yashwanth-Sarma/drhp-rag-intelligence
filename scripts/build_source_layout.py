"""Build source-bound native layout on explicitly selected physical pages."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.runtime'), str(ROOT)]
from finsight.store import Store
from finsight.structure import build_layout

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('document_id')
    parser.add_argument('pages', nargs='+', type=int)
    args = parser.parse_args()
    inventory = build_layout(Store(ROOT / 'data'), args.document_id, args.pages)
    print(json.dumps([dict(page=p['page'], headings=len(p['headings']), tables=len(p['tables']),
                           parser_version=p['parser_version']) for p in inventory], indent=2))
