import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.runtime'), str(ROOT)]
from finsight.store import Store
from finsight.local_qwen import draft_answer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draft against an already running local Qwen server.')
    parser.add_argument('--company', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--document-id')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000/v1')
    args = parser.parse_args()
    print(json.dumps(draft_answer(Store(ROOT/'data'), args.question, args.company,
                                  document_id=args.document_id, endpoint=args.endpoint), indent=2))
