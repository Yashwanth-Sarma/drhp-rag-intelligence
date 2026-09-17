"""Record/check local downloaded model artifacts without contacting a network."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    cache = (ROOT/'data/models').resolve()
    records = []
    for path in sorted(cache.glob('models--*/snapshots/**/*')):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(cache):
            raise ValueError('Model artifact resolves outside the cache.')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        records.append(dict(path=path.relative_to(cache).as_posix(), bytes=path.stat().st_size,
                            sha256=digest.hexdigest()))
    if not records:
        raise ValueError('No downloaded snapshot files found.')
    return dict(format='finsight-model-inventory-v1', artifacts=records,
                limitation='Records local bytes and upstream snapshot IDs; not an independent authenticity attestation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='Create the initial manifest; never overwrite it.')
    args = parser.parse_args()
    path = ROOT/'evals/model-lock.json'
    current = inventory()
    if args.record:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(current, handle, indent=2)
        print(f'Recorded {len(current["artifacts"])} local model artifacts.')
    else:
        expected = json.loads(path.read_text(encoding='utf-8'))
        if current != expected:
            raise SystemExit('Model inventory changed: review weights/configuration before reusing embeddings.')
        print(f'PASS: {len(current["artifacts"])} model artifacts match the recorded manifest.')
