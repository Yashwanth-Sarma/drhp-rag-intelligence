"""Private parser subprocess; never executes user-supplied commands."""
import json
from pathlib import Path
import sys
from .ingest import extract_native

if __name__ == '__main__':
    source, output, document_id = sys.argv[1:]
    try:
        result = extract_native(source, document_id)
    except ValueError as exc:
        result = {'error': str(exc)}
    except Exception:
        result = {'error': 'Unable to parse this PDF. No evidence was published.'}
    Path(output).write_text(json.dumps(result), encoding='utf-8')
