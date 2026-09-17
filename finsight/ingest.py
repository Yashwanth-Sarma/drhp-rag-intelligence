"""Bounded native extraction, then atomic publication. No table inference or OCR."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import uuid
from .contracts import EXTRACTION_VERSION, FilingMetadata
from .store import identity

MAX_BYTES = 35 * 1024 * 1024
MAX_PAGES = 1500
MAX_TEXT_CHARS = 12_000_000
PARSER_TIMEOUT = 120
_ingestion_slot = threading.BoundedSemaphore(1)

class IngestionBusy(ValueError):
    pass

def extract_native(path, did):
    """Worker-only parser. Preserve text exactly as emitted by PyMuPDF."""
    import pymupdf
    blocks, pages, warnings = [], [], []
    chars = 0
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass or len(pdf) > MAX_PAGES:
            raise ValueError('Encrypted PDFs and documents over 1,500 pages are unsupported.')
        for index, page in enumerate(pdf):
            if page.rect.width > 3000 or page.rect.height > 3000:
                raise ValueError('Page dimensions exceed the supported rendering limit.')
            items = [b for b in page.get_text('blocks', sort=True) if b[6] == 0 and b[4].strip()]
            if not items:
                warnings.append(f'PDF page {index + 1}: no native text; inspect whether OCR is required.')
            pages.append(dict(document_id=did, page=index+1, label=page.get_label() or '',
                              width=page.rect.width, height=page.rect.height, rotation=page.rotation,
                              status='native_text' if items else 'no_native_text', block_count=len(items)))
            for j, b in enumerate(items):
                text = b[4]
                chars += len(text)
                if chars > MAX_TEXT_CHARS:
                    raise ValueError('Extracted text exceeds the supported size limit.')
                display_rect = pymupdf.Rect(b[:4]) * page.rotation_matrix
                geometry = dict(rect=list(b[:4]), display_rect=list(display_rect),
                                width=page.rect.width, height=page.rect.height,
                                origin='top-left', unit='pt', coordinates='unrotated-page')
                blocks.append(dict(id=identity(did, EXTRACTION_VERSION, index, j, text), document_id=did,
                                   page=index+1, section='Unclassified', text=text, bbox=json.dumps(geometry)))
        count = len(pdf)
    if not blocks:
        raise ValueError('No native text found. OCR is required before this document can be indexed.')
    return dict(blocks=blocks, pages=pages, warnings=warnings, page_count=count,
                parser_version=pymupdf.VersionBind, extraction_version=EXTRACTION_VERSION)

def parse_isolated(path, did):
    output = path.with_suffix('.json')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), *sys.path])
    try:
        completed = subprocess.run([sys.executable, '-m', 'finsight.pdf_worker', str(path), str(output), did],
                                   capture_output=True, timeout=PARSER_TIMEOUT, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('PDF extraction timed out. No document was published.') from exc
    if completed.returncode or not output.exists():
        raise ValueError('PDF parsing failed. Check that the file is a valid, supported PDF.')
    result = json.loads(output.read_text(encoding='utf-8'))
    if 'error' in result:
        raise ValueError(result['error'])
    return result

def ingest_pdf(store, raw, company, name, kind, filing_date):
    metadata = FilingMetadata(company=company, name=name, kind=kind, filing_date=filing_date)
    if len(raw) > MAX_BYTES or not raw.startswith(b'%PDF-'):
        raise ValueError('Expected a PDF no larger than 35 MB.')
    if not _ingestion_slot.acquire(blocking=False):
        raise IngestionBusy('Another document is being processed. Retry after it finishes.')
    jid = uuid.uuid4().hex
    digest = hashlib.sha256(raw).hexdigest()
    did = identity('pdf-v1', digest)
    document = dict(id=did, **metadata.model_dump(mode='json'))
    try:
        store.job(jid, 'received', 'PDF received; metadata supplied by user.', did)
        existing = store.check_existing(document)
        target = store.original_path(did)
        if existing and target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==digest:
            store.job(jid, 'complete', 'Already indexed; original hash checked.', did)
            return did
        store.job(jid, 'extracting', 'Extracting native page blocks in an isolated process.', did)
        with tempfile.TemporaryDirectory(prefix='finsight-', dir=store.root) as temp:
            source = Path(temp)/'source.pdf'
            source.write_bytes(raw)
            extracted = parse_isolated(source, did)
            store.job(jid, 'publishing', 'Publishing source and index.', did)
            # A crash may leave an orphan source file, never a new index without its PDF.
            os.replace(source, target)
            doc = dict(**document, status='partial_text' if extracted['warnings'] else 'source_available',
                       page_count=extracted['page_count'], sha256=digest, warnings=json.dumps(extracted['warnings']))
            manifest = dict(sha256=digest, extraction_version=EXTRACTION_VERSION,
                            parser='PyMuPDF', parser_version=extracted['parser_version'],
                            metadata_origin='user_supplied', source_verification='original_bytes_preserved',
                            limitations=['Native text only; reading order and tables require review.',
                                         'Page labels are PDF metadata, not visually verified printed labels.'])
            store.publish(doc, extracted['blocks'], extracted['pages'], manifest)
        store.job(jid, 'complete', 'Document published.', did)
        return did
    except Exception as exc:
        store.job(jid, 'failed', str(exc)[:500], did)
        raise
    finally:
        _ingestion_slot.release()
