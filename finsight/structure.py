"""Optional native PDF layout inventory; never changes canonical evidence."""
import hashlib
import json
import inspect
from statistics import median
from .store import identity

VERSION = 'native-layout-v1'
SCHEMA = '''CREATE TABLE IF NOT EXISTS source_layout (
 document_id TEXT NOT NULL REFERENCES documents(id), page INTEGER NOT NULL,
 payload TEXT NOT NULL, PRIMARY KEY(document_id,page));'''


def intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def page_layout(store, document_id, page):
    """Read an optional inventory without running a PDF parser in a request."""
    doc = store.document(document_id)
    if not doc:
        raise LookupError('Document not found.')
    if not 1 <= page <= doc['page_count']:
        raise ValueError('Physical page is outside the document.')
    result = dict(document_id=document_id, page=page, status='not_indexed', inventory=None)
    if doc['status'] == 'legacy_unverified':
        return dict(result, status='source_unavailable')
    if not store.rows("SELECT name FROM sqlite_master WHERE name='source_layout'"):
        return result
    rows = store.rows('SELECT payload FROM source_layout WHERE document_id=? AND page=?',
                      (document_id, page))
    if not rows:
        return result
    payload = json.loads(rows[0]['payload'])
    if (payload['version'] != VERSION or payload['document_sha256'] != doc['sha256'] or
            payload['document_id'] != document_id or payload['page'] != page):
        return dict(result, status='stale_rebuild_required')
    return dict(result, status='available', inventory=payload)


def layout_coverage(store, document_id):
    doc = store.document(document_id)
    if not doc:
        raise LookupError('Document not found.')
    rows = []
    if store.rows("SELECT name FROM sqlite_master WHERE name='source_layout'"):
        rows = store.rows('SELECT page,payload FROM source_layout WHERE document_id=? ORDER BY page', (document_id,))
    pages = []
    for row in rows:
        payload = json.loads(row['payload'])
        current = (doc['status'] != 'legacy_unverified' and payload['version'] == VERSION and
                   payload['document_sha256'] == doc['sha256'] and payload['document_id'] == document_id and
                   payload['page'] == row['page'])
        pages.append(dict(page=row['page'], status='available' if current else 'stale_rebuild_required',
                          headings=len(payload['headings']) if current else 0,
                          tables=len(payload['tables']) if current else 0))
    return dict(document_id=document_id, total_pages=doc['page_count'], pages=pages,
                indexed_pages=sum(p['status'] == 'available' for p in pages),
                review_status='unreviewed', limitation='Counts describe parser coverage, not extraction accuracy.')


def build_layout(store, document_id, pages):
    """Explicit offline build for selected pages, published atomically per document."""
    import pymupdf
    doc = store.document(document_id)
    if not doc or doc['status'] == 'legacy_unverified':
        raise ValueError('Native layout requires an original source document.')
    raw = store.original_path(document_id).read_bytes()
    if hashlib.sha256(raw).hexdigest() != doc['sha256']:
        raise ValueError('Original PDF hash mismatch; layout was not published.')
    pages = sorted(set(pages))
    if not pages or len(pages) > 40 or any(p < 1 or p > doc['page_count'] for p in pages):
        raise ValueError('Select between 1 and 40 valid physical pages per build.')
    inventories = []
    with pymupdf.open(stream=raw, filetype='pdf') as pdf:
        outline = pdf.get_toc()
        for number in pages:
            page = pdf[number - 1]
            blocks = store.rows('SELECT id,text,bbox FROM evidence WHERE document_id=? AND page=?',
                                (document_id, number))
            rectangles = [(b['id'], json.loads(b['bbox'])['rect']) for b in blocks if b['bbox']]
            def bindings(rect):
                return [eid for eid, box in rectangles if intersects(rect, box)]
            lines = []
            for block in page.get_text('dict', flags=pymupdf.TEXTFLAGS_TEXT)['blocks']:
                for line in block.get('lines', []):
                    spans = [{k: s[k] for k in ('text', 'bbox', 'font', 'size', 'flags')}
                             for s in line['spans']]
                    lines.append(dict(bbox=list(line['bbox']), spans=spans,
                                      text=''.join(s['text'] for s in spans)))
            sizes = [s['size'] for line in lines for s in line['spans'] if s['text'].strip()]
            body_size = median(sizes) if sizes else 0
            headings = []
            for index, line in enumerate(lines):
                spans = [s for s in line['spans'] if s['text'].strip()]
                if spans and any(c.isalpha() for c in line['text']) and len(line['text']) <= 180 and (
                    all(s['flags'] & 16 for s in spans) or max(s['size'] for s in spans) > body_size * 1.15
                ):
                    headings.append(dict(id=identity(document_id, number, VERSION, 'heading', index),
                        kind='native_heading_candidate', **line, evidence_ids=bindings(line['bbox']),
                        basis='native_bold_or_larger_font'))
            tables = []
            # Explicit ruled-line detection: no optional ML layout engine or downloads.
            options = {'use_layout': False} if 'use_layout' in inspect.signature(page.find_tables).parameters else {}
            finder = page.find_tables(strategy='lines', **options)
            for index, table in enumerate(finder.tables):
                tables.append(dict(id=identity(document_id, number, VERSION, 'table', index),
                    kind='native_table_candidate', bbox=list(table.bbox),
                    evidence_ids=bindings(table.bbox), cells=table.extract(),
                    cell_bboxes=[[list(cell) if cell else None for cell in row.cells] for row in table.rows],
                    header=dict(bbox=list(table.header.bbox), names=table.header.names,
                                cells=table.header.cells, external=table.header.external),
                    basis='pymupdf_ruled_lines'))
                tables[-1]['margin_lines'] = [line for line in lines if (
                    line['bbox'][0] < table.bbox[2] and line['bbox'][2] > table.bbox[0] and (
                        0 <= table.bbox[1] - line['bbox'][3] <= 24 or
                        0 <= line['bbox'][1] - table.bbox[3] <= 24))]
            headings = [h for h in headings if not any(intersects(h['bbox'], t['bbox']) for t in tables)]
            payload = dict(document_id=document_id, document_sha256=doc['sha256'], page=number,
                version=VERSION, parser_version=pymupdf.VersionBind, review_status='unreviewed',
                coordinates='unrotated-page', rotation=page.rotation,
                width=page.rect.width, height=page.rect.height,
                outline=[dict(level=level, title=title, page=p) for level, title, p in outline if p == number],
                lines=lines, headings=headings, tables=tables,
                limitation='Native parser candidates. Font style and geometric overlap do not establish hierarchy, header meaning, continuation, or claim support.')
            inventories.append(payload)
    with store.connect() as c:
        c.executescript(SCHEMA)
        c.executemany('INSERT OR REPLACE INTO source_layout VALUES (?,?,?)',
                      [(document_id, p['page'], json.dumps(p)) for p in inventories])
    return inventories


def structure_context(store, anchor, max_chars):
    result = dict(status='not_indexed', items=[], omitted_ids=[], characters=0,
                  relation='native_page_layout_candidates',
                  limitation='Parser candidates and geometric bindings require source inspection; no semantic support or verified hierarchy is implied.')
    layout = page_layout(store, anchor['document_id'], anchor['page'])
    if layout['status'] != 'available':
        return dict(result, status=layout['status'])
    payload = layout['inventory']
    rect = json.loads(anchor['bbox'])['rect']
    tables = [t for t in payload['tables'] if anchor['id'] in t['evidence_ids']]
    headings = [h for h in payload['headings'] if h['bbox'][1] <= rect[1]]
    candidates = tables + sorted(headings, key=lambda h: h['bbox'][1], reverse=True)[:2]
    for item in candidates:
        # Charge the full serialized candidate, including cell/header metadata.
        cost = len(json.dumps(item, ensure_ascii=False))
        if result['characters'] + cost > max_chars:
            result['omitted_ids'].append(item['id'])
        else:
            result['items'].append(item)
            result['characters'] += cost
    result.update(status='partial' if result['omitted_ids'] else 'available',
                  document_sha256=payload['document_sha256'], version=VERSION,
                  parser_version=payload['parser_version'])
    return result
