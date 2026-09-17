import hashlib
import json
import subprocess
from unittest.mock import patch
import pytest
import pymupdf
from finsight.ingest import ingest_pdf, extract_native, parse_isolated, IngestionBusy, _ingestion_slot
from finsight.store import MetadataConflict

def ingest(store,raw,**kw):
    return ingest_pdf(store,raw,kw.get('company','Example Limited'),'example.pdf','DRHP',kw.get('date','2024-06-01'))

def test_exact_numeric_text_page_coverage_and_source_hash(store,pdf_bytes):
    did=ingest(store,pdf_bytes)
    d=store.document(did)
    assert d['status']=='partial_text'
    assert d['sha256']==hashlib.sha256(pdf_bytes).hexdigest()
    assert store.original_path(did).read_bytes()==pdf_bytes
    blocks=store.rows('SELECT * FROM evidence WHERE document_id=?',(did,))
    assert any('\n1000\n0\n(5)\n' in b['text'] for b in blocks)
    pages=store.rows('SELECT * FROM document_pages WHERE document_id=? ORDER BY page',(did,))
    assert [p['status'] for p in pages]==['native_text','no_native_text']
    assert pages[0]['label']=='i'
    assert all(json.loads(b['bbox'])['coordinates']=='unrotated-page' for b in blocks)
    assert store.rows('SELECT state FROM ingestion_jobs')[0]['state']=='complete'

def test_reingest_deduplicates_and_conflicts_do_not_mutate(store,pdf_bytes):
    did=ingest(store,pdf_bytes)
    count=len(store.rows('SELECT * FROM search'))
    assert ingest(store,pdf_bytes)==did
    assert len(store.rows('SELECT * FROM search'))==count
    for kw in [{'company':'Other Limited'},{'date':'2025-01-01'}]:
        with pytest.raises(MetadataConflict): ingest(store,pdf_bytes,**kw)
    assert store.document(did)['company']=='Example Limited'
    assert store.original_path(did).read_bytes()==pdf_bytes

@pytest.mark.parametrize('raw',[b'not a pdf',b'%PDF-1.7 corrupted'])
def test_invalid_input_never_publishes(store,raw):
    with pytest.raises(ValueError): ingest(store,raw)
    assert store.documents()==[]
    assert store.rows('SELECT * FROM search')==[]

def test_all_blank_requires_ocr(store):
    with pymupdf.open() as p:
        p.new_page(); raw=p.tobytes()
    with pytest.raises(ValueError,match='OCR'): ingest(store,raw)
    assert not store.documents()

def test_missing_original_can_be_restored(store,pdf_bytes):
    did=ingest(store,pdf_bytes)
    store.original_path(did).unlink()
    assert ingest(store,pdf_bytes)==did
    assert store.original_path(did).read_bytes()==pdf_bytes

def test_parser_timeout_is_explicit(tmp_path):
    p=tmp_path/'source.pdf';p.write_bytes(b'%PDF-')
    with patch('finsight.ingest.subprocess.run',side_effect=subprocess.TimeoutExpired('worker',1)):
        with pytest.raises(ValueError,match='timed out'): parse_isolated(p,'a'*32)

def test_failed_publication_keeps_index_empty(store,pdf_bytes):
    with patch.object(store,'publish',side_effect=RuntimeError('simulated disk failure')):
        with pytest.raises(RuntimeError): ingest(store,pdf_bytes)
    assert not store.documents()
    assert not store.rows('SELECT * FROM search')
    assert store.rows('SELECT state FROM ingestion_jobs')[0]['state']=='failed'
    # Recovery is idempotent even with an orphan original left by failed publication.
    assert ingest(store,pdf_bytes)

def test_busy_limit(store,pdf_bytes):
    _ingestion_slot.acquire()
    try:
        with pytest.raises(IngestionBusy): ingest(store,pdf_bytes)
    finally: _ingestion_slot.release()

def test_rotated_page_geometry(tmp_path):
    path=tmp_path/'rotated.pdf'
    with pymupdf.open() as pdf:
        p=pdf.new_page(width=300,height=500)
        p.insert_text((40,70),'Revenue 1000')
        p.set_rotation(90);pdf.save(path)
    result=extract_native(path,'a'*32)
    box=json.loads(result['blocks'][0]['bbox'])
    assert result['pages'][0]['rotation']==90
    assert box['width']==500 and box['height']==300
    x0,y0,x1,y1=box['display_rect']
    assert 0<=x0<x1<=500 and 0<=y0<y1<=300

def test_source_path_rejects_traversal(store):
    with pytest.raises(ValueError): store.original_path('../secret')
