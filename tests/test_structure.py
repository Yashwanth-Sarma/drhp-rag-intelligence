import json
import pymupdf
import pytest
from finsight.ingest import ingest_pdf
from finsight.structure import build_layout
from finsight.source_context import source_context


def source(store):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 40), 'FINANCIAL INFORMATION', fontname='hebo', fontsize=15)
        page.insert_text((230, 70), 'INR million', fontsize=9)
        for x in (50, 220, 330):
            page.draw_line((x, 80), (x, 170))
        for y in (80, 110, 140, 170):
            page.draw_line((50, y), (330, y))
        for x, y, text in [(60,100,'Particulars'), (230,100,'FY2024'),
                           (60,130,'Profit'), (230,130,'(5)'),
                           (60,160,'Balance'), (230,160,'0')]:
            page.insert_text((x,y), text)
        raw = pdf.tobytes()
    return ingest_pdf(store, raw, 'Fixture', 'fixture.pdf', 'DRHP', '2024-01-01')


def test_native_layout_preserves_cells_and_evidence_identity(store):
    did = source(store)
    before = store.rows('SELECT * FROM evidence WHERE document_id=?', (did,))
    inventory = build_layout(store, did, [1])[0]
    assert inventory['headings'][0]['text'] == 'FINANCIAL INFORMATION'
    table = inventory['tables'][0]
    assert table['cells'] == [['Particulars', 'FY2024'], ['Profit', '(5)'], ['Balance', '0']]
    assert len(table['cell_bboxes']) == 3
    assert any(line['text'] == 'INR million' for line in table['margin_lines'])
    assert before == store.rows('SELECT * FROM evidence WHERE document_id=?', (did,))
    anchor = next(b for b in before if '(5)' in b['text'])
    context = source_context(store, anchor['id'])
    assert context['structure_context']['status'] == 'available'
    assert any(i['kind'] == 'native_table_candidate' for i in context['structure_context']['items'])
    empty = source_context(store, anchor['id'], max_chars=0)
    assert empty['characters'] == 0 and not empty['structure_context']['items']
    assert empty['structure_context']['omitted_ids']
    assert context['characters'] <= 12000


def test_layout_refuses_hash_mismatch_and_retains_previous_inventory(store):
    did = source(store)
    build_layout(store, did, [1])
    before = store.rows('SELECT * FROM source_layout')
    store.original_path(did).write_bytes(b'%PDF-changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        build_layout(store, did, [1])
    assert before == store.rows('SELECT * FROM source_layout')


def test_layout_bounds_and_stale_state(store):
    did = source(store)
    with pytest.raises(ValueError, match='physical pages'):
        build_layout(store, did, [0])
    build_layout(store, did, [1])
    with store.connect() as c:
        c.execute('UPDATE documents SET sha256=? WHERE id=?', ('b'*64, did))
    anchor = store.rows('SELECT id FROM evidence WHERE document_id=?', (did,))[0]
    assert source_context(store, anchor['id'])['structure_context']['status'] == 'stale_rebuild_required'
