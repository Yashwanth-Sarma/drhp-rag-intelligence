from unittest.mock import patch
from finsight.structure import build_layout
from test_structure import source


def test_layout_coverage_and_full_inventory_are_read_only(client):
    store = client.app.state.store
    did = source(store)
    url = f'/api/documents/{did}'
    assert client.get(url).json()['layout_coverage']['indexed_pages'] == 0
    assert client.get(url+'/pages/1/layout').json()['status'] == 'not_indexed'
    before = store.rows('SELECT * FROM evidence')
    inventory = build_layout(store, did, [1])[0]
    with patch('pymupdf.open', side_effect=AssertionError('Read must not parse PDFs')):
        coverage = client.get(url).json()['layout_coverage']
        result = client.get(url+'/pages/1/layout').json()
    assert coverage['indexed_pages'] == 1 and coverage['total_pages'] == 1
    assert coverage['pages'][0]['tables'] == 1
    assert result['status'] == 'available'
    assert result['inventory']['tables'][0]['cells'] == inventory['tables'][0]['cells']
    assert before == store.rows('SELECT * FROM evidence')
    assert client.get(url+'/pages/0/layout').status_code == 422
    assert client.get(url+'/pages/2/layout').status_code == 422
    assert client.get('/api/documents/'+'f'*32+'/pages/1/layout').status_code == 404


def test_stale_inventory_is_not_exported_as_current(client):
    store = client.app.state.store
    did = source(store)
    build_layout(store, did, [1])
    with store.connect() as c:
        c.execute('UPDATE documents SET sha256=? WHERE id=?', ('b'*64, did))
    result = client.get(f'/api/documents/{did}/pages/1/layout').json()
    assert result['status'] == 'stale_rebuild_required' and result['inventory'] is None
    assert client.get(f'/api/documents/{did}').json()['layout_coverage']['indexed_pages'] == 0


def test_source_inspection_rejects_changed_original_bytes(client):
    store = client.app.state.store
    did = source(store)
    eid = store.rows('SELECT id FROM evidence WHERE document_id=?', (did,))[0]['id']
    assert client.get(f'/api/documents/{did}/pdf').status_code == 200
    store.original_path(did).write_bytes(b'%PDF-modified')
    assert client.get(f'/api/documents/{did}/pdf').status_code == 409
    assert client.get(f'/api/evidence/{eid}/image').status_code == 409
