import json
from urllib.parse import urlencode
from finsight.store import identity

def upload(client,pdf_bytes):
    return client.post('/api/documents/pdf?'+urlencode(dict(company='Example Limited',name='example.pdf',kind='DRHP',filing_date='2024-06-01')),content=pdf_bytes,headers={'Content-Type':'application/pdf'})

def test_upload_search_source_image_export_and_report(client,pdf_bytes):
    r=upload(client,pdf_bytes);assert r.status_code==200,r.text
    did=r.json()['id']
    r=client.post('/api/research',json={'question':'revenue','company':'Example Limited'})
    assert r.status_code==200,r.text
    result=r.json();eid=result['evidence'][0]['id']
    e=client.get('/api/evidence/'+eid).json()
    assert e['page_metadata']['page']==1
    image=client.get(f'/api/evidence/{eid}/image')
    assert image.status_code==200 and image.content.startswith(b'\x89PNG')
    source=client.get(f'/api/documents/{did}/pdf')
    assert source.content==pdf_bytes
    assert client.get('/api/runs/'+result['id']).json()==result
    assert client.get(f'/api/documents/{did}').json()['manifest']['parser']=='PyMuPDF'
    assert client.post('/api/reports',json={'company':'Example Limited'}).status_code==200
    assert client.get('/').status_code==200

def test_request_boundaries(client,pdf_bytes):
    assert client.post('/api/import-legacy',headers={'X-FinSight-Client':''}).status_code==403
    assert client.post('/api/import-legacy',headers={'Origin':'https://evil.test'}).status_code==403
    assert client.get('/api/status',headers={'Host':'evil.test'}).status_code==400
    assert client.post('/api/research',json={'question':'revenue','unexpected':True}).status_code==422
    assert client.post('/api/research',json={'question':'  '}).status_code==422
    assert client.get('/api/evidence/not-an-id').status_code==422
    assert client.get('/api/documents/'+'f'*32+'/pdf').status_code==404
    assert client.post('/api/research',json={'question':'revenue','as_of':'bad'}).status_code==422
    r=client.get('/api/status')
    assert "script-src 'self'" in r.headers['content-security-policy']

def test_metadata_conflict_is_409(client,pdf_bytes):
    assert upload(client,pdf_bytes).status_code==200
    r=client.post('/api/documents/pdf?company=Other%20Limited&name=x.pdf&kind=DRHP&filing_date=2024-06-01',content=pdf_bytes)
    assert r.status_code==409
