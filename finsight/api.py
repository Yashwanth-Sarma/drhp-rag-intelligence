import json
import os
from datetime import date
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from .store import Store, MetadataConflict
from .ingest import ingest_pdf, MAX_BYTES, IngestionBusy
from .research import research, report
from .finance import Observation, add_observation, compare
from .contracts import Query, ReportRequest, Comparison, Identifier, DocumentKind

ROOT=Path(__file__).resolve().parents[1]

def create_app(data_dir=None):
    store=Store(data_dir or os.environ.get('FINSIGHT_DATA', ROOT/'data'))
    app=FastAPI(title='FinSight',version='0.2.0',docs_url=None,redoc_url=None)
    app.state.store=store
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])

    @app.middleware('http')
    async def local_boundary(request, call_next):
        origin=request.headers.get('origin')
        if request.method not in ('GET','HEAD','OPTIONS'):
            if (origin and origin not in ('http://127.0.0.1:8765','http://localhost:8765')) or request.headers.get('x-finsight-client')!='local-ui':
                return Response('Local client header and same-origin request required',status_code=403)
        length=request.headers.get('content-length','0')
        if not length.isdigit() or int(length)>MAX_BYTES:
            return Response('Request exceeds the supported size',status_code=413)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
        response.headers['Cache-Control']='no-store'
        return response

    @app.get('/api/status')
    def status():
        docs=store.documents()
        return dict(documents=len(docs),blocks=sum(d['blocks'] for d in docs),companies=sorted(set(d['company'] for d in docs)),mode='Local · no paid API calls',generation=False,version='0.2.0')

    @app.get('/api/documents')
    def documents(): return store.documents()

    @app.post('/api/import-legacy')
    def legacy():
        path=ROOT/'legacy-review/data/embeddings_stage1/chroma.sqlite3'
        if not path.exists(): raise HTTPException(404,'Legacy artifact is unavailable.')
        return dict(imported=store.import_legacy(path))

    @app.post('/api/documents/pdf')
    async def upload(request: Request, company: str, name: str='filing.pdf', kind: DocumentKind='DRHP', filing_date: date | None=None):
        if not company.strip() or len(company)>120 or len(name)>200:
            raise HTTPException(422,'Provide a valid company and filename.')
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>MAX_BYTES: raise HTTPException(413,'PDF exceeds 35 MB.')
        try:
            did=await run_in_threadpool(ingest_pdf,store,bytes(body),company,Path(name).name,kind,filing_date.isoformat() if filing_date else None)
        except IngestionBusy as e: raise HTTPException(429,str(e),headers={'Retry-After':'5'})
        except MetadataConflict as e: raise HTTPException(409,str(e))
        except (ValueError,RuntimeError) as e: raise HTTPException(422,str(e))
        except ImportError: raise HTTPException(503,'PDF parser unavailable. Install the application requirements.')
        return dict(id=did)

    @app.post('/api/research')
    def query(q: Query):
        try: return research(store,**q.model_dump(mode='json'))
        except ValueError as e: raise HTTPException(422,str(e))

    @app.post('/api/reports')
    def brief(q: ReportRequest):
        try: return report(store,**q.model_dump(mode='json'))
        except ValueError as e: raise HTTPException(422,str(e))

    @app.get('/api/evidence/{eid}')
    def evidence(eid: Identifier):
        e=store.evidence(eid)
        if not e: raise HTTPException(404,'Evidence not found.')
        pages=store.rows('SELECT * FROM document_pages WHERE document_id=? AND page=?',(e['document_id'],e['page']))
        e['page_metadata']=pages[0] if pages else None
        return e

    @app.get('/api/evidence/{eid}/image')
    def image(eid: Identifier):
        e=store.evidence(eid)
        if not e or not e['bbox']: raise HTTPException(404,'Source image unavailable for legacy evidence.')
        import pymupdf
        path=store.original_path(e['document_id'])
        if not path.exists(): raise HTTPException(404,'Original PDF is missing; reimport the same document to restore it.')
        with pymupdf.open(path) as pdf:
            page=pdf[e['page']-1]
            annotation=page.add_rect_annot(pymupdf.Rect(json.loads(e['bbox'])['rect']))
            annotation.set_colors(stroke=(0,0.55,0.45))
            annotation.set_border(width=2)
            annotation.update()
            return Response(page.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).tobytes('png'),media_type='image/png')

    @app.get('/api/documents/{did}/pdf')
    def original(did: Identifier):
        rows=store.rows('SELECT id FROM documents WHERE id=?',(did,))
        if not rows: raise HTTPException(404,'Document not found.')
        path=store.original_path(did)
        if not path.exists(): raise HTTPException(404,'Original PDF unavailable.')
        return FileResponse(path,media_type='application/pdf',filename='source.pdf',content_disposition_type='inline')

    @app.get('/api/documents/{did}')
    def document(did: Identifier):
        d=store.document(did)
        if not d: raise HTTPException(404,'Document not found.')
        d['pages']=store.rows('SELECT * FROM document_pages WHERE document_id=? ORDER BY page',(did,))
        manifests=store.rows('SELECT payload FROM document_manifests WHERE document_id=?',(did,))
        d['manifest']=json.loads(manifests[0]['payload']) if manifests else None
        return d

    @app.get('/api/documents/{did}/pages/{page}')
    def page_evidence(did: Identifier,page: int):
        if not store.document(did): raise HTTPException(404,'Document not found.')
        return store.rows('SELECT id,document_id,page,section,text,bbox FROM evidence WHERE document_id=? AND page=? ORDER BY rowid',(did,page))

    @app.get('/api/jobs')
    def jobs(): return store.rows('SELECT * FROM ingestion_jobs ORDER BY created_at DESC,id DESC LIMIT 50')

    @app.post('/api/observations')
    def observation(o: Observation):
        try: return add_observation(store,o)
        except ValueError as e: raise HTTPException(422,str(e))

    @app.get('/api/observations')
    def observations(): return [json.loads(r['payload']) for r in store.rows('SELECT payload FROM observations')]

    @app.post('/api/compare')
    def comparison(q: Comparison):
        obs={o['id']:o for o in observations()}
        if q.first not in obs or q.second not in obs: raise HTTPException(404,'Observation not found.')
        try: return compare(obs[q.first],obs[q.second],q.mode)
        except ValueError as e: raise HTTPException(422,str(e))

    @app.get('/api/runs/{rid}')
    def run(rid: str):
        rows=store.rows('SELECT payload FROM runs WHERE id=?',(rid,))
        if not rows: raise HTTPException(404,'Research run not found.')
        return Response(rows[0]['payload'],media_type='application/json',headers={'Content-Disposition':'attachment; filename="finsight-evidence.json"'})

    app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
    return app
