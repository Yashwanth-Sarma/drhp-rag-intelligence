"""Import downloaded official fixtures and retain acquisition provenance."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime'),str(ROOT)]
from finsight.ingest import ingest_pdf
from finsight.store import Store

REFERENCES=[
 ('ola-drhp-2023.pdf','Ola Electric Mobility Limited','2023-12-22','2023-12-26','https://www.sebi.gov.in/sebi_data/attachdocs/dec-2023/1703580725954.PDF'),
 ('swiggy-udrhp-2024.pdf','Swiggy Limited','2024-09-26','2024-09-27','https://www.sebi.gov.in/sebi_data/attachdocs/sep-2024/1727416339523.pdf')]

if __name__=='__main__':
    store=Store(ROOT/'data')
    results=[]
    for name,company,dated,listed,url in REFERENCES:
        path=ROOT/'data/reference'/name
        if not path.exists(): raise SystemExit(f'Download {url} to {path} first.')
        did=ingest_pdf(store,path.read_bytes(),company,name,'DRHP',dated)
        with store.connect() as c:
            manifest=json.loads(c.execute('SELECT payload FROM document_manifests WHERE document_id=?',(did,)).fetchone()[0])
            manifest.update(source_url=url,source_listing_date=listed,acquired_on='2026-09-15',document_variant='Updated DRHP I' if 'swiggy' in name else 'DRHP')
            c.execute('UPDATE document_manifests SET payload=? WHERE document_id=?',(json.dumps(manifest),did))
        results.append(dict(document_id=did,filename=name,sha256=store.document(did)['sha256'],pages=store.document(did)['page_count']))
    (ROOT/'data/reference/import-results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
