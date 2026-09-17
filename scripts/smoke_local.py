"""Exercise real-corpus API responses without starting a public server."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.runtime'), str(ROOT)]
from fastapi.testclient import TestClient
from finsight.api import create_app


if __name__ == '__main__':
    with TestClient(create_app(ROOT/'data'), headers={'X-FinSight-Client':'local-ui'}) as client:
        response = client.post('/api/research', json={'company':'Ola Electric Mobility Limited',
                              'question':'What losses and negative cash flows are disclosed?'})
        response.raise_for_status()
        result = response.json()
        if result['retrieval']['dense_status'] != 'ready' or not result['evidence']:
            raise SystemExit('Real-corpus hybrid retrieval is unavailable.')
        source = result['evidence'][0]
        rendered = client.get('/api/evidence/'+source['id']+'/image')
        rendered.raise_for_status()
        if not rendered.content.startswith(b'\x89PNG'):
            raise SystemExit('Source image did not render as PNG.')
        report = client.post('/api/reports',json={'company':'Swiggy Limited'})
        report.raise_for_status()
        pack = report.json()
        if len(pack['sections']) != 6:
            raise SystemExit('Expected six extractive report sections.')
        for section in pack['sections']:
            if any(e['company'] != 'Swiggy Limited' for e in section['evidence']):
                raise SystemExit('Cross-issuer evidence leaked into report.')
        summary = dict(research_run_id=result['id'], report_run_id=pack['id'],
                       dense_status=result['retrieval']['dense_status'],
                       evidence_count=len(result['evidence']),
                       knowledge_concepts=len(result['knowledge_context']['concepts']),
                       source_image_bytes=len(rendered.content), report_sections=len(pack['sections']),
                       limitation='API integration smoke, not visual UI inspection or financial answer validation.')
        out = ROOT/'evals/results'; out.mkdir(parents=True,exist_ok=True)
        (out/'api-smoke.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))
