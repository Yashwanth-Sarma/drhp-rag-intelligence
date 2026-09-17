"""Small, transparent SEBI page-location smoke set. Not a held-out benchmark."""
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.runtime'), str(ROOT)]
from finsight.store import Store
from finsight.research import search
from finsight.retrieval import retrieve
from finsight.rerank import rerank
from finsight.evaluation import page_metrics, dataset_hash, require_dense_ready

CASES = [
    ('ola-assets', 'Ola Electric Mobility Limited', 'summary restated consolidated balance sheet total assets', [71]),
    ('ola-financials', 'Ola Electric Mobility Limited', 'summary restated consolidated financial information', [70,71,72,73]),
    ('ola-risk', 'Ola Electric Mobility Limited', 'risk factors losses negative cash flows', [32,33,34,35]),
    ('swiggy-assets', 'Swiggy Limited', 'summary restated consolidated statement assets liabilities equity', [81]),
    ('swiggy-financials', 'Swiggy Limited', 'summary restated consolidated financial information', [80,81,82,83]),
    ('swiggy-risk', 'Swiggy Limited', 'risk factors losses negative cash flows', [39,40,41,42]),
]

if __name__ == '__main__':
    store, results = Store(ROOT/'data'), []
    for name, company, query, pages in CASES:
        start = time.perf_counter()
        lexical = search(store, query, company=company, limit=12)
        hybrid = retrieve(store, query, company=company, limit=40)
        require_dense_ready(hybrid)
        reranked = rerank(store, query, hybrid['evidence'], limit=12)
        row = dict(id=name, company=company, question=query, target_pages=pages,
                   dense_status=hybrid['dense_status'], latency_ms=round((time.perf_counter()-start)*1000))
        for mode, evidence in [('lexical', lexical), ('hybrid', hybrid['evidence'][:12]), ('reranked', reranked)]:
            row[mode] = page_metrics(evidence, company, pages)
        results.append(row)
    output = dict(dataset='sebi-page-smoke-v1', dataset_sha256=dataset_hash(CASES),
                  created_at=datetime.now(timezone.utc).isoformat(),
                  corpus=[{k:d[k] for k in ['id','sha256','company','blocks']} for d in store.documents()],
                  dense_manifest=json.loads(store.rows('SELECT payload FROM dense_manifest WHERE id=1')[0]['payload']),
                  packages={name:version(name) for name in ['fastembed','onnxruntime','numpy']},
                  reranker='Xenova/ms-marco-MiniLM-L-6-v2',
                  limitation='Developer-authored page-location checks, not expert-labelled answer relevance or held-out evaluation. Page matches may be headings rather than answer-bearing spans.',
                  cases=results, summary={mode:dict(page_hit_rate_at_12=sum(r[mode]['page_hit_at_12'] for r in results)/len(results),
                  mean_reciprocal_rank=sum(r[mode]['reciprocal_rank'] for r in results)/len(results)) for mode in ['lexical','hybrid','reranked']})
    out = ROOT/'evals/results'; out.mkdir(parents=True, exist_ok=True)
    (out/'retrieval-smoke.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output['summary'], indent=2))
