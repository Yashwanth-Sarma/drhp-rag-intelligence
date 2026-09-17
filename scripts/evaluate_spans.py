"""Evaluate exact developer-labelled evidence IDs, not just page membership."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime'),str(ROOT)]
from finsight.store import Store
from finsight.research import search
from finsight.retrieval import retrieve
from finsight.rerank import rerank
from finsight.source_context import source_context
from finsight.evaluation import validate_span_label, require_dense_ready, dataset_hash

if __name__=='__main__':
    dataset=json.loads((ROOT/'evals/datasets/disclosure-spans-v1.json').read_text(encoding='utf-8'))
    store=Store(ROOT/'data')
    for case in dataset['cases']: validate_span_label(store,case)
    rows=[]
    for case in dataset['cases']:
        scope=dict(company=case['company'],document_id=case['document_id'])
        hybrid=retrieve(store,case['question'],**scope,limit=40)
        require_dense_ready(hybrid)
        channels=dict(lexical=search(store,case['question'],**scope,limit=12),
                      hybrid=hybrid['evidence'][:12],
                      reranked=rerank(store,case['question'],hybrid['evidence'],limit=12))
        result=dict(id=case['id'],question=case['question'],expected_id=case['evidence_id'],channels={})
        for name,items in channels.items():
            if any(e['document_id']!=case['document_id'] for e in items):
                raise ValueError('Document leakage in evaluation.')
            ids=[e['id'] for e in items]
            expanded=set(ids)
            for eid in ids:
                expanded.update(e['id'] for e in source_context(store,eid)['neighbors'])
            rank=ids.index(case['evidence_id'])+1 if case['evidence_id'] in ids else None
            result['channels'][name]=dict(rank=rank,ids=ids,direct_hit=rank is not None,
                                         hit_with_neighbors=case['evidence_id'] in expanded)
        rows.append(result)
    output=dict(dataset=dataset['name'],dataset_sha256=dataset_hash(dataset),annotation=dataset['annotation'],
                at=datetime.now(timezone.utc).isoformat(),cases=rows,
                summary={mode:{metric:sum(r['channels'][mode][metric] for r in rows)/len(rows)
                         for metric in ['direct_hit','hit_with_neighbors']} for mode in channels},
                limitation='Four development cases across three source blocks. Context-expanded hits use a larger evidence budget. Neither hit metric proves answer faithfulness.')
    path=ROOT/'evals/results/disclosure-spans-v1.json'
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output['summary'],indent=2))
