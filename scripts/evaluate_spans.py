"""Evaluate exact developer-labelled evidence IDs, not just page membership."""
import json
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime'),str(ROOT)]
from finsight.store import Store
from finsight.research import search
from finsight.retrieval import retrieve
from finsight.rerank import rerank, VERSION as RERANK_VERSION
from finsight.source_context import source_context
from finsight.evaluation import validate_span_label, require_dense_ready, dataset_hash, pack_context


def expanded_blocks(store, ranked):
    for item in ranked:
        yield item
        yield from source_context(store, item['id'])['neighbors']


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='evals/datasets/disclosure-spans-v2.json')
    parser.add_argument('--output', default='evals/results/disclosure-spans-v2.json')
    parser.add_argument('--max-chars', type=int, default=12000)
    args = parser.parse_args()
    dataset=json.loads((ROOT/args.dataset).read_text(encoding='utf-8'))
    store=Store(ROOT/'data')
    for case in dataset['cases']: validate_span_label(store,case)
    rows=[]
    for case in dataset['cases']:
        scope=dict(company=case['company'],document_id=case['document_id'])
        hybrid=retrieve(store,case['question'],**scope,limit=40)
        require_dense_ready(hybrid)
        channels=dict(lexical=search(store,case['question'],**scope,limit=40),
                      hybrid=hybrid['evidence'],
                      reranked=rerank(store,case['question'],hybrid['evidence'],limit=40))
        result=dict(id=case['id'],question=case['question'],expected_id=case['evidence_id'],channels={})
        for name,items in channels.items():
            if any(e['document_id']!=case['document_id'] for e in items):
                raise ValueError('Document leakage in evaluation.')
            ids=[e['id'] for e in items[:12]]
            expanded=set(ids)
            for eid in ids:
                expanded.update(e['id'] for e in source_context(store,eid)['neighbors'])
            direct_context=pack_context(items,args.max_chars)
            expanded_context=pack_context(expanded_blocks(store,items),args.max_chars)
            rank=ids.index(case['evidence_id'])+1 if case['evidence_id'] in ids else None
            result['channels'][name]=dict(rank=rank,ids=ids,direct_hit=rank is not None,
                                         candidate_ids=[e['id'] for e in items],
                                         hit_with_neighbors=case['evidence_id'] in expanded,
                                         direct_context=direct_context, expanded_context=expanded_context,
                                         direct_budget_hit=case['evidence_id'] in direct_context['ids'],
                                         expanded_budget_hit=case['evidence_id'] in expanded_context['ids'])
        rows.append(result)
    output=dict(dataset=dataset['name'],dataset_sha256=dataset_hash(dataset),annotation=dataset['annotation'],
                at=datetime.now(timezone.utc).isoformat(),cases=rows,
                dense_manifest=json.loads(store.rows('SELECT payload FROM dense_manifest WHERE id=1')[0]['payload']),
                reranker_version=RERANK_VERSION,
                budget=dict(unit='source_text_characters',maximum=args.max_chars,
                            candidate_depth=40,
                            policy='Both budget arms start from up to 40 ranked candidates. Whole blocks; skip oversized blocks, deduplicate IDs, retain order. Expanded order is each direct candidate then its same-page neighbors. Metadata excluded. Native layout not scored. Legacy hit metrics retain top 12.'),
                summary={mode:{metric:sum(r['channels'][mode][metric] for r in rows)/len(rows)
                         for metric in ['direct_hit','hit_with_neighbors','direct_budget_hit','expanded_budget_hit']} for mode in channels},
                limitation='Development labels only, not expert-reviewed gold. Budget metrics share the same character cap, with actual consumption recorded; this is not a model-token budget. Legacy unbounded neighbor hits are descriptive only. No hit metric proves answer faithfulness.')
    path=ROOT/args.output
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output['summary'],indent=2))
