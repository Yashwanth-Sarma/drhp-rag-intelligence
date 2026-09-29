"""Retrieval evaluation invariants, separate from answer correctness."""
import hashlib
import json


def dataset_hash(cases):
    return hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def page_metrics(evidence, company, pages, k=12):
    if k < 1 or not pages:
        raise ValueError('Evaluation requires positive k and target pages.')
    if any(e['company'] != company for e in evidence):
        raise ValueError('Issuer leakage in evaluation results.')
    ids = [e['id'] for e in evidence]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate evidence IDs invalidate ranked metrics.')
    ranked = evidence[:k]
    ranks = [i for i, e in enumerate(ranked, 1) if e['page'] in pages]
    return dict(**{f'page_hit_at_{k}': bool(ranks)}, reciprocal_rank=1/min(ranks) if ranks else 0,
                pages=[e['page'] for e in ranked], ids=[e['id'] for e in ranked])


def require_dense_ready(result):
    if result.get('dense_status') != 'ready':
        raise ValueError('Hybrid evaluation requires a current dense index; fallback is not a hybrid result.')


def validate_span_label(store, label):
    doc = store.document(label['document_id'])
    source = store.evidence(label['evidence_id'])
    if not doc or doc['sha256'] != label['document_sha256']:
        raise ValueError('Evaluation source document hash mismatch.')
    if (not source or source['document_id'] != doc['id'] or
        source['company'] != label['company'] or source['page'] != label['page']):
        raise ValueError('Evaluation source scope mismatch.')
    if not label['quote'].strip() or label['quote'] not in source['text']:
        raise ValueError('Evaluation quote is absent from source.')
    return source


def pack_context(blocks, max_chars=12000):
    """One fixed character cap; include whole immutable blocks and record misses."""
    if max_chars < 0:
        raise ValueError('Context budget must be nonnegative.')
    result = dict(ids=[], omitted_ids=[], characters=0, max_chars=max_chars)
    seen = set()
    for block in blocks:
        if block['id'] in seen:
            continue
        seen.add(block['id'])
        cost = len(block['text'])
        if result['characters'] + cost > max_chars:
            result['omitted_ids'].append(block['id'])
            continue
        result['ids'].append(block['id'])
        result['characters'] += cost
    return result
