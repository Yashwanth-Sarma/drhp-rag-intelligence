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
    return dict(page_hit_at_12=bool(ranks), reciprocal_rank=1/min(ranks) if ranks else 0,
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
