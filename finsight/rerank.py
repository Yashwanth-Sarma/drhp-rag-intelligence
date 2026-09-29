"""Optional local cross-encoder experiment; disabled until corpus evaluation."""
import math
import threading
from functools import lru_cache

MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2'
VERSION = 'token-window-query-budget-v1'
_lock = threading.Lock()


@lru_cache(maxsize=2)
def model(cache, download=False):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=MODEL, cache_dir=cache, threads=4,
                            providers=['CPUExecutionProvider'], local_files_only=not download)


def rerank(store, question, evidence, limit=12, encoder=None):
    if not evidence:
        return []
    encoder = encoder or model(str(store.root/'models'))
    from .retrieval import windows, window_tokenizer
    if not 1 <= limit <= 50:
        raise ValueError('Reranker limit must be between 1 and 50.')
    candidates = evidence[:40]
    tokenizer = window_tokenizer(encoder)
    backend = getattr(encoder, 'model', encoder)
    truncation = backend.tokenizer.truncation
    max_tokens = truncation['max_length'] if truncation else 512
    query_tokens = len(tokenizer.encode(question, add_special_tokens=False).ids)
    available = max_tokens - query_tokens - tokenizer.num_special_tokens_to_add(True) - 8
    window_size = min(180, available)
    if window_size <= 30:
        raise ValueError('Question leaves insufficient reranker passage capacity.')
    # Score overlapping spans rather than silently dropping each block's tail.
    # Bound work for unusually large blocks and expose any remaining omission.
    passages, owners, coverage = [], [], []
    for index, item in enumerate(candidates):
        spans = list(windows(item['text'], tokenizer, size=window_size))
        chosen = spans[:32]
        if any(len(tokenizer.encode(question, span).ids) > max_tokens for span in chosen):
            raise ValueError('Reranker query/passage pair exceeds tokenizer capacity.')
        coverage.append(dict(windows_scored=len(chosen), windows_total=len(spans),
                             complete=len(chosen)==len(spans), window_tokens=window_size,
                             query_tokens=query_tokens, pair_token_limit=max_tokens, version=VERSION))
        passages.extend(chosen)
        owners.extend([index]*len(chosen))
    with _lock:
        scores = list(encoder.rerank(question, passages, batch_size=8))
    if len(scores) != len(passages) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError('Invalid reranker output.')
    best = {}
    for owner, score in zip(owners, scores):
        best[owner] = max(best.get(owner, -math.inf), float(score))
    ranked = [dict(e, rerank_score=best[i], rerank_coverage=coverage[i])
              for i,e in enumerate(candidates) if i in best]
    return sorted(ranked, key=lambda e:(-e['rerank_score'], e['id']))[:limit]
