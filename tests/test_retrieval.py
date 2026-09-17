import numpy as np
import pytest
from finsight.store import identity
from finsight.retrieval import build_index, dense_search, fuse, windows


class FakeEmbedding:
    def passage_embed(self, texts, batch_size=32):
        return [self.vector(t) for t in texts]

    def query_embed(self, text):
        return [self.vector(text)]

    def vector(self, text):
        return np.array([1.0, 0.0] if 'supplier' in text else [0.0, 1.0])


def seed(store, issuer, date, text):
    did, eid = identity(issuer, date, text), identity('evidence', issuer, date, text)
    store.publish(dict(id=did, company=issuer, name='filing.pdf', kind='DRHP',
                       filing_date=date, status='native_text', page_count=1,
                       sha256='a'*64, warnings='[]'),
                  [dict(id=eid, document_id=did, page=1, section='Unknown', text=text, bbox=None)])
    return did, eid


def test_dense_scope_before_ranking_and_unknown_dates(store):
    _, target = seed(store, 'Alpha', '2024-01-01', 'supplier concentration')
    seed(store, 'Beta', '2024-01-01', 'supplier concentration')
    seed(store, 'Alpha', None, 'supplier concentration unknown')
    seed(store, 'Alpha', '2025-01-01', 'supplier concentration future')
    build_index(store, FakeEmbedding())
    rows, state = dense_search(store, 'supplier', company='Alpha', as_of='2024-12-31', model=FakeEmbedding())
    assert state == 'ready' and [r['id'] for r in rows] == [target]


def test_stale_index_never_silently_searches_partial_corpus(store):
    seed(store, 'Alpha', None, 'supplier')
    build_index(store, FakeEmbedding())
    seed(store, 'Alpha', None, 'new financial evidence')
    assert dense_search(store, 'supplier', model=FakeEmbedding()) == ([], 'stale_rebuild_required')


def test_failed_rebuild_preserves_previous_index(store):
    seed(store, 'Alpha', None, 'supplier')
    build_index(store, FakeEmbedding())
    class Broken(FakeEmbedding):
        def vector(self, text): return np.array([float('nan'), 0])
    with pytest.raises(ValueError): build_index(store, Broken())
    assert dense_search(store, 'supplier', model=FakeEmbedding())[1] == 'ready'


def test_rank_fusion_deduplicates_and_is_deterministic():
    a, b = dict(id='a'), dict(id='b')
    result = fuse([a, a, b], [b, a])
    assert len(result) == 2
    assert result[0]['retrieval_ranks'] == {'lexical': 1, 'dense': 2}


def test_windows_keep_all_tokens_and_numeric_signs():
    text = ' '.join(str(i) for i in range(500)) + ' (5) 0'
    parts = list(windows(text))
    assert all(p in text for p in parts)
    assert parts[0].startswith('0 ') and parts[-1].endswith('(5) 0')
    assert set(text.split()) == set(' '.join(parts).split())
