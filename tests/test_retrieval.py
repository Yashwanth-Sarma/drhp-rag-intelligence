import numpy as np
import pytest
from finsight.store import identity
from finsight.retrieval import build_index, dense_search, fuse, windows


class FakeEmbedding:
    def __init__(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        self.tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

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


def test_windows_use_tokenizer_offsets_when_available():
    class Encoded:
        offsets = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return Encoded()

    assert list(windows('abcdefghij', Tokenizer(), size=3, overlap=1)) == ['abcdef', 'efghij']


def test_window_tokenizer_unwraps_and_does_not_truncate_or_mutate():
    from types import SimpleNamespace
    from finsight.retrieval import window_tokenizer
    from tokenizers import processors
    backend = FakeEmbedding()
    backend.tokenizer.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    backend.tokenizer.enable_truncation(max_length=12)
    backend.tokenizer.enable_padding(length=12)
    tokenizer = window_tokenizer(SimpleNamespace(model=backend))
    text = ' '.join(str(i) for i in range(700))
    parts = list(windows(text, tokenizer))
    assert parts[-1].endswith('699')
    assert all(len(tokenizer.encode(p, add_special_tokens=False).ids) <= 180 for p in parts)
    assert backend.tokenizer.truncation['max_length'] == 12
    assert backend.tokenizer.padding['length'] == 12
    with pytest.raises(ValueError, match='tokenizer unavailable'):
        window_tokenizer(object())


def test_tokenless_normalized_blocks_are_reported(store):
    from tokenizers import normalizers
    model = FakeEmbedding()
    model.tokenizer.normalizer = normalizers.BertNormalizer()
    _, empty = seed(store, 'Alpha', None, '\x01')
    seed(store, 'Alpha', None, 'supplier')
    manifest = build_index(store, model)
    assert manifest['tokenless_ids'] == [empty]
