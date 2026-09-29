import math
import pytest
from finsight.rerank import rerank
from test_retrieval import FakeEmbedding


class Encoder(FakeEmbedding):
    def rerank(self, query, passages, batch_size=8):
        return [10 if 'tail_evidence' in p else 0 for p in passages]


def test_evidence_at_tail_is_scored_without_mutating_source(store):
    text = 'unrelated ' * 400 + 'tail_evidence'
    items = [dict(id='a',text='no match'),dict(id='b',text=text)]
    result = rerank(store, 'question', items, encoder=Encoder())
    assert result[0]['id'] == 'b'
    assert result[0]['text'] == text
    assert result[0]['rerank_coverage']['complete']
    assert 'rerank_score' not in items[0]


def test_rerank_work_limit_is_visible(store):
    result = rerank(store, 'q', [dict(id='a',text='word '*6000)], encoder=Encoder())
    assert not result[0]['rerank_coverage']['complete']
    assert result[0]['rerank_coverage']['windows_scored'] == 32


def test_invalid_scores_rejected(store):
    class Broken(Encoder):
        def rerank(self,*args,**kwargs): return [math.nan]
    with pytest.raises(ValueError): rerank(store,'q',[dict(id='a',text='evidence')],encoder=Broken())


def test_reranker_reserves_question_capacity_and_refuses_oversize_query(store):
    encoder = Encoder()
    encoder.tokenizer.enable_truncation(max_length=100)
    result = rerank(store, 'query '*40, [dict(id='a',text='word '*150)], encoder=encoder)
    coverage = result[0]['rerank_coverage']
    assert coverage['window_tokens'] == 52 and coverage['pair_token_limit'] == 100
    assert coverage['complete'] and coverage['windows_scored'] > 1
    assert encoder.tokenizer.truncation['max_length'] == 100
    with pytest.raises(ValueError, match='insufficient'):
        rerank(store, 'query '*95, [dict(id='a',text='word')], encoder=encoder)
