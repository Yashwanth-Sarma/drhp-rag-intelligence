import json
import httpx
import pytest
from finsight.local_qwen import draft_answer
from test_retrieval import seed


def test_local_draft_and_false_paraphrase(store):
    _, eid = seed(store, 'Alpha', '2024-01-01', 'Revenue declined by 5%.')
    def handler(request):
        payload = json.loads(request.content)
        assert payload['model'] == 'Qwen/Qwen3.6-35B-A3B'
        draft = dict(claims=[dict(text='Revenue increased by 5%.', citations=[dict(evidence_id=eid, quote='Revenue declined by 5%.')])], gaps=[])
        return httpx.Response(200, json={'choices':[{'message':{'content':json.dumps(draft)}}]})
    result = draft_answer(store, 'revenue', 'Alpha', transport=httpx.MockTransport(handler))
    assert result['status'] == 'draft_requires_review'
    assert result['claims'][0]['status'] == 'needs_semantic_review'


@pytest.mark.parametrize('endpoint', ['https://example.com/v1','http://127.0.0.1@example.com/v1','http://localhost/v1?secret=x'])
def test_no_remote_endpoint(store, endpoint):
    with pytest.raises(ValueError): draft_answer(store, 'revenue', 'Alpha', endpoint=endpoint)


def test_no_evidence_no_model_call(store):
    def handler(request): raise AssertionError('Model must not be called')
    assert draft_answer(store, 'revenue', 'Alpha', transport=httpx.MockTransport(handler))['status'] == 'insufficient_evidence'
