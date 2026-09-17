from datetime import datetime, timezone
import pytest
from finsight.okf import read_concept, load_bundle
from finsight.grounding import check_claim
from test_retrieval import seed


def write(tmp_path, front, body='Example'):
    path = tmp_path/'metric.md'
    path.write_text('---\n'+front+'\n---\n'+body, encoding='utf-8')
    return path


def test_unknown_fields_preserved_and_not_authenticated(tmp_path):
    path = write(tmp_path, 'type: Novel\ncustom: 42\nverified: {by: "human:someone", at: "2026-01-01T00:00:00Z"}')
    c = read_concept(path, tmp_path)
    assert c['declared_trust'] == 'human-reviewed'
    assert not c['verification_authenticated']
    assert c['metadata']['custom'] == 42


def test_native_yaml_timestamp_is_json_serializable(tmp_path):
    import json
    concept = read_concept(write(tmp_path, 'type: X\ngenerated: {by: app/1, at: 2026-01-01T00:00:00Z}'), tmp_path)
    assert '2026-01-01T00:00:00+00:00' in json.dumps(concept)


@pytest.mark.parametrize('front', ['type: X\ntype: Y', 'type: X\ncustom: &a [*a]', 'type: X\nstale_after: "2026-01-01"', 'type: X\nsources: [{id: a}]'])
def test_invalid_metadata_rejected(tmp_path, front):
    with pytest.raises(ValueError): read_concept(write(tmp_path, front), tmp_path)


def test_stale_deprecated_and_missing_attribution(tmp_path):
    c = read_concept(write(tmp_path, 'type: X\nstale_after: "2026-01-01T00:00:00Z"'), tmp_path, datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert not c['eligible'] and c['stale']
    assert not read_concept(write(tmp_path, 'type: X\nstatus: deprecated'), tmp_path)['eligible']
    assert not read_concept(write(tmp_path, 'type: X', 'Claim[^missing]'), tmp_path)['eligible']


def test_no_path_escape_and_reserved_index(tmp_path):
    root = tmp_path/'bundle'; root.mkdir()
    path = write(tmp_path, 'type: X')
    with pytest.raises(ValueError): read_concept(path, root)
    (root/'index.md').write_text('listing')
    assert load_bundle(root) == dict(concepts=[], errors=[])


def test_quote_integrity_and_paraphrase_not_proven(store):
    did, eid = seed(store, 'Alpha', '2024-01-01', 'Revenue declined by 5%.')
    claim = dict(text='Revenue declined by 5%.', citations=[dict(evidence_id=eid, quote='Revenue declined by 5%.')])
    assert check_claim(store, claim, [eid], 'Alpha')['status'] == 'validated_extract'
    claim['text'] = 'Revenue grew by 5%.'
    assert check_claim(store, claim, [eid], 'Alpha')['status'] == 'needs_semantic_review'
    assert check_claim(store, claim, [eid], 'Beta')['status'] == 'rejected'
    assert check_claim(store, claim, [], 'Alpha')['status'] == 'rejected'
    assert check_claim(store, claim, [eid], 'Alpha', as_of='2023-01-01')['status'] == 'rejected'
    claim['citations'][0]['quote'] = 'Revenue grew by 5%.'
    assert 'quote_not_in_source' in check_claim(store, claim, [eid], 'Alpha')['errors']
