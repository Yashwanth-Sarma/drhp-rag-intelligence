import pytest
from finsight.evaluation import page_metrics, require_dense_ready, dataset_hash


def test_metrics_respect_rank_and_cutoff():
    results = [dict(id=str(i), company='Alpha', page=i) for i in range(1,14)]
    assert page_metrics(results, 'Alpha', [3])['reciprocal_rank'] == 1/3
    assert not page_metrics(results, 'Alpha', [13])['page_hit_at_12']


def test_evaluation_rejects_fallback_duplicates_and_issuer_leak():
    with pytest.raises(ValueError): require_dense_ready({'dense_status':'unavailable'})
    row = dict(id='1', company='Alpha', page=1)
    with pytest.raises(ValueError): page_metrics([row,row], 'Alpha', [1])
    with pytest.raises(ValueError): page_metrics([row], 'Beta', [1])


def test_dataset_fingerprint_tracks_label_changes():
    assert dataset_hash([{'question':'q','pages':[1]}]) != dataset_hash([{'question':'q','pages':[2]}])
