import json
import pytest
from finsight.source_context import source_context
from finsight.store import identity


def populate(store, issuer='Alpha', page=1):
    did = identity(issuer)
    blocks = [dict(id=identity(issuer,i), document_id=did, page=page, section='Unknown',
                   text=f'Exact source block {i}\n(5) 0', bbox=json.dumps({'rect':[20,i*30,300,i*30+20]}))
              for i in range(7)]
    store.publish(dict(id=did,company=issuer,name='filing.pdf',kind='DRHP',filing_date='2024-01-01',
                       status='source_available',page_count=2,sha256='a'*64,warnings='[]'),blocks)
    return blocks


def test_context_preserves_sources_and_cannot_cross_issuer(store):
    blocks = populate(store)
    populate(store,'Beta')
    context = source_context(store,blocks[3]['id'])
    assert [r['id'] for r in context['neighbors']] == [blocks[i]['id'] for i in [1,2,4,5]]
    assert all(r['company']=='Alpha' and r['page']==1 for r in context['neighbors'])
    assert context['neighbors'][0]['text'] == blocks[1]['text']


def test_context_never_crosses_page(store):
    blocks = populate(store)
    with store.connect() as c:
        c.execute('UPDATE evidence SET page=2 WHERE id=?',(blocks[4]['id'],))
    context = source_context(store,blocks[3]['id'])
    assert blocks[4]['id'] not in [r['id'] for r in context['neighbors']]


def test_budget_omissions_are_explicit(store):
    blocks = populate(store)
    context = source_context(store,blocks[3]['id'],max_chars=0)
    assert context['status']=='partial' and len(context['omitted_ids'])==4
    assert context['neighbors']==[]


def test_missing_geometry_is_not_invented(store):
    blocks=populate(store)
    with store.connect() as c: c.execute('UPDATE evidence SET bbox=NULL WHERE id=?',(blocks[3]['id'],))
    assert source_context(store,blocks[3]['id'])['status']=='geometry_unavailable'
    with pytest.raises(ValueError): source_context(store,'missing')
