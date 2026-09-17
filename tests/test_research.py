import json
import pytest
from finsight.research import search,research,report
from finsight.store import identity

def seed(store,company,date,text):
    did=identity(company,date,text);eid=identity(did,'block')
    store.publish(dict(id=did,company=company,name=company+'.pdf',kind='DRHP',filing_date=date,status='legacy_unverified',page_count=1,sha256=None,warnings='[]'),
                  [dict(id=eid,document_id=did,page=1,section='General',text=text,bbox=None)])
    return did,eid

def test_scope_date_and_no_unknown_date_leak(store):
    a,eid=seed(store,'Alpha','2024-01-01','Revenue operations risk')
    seed(store,'Beta','2024-01-01','Revenue operations risk')
    seed(store,'Alpha',None,'Revenue operations risk unknown')
    seed(store,'Alpha','2025-01-01','Revenue operations risk later')
    result=research(store,'revenue',company='Alpha',as_of='2024-12-31')
    assert [e['id'] for e in result['evidence']]==[eid]
    assert result['corpus']['unknown_date_excluded']==1
    assert result['mode']=='evidence_search'
    assert json.loads(store.rows('SELECT payload FROM runs')[0]['payload'])==result
    with pytest.raises(ValueError,match='different issuer'): search(store,'revenue',company='Beta',document_id=a)

@pytest.mark.parametrize('query',['" OR * --','AND NOT NEAR()','what is the','nonexistentphrasexyz'])
def test_safe_empty_or_unmatched_queries(store,query):
    seed(store,'Alpha',None,'revenue only')
    assert search(store,query)==[]

def test_report_has_six_sections_and_explicit_limits(store):
    seed(store,'Alpha',None,'revenue risk operations')
    r=report(store,'Alpha')
    assert len(r['sections'])==6
    assert r['status']=='partial_research_pack'
    assert any(s['status']=='gap' for s in r['sections'])
    assert r['observations']==[]

def test_publication_rollback(store):
    did=identity('broken')
    doc=dict(id=did,company='Alpha',name='x',kind='DRHP',filing_date=None,status='legacy_unverified',page_count=1,sha256=None,warnings='[]')
    block=dict(id='b'*32,document_id=did,page=1,section='General',text='revenue',bbox=None)
    with pytest.raises(Exception):store.publish(doc,[block,block])
    assert store.documents()==[] and store.rows('SELECT * FROM search')==[]
