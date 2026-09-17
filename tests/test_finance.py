import pytest
from pydantic import ValidationError
from finsight.finance import Observation,add_observation,compare
from finsight.ingest import ingest_pdf

def observation(**updates):
    result=dict(company='Example Limited',metric='revenue_operations',value='1000',currency='INR',scale='million',period_start='2023-04-01',period_end='2024-03-31',basis='consolidated',evidence_id='a'*32,quote='1000',reviewer='Test reviewer',review_confirmed=True)
    result.update(updates)
    return result

@pytest.mark.parametrize('value',['NaN','Infinity','-Infinity','1e-999','1e999'])
def test_numeric_bounds(value):
    with pytest.raises(ValidationError): Observation(**observation(value=value))

def test_user_attestation_requires_exact_source_and_value(store,pdf_bytes):
    did=ingest_pdf(store,pdf_bytes,'Example Limited','test.pdf','DRHP','2024-06-01')
    block=next(b for b in store.rows('SELECT * FROM evidence') if '1000' in b['text'])
    o=Observation(**observation(evidence_id=block['id'],quote=block['text']))
    saved=add_observation(store,o)
    assert saved['validation']=='user_attested'
    with pytest.raises(ValueError,match='does not occur'):
        add_observation(store,o.model_copy(update={'value':'999'}))
    with pytest.raises(ValueError,match='issuer'):
        add_observation(store,o.model_copy(update={'company':'Other'}))
    with pytest.raises(ValueError,match='exact substring'):
        add_observation(store,o.model_copy(update={'quote':'fabricated 1000'}))

def test_negative_zero_and_scale_comparison():
    a=dict(observation(value='-5'),id='a')
    b=dict(observation(value='0',scale='crore'),id='b')
    r=compare(a,b)
    assert r['first']=='-5000000' and r['second']=='0' and r['difference']=='5000000'

@pytest.mark.parametrize('field,value',[('metric','total_income'),('basis','standalone'),('currency','USD'),('period_start','2023-07-01')])
def test_incomparable_rejected(field,value):
    a=dict(observation(),id='a');b=dict(observation(),id='b');b[field]=value
    with pytest.raises(ValueError):compare(a,b)

def test_growth_and_invalid_baseline():
    a=dict(observation(),id='a')
    b=dict(observation(value='1200',period_start='2024-04-01',period_end='2025-03-31'),id='b')
    assert compare(a,b,'growth')['percent_change']=='20.00'
    a['value']='0'
    with pytest.raises(ValueError,match='baseline'):compare(a,b,'growth')
