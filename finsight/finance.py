import json
from datetime import date
from decimal import Decimal, InvalidOperation
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal
from .store import identity
from .contracts import Contract, Identifier

class Observation(Contract):
    company: str = Field(min_length=1,max_length=120)
    metric: Literal['revenue_operations','total_income','profit_after_tax','operating_cash_flow']
    value: str = Field(min_length=1, max_length=40)
    currency: Literal['INR','USD'] = 'INR'
    scale: Literal['units','thousand','lakh','million','crore'] = 'million'
    period_start: date
    period_end: date
    basis: Literal['consolidated','standalone']
    evidence_id: Identifier
    quote: str = Field(min_length=1,max_length=5000)
    reviewer: str = Field(min_length=2,max_length=120)
    review_confirmed: Literal[True]

    @field_validator('value')
    @classmethod
    def finite(cls, v):
        try:
            d=Decimal(v)
        except InvalidOperation:
            raise ValueError('Enter an unscaled decimal value, without commas.')
        if not d.is_finite() or abs(d)>Decimal('1e18'):
            raise ValueError('Value must be finite and within supported bounds.')
        if d.as_tuple().exponent < -8:
            raise ValueError('At most eight decimal places are supported.')
        return str(d)

    @model_validator(mode='after')
    def periods(self):
        if self.period_start > self.period_end:
            raise ValueError('Period start must precede period end.')
        return self

SCALES = {'units':1,'thousand':1000,'lakh':100000,'million':1000000,'crore':10000000}

def add_observation(store, o):
    e = store.evidence(o.evidence_id)
    if not e or e['company'] != o.company:
        raise ValueError('Evidence must belong to the selected issuer.')
    if e['status'] == 'legacy_unverified':
        raise ValueError('Attach the original PDF before reviewing financial observations.')
    if o.quote not in e['text']:
        raise ValueError('Quote must be an exact substring of the source block.')
    # Numeric membership is only an integrity check. The reviewer attests metric/period/header association.
    import re
    values = []
    for token in re.findall(r'\(?-?\d[\d,]*(?:\.\d+)?\)?', o.quote):
        try:
            values.append(Decimal(token.replace(',','').replace('(','-').replace(')','')))
        except InvalidOperation:
            pass
    if Decimal(o.value) not in values:
        raise ValueError('Entered value does not occur in the quoted evidence.')
    payload = o.model_dump(mode='json')
    payload['id'] = identity('observation', payload)
    payload['validation'] = 'user_attested'
    payload['limitations'] = 'Quote and value membership checked; metric, period and unit association are attested by the user, not independently verified.'
    with store.connect() as c:
        c.execute('INSERT OR IGNORE INTO observations VALUES (?,?)', (payload['id'],json.dumps(payload)))
    return payload

def compare(a,b, mode='peer'):
    if mode not in ('peer','growth'):
        raise ValueError('Unknown comparison mode.')
    for key in ['metric','currency','basis']:
        if a[key] != b[key]:
            raise ValueError(f'Incomparable {key}: {a[key]} / {b[key]}')
    if mode=='peer':
        if (a['period_start'],a['period_end']) != (b['period_start'],b['period_end']):
            raise ValueError('Peer comparison requires identical reporting periods.')
    else:
        if a['company'] != b['company']:
            raise ValueError('Growth requires the same issuer.')
        da = (date.fromisoformat(a['period_end'])-date.fromisoformat(a['period_start'])).days
        db = (date.fromisoformat(b['period_end'])-date.fromisoformat(b['period_start'])).days
        if abs(da-db)>1 or a['period_end']>=b['period_start']:
            raise ValueError('Growth requires ordered, non-overlapping periods of equal duration.')
        if (a['period_start'][5:],a['period_end'][5:]) != (b['period_start'][5:],b['period_end'][5:]):
            raise ValueError('Growth requires the same fiscal-calendar boundaries.')
    x=Decimal(a['value'])*SCALES[a['scale']]
    y=Decimal(b['value'])*SCALES[b['scale']]
    result = dict(input_ids=[a['id'],b['id']], first=str(x),second=str(y),difference=str(y-x),currency=a['currency'],unit='units',formula='second - first',percent_change=None)
    if mode=='growth':
        if x<=0:
            raise ValueError('Percentage growth from a zero or negative baseline is unsupported.')
        result['percent_change']=str(((y-x)/x*100).quantize(Decimal('0.01')))
        result['formula']='(second - first) / first × 100'
    return result
