import json
import re
import time
import uuid
from pathlib import Path
from .contracts import EXTRACTION_VERSION
from .source_context import source_context

STOP = set('what is are was were the a an of to in for and how did does do with from about company its their it as by explain describe'.split())

def search(store, question, company=None, document_id=None, as_of=None, limit=12):
    validate_scope(store, company, document_id)
    if not 1 <= limit <= 50:
        raise ValueError('Search limit must be between 1 and 50.')
    tokens = list(dict.fromkeys(t.lower() for t in re.findall(r'[\w]+', question) if len(t)>1 and t.lower() not in STOP))[:32]
    if not tokens:
        return []
    query = ' OR '.join('"'+t+'"' for t in tokens)
    clauses, args = ['search MATCH ?'], [query]
    if company:
        clauses.append('d.company=?'); args.append(company)
    if document_id:
        clauses.append('d.id=?'); args.append(document_id)
    if as_of:
        clauses.append('d.filing_date IS NOT NULL AND d.filing_date<=?'); args.append(as_of)
    args.append(limit)
    sql = '''SELECT e.*,d.company,d.name,d.kind,d.status,d.filing_date,
             bm25(search) AS retrieval_score FROM search JOIN evidence e ON e.id=search.evidence_id
             JOIN documents d ON d.id=e.document_id WHERE ''' + ' AND '.join(clauses) + ' ORDER BY retrieval_score,e.id LIMIT ?'
    return store.rows(sql, args)

def validate_scope(store, company, document_id):
    if document_id:
        doc = store.document(document_id)
        if not doc:
            raise ValueError('Selected document does not exist.')
        if company and doc['company'] != company:
            raise ValueError('Selected document belongs to a different issuer.')

def corpus_snapshot(store, company=None, document_id=None, as_of=None):
    docs = [d for d in store.documents() if (not company or d['company']==company) and (not document_id or d['id']==document_id)]
    excluded = sum(1 for d in docs if as_of and not d['filing_date'])
    selected = [d for d in docs if not as_of or (d['filing_date'] and d['filing_date']<=as_of)]
    return dict(documents=[{k:d[k] for k in ('id','name','sha256','status','filing_date','blocks')} for d in selected],
                unknown_date_excluded=excluded, date_policy='User-supplied filing dates; publication availability not independently verified.',
                search_version='sqlite-fts5-bm25-v1', extraction_version=EXTRACTION_VERSION)

def research(store, question, company=None, document_id=None, as_of=None):
    start = time.perf_counter()
    retrieval = retrieve_evidence(store, question, company, document_id, as_of)
    evidence = retrieval['evidence']
    result = dict(id=str(uuid.uuid4()), question=question, scope=dict(company=company, document_id=document_id, as_of=as_of),
                  mode='evidence_search', status='evidence_found' if evidence else 'insufficient_evidence',
                  message='Relevant excerpts for your review. These are search results, not a verified answer.' if evidence else 'No matching evidence in the selected scope. This does not establish that the disclosure is absent.',
                  evidence=evidence, latency_ms=round((time.perf_counter()-start)*1000),
                  provider=None, cost=0, pipeline_version=retrieval['pipeline'])
    result['retrieval']={k:v for k,v in retrieval.items() if k!='evidence'}
    result['knowledge_context']=knowledge_context(question)
    result['source_context']=[source_context(store, e['id']) for e in evidence]
    result['corpus']=corpus_snapshot(store, company, document_id, as_of)
    result['coverage_note']='Ranked excerpts only; results do not establish support or complete disclosure coverage. Knowledge guidance is not issuer evidence.'
    result['latency_ms']=round((time.perf_counter()-start)*1000)
    with store.connect() as c:
        c.execute('INSERT INTO runs(id,payload) VALUES (?,?)', (result['id'], json.dumps(result)))
    return result

def retrieve_evidence(store, question, company=None, document_id=None, as_of=None, limit=12):
    try:
        from .retrieval import retrieve
    except ImportError:
        return dict(evidence=search(store, question, company, document_id, as_of, limit),
                    pipeline='lexical-fallback-v1', dense_status='runtime_not_installed')
    return retrieve(store, question, company, document_id, as_of, limit)

def knowledge_context(question):
    try:
        from .okf import retrieve_context
    except ImportError:
        return dict(concepts=[], errors=[dict(error='OKF runtime not installed')])
    return retrieve_context(Path(__file__).resolve().parents[1]/'knowledge', question)

SECTIONS = [
    ('Business and offering', 'business operations products services objects offer proceeds'),
    ('Financial performance', 'revenue income profit loss cash flows financial statements'),
    ('Operating model and dependencies', 'customers suppliers concentration segments dependency'),
    ('Risks and governance', 'risk litigation related party governance regulatory'),
    ('Management and outlook', 'management strategy growth outlook expectations'),
    ('Open questions and evidence gaps', 'restated qualifications contingencies material uncertainty')]

def report(store, company, document_id=None, as_of=None):
    sections = []
    for title, query in SECTIONS:
        retrieval = retrieve_evidence(store, query, company, document_id, as_of, limit=4)
        blocks = retrieval['evidence']
        sections.append(dict(title=title, evidence=blocks, status='excerpts_found' if blocks else 'gap',
                             retrieval={k:v for k,v in retrieval.items() if k!='evidence'},
                             knowledge_context=knowledge_context(query)))
    observations = [json.loads(r['payload']) for r in store.rows('SELECT payload FROM observations')]
    observations = [o for o in observations if o['company']==company and (not document_id or store.evidence(o['evidence_id'])['document_id']==document_id)]
    if as_of:
        observations = [o for o in observations if (store.evidence(o['evidence_id'])['filing_date'] or '9999')<=as_of]
    result = dict(id=str(uuid.uuid4()), company=company, document_id=document_id, as_of=as_of,
                  title=f'{company} | Company research brief', status='partial_research_pack',
                  limitation='Extractive research pack. Analytical synthesis and exhaustive coverage are not established. Source excerpts retain their original wording. Financial charts use manually reviewed observations only.',
                  sections=sections, observations=observations, pipeline_version='report-extractive-v1')
    result['corpus']=corpus_snapshot(store, company, document_id, as_of)
    with store.connect() as c:
        c.execute('INSERT INTO runs(id,payload) VALUES (?,?)', (result['id'], json.dumps(result)))
    return result
