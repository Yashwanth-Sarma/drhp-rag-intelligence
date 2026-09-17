"""Local dense retrieval + reciprocal rank fusion. Scores are ranks, not truth."""
import hashlib
import json
import re
import threading
from functools import lru_cache
import numpy as np

MODEL = 'BAAI/bge-small-en-v1.5'
VERSION = 'block-window-180-30-v1'
SCHEMA = '''CREATE TABLE IF NOT EXISTS dense_vectors (
 evidence_id TEXT NOT NULL REFERENCES evidence(id), window INTEGER NOT NULL,
 text_hash TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(evidence_id,window));
 CREATE TABLE IF NOT EXISTS dense_manifest (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);'''
_lock = threading.Lock()


@lru_cache(maxsize=2)
def embedding(cache, download=False):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=MODEL, cache_dir=cache, threads=4,
                         providers=['CPUExecutionProvider'], local_files_only=not download)


def windows(text):
    """Keep original spans; overlap is for retrieval only, never new evidence."""
    words = list(re.finditer(r'\S+', text))
    for start in range(0, len(words), 150):
        end = min(start + 180, len(words))
        yield text[words[start].start():words[end-1].end()]
        if end == len(words):
            break


def fingerprint(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps([row['id'], row['text']], ensure_ascii=False).encode())
    return h.hexdigest()


def corpus(store):
    return store.rows("SELECT e.* FROM evidence e JOIN documents d ON d.id=e.document_id WHERE d.status!='legacy_unverified' ORDER BY e.id")


def build_index(store, model=None, download=False):
    rows = corpus(store)
    if not rows:
        raise ValueError('Ingest source PDFs before building a dense index.')
    model = model or embedding(str(store.root / 'models'), download)
    entries = [(r['id'], i, span) for r in rows for i, span in enumerate(windows(r['text']))]
    # Stage in memory, then publish all vectors and their manifest atomically.
    vectors = list(model.passage_embed([e[2] for e in entries], batch_size=32))
    if len(vectors) != len(entries):
        raise ValueError('Embedding count mismatch; previous index retained.')
    payloads, dimension = [], None
    for (eid, index, text), vector in zip(entries, vectors):
        v = np.asarray(vector, dtype='<f4')
        if v.ndim != 1 or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
            raise ValueError('Invalid embedding; previous index retained.')
        dimension = dimension or len(v)
        if len(v) != dimension:
            raise ValueError('Inconsistent embedding dimensions.')
        v = v / np.linalg.norm(v)
        payloads.append((eid, index, hashlib.sha256(text.encode()).hexdigest(), v.tobytes()))
    manifest = dict(model=MODEL, dimension=dimension, chunk_version=VERSION,
                    corpus_hash=fingerprint(rows), blocks=len(rows), windows=len(entries))
    with store.connect() as c:
        c.executescript(SCHEMA)
        c.execute('BEGIN IMMEDIATE')
        current = [dict(r) for r in c.execute("SELECT e.* FROM evidence e JOIN documents d ON d.id=e.document_id WHERE d.status!='legacy_unverified' ORDER BY e.id")]
        if fingerprint(current) != manifest['corpus_hash']:
            raise ValueError('Corpus changed while indexing; rebuild required.')
        c.execute('DELETE FROM dense_vectors')
        c.executemany('INSERT INTO dense_vectors VALUES (?,?,?,?)', payloads)
        c.execute('INSERT OR REPLACE INTO dense_manifest VALUES (1,?)', (json.dumps(manifest),))
    return manifest


def dense_search(store, question, company=None, document_id=None, as_of=None, limit=40, model=None):
    tables = store.rows("SELECT name FROM sqlite_master WHERE name='dense_manifest'")
    if not tables:
        return [], 'not_built'
    manifests = store.rows('SELECT payload FROM dense_manifest WHERE id=1')
    if not manifests:
        return [], 'not_built'
    manifest = json.loads(manifests[0]['payload'])
    if manifest['model'] != MODEL or manifest['chunk_version'] != VERSION or manifest['corpus_hash'] != fingerprint(corpus(store)):
        return [], 'stale_rebuild_required'
    clauses, args = [], []
    for value, clause in [(company, 'd.company=?'), (document_id, 'd.id=?'), (as_of, 'd.filing_date IS NOT NULL AND d.filing_date<=?')]:
        if value:
            clauses.append(clause); args.append(str(value))
    rows = store.rows('SELECT v.evidence_id,v.vector FROM dense_vectors v JOIN evidence e ON e.id=v.evidence_id JOIN documents d ON d.id=e.document_id' + (' WHERE ' + ' AND '.join(clauses) if clauses else ''), args)
    if not rows:
        return [], 'ready'
    model = model or embedding(str(store.root / 'models'))
    with _lock:
        query = np.asarray(next(iter(model.query_embed(question))), dtype=np.float32)
    if query.shape != (manifest['dimension'],) or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
        raise ValueError('Invalid query embedding.')
    query /= np.linalg.norm(query)
    best = {}
    for row in rows:
        vector = np.frombuffer(row['vector'], dtype='<f4')
        if vector.shape != query.shape or not np.isfinite(vector).all():
            raise ValueError('Corrupt dense index; rebuild required.')
        score = float(vector @ query)
        best[row['evidence_id']] = max(best.get(row['evidence_id'], -2), score)
    ranked = sorted(best, key=lambda eid: (-best[eid], eid))[:limit]
    return [dict(store.evidence(eid), dense_similarity=best[eid]) for eid in ranked], 'ready'


def fuse(lexical, dense, limit=12):
    scores, records, ranks = {}, {}, {}
    for channel, items in [('lexical', lexical), ('dense', dense)]:
        seen = set()
        for rank, row in enumerate(items, 1):
            eid = row['id']
            if eid in seen:
                continue
            seen.add(eid)
            scores[eid] = scores.get(eid, 0) + 1 / (60 + rank)
            records.setdefault(eid, {}).update(row)
            ranks.setdefault(eid, {})[channel] = rank
    return [dict(records[eid], fusion_score=scores[eid], retrieval_ranks=ranks[eid])
            for eid in sorted(scores, key=lambda eid: (-scores[eid], eid))[:limit]]


def retrieve(store, question, company=None, document_id=None, as_of=None, limit=12):
    from .research import search
    lexical = search(store, question, company, document_id, as_of, limit=40)
    if not 1 <= limit <= 50:
        raise ValueError('Search limit must be between 1 and 50.')
    try:
        dense, state = dense_search(store, question, company, document_id, as_of)
    except (ImportError, OSError, ValueError, RuntimeError):
        dense, state = [], 'unavailable'
    return dict(evidence=fuse(lexical, dense, limit), dense_status=state,
                pipeline='hybrid-rrf-v1' if state == 'ready' else 'lexical-fallback-v1',
                score_policy='Ranking signals only; not confidence, support, or completeness.')
