import hashlib
import json
import sqlite3
import re
from contextlib import contextmanager
from pathlib import Path

SCHEMA = '''
CREATE TABLE IF NOT EXISTS documents (
 id TEXT PRIMARY KEY, company TEXT NOT NULL, name TEXT NOT NULL,
 kind TEXT NOT NULL, filing_date TEXT, status TEXT NOT NULL,
 page_count INTEGER NOT NULL, sha256 TEXT, warnings TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence (
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
 page INTEGER NOT NULL, section TEXT NOT NULL, text TEXT NOT NULL, bbox TEXT);
CREATE INDEX IF NOT EXISTS evidence_document ON evidence(document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
 evidence_id UNINDEXED, text, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS observations (
 id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
 id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS document_pages (
 document_id TEXT NOT NULL REFERENCES documents(id), page INTEGER NOT NULL CHECK(page>0),
 label TEXT NOT NULL, width REAL NOT NULL, height REAL NOT NULL,
 rotation INTEGER NOT NULL, status TEXT NOT NULL, block_count INTEGER NOT NULL,
 PRIMARY KEY(document_id,page));
CREATE TABLE IF NOT EXISTS document_manifests (
 document_id TEXT PRIMARY KEY REFERENCES documents(id), payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ingestion_jobs (
 id TEXT PRIMARY KEY, state TEXT NOT NULL, document_id TEXT,
 message TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
'''

def identity(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:32]

class MetadataConflict(ValueError):
    """Existing byte-identical document has conflicting metadata."""

class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'originals').mkdir(exist_ok=True)
        self.path = self.root / 'finsight.sqlite3'
        with self.connect() as c:
            c.executescript(SCHEMA)
            c.execute('PRAGMA user_version=2')

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.path, timeout=20)
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA foreign_keys=ON')
        try:
            with c:
                yield c
        finally:
            c.close()

    def rows(self, sql, args=()):
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, args)]

    def documents(self):
        return self.rows('SELECT d.*, count(e.id) AS blocks FROM documents d LEFT JOIN evidence e ON e.document_id=d.id GROUP BY d.id ORDER BY company,name')

    def evidence(self, eid):
        rows = self.rows('SELECT e.*, d.company,d.name,d.kind,d.status,d.filing_date FROM evidence e JOIN documents d ON d.id=e.document_id WHERE e.id=?', (eid,))
        return rows[0] if rows else None

    def original_path(self, did):
        if not re.fullmatch(r'[a-f0-9]{32}', did):
            raise ValueError('Invalid document identifier.')
        return self.root / 'originals' / f'{did}.pdf'

    def document(self, did):
        rows = self.rows('SELECT * FROM documents WHERE id=?', (did,))
        return rows[0] if rows else None

    def check_existing(self, document):
        existing = self.document(document['id'])
        if existing:
            for key in ('company', 'kind', 'filing_date'):
                if existing[key] != document[key]:
                    raise MetadataConflict(f'Identical PDF already registered with different {key}. Metadata edits require a separate reviewed revision.')
        return existing

    def publish(self, document, blocks, pages=(), manifest=None):
        """A document and its search entries are published in one transaction."""
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            existing = c.execute('SELECT * FROM documents WHERE id=?', (document['id'],)).fetchone()
            if existing:
                for key in ('company', 'kind', 'filing_date'):
                    if existing[key] != document[key]:
                        raise MetadataConflict(f'Document already registered with different {key}.')
                return document['id']
            c.execute('INSERT INTO documents VALUES (:id,:company,:name,:kind,:filing_date,:status,:page_count,:sha256,:warnings)', document)
            for b in blocks:
                c.execute('INSERT INTO evidence VALUES (:id,:document_id,:page,:section,:text,:bbox)', b)
                c.execute('INSERT INTO search(evidence_id,text) VALUES (?,?)', (b['id'], b['text']))
            for p in pages:
                c.execute('INSERT INTO document_pages VALUES (:document_id,:page,:label,:width,:height,:rotation,:status,:block_count)', p)
            if manifest:
                c.execute('INSERT INTO document_manifests VALUES (?,?)', (document['id'], json.dumps(manifest)))
        return document['id']

    def job(self, jid, state, message, document_id=None):
        with self.connect() as c:
            c.execute('INSERT INTO ingestion_jobs(id,state,document_id,message) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,document_id=excluded.document_id,message=excluded.message,updated_at=CURRENT_TIMESTAMP', (jid,state,document_id,message))

    def import_legacy(self, path):
        """Read text/metadata only. Never load untrusted pickle or tensor artifacts."""
        c = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
        try:
            records = {}
            for row in c.execute('SELECT id,key,string_value,int_value,float_value FROM embedding_metadata'):
                records.setdefault(row[0], {})[row[1]] = next((v for v in row[2:] if v is not None), None)
        finally:
            c.close()
        groups = {}
        for r in records.values():
            if r.get('chroma:document'):
                groups.setdefault(r.get('source_file') or r.get('company_key', 'legacy'), []).append(r)
        for name, records in groups.items():
            records.sort(key=lambda r: str(r.get('chunk_id','')))
            did = identity('legacy-v2', name, records)
            doc = dict(id=did, company=records[0].get('company_name', 'Unknown'), name=name,
                       kind=records[0].get('doc_type', 'Unknown'), filing_date=None,
                       status='legacy_unverified', page_count=max(int(r.get('page_number', 1)) for r in records),
                       sha256=None, warnings=json.dumps(['Original PDF unavailable; legacy cleaning may have removed numbers. Filing date unverified. Do not use for verified financial calculations.']))
            blocks = [dict(id=identity(did, r.get('chunk_id', i)), document_id=did,
                           page=int(r.get('page_number', 1)), section=r.get('section', 'General'),
                           text=r.get('original_text') or r['chroma:document'], bbox=None)
                      for i, r in enumerate(records)]
            self.publish(doc, blocks)
        return len(groups)
