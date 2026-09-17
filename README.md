# FinSight

**Current:** [Stage 09 source context](docs/stages/09_SOURCE_CONTEXT.md), 63 passing
tests, source-bound development evaluation and inspectable nearby passages.
Financial table interpretation and live Qwen validation remain outstanding.

**Latest verified stage:** [Stage 08](docs/stages/08_RETRIEVAL_VERIFICATION.md).
58 tests pass; the dense index contains 18,986 windows. Real-filing retrieval,
OKF/API integration and citation rendering passed smoke checks. The six-case
benchmark does not establish financial accuracy or production readiness.

Local evidence research foundation for a professional Indian IPO research application.
Run from this directory:

```powershell
python run.py
```

Open http://127.0.0.1:8765. The local workspace runtime is installed. For a fresh machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python run.py
```

Use Python 3.12–3.14; current verification used Windows Python 3.14. Dependencies need a clean-environment lock/CI stage before deployment.

```powershell
python scripts/test.py --tb=short
python scripts/import_reference.py
```

The second command imports two official PDFs already downloaded in `data/reference`; it does not fetch arbitrary URLs. Originals and generated artifacts are ignored by Git. Legacy code is untouched.

## Available

PDF upload, hash-based duplicate handling, isolated parser timeout, all-page coverage records, exact native text blocks, source geometry, original PDF and highlighted page inspection, scoped BM25 search, immutable search exports, six-section extractive research packs, ingestion history and a local interface. Financial observation/comparison endpoints are experimental user-attested primitives, not independently validated facts.

## Not yet available

Table-cell reconstruction, OCR, live-tested Qwen synthesis, professional report/verified chart generation, authenticated external deployment and production validation. Printed page labels are not inferred from arbitrary numeric lines. Date filtering uses the supplied document date, not verified historical public availability. Do not publish this local development server externally.

Start a new session with [START_HERE.md](docs/START_HERE.md). Source-based design observations are in [filing study](docs/SEBI_FILING_STUDY.md).
# September 16 retrieval update

Local hybrid retrieval, OKF methodology context, citation integrity checks and a
Qwen draft adapter are implemented. Read [the architecture](docs/RAG_OKF_ARCHITECTURE.md)
and [the exact verification/resume status](docs/stages/08_RETRIEVAL_VERIFICATION.md) before
continuing. Both official reference filings are imported. Model downloads
completed; dense-index publication and the first real-corpus evaluation are confirmed.
The current UI produces evidence packs, not validated investment research.
