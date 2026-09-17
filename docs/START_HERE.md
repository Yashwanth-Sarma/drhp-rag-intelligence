# Session entry point

Updated 2026-09-16. Read this file and only the relevant stage file first.

**Current verified status:** [Stage 08](stages/08_RETRIEVAL_VERIFICATION.md)
supersedes the historical Stage 07 uncertainty below. The dense index is confirmed,
58 tests pass, real-filing retrieval and API checks completed, and model hashes
were recorded/checked. Keyword/hybrid/reranked page MRR on six smoke cases is
0.7222/0.5238/0.8056, not financial accuracy. Next: span-level labels and
token/heading-aware context. Historical status follows for traceability.

**Latest implementation:** read [RAG_OKF_ARCHITECTURE.md](RAG_OKF_ARCHITECTURE.md) and
[Stage 07 handoff](stages/07_RAG_OKF_HANDOFF.md). Hybrid retrieval, OKF context,
grounding gates and an opt-in local Qwen adapter have been added. Both reference
filings are imported (976 pages, 18,198 blocks). Free embedding and reranker
models were downloaded. The dense build was launched, but its completion could
not be confirmed after the execution session became unavailable. Do not assume
the dense index or real-filing evaluation is complete.

**Current direction:** read [PRODUCT_DIRECTION_V2.md](PRODUCT_DIRECTION_V2.md) before a stage. User fixed Qwen3.6-35B-A3B and requested financial fine-tuning plus hybrid RAG and canonical Google OKF. The new PDF is provisional architecture, not binding implementation instructions. Single-company professional reports are the proposed V1; cross-company/portfolio scope is deferred.

**Operational status:** the earlier import blockage was resolved during this session
and both PDFs were successfully imported. Later, automatic approval review again
reported no workspace credits and rejected an execution check. The last completed
test run passed 51 tests. A subsequent OKF timestamp serialization fix, one added
test and the latest UI changes have not been re-run/visually verified. No actual
Qwen server test or completed retrieval-quality result is available.

## Product

Professional company research for bankers, analysts and CFOs: a roughly six-page statistical, visual, evidence-linked report from DRHP/IPO filings and later earnings-call transcripts. Chat supports that workflow. Never represent retrieval relevance as confidence in truth.

## Current state

Stage 01 foundation and the Stage 07 retrieval increment are implemented. Search
uses hybrid ranking when a current dense index is available, otherwise an explicit
lexical fallback. Current brief remains an extractive source pack, not a final
analytical report. No paid inference API calls, reset-credit use, or deployment.

Files: `finsight/contracts.py` input schemas; `store.py` SQLite registry; `ingest.py` bounded extraction/publication; `pdf_worker.py` subprocess; `research.py` search/report scaffolding; `api.py` application factory; `finance.py` experimental arithmetic/attestation; `web/` no-build interface; `tests/` 33 invariant and API cases.

## Run

`python run.py` → localhost port 8765. `python scripts/test.py --tb=short` → tests. `python scripts/import_reference.py` → idempotent import of downloaded reference PDFs. On this machine shell sandbox may deny temp file writes; elevated test execution was needed. This is an execution environment restriction, not an application requirement.

## Stage handoffs

1. [Foundation](stages/01_FOUNDATION.md) — delivered with explicit limitations.
2. [Structure and tables](stages/02_STRUCTURE_AND_TABLES.md) — next priority.
3. [Retrieval and evaluation](stages/03_RETRIEVAL.md).
4. [Claims and financial reasoning](stages/04_CLAIMS_AND_FINANCE.md).
5. [Professional reports](stages/05_REPORTS.md).
6. [Pilot hardening](stages/06_PILOT.md).

Suggested new-session prompt: “Read docs/START_HERE.md and docs/stages/02_STRUCTURE_AND_TABLES.md. Implement the next bounded module, inspect real filing pages, test it, and update the handoff. Do not reread the full old conversation or use paid APIs without a configured budget.”

## Rules for continuation

User prioritizes careful stages over one-shot completion. Use existing modules only where sound. Inspect real source material for each feature. Preserve originals and legacy baseline. State test scope honestly. Update a stage file with changed files, commands, results, limitations and next action after each session. Work within available Codex allowance; do not redeem reset credits implicitly.

## Known limitations / next fixes

Date filter is document-date based: introduce public-availability dates before historical backtesting. Add reviewed metadata corrections/versioning. Jobs have durable status but no automatic crash resumption; an interrupted state must be reconciled against publication. Native extraction is isolated by process/time limit, not a complete OS sandbox/memory limit. SQLite is single local workspace; no multi-tenant authorization. Review source-hash integrity on reads before treating restored/mutated files as verified. Financial attestation is not independent numeric validation. Improve source context expansion: table headers can be separate blocks. Build real gold evidence labels, not answer-string-only judge scores.
