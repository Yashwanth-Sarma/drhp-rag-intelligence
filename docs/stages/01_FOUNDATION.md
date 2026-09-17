# Stage 01 — document and evidence foundation

Status: implemented local foundation; 33 automated tests pass (2026-09-15). Not industry-grade deployment approval.

## Delivered

Strict input contracts; content-derived identifiers; atomic SQLite document/block/search publication; immutable source bytes; identical-PDF metadata conflict rejection; missing-original restoration; per-page native-text coverage; parser version manifests; bounded subprocess extraction; concurrency limit; source image with bounding rectangle; local-origin write guard and security headers; read-only legacy import; lexical research and evidence exports; functional filing/research/source-pack UI.

## Verification

`python scripts/test.py --tb=short -p no:cacheprovider` passed 33 tests. Tested numeric-only lines, zero/negative values, blank pages, corrupt PDFs, duplicates, metadata conflicts, source restoration, timeout, failed publication/retry, parser concurrency, rotated geometry, scope/date exclusion, malformed queries, JSON validation, source image/PDF, run export, fiscal compatibility and value/quote membership. One upstream Starlette deprecation warning remains.

Tests use synthetic fixtures to isolate invariants. They do not establish table semantic accuracy. Actual Ola and Swiggy PDFs are separate inspection/integration fixtures. See `../SEBI_FILING_STUDY.md`.

## Remaining foundation work

Schema migration discipline and immutable metadata corrections; explicit public availability separate from document date; jobs crash reconciliation; parser memory quota and additional malformed/encrypted fixtures; full source hash verification on read; restoration audit; license review for commercial parser use; clean dependency lock and CI. Current frontend uses vanilla JS to complete a low-dependency vertical slice; a React migration is optional after contracts stabilize.

## Exit meaning

Usable local source inspection and search. No OCR/table correctness, semantic answer quality, investment conclusions or external-user readiness claimed.
