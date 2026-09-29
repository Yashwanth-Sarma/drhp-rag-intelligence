# FinSight ML Systems Council Review — 2026-09-29

## Scope and decision

This is a focused, read-only architecture and failure-mode review of the
existing FinSight implementation, followed by one small evaluator correction.
The council adapted the role-based process from the supplied example to this
project:

| Role | Focus | Outcome |
|---|---|---|
| Codebase auditor | Data flow, correctness, maintainability | Financial observations are explicitly user-attested, but downstream reports must retain that status; parser resource limits and layout integrity need more work. |
| Retrieval and evaluation specialist | Search quality, labels, experiment design | The benchmark is useful for development, too small and unreviewed for quality claims; extend labels and evaluate retrieval at multiple depths and equal context budgets. |
| Financial safety and API specialist | Provenance, privacy, hostile input, deployment | Keep local-only positioning; public-availability chronology, parser resource containment, retention, and real authentication are release gates for broader use. |

The council agrees on the order: establish reviewed evaluation data and source
integrity first, then harden ingestion and metadata, then change retrieval or
financial interpretation only behind controlled evaluations. More model
complexity is not justified by the current evidence.

## Existing system and baseline

FinSight ingests native PDF text in an isolated worker, preserves original
bytes and hashes, stores evidence blocks in SQLite/FTS, supports a local dense
index and reciprocal-rank fusion, and exposes source inspection and extractive
research packs. It does not currently provide independently verified table
semantics or a production multi-user security boundary.

The checked-in `disclosure-spans-v2` result is a development baseline, not a
new run in this review. It contains eight agent-authored exact-evidence labels
across three issuers, without financial-expert review or exhaustive alternative
supporting spans:

| Retrieval path | Direct top-12 hit | Hit with same-page neighbors | Fixed-character direct budget hit | Fixed-character expanded budget hit |
|---|---:|---:|---:|---:|
| Lexical | 0.625 | 0.625 | 1.000 | 0.625 |
| Hybrid | 0.875 | 1.000 | 1.000 | 0.875 |
| Hybrid + reranker | 0.875 | 1.000 | 1.000 | 1.000 |

These are evidence-location hit rates, not answer quality, factual accuracy,
financial accuracy, or production generalization. The development sample does
not establish a reliable winner. Expanding with neighbors can also consume the
fixed character budget before a directly ranked passage is included.

## Council findings and risk order

| Priority | Finding | Real-world consequence | Decision |
|---|---|---|---|
| P0 | Historical `as_of` filters use user-entered filing date, not independently verified public availability. | A backtest can use a filing before it was actually public. | Add a separate provenance-backed availability timestamp; exclude unknown availability from historical scopes. |
| P0 | Financial observations validate quote/value membership, while metric, period, scale, and statement basis are attested by the user. | A wrong table-header association can produce a plausible but misleading comparison/chart. | Preserve `user_attested` labels throughout every export and view; build reviewable table-cell associations before automated fact extraction. |
| P0 | Parser has byte/page/text/time bounds but no OS memory/CPU quota; subprocess output capture is unbounded. | Pathological untrusted PDFs can exhaust resources despite timeout. | Add platform resource containment and bounded worker output before accepting broader untrusted uploads. |
| P1 | GET APIs expose documents, originals, jobs, observations, and research runs without identity checks. | Loopback-only operation is not suitable for shared or remotely exposed deployments. | Keep loopback-only guarantee; add authentication, tenant scoping and read authorization before deployment. |
| P1 | Spatial neighbor order is a y/x heuristic and can cross columns. | Nearby passages may appear related when they are not. | Keep it explicitly labeled as spatial context; add reviewed column/table structure before automated synthesis. |
| P1 | The eight-case benchmark has one expected block per question and no expert review, negative/no-answer slice, or broad query-type breakdown. | Tuning may overfit the small label set and miss singleton/abstention behavior. | Expand source-hash-bound labels to multiple valid spans and reviewed table/date/negative cases; retain issuer-held-out evaluation. |
| P1 | Reranking inspects bounded windows and exposes incomplete coverage, but the evaluator does not report gold-span omission at candidate depth. | A relevant answer late in a long block can disappear before ranking. | Report Recall@40, rank metrics, and per-case reranker coverage/omission before tuning its budget. |
| P2 | Research runs and uploaded originals have no user-facing retention/deletion lifecycle. | Sensitive deal research can persist indefinitely on disk and backups. | Define export/delete/retention behavior before a professional pilot. |
| P2 | Layout state can be based on stored hashes without checking the current original bytes. | A changed PDF may leave derived layout appearing current even if source inspection rejects it. | Couple layout-current status to original hash verification. |

## Small implementation change

`page_metrics` accepted arbitrary `k` while returning the fixed key
`page_hit_at_12`. It now returns a key matching the requested cutoff, with tests
for `k=3` and the existing top-12 behavior. This fixes evaluator output
semantics; it does not improve retrieval quality.

## Experiments and verification

- Full existing test suite after the correction: **75 passed**, one upstream
  Starlette deprecation warning, 27.40 seconds.
- Focused cutoff test: **1 passed**.
- Existing v2 retrieval result remains the baseline above; no new dense-index
  or reranker benchmark was run during this review.
- The initial sandboxed pytest attempts could not access/create pytest temp
  directories. The full run completed when run with the required filesystem
  access to pytest temporary storage.

## Next gated work

1. Build a reviewed benchmark of at least 30–50 questions over at least three
   issuers, including multiple valid evidence spans, table/header association,
   period/basis/unit, no-answer, and as-of availability cases.
2. Add immutable source availability and original-integrity status to the
   document contract before historical backtesting or layout-current claims.
3. Bound the parser worker's memory, CPU, stdout/stderr and output JSON; add
   malformed, encrypted, truncated, complex, timeout, and restart-recovery
   fixtures.
4. Benchmark lexical, dense, hybrid, and reranked paths at candidate depths 12
   and 40, under equal context budgets; report per-case misses and latency.
5. Only after these gates, implement table-cell extraction and controlled
   financial comparison workflows. Keep synthesis opt-in and human-reviewed.

## Explicitly not established

No validation score for financial answer accuracy, investment conclusions,
table extraction correctness, OCR quality, production security, multi-user
isolation, or a live Qwen deployment is claimed. No external lookup or paid
model call was used in this review.
