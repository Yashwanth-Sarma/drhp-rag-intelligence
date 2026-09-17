# Stage 08 — Retrieval verification and correction

Completed 2026-09-16. Supersedes Stage 07's uncertain index/test status.

## Verified state

Execution access worked again. No reset credit was redeemed.
Dense index: **18,986 windows, 18,198 blocks, 384 dimensions**.
All **58 tests passed**, with one upstream Starlette deprecation warning.
Ten model files were hashed in `evals/model-lock.json`; the subsequent comparison
passed. JavaScript syntax passed `node --check web/app.js`.

The real-corpus API smoke passed: Ola hybrid retrieval, three OKF concepts,
source-page PNG rendering and a six-section Swiggy extractive brief. This is not
visual browser inspection or validation of financial analysis.

## Results

The six existing page-location cases were unchanged. They are developer smoke
checks with queries similar to source headings, not a held-out expert benchmark.

| Method | Target page in top 12 | Mean reciprocal rank |
|---|---:|---:|
| Keyword BM25 | 6/6 | 0.7222 |
| Hybrid RRF | 6/6 | 0.5238 |
| Hybrid then MiniLM reranking | 6/6 | 0.8056 |

Hybrid ranks worse than keyword search on this set. Reranking leads overall, but
not on every case. A page hit may be a heading rather than an answer-bearing span;
alternative relevant pages are not exhaustively labelled. No default reranker
promotion was made from these six cases. Existing hybrid application behaviour
remains experimental; do not describe it as a proven quality improvement.

Artifacts:

* `evals/results/retrieval-smoke-prefix-baseline.json`: first measurement.
* `evals/results/retrieval-smoke.json`: current measurement with dataset hash,
  document hashes, dense manifest, package versions, evidence IDs and ranks.
* `evals/results/api-smoke.json`: integration results and stored research run IDs.
* `evals/model-lock.json`: snapshot paths, sizes and SHA-256 hashes. Records local
  bytes, not independent authenticity; runtime does not yet enforce this audit.

## Correction and evaluation gates

The reranker previously discarded everything after 2,000 characters per block.
It now scores overlapping spans and uses their maximum score. A 32-window cap
per candidate bounds work; output records scored/total windows and omissions.
Tests verify tail evidence, work limits and rejection of invalid model scores.
The six smoke metrics stayed unchanged: this fixes a demonstrated coverage
defect, without claiming an aggregate quality gain.

Word windows still do not guarantee subword-token coverage. Heading/table context
is still incomplete. Larger chunking changes need evaluation before promotion.

The evaluator now rejects dense fallback, duplicate IDs and issuer leakage using
explicit exceptions, not Python assertions that can be disabled. Rank/cutoff and
dataset-fingerprint behaviour are tested in `tests/test_evaluation.py`.

## Reproduce

```powershell
python scripts/test.py --tb=short -x -p no:cacheprovider
python scripts/audit_models.py
python scripts/evaluate_retrieval.py
python scripts/smoke_local.py
```

The initial `audit_models.py --record` refuses overwrite. Intentional model
changes need a reviewed new manifest and a rebuilt index. No rebuild was needed
for this stage because source vectors and embeddings were unchanged.

## Next stage

1. Create auditable span-level labels for natural analyst questions, paraphrases,
   duplicate disclosures, periods and unsupported questions. Preserve source
   hashes/pages/exact spans. No finance professional has reviewed the project.
2. Implement tokenizer-aware chunks and heading/parent context while retaining
   original evidence IDs. Do not treat neighbouring text as automatically cited.
3. Compare against the saved baseline and retain regressions in the report.
4. Annotate financial tables before parser selection: headers, units, signs,
   periods, accounting basis and footnotes need separate checks.
5. Connect Qwen only after serving is available. Semantic claim review, reliable
   table interpretation, final report rendering and production controls remain.

## Next-session prompt

Read docs/START_HERE.md and docs/stages/08_RETRIEVAL_VERIFICATION.md. Continue
FinSight autonomously with source-aware retrieval: create auditable span-level
labels from the existing Ola and Swiggy filings, then implement token-aware
chunking and heading/parent context while preserving evidence IDs. Compare with
the saved baseline and retain failures. Run appropriate tests and write the next
stage handoff. Do not claim financial verification from retrieval relevance.
