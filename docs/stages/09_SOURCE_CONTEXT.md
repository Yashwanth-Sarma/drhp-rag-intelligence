# Stage 09 — Source context and exact-span evaluation

Completed 2026-09-17. This is a bounded increment, not complete structural parsing.

## Implemented and checked

`finsight/source_context.py` expands each research hit to at most two preceding
and two following geometrically ordered blocks on the same physical page of the
same document. Blocks keep their original IDs, exact text and source coordinates.
The 12,000-character context budget records omitted IDs. Missing/legacy geometry
produces an explicit unavailable state. No headings, table relationships or
claim support are inferred from proximity.

Research API exports retain this context. The browser offers “Read nearby source
passages” below each hit, with individual source inspection for every neighbour.
The Qwen adapter still consumes direct hits only; it has not been expanded to use
neighbours automatically. Reports retain their existing extractive behaviour.
Geometry order is a simple y/x ordering and may not reflect multi-column reading
order. It is explicitly spatial context rather than a reconstructed document tree.

Added four agent-authored natural-language questions covering three exact risk
disclosure blocks in Ola and Swiggy. Each label binds document SHA-256, issuer,
document ID, physical page, evidence ID and an exact quote. The evaluator checks
all bindings before running, and refuses stale dense fallback. These are
development labels, not expert-reviewed or held-out gold data.

## Measurements

| Method | Direct labelled-span hits, top 12 | Hits including spatial neighbours |
|---|---:|---:|
| Keyword | 2/4 | 3/4 |
| Hybrid | 3/4 | 4/4 |
| Hybrid plus reranker | 3/4 | 4/4 |

Expanded context uses a larger evidence budget, so this is not an equal-budget
ranking comparison. Valid alternative supporting spans have not been exhaustively
labelled. The small sample does not establish financial answer accuracy or a
production model choice. Preserve the misses and add broader cases next.

Artifacts:

* `evals/datasets/disclosure-spans-v1.json`: questions and source-bound labels.
* `evals/results/disclosure-spans-v1.json`: ranked IDs, direct/context hits and
  dataset fingerprint.
* `scripts/evaluate_spans.py`: reproducible evaluator.

Validation: **63 tests passed**, one upstream Starlette warning. Real-corpus API
smoke and browser JavaScript syntax checks passed after context integration.
Browser visual inspection and live Qwen serving remain unverified.

## GitHub checkpoint

Stages 01–08 were pushed as commit `172ed26` on branch
`codex/finsight-rag-okf-stages-01-08` of
`https://github.com/Yashwanth-Sarma/drhp-rag-intelligence`.
Publication uses `tmp/github-publish`, a separate checkout preserving the original
repository history. The workspace root still has an unborn local Git repository;
do not assume a root `git push` will work. Local PDFs, vectors and downloaded
weights were not added. Prior dependency requirements were preserved in
`requirements-legacy.txt` in the publishing checkout. Stage 09 is a follow-up
checkpoint on the same branch; inspect its Git log for the final hash.

## Reproduce

```powershell
python scripts/test.py --tb=short -x -p no:cacheprovider
python scripts/evaluate_spans.py
python scripts/smoke_local.py
```

## Next-session prompt

Read docs/START_HERE.md and docs/stages/09_SOURCE_CONTEXT.md. Continue with
tokenizer-aware retrieval windows and real heading/table-context preservation.
Keep existing evidence IDs and original text immutable. Freeze the current
evaluation outputs before changes; expand span-level cases beyond the current
three blocks and two issuers, and preserve failures. Compare at equal context
budgets before claiming improvements. Do not claim expert or semantic validation
from page, quote or neighbour membership. Run appropriate checks and write the
next stage handoff. GitHub publishing checkout is tmp/github-publish.
