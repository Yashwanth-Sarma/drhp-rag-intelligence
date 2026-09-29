# Stage 11 - Filing layout inspection

Completed 2026-09-18. Review Stage 10 for tokenizer, corpus and evaluation changes.

## Delivered

The filing library now shows native-layout coverage separately from native text
coverage. Indexed pages show table/heading candidate counts; other pages say
`Not indexed` or `Rebuild required`. Zero detected tables on an indexed page is
distinct from an unindexed page.

Each indexed page has a full layout inspector with table cells, nearby source
lines, font-based heading candidates, native text lines, an original-page link,
JSON export and navigation back to page coverage. Full inventories remain
accessible when a search context budget omits a table. Missing detected cells
are distinguished from empty strings; values are not normalized or converted.
Wide tables scroll within the panel and numeric strings remain intact.

`GET /api/documents/{id}` includes layout coverage.
`GET /api/documents/{id}/pages/{page}/layout` returns an explicit availability
state and the full source-bound inventory when current. Requests never run the
PDF parser or rebuild indexes. Stale source hashes/versions suppress inventories.
Document and physical-page bounds are checked.

Review also found that original-PDF and highlighted-image responses did not
check the file against its registered hash. Both now reject mismatches with
HTTP 409 before normal inspection. Cached text/inventories remain derived from
the registered original; these checks are not semantic validation.

## Verification

- Full Python suite: 75 passed, one upstream Starlette/anyio deprecation warning.
- Source-bound span evaluation: eight cases, three issuers; see Stage 10's
  equal-cap results and preserved misses.
- Real-corpus API smoke: dense `ready`, source PNG rendered, six report sections.
- Research browser: desktop/mobile native table display and source inspection,
  no page overflow or JavaScript errors.
- Layout browser: desktop/mobile inspector, all four Hyundai table fragments
  retained, JSON export and back navigation, no dialog overflow or JS errors.
- JavaScript syntax and Python compilation checks passed.

Browser screenshots were inspected locally in `tmp/stage10-*.png` and
`tmp/stage11-*.png`. Machine-readable results are in
`evals/results/browser-stage10.json` and `browser-stage11.json`.
The app is served locally at `http://127.0.0.1:8765`.

```powershell
python scripts/test.py --tb=short -x -p no:cacheprovider
python scripts/evaluate_spans.py
python scripts/smoke_local.py
# With run.py serving and Node/Playwright available:
node scripts/verify_research_browser.cjs
node scripts/verify_layout_browser.cjs
```

The OneDrive sandbox required approved execution for tests/model runtimes and
some original-file reads. A temporary approval-service credit failure interrupted
verification on September 17; the remaining checks completed September 18.

## Limits and next module

Layout covers 14 selected pages, with only three visually inspected this stage.
The known Hyundai table fragmentation is deliberately retained. No verified
hierarchy, automatic continuation linking, expert financial validation, model
selection claim or live Qwen validation is provided. This is an inspection
interface, not a table-correction or financial-fact approval workflow.

Next: build an exact-cell annotation/evaluation module on at least 20 reviewed
pages. Bind labels to document hash, physical page, region and raw cell text;
include missing cells, parentheses, zero, dash, annual/interim headers and
cross-page continuations. Compare native table strategies or a layout parser
against those labels, preserving failures. Freeze these results first, keep
canonical evidence immutable and do not silently merge the known fragments.

Changes remain local. Publication checkout: `tmp/github-publish`; last existing
checkpoint `f467fbf`, branch `codex/finsight-rag-okf-stages-01-08`. No root push.
