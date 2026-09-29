# Stage 10 - Token windows and native layout context

Completed 2026-09-18 after implementation and review on September 17-18.
This is a bounded increment; full structural parsing remains unfinished.

## Implemented

- Dense retrieval uses the actual FastEmbed backend tokenizer, cloned with
  padding and truncation disabled, with 180-token windows and 30-token overlap.
  Source substrings remain exact. Missing tokenizers fail the build explicitly.
- Reranking uses its own tokenizer and reserves capacity for the question and
  special tokens. Every scored pair is checked against the encoder limit.
  The existing 32-window work limit remains explicit in coverage metadata.
- The rebuilt index contains 25,694 blocks and 28,054 windows across Ola,
  Swiggy and Hyundai. Ten tokenless blocks are explicitly listed in its manifest.
  Chunk version is `block-token-window-180-30-v3`; downloads remain opt-in.
- `finsight/structure.py` preserves native font spans, outline entries,
  detected table cells, cell rectangles, detected headers and nearby margin
  lines in a separate SQLite inventory. Original evidence is never rewritten.
  Heading candidates use font properties. Tables use native ruled-line detection.
- Selected-page inventories are source-hash bound. Research context exports
  table candidates overlapping an anchor and at most two preceding font-based
  heading candidates. Neighbours and serialized layout candidates share the
  existing 12,000-character expansion cap; omissions are explicit.
- Browser search results expose native layout candidates and individual source
  inspection. Full page inventory inspection follows in Stage 11.

The original 18,198 Ola/Swiggy evidence records retain fingerprint
`d43e7f5088844e3b2bd2bae87d60207fe56d1a6b5b9c8e13552339a20d8358a6`.
The three-issuer index corpus fingerprint is
`147c39b4cbc460aa20720409a25ea4475b2c87ee2c2ac6d28ad5dadd23829d86`.

## Source and evaluation

Added the [Hyundai DRHP listed by SEBI](https://www.sebi.gov.in/filings/public-issues/jun-2024/hyundai-motor-india-limited-drhp_84186.html),
dated June 14, 2024, listed June 18, 2024. Its original PDF has 436 pages and
SHA-256 `5b2b9bda4d9a4cb849a672be432253888dc0254633b8500b497ebc4487102a21`.
Acquisition provenance is in `scripts/import_reference.py` and the local manifest.
Original PDFs, models and vectors remain untracked.

Stage 09 dataset/results remain unchanged; explicit frozen copies retain the
span results and API smoke. `disclosure-spans-v2.json` has eight agent-authored
questions covering seven evidence blocks across three issuers. All eight source
bindings pass validation. Labels remain development data with incomplete
alternative-span coverage and no independent expert review.

| Method | Direct top-12 hits | Direct, 12,000-char cap | Neighbours, same cap |
|---|---:|---:|---:|
| Keyword | 5/8 | 8/8 | 5/8 |
| Hybrid | 7/8 | 8/8 | 7/8 |
| Hybrid plus reranker | 7/8 | 8/8 | 8/8 |

Both budget arms start from the same up-to-40 ranked candidates for each method.
Whole blocks are deduplicated; oversized blocks are skipped and recorded.
Expanded order is each ranked hit followed by its spatial neighbours. Actual
consumption, selected IDs, omitted IDs and ranked candidates are recorded.
These are equal maximum character budgets, not identical consumption or model
token budgets. Native layout metadata is not scored. Legacy top-12 neighbour
metrics use more evidence and remain descriptive only.

Preserved failures: direct top-12 hybrid/reranked search misses Ola's loss-period
span; hybrid neighbour packing displaces the Swiggy licensing span. Keyword
neighbour packing misses Swiggy loss history and both Hyundai cases. Do not
claim improvement from context expansion or compare this changed corpus directly
with Stage 09 as a controlled ranking experiment.

## Layout inspection

Fourteen selected pages have inventories. Three rendered pages were visually
inspected: Ola physical 32, Swiggy physical 81 and Hyundai physical 36.
`evals/results/source-layout-v1.json` retains source hashes, table bounds,
candidate counts and cell fingerprints.

The Hyundai litigation table on physical page 36 is visually continuous, but
native detection splits it into four fragments and splits header text into
rows. This failure is retained. Multi-column ordering, merged-cell semantics,
header associations, units, footnotes and cross-page continuations remain
unverified. Margin lines preserve nearby text without asserting a relationship.
This does not meet Stage 02's 20-page review or parser-comparison acceptance.

## Verification and reproduction

The Stage 10 full suite passed 71 tests before the final reranker correction;
the combined Stage 10/11 verification is recorded in Stage 11. Real-corpus API
smoke reports dense status `ready`, a rendered source image and six extractive
report sections. Research browser checks pass desktop/mobile source inspection,
native table display, no page overflow and no JavaScript errors. Qwen serving
remains untested; generation and reports retain their existing behavior.

```powershell
python scripts/import_reference.py
python scripts/build_retrieval.py
python scripts/inspect_layout.py
python scripts/evaluate_spans.py
python scripts/test.py --tb=short -x -p no:cacheprovider
python scripts/smoke_local.py
```

The importer requires the official reference PDFs at the paths it names.
The offline CPU index rebuild is slow and publishes atomically only on success.
Browser checks require the app on port 8765 and Node with Playwright available:
`node scripts/verify_research_browser.cjs`.

## Checkpoint

Stage 09 is commit `f467fbf` in `tmp/github-publish`, branch
`codex/finsight-rag-okf-stages-01-08`. Stage 10 changes are local and not pushed.
The workspace root still has an unborn repository; publication must use the
separate checkout. Read [Stage 11](11_LAYOUT_INSPECTION.md) next.
