# Stage 02 — source-driven structure and table extraction

Status: next. Read `../SEBI_FILING_STUDY.md`, contracts/store/ingest and ingestion tests. Do not start with a new vector database.

## First bounded module

Add a structure inventory preserving section hierarchy, table regions, native words/cells, page labels and continuations. Benchmark native table detection versus a layout parser on inspected Ola and Swiggy pages. Keep parser outputs as candidates with provenance, never auto-approved financial observations.

## Inputs

Ola 444-page DRHP and Swiggy 532-page Updated DRHP I in `data/reference`. Financial statement examples: Ola physical page 71, printed 67; Swiggy physical page 81, printed 75. Include adjoining profit/loss and cash-flow pages, risk lists, offer-use tables, definitions and footnotes. Offsets must be checked per region rather than assumed globally.

## Contracts

Table ID, source document hash, physical page, region geometry, row/column spans, merged headers, raw cell text, normalized value candidate, unit/currency, period header, instant/duration, statement scope, footnote links, extraction version and validation status. Separate document date and source publication availability. Add Updated DRHP I as a real variant rather than silently collapsing it into DRHP.

## Acceptance

At least 20 manually inspected representative pages and a committed annotation manifest using source hashes/page/cell coordinates. Correct preservation of negative parentheses, zero, dash/unknown (never auto-zero), currency scale, annual/interim columns, subtotal labels and continuation headers. Compare parsers on label/cell bindings and extraction time. Reject ambiguous mappings. Fixture download paths are local, not tracked raw PDFs.

## Deliverables

Module, tests, parser comparison JSON, source manifest and updated status. Add no LLM calls unless a measured failing slice justifies them and a budget is configured.
