# Source-driven filing study — 2026-09-15

Official PDFs downloaded, parsed and selected pages visually inspected. This is a structure study, not investment analysis.

| Filing | Official listing | Document date / listing date | Physical pages |
|---|---|---|---|
| Ola Electric Mobility DRHP | https://www.sebi.gov.in/filings/public-issues/dec-2023/ola-electric-mobility-limited-drhp_80215.html | 2023-12-22 / 2023-12-26 | 444 |
| Swiggy Updated DRHP I | https://www.sebi.gov.in/filings/public-issues/sep-2024/swiggy-limited-updated-drhp-i_87047.html | 2024-09-26 / 2024-09-27 | 532 |

Original URLs and hashes are recorded by `scripts/import_reference.py` in application manifests and `data/reference/import-results.json`.

## Observations that change implementation

- Ola table of contents is physical page 4; sampled printed page 1 starts at physical 5. Risk section physical 32/printed 28; financial summary physical 70/printed 66; objects physical 101/printed 97.
- Swiggy contents is physical page 6, with extra front matter including an intentionally blank page. Risk section physical 39/printed 33; financial summary physical 80/printed 74; objects physical 138/printed 132. Never apply one global offset across all filings.
- Visually inspected Ola physical 71 is portrait with dense ruled rows and nested asset/liability labels. Swiggy physical 81 is landscape with five date columns and nested labels. Geometry and row/header association are essential.
- Ola assets/liabilities columns include June 2023 and three March year-ends. Swiggy includes June 2024 and June 2023 alongside March year-ends. Balance-sheet dates are instants; nearby income/cash-flow tables have duration periods. Do not normalize them into a single generic FY field.
- Amount scales appear above tables; negative parentheses and dash entries occur. Dashes must not silently become zero. Continuation/header context is required before facts can be accepted.
- Offer sections distinguish fresh issue from offer for sale. A report must not attribute selling-shareholder proceeds to company funding.
- The PDF document date differs from SEBI listing date in both examples. Current date filter is explicitly document-date only. Historical public-availability filtering remains a necessary next-stage change.

Rendered inspection images are under `data/reference/inspection`. The current native-block parser preserves evidence geometry/text but does not establish cell semantics. Stage 02 must benchmark actual table extraction and use reviewer-anchored annotations.
