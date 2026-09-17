# Stage 03 — retrieval and gold evaluation

Prerequisite: source evidence contracts from Stage 02. Current BM25 path is an explicit baseline.

Implement structure-aware retrieval chunks pointing to immutable evidence blocks; normalized financial terms; parent/table-header expansion; company/document/availability-date filtering. Add a dense adapter only with known model, dimensionality, input type, corpus hash and cost configuration. Version indexes and fuse ranks with RRF. Reranking remains an ablation, not assumed benefit.

Label first 30 real questions with required evidence groups, exact source pages/cells and answerability, then expand toward held-out filing/layout coverage. Include OFS versus fresh issue, interim versus annual periods, conflicting definitions, multi-company starvation, and questions absent from corpus. Compare BM25, dense, fusion and reranking at equal context budgets. Report Recall@20, nDCG@10, required-group coverage and latency by slice. No scalar judge substitute for evidence labels.

Acceptance: zero scope leakage on fixtures; deterministic replay; reliable empty/failure behavior; measured quality improvement before promoting extra complexity. Initial recall target 0.90 is provisional and must include counts/uncertainty. Deliver a machine-readable benchmark and one-command runner with no generation calls during retrieval evaluation.
