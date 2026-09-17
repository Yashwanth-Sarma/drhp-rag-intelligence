# FinSight — revised product direction

User clarification and seven-page IPO_ANALYSIS_SYSTEM_V1_SPECIFICATION reviewed 2026-09-15. This is the current planning overlay. Application code was not changed in this review.

## Fixed user intent

Build a professional IPO analysis system for investment bankers, analysts and due-diligence teams. Ingest a company dossier anchored in its DRHP, with related IPO filings, financial reports and optionally call transcripts. Produce a grounded, detailed, visually clear 5–6-page report tailored to the requested analysis, plus traceable structured exports. Preserve the original evidence-first philosophy.

The selected base model is **Qwen/Qwen3.6-35B-A3B**. Financial fine-tuning and curated hybrid RAG plus Google's Open Knowledge Format are part of the intended product. The user explicitly says Claude's architecture is provisional; the PDF's “APPROVED” labels are not independent authorization or proof of technical validity.

## Proposed V1 scope

One company dossier per report. Financial trends cover the periods actually supplied, without promising long-term history. Report sections: executive summary, financial health, risks, business/market, management/governance, and due-diligence questions. Support both whole-company summaries and focused research requests. Show evidence gaps, attribution, dates and qualifications. A draft recommendation must distinguish a research assessment from an investment/trading decision and remain subject to analyst review.

Defer portfolio-scale research, interactive analytical dashboards, broad relationship graphs and historical anomaly detection. Keep the existing source viewer as supporting review infrastructure. Basic conflict handling within the input dossier remains necessary even though broad cross-filing contradiction discovery is deferred. Never quietly choose between incompatible financial values.

## Responsibilities of each layer

1. **Originals and evidence registry:** immutable PDF bytes, hashes, issuer/version identity, physical pages, printed labels, geometry, tables/cells, and separate document/public-availability dates.
2. **Financial observations:** exact values, signs, currencies, scales, fiscal periods, accounting basis, definitions and source anchors. Deterministic arithmetic operates here.
3. **OKF:** versioned human-readable domain definitions, metric comparability rules, company/filing maps and reviewed contextual notes. Export approved evidence-backed assertions with source IDs, freshness and review state. Pin the canonical specification version before implementing validation. Generated notes cannot promote themselves to verified facts.
4. **Retrieval:** lexical plus dense search, metadata constraints, table/section context expansion, measured reranking and bounded query planning. Fetch source evidence as well as relevant curated knowledge.
5. **Qwen plus financial adapter:** interpret the question and supplied evidence; draft structured claims, analytical explanations, qualifiers and report sections. Do not use model memory as authority for issuer-specific facts.
6. **Validation and report assembly:** resolve citations, validate calculations/scope, assess semantic support, expose conflicts and abstentions, then render charts/report. Generate executive summary from accepted section claims.

OKF is a portable representation, not a replacement for search or a guarantee that relationships survive extraction. Raw source data remains authoritative; linked knowledge helps interpretation. Use one authoritative record per fact and generated projections to avoid divergent database/Markdown copies.

## Fine-tuning plan

Establish an untuned Qwen baseline using a small working evidence pipeline before training. Freeze a held-out set by company/filing and period, not random question rows. Similar questions from one filing must not leak across train/test boundaries.

Curate evidence-conditioned examples: question, dossier scope, source passages/table cells, normalized facts, desired structured answer, citations, missing-evidence response and rubric. Include financial definitions, multi-period interpretation, fresh issue versus OFS, loss/zero handling, management attribution, ambiguous/incomplete evidence, prompt injection and unsupported questions. Train on reviewed outputs, not unverified model-generated reasoning.

The proposed 1,000–2,000 examples across 10–20 filings are a pilot dataset target, not proof of sufficient coverage. Verify usage rights and annotation consistency. Compare base Qwen and adapter on the same RAG/OKF inputs, including held-out companies. Require gains in relevant task quality without worsened unsupported claims, numeric errors or abstention behavior.

LoRA/QLoRA, module targets, rank, sequence length, quantization, batch/accumulation and optimizer are experimental choices. Verify support for this exact MoE checkpoint before committing. Begin with a small smoke training run and measure memory/throughput before any full run. Preserve base checkpoint, adapter, dataset hash, serving config and evaluation artifact for rollback.

## Corrections to the supplied specification

- AIME/GPQA are not IPO-finance evaluations. They cannot justify banker-grade accuracy. The official card currently lists AIME26 92.7 for this model, rather than the PDF's 92.6.
- A 262K context window is a token limit, not a promise that 400 or 1,200 PDF pages fit or are processed reliably. Tokenize representative dossiers, reserve output space, and measure recall/cost. Retrieval remains necessary.
- 3B active parameters do not mean only 3B parameters require storage. Approximate raw 35B weight storage is 70 GB at 16 bits or 35 GB at 8 bits, before runtime/cache/activation overhead. Actual serving/training requirements depend on implementation and hardware.
- The stated BF16 A100 40GB training fit, 3–4-hour run, FP8 throughput of 200 tokens/s and zero self-hosting cost are unverified. Do not plan GPU purchases/rentals around them. Verify exact quantization/kernel compatibility.
- Rank 8, two epochs and learning rate 2e-4 are trial settings, not established safe choices.
- Structured JSON decoding constrains syntax, not claim truth or citation support. Serving-engine backtracking does not validate the semantics of a reference. Benchmark SGLang and alternatives if needed without changing the selected base model.
- Temperature zero does not guarantee reproducible output across hardware/kernels/versions. Store actual generated outputs and configurations.
- Four plausibility checks with +0.2 scores cannot establish correctness or calibrated confidence. Domain plausibility must not override unusual but correctly sourced facts. Verification failures must block/qualify output, not be averaged away.
- Native reasoning text is not an audit trail proving correctness. Audit evidence, tool calls, calculation inputs, versions and validation outcomes; provide concise evidence-based explanations.
- Zero hallucination risk and perfect reports are aspirations, not defensible guarantees. Measure unsupported material claims, numerical correctness, citation entailment, coverage and abstention separately. Human review remains a release gate.
- Cold ingestion and warm report generation need separate latency targets. A sub-five-minute total for 800–1,200 pages is an experiment on specified hardware, not a current commitment.
- The listed document page ranges do not themselves sum to the stated 800–1,200-page dossier; test real dossier sizes rather than relying on that estimate.

## Revised work order

1. Finish source-based extraction/table contracts and a small gold evidence set (existing Stage 02).
2. Add canonical OKF schema validation, reviewed concepts, provenance and freshness behavior.
3. Establish untuned Qwen serving feasibility and hardware measurements; connect bounded retrieval and record baseline results.
4. Curate and review adapter-training data; run pilot fine-tuning and controlled base/adapter comparisons.
5. Integrate validated claims/calculations and build professional report rendering; evaluate full dossiers.
6. Banker review, operational/privacy controls, reproducible deployment and monitoring.

Data preparation and serving feasibility can proceed alongside extraction planning; final training should wait until labels and baseline are meaningful. Existing Stage 03–06 files remain useful module backlogs, but this overlay changes their scheduling and fixes Qwen as the model choice.

## Open decisions

User update: plans to rent suitable GPU hardware, possibly DGX Spark; no finance professional is currently available. Hardware remains a candidate, not a booked or benchmarked resource. Validate exact model training/serving support, usable memory, architecture, throughput and rental costs before selecting a host. Do not equate unified system memory with dedicated GPU VRAM or extrapolate A100 throughput to DGX Spark.

Without professional review, create source-anchored labels with explicit author/reviewer status and independent checks, but do not call them banker-reviewed. Numerical and citation integrity can be tested; analyst usefulness, domain judgments and real-deal suitability remain unvalidated. Start as a research prototype/private evaluation product. Expert review remains an unmet gate for claims of banker-grade readiness, rather than silently substituting an LLM judge for a professional.

GPU/VRAM or cloud access, hosting/training budget, expert reviewer availability, handling of confidential deal documents, and meaning of “overall recommendation” for the first release. No GPU download/training or paid infrastructure is authorized merely by the PDF. Proceed with design/data contracts while these are unresolved.

## Sources checked

- Qwen model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B — model identity, architecture, context, serving examples and benchmark categories. Serving examples are not minimum hardware guarantees.
- Canonical OKF repository: https://github.com/GoogleCloudPlatform/open-knowledge-format — Markdown/YAML representation, portability and v0.2 provenance/freshness direction. Read and pin SPEC.md before implementation.

## Previous-session operational handoff

33 automated tests passed before this revision; no new application tests run for this documentation-only change. Original Ola/Swiggy PDFs were downloaded and selected pages inspected. The reference import command was rejected by automatic approval review because the workspace was reported out of credits; do not claim the real filings were indexed or that import-results.json was generated. Resolve that prerequisite before resuming the command. Only the empty browser page was verified; full browser interaction remains pending.
