# Stage 07 — Local RAG and OKF

Updated 2026-09-16. Read `../RAG_OKF_ARCHITECTURE.md` for the design and boundaries.

## Implemented

* `finsight/retrieval.py`: local BGE embedding, overlapping evidence windows,
  atomic SQLite vector publication, corpus fingerprints, scoped exact vector
  search and reciprocal rank fusion. Explicit lexical fallback for missing,
  stale or unavailable dense runtime/index.
* `finsight/okf.py`: restricted safe YAML reader for canonical OKF concepts,
  metadata/source checks, declared trust, lifecycle exclusion, independent
  methodology context retrieval. No arbitrary code execution or URL fetching.
* `knowledge/`: three application-authored, unverified methodology playbooks.
* `finsight/grounding.py`: typed claims, exact quote membership, retrieved-ID,
  issuer/document/date and original-source checks. Paraphrases require review.
* `finsight/local_qwen.py`: opt-in bounded loopback adapter for the fixed Qwen
  model; model output remains a draft. No Qwen weights downloaded/server started.
* `finsight/rerank.py`: downloaded MiniLM cross-encoder evaluation candidate,
  deliberately not enabled in the application pipeline pending evaluation.
* `research.py` and `web/app.js`: retrieval metadata and separate OKF methodology
  in research results; reports use the same retrieval layer.
* Tests cover leakage, stale indices, failed atomic rebuild, corrupted embeddings,
  YAML hazards, invalid attribution, citation checks and mocked Qwen output.

## Local state and verification

Python 3.14, approximately 32 GB physical RAM. Free FastEmbed 0.8.0, ONNX Runtime,
NumPy and PyYAML were installed into `.runtime`. Models were downloaded to
`data/models`; downloads completed for both BGE-small and MiniLM reranker.

Official filings imported successfully:

| Filing | Document ID | Pages | Blocks |
|---|---|---:|---:|
| Ola DRHP 2023 | a5b8cce713093f98ee7819e0a9fe26db | 444 | 8,145 |
| Swiggy Updated DRHP I 2024 | f8d05e110527a3b14987ae5a7fd2a11e | 532 | 10,053 |

The last completed regression run passed **51 tests** with one upstream Starlette
deprecation warning. After that run, OKF timestamp-to-JSON normalization, one test
and UI result rendering changed; these latest changes remain unverified.

The real dense build ran in execution session 55226. It downloaded the model and
was observed doing CPU work, but no completion output was received. After an
automatic approval review reported workspace credits exhausted, a later attempt
to read the existing session returned `Unknown process id`. The index may or may
not have published. Inspect it before rebuilding; do not claim it is ready.

The six-case evaluator exists but has **not run**. Its targets are developer-chosen
summary/risk page locations, not exhaustive relevant evidence or expert labels.
Risk disclosures can have valid alternatives outside these pages. Do not report
its future hit rate as overall answer accuracy. The reranker has no application
quality endorsement. UI browser verification and a live Qwen test remain open.

## Resume commands

Run from the project root. These commands require execution access; do not work
around the workspace-credit approval rejection through another tool or account.

```powershell
python scripts/test.py --tb=short -x -p no:cacheprovider
python scripts/build_retrieval.py
python scripts/evaluate_retrieval.py
python run.py
```

`build_retrieval.py` uses locally cached weights unless `--download` is explicitly
supplied. Rebuilding currently embeds the whole corpus; it is not incremental.
For a new environment, install `requirements-retrieval.txt` and
`requirements-dev.txt` into a virtual environment (or `.runtime` with `--target`).
Acquire the model with `python scripts/build_retrieval.py --download`. The optional
reranker also needs an explicit acquisition before running the evaluation script.

Once the user's Qwen server is running under the exact served model name:

```powershell
python scripts/draft_answer.py --company "Ola Electric Mobility Limited" --question "What losses and negative cash flows are disclosed?"
```

Default endpoint: `http://127.0.0.1:8000/v1`. The adapter does not start the server.
For a rented remote server, use a deliberately configured local tunnel; do not
remove loopback restrictions merely to make a request succeed. No API key is used.

## Required next work, in order

1. Confirm dense index publication, rerun the full suite, run real-file evaluation
   and preserve the result under `evals/results/retrieval-smoke.json`.
2. Review every smoke miss against the PDF; add acceptable alternative spans.
   Keep a separate held-out set before tuning ranking. Do not silently drop failures.
3. Pin downloaded model revisions/hashes and resolved dependency versions. Index
   metadata currently pins the logical model name, chunk version and corpus hash,
   but does not detect a replacement of weights under the same model name.
4. Add token-aware hierarchical chunks and retain full heading/table context.
   The current word-window baseline can truncate at tokenizer limits, and the
   experimental reranker sees only the first 2,000 characters of each block.
5. Bind financial table cells to headers, signs, units, dates and footnotes before
   any automatic numeric analysis. Follow Stage 02 on annotated real pages.
6. Expand OKF provenance and controlled issuer-concept export. Add authenticated
   reviews separately from declaration parsing. Do not fabricate expert approval.
7. Run Qwen baseline evaluations before training. Add semantic support review and
   publication gates; the current adapter is intentionally a draft path only.

## Suggested next-session prompt

Read docs/START_HERE.md and docs/stages/07_RAG_OKF_HANDOFF.md. Check whether the
dense index was published; complete the remaining tests and real-filing retrieval
evaluation before changing architecture. Preserve failures, compare BM25/hybrid/
reranked results, and improve heading-aware chunking from source evidence. Do not
claim production readiness or semantic validation from citation membership.
