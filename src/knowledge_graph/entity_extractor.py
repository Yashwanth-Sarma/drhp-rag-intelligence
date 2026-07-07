"""
src/knowledge_graph/entity_extractor.py

Extracts financial entities and relationships from document chunks.

Key improvements:
- BATCHING: 5 chunks per LLM call instead of 1 — 5x fewer API calls
- Full chunk text up to 4000 chars (not truncated to 700)
- Progress saved after EVERY chunk — no data loss on crash
- JSONL flushed + fsync'd after every write for crash safety
- Duplicate detection via both progress file AND JSONL scan
- Batched runtime: ~48 min for 1178 chunks vs ~4 hours single-chunk
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from src.configuration.config import DATA_DIR
from src.llm.provider_router import get_router, TaskType
from src.shared.exceptions import FinSightError

logger = logging.getLogger(__name__)

ENTITIES_DIR = DATA_DIR / "knowledge_graph"
ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
ENTITIES_FILE = ENTITIES_DIR / "extracted_entities.jsonl"
PROGRESS_FILE = ENTITIES_DIR / "extraction_progress.json"

HIGH_VALUE_SECTIONS = [
    "Risk Factors",
    "Business Overview",
    "Financial Statements",
    "Capital Structure",
    "Management",
    "Objects of the Offer",
    "Related Party Transactions",
    "Industry Overview",
]

CALL_DELAY_SECONDS: float = 12.0
BATCH_SIZE: int = 5
MAX_CHARS_PER_CHUNK: int = 4000

BATCH_EXTRACTION_PROMPT = """Extract financial entities and relationships from each numbered chunk below.

Return ONLY valid JSON — no explanation, no markdown, just the JSON array.

Format:
[
  {{
    "chunk_index": 1,
    "entities": [
      {{"name": "entity name", "type": "COMPANY|PERSON|METRIC|PRODUCT|REGULATOR|SUBSIDIARY", "value": "optional"}}
    ],
    "relationships": [
      {{"from": "entity1", "relation": "OWNS|COMPETES_WITH|INVESTED_IN|REGULATED_BY|ACQUIRED|REPORTED|PART_OF|OPERATES_IN", "to": "entity2", "context": "one sentence"}}
    ]
  }}
]

If a chunk has nothing meaningful: {{"chunk_index": N, "entities": [], "relationships": []}}

CHUNKS:
{chunks_block}

JSON array:"""


def _format_chunk_for_batch(index: int, chunk: Document) -> str:
    meta = chunk.metadata
    text = meta.get("original_text", chunk.page_content)[:MAX_CHARS_PER_CHUNK]
    return (
        f"[CHUNK {index}]\n"
        f"Company: {meta.get('company_name', 'Unknown')} | "
        f"Type: {meta.get('doc_type', 'Unknown')} | "
        f"Year: {meta.get('year', 'Unknown')} | "
        f"Section: {meta.get('section', 'Unknown')} | "
        f"Page: {meta.get('page_number', 'Unknown')}\n"
        f"{text}"
    )


def _load_existing_jsonl_chunk_ids() -> set[str]:
    """
    Scan JSONL file and return all chunk_ids already written.
    Used to prevent duplicates if progress file is ever lost.
    """
    if not ENTITIES_FILE.exists():
        return set()
    ids: set[str] = set()
    with open(ENTITIES_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                cid = record.get("chunk_id")
                if cid:
                    ids.add(cid)
            except Exception:
                continue
    return ids


class EntityExtractor:
    """
    Extracts financial entities from document chunks via batched LLM calls.

    Batching: 5 chunks per call reduces API calls by 5x.
    Crash safety: progress saved and JSONL fsync'd after every chunk.
    Resumable: restarts pick up exactly where they left off.
    """

    def __init__(self) -> None:
        self.router = get_router()
        logger.info(
            "EntityExtractor initialized",
            extra={"providers": self.router.available_providers()},
        )

    def _load_progress(self) -> set[str]:
        if not PROGRESS_FILE.exists():
            return set()
        try:
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
            return set(data.get("processed_chunk_ids", []))
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}. Will use JSONL scan.")
            return set()

    def _save_progress(self, processed_ids: set[str]) -> None:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"processed_chunk_ids": list(processed_ids)}, f)

    def _append_to_jsonl(self, chunk_id: str, data: dict, meta: dict) -> None:
        """Write one record to JSONL, flush, and fsync for crash safety."""
        record = {
            "chunk_id": chunk_id,
            "company_name": meta.get("company_name"),
            "doc_type": meta.get("doc_type"),
            "year": meta.get("year"),
            "page_number": meta.get("page_number"),
            "section": meta.get("section"),
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }
        with open(ENTITIES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _extract_batch(self, chunks: list[Document]) -> list[dict]:
        """
        Send one batch of chunks to the LLM, get back extraction results.
        Returns empty results for all chunks if the call fails entirely.
        """
        chunks_block = "\n\n".join(
            _format_chunk_for_batch(i + 1, chunk)
            for i, chunk in enumerate(chunks)
        )
        prompt = BATCH_EXTRACTION_PROMPT.format(chunks_block=chunks_block)
        estimated_output_tokens = len(chunks) * 150

        try:
            raw = self.router.generate(
                task=TaskType.ENTITY_EXTRACTION,
                prompt=prompt,
                temperature=0.0,
                max_tokens=min(2000, estimated_output_tokens + 500),
            )

            text = raw.strip()
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else parts[0]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            result = json.loads(text)

            if not isinstance(result, list):
                logger.warning("Batch returned non-list — wrapping or discarding")
                result = [result] if isinstance(result, dict) else []

            validated = []
            for entry in result:
                if not isinstance(entry, dict):
                    continue
                validated.append({
                    "chunk_index": entry.get("chunk_index", 0),
                    "entities": (
                        entry.get("entities", [])
                        if isinstance(entry.get("entities"), list) else []
                    ),
                    "relationships": (
                        entry.get("relationships", [])
                        if isinstance(entry.get("relationships"), list) else []
                    ),
                })
            return validated

        except json.JSONDecodeError:
            logger.debug("Batch JSON parse failed — chunk may have no structured entities")
            return [
                {"chunk_index": i + 1, "entities": [], "relationships": []}
                for i in range(len(chunks))
            ]

        except Exception as e:
            logger.warning(f"Batch extraction error: {str(e)[:100]}")
            return [
                {"chunk_index": i + 1, "entities": [], "relationships": []}
                for i in range(len(chunks))
            ]

    def extract_from_chunks(
        self,
        chunks: list[Document],
        sections_filter: Optional[list[str]] = None,
    ) -> int:
        """
        Extract entities from all chunks using batched LLM calls.

        Args:
            chunks:          All chunks from load_all_pdfs().
            sections_filter: Sections to process.
                             None  = HIGH_VALUE_SECTIONS (saves ~60% tokens)
                             []    = ALL sections

        Returns:
            Number of chunks processed this session.
        """
        if sections_filter is None:
            target_sections = HIGH_VALUE_SECTIONS
        elif len(sections_filter) == 0:
            target_sections = None
        else:
            target_sections = sections_filter

        filtered = (
            [c for c in chunks if c.metadata.get("section") in target_sections]
            if target_sections else chunks
        )

        # Combine progress file + JSONL scan for maximum duplicate safety
        progress_ids = self._load_progress()
        jsonl_ids = _load_existing_jsonl_chunk_ids()
        processed_ids = progress_ids | jsonl_ids

        if len(jsonl_ids) > len(progress_ids):
            logger.info(
                f"JSONL has {len(jsonl_ids)} records, progress file has {len(progress_ids)}. "
                "Using union as source of truth."
            )

        pending = [
            c for c in filtered
            if c.metadata.get("chunk_id") not in processed_ids
        ]

        already_done = len(filtered) - len(pending)
        total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
        est_minutes = round(total_batches * CALL_DELAY_SECONDS / 60, 1)

        print(f"\nEntity extraction plan:")
        print(f"  Target sections:     {target_sections or 'ALL'}")
        print(f"  Chunks in scope:     {len(filtered)}")
        print(f"  Already processed:   {already_done} (skipping)")
        print(f"  Pending chunks:      {len(pending)}")
        print(f"  Batch size:          {BATCH_SIZE} chunks per LLM call")
        print(f"  Total batches:       {total_batches}")
        print(f"  Delay per batch:     {CALL_DELAY_SECONDS}s")
        print(f"  Estimated time:      ~{est_minutes} min")
        print(f"  Provider status:     {self.router.status_report()}\n")

        if not pending:
            print("All chunks already processed.")
            self._print_stats()
            return 0

        processed_this_session = 0
        total_entities = 0
        total_rels = 0
        batch_count = 0

        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start: batch_start + BATCH_SIZE]
            batch_count += 1

            try:
                batch_results = self._extract_batch(batch)
                results_by_index = {r["chunk_index"]: r for r in batch_results}

                for i, chunk in enumerate(batch):
                    chunk_id = chunk.metadata.get("chunk_id", f"unknown_{batch_start + i}")

                    if chunk_id in processed_ids:
                        continue

                    result = results_by_index.get(
                        i + 1, {"entities": [], "relationships": []}
                    )

                    self._append_to_jsonl(chunk_id, result, chunk.metadata)

                    processed_ids.add(chunk_id)
                    self._save_progress(processed_ids)
                    processed_this_session += 1

                    total_entities += len(result.get("entities", []))
                    total_rels += len(result.get("relationships", []))

                if batch_count % 10 == 0 or batch_start + BATCH_SIZE >= len(pending):
                    chunks_done = min(batch_start + BATCH_SIZE, len(pending))
                    pct = round(chunks_done / len(pending) * 100)
                    eta = round((total_batches - batch_count) * CALL_DELAY_SECONDS / 60, 1)
                    print(
                        f"  Batch {batch_count}/{total_batches} | "
                        f"Chunks: {chunks_done}/{len(pending)} ({pct}%) | "
                        f"Entities: {total_entities} | "
                        f"Relations: {total_rels} | "
                        f"ETA: {eta}min | "
                        f"Status: {self.router.status_report()}"
                    )

                time.sleep(CALL_DELAY_SECONDS)

            except KeyboardInterrupt:
                print(
                    f"\nInterrupted after batch {batch_count}. "
                    f"{processed_this_session} chunks saved."
                )
                print("Restart anytime — progress is fully saved.")
                self._print_stats()
                return processed_this_session

            except Exception as e:
                logger.error(f"Unexpected error in batch {batch_count}: {e}")
                continue

        print(f"\nExtraction complete!")
        print(f"  Processed this session: {processed_this_session}")
        print(f"  Total entities:         {total_entities}")
        print(f"  Total relationships:    {total_rels}")
        self._print_stats()
        return processed_this_session

    def _print_stats(self) -> None:
        if not ENTITIES_FILE.exists():
            print("No entities extracted yet.")
            return
        entity_count = 0
        rel_count = 0
        chunk_count = 0
        companies: set[str] = set()
        with open(ENTITIES_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    entity_count += len(r.get("entities", []))
                    rel_count += len(r.get("relationships", []))
                    chunk_count += 1
                    if r.get("company_name"):
                        companies.add(r["company_name"])
                except Exception:
                    continue
        print(
            f"\nEntity Extraction Progress:"
            f"\n  Chunks in JSONL:     {chunk_count}"
            f"\n  Entities extracted:  {entity_count}"
            f"\n  Relationships found: {rel_count}"
            f"\n  Companies covered:   {sorted(companies)}"
            f"\n  Output file:         {ENTITIES_FILE}"
        )