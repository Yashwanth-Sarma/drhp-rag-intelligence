"""
src/knowledge_graph/entity_extractor.py

Extracts financial entities and relationships from document chunks.
Uses ProviderRouter (Cerebras) instead of direct Groq calls.
Fully resumable — tracks processed chunk_ids in a progress file.
Targets key sections only to conserve API tokens.
"""

import json
import time
import logging
from typing import Optional
from pathlib import Path

from langchain_core.documents import Document

from src.configuration.config import DATA_DIR
from src.llm.provider_router import get_router, TaskType
from src.shared.exceptions import FinSightError

logger = logging.getLogger(__name__)

ENTITIES_DIR = DATA_DIR / "knowledge_graph"
ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
ENTITIES_FILE = ENTITIES_DIR / "extracted_entities.jsonl"
PROGRESS_FILE = ENTITIES_DIR / "extraction_progress.json"

# Sections most likely to contain useful entities and relationships
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

EXTRACTION_PROMPT = """You are a financial document analyst. Extract structured entities and relationships from the chunk below.

Return ONLY valid JSON in this exact format with no extra text:
{{
  "entities": [
    {{"name": "entity name", "type": "COMPANY|PERSON|METRIC|PRODUCT|REGULATOR|SUBSIDIARY", "value": "optional numeric value or description"}}
  ],
  "relationships": [
    {{"from": "entity1", "relation": "OWNS|COMPETES_WITH|INVESTED_IN|REGULATED_BY|ACQUIRED|REPORTED|PART_OF", "to": "entity2", "context": "one sentence context"}}
  ]
}}

If nothing meaningful: {{"entities": [], "relationships": []}}

DOCUMENT CONTEXT:
Company: {company_name} | Type: {doc_type} | Year: {year} | Section: {section} | Page: {page_number}

TEXT:
{chunk_text}

JSON:"""


class EntityExtractor:
    """
    Extracts financial entities from document chunks using the ProviderRouter.
    Routes to Cerebras (1M tokens/day) for bulk extraction work.
    Fully resumable — tracks progress per chunk_id.
    """

    DELAY_BETWEEN_CALLS: float = 6.0

    def __init__(self) -> None:
        self.router = get_router()
        logger.info("EntityExtractor initialized")

    def _load_progress(self) -> set[str]:
        """Load set of already-processed chunk_ids from progress file."""
        if not PROGRESS_FILE.exists():
            return set()
        try:
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
            return set(data.get("processed_chunk_ids", []))
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}. Starting fresh.")
            return set()

    def _save_progress(self, processed_ids: set[str]) -> None:
        """Save progress so extraction is always resumable."""
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"processed_chunk_ids": list(processed_ids)}, f)

    def _append_to_jsonl(self, chunk_id: str, data: dict, meta: dict) -> None:
        """Append one extraction result to the JSONL output file."""
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

    def extract_from_chunk(self, chunk: Document) -> dict:
        """
        Extract entities and relationships from one chunk.
        Returns dict with 'entities' and 'relationships'.
        Returns empty dict if extraction fails — never raises.
        """
        meta = chunk.metadata
        prompt = EXTRACTION_PROMPT.format(
            company_name=meta.get("company_name", "Unknown"),
            doc_type=meta.get("doc_type", "Unknown"),
            year=meta.get("year", "Unknown"),
            section=meta.get("section", "Unknown"),
            page_number=meta.get("page_number", "Unknown"),
            chunk_text=chunk.page_content[:700],
        )

        try:
            raw = self.router.generate(
                task=TaskType.ENTITY_EXTRACTION,
                prompt=prompt,
                temperature=0.0,
                max_tokens=400,
            )
            # Strip markdown fences if model adds them
            text = raw.strip()
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else parts[0]
                if text.startswith("json"):
                    text = text[4:]

            result = json.loads(text.strip())
            # Validate structure
            if not isinstance(result.get("entities"), list):
                result["entities"] = []
            if not isinstance(result.get("relationships"), list):
                result["relationships"] = []
            return result

        except json.JSONDecodeError:
            logger.debug(
                f"JSON parse failed for chunk {meta.get('chunk_id')} — "
                "returning empty (chunk may have no structured entities)"
            )
            return {"entities": [], "relationships": []}

        except Exception as e:
            logger.warning(
                f"Extraction failed for chunk {meta.get('chunk_id')}: "
                f"{str(e)[:80]}"
            )
            return {"entities": [], "relationships": []}

    def extract_from_chunks(
        self,
        chunks: list[Document],
        sections_filter: Optional[list[str]] = None,
    ) -> int:
        """
        Extract entities from all chunks. Fully resumable.

        Args:
            chunks:          All chunks from load_all_pdfs()
            sections_filter: Only process these sections. Defaults to HIGH_VALUE_SECTIONS.
                             Pass None to process all sections.

        Returns:
            Number of chunks processed in this session.
        """
        filter_sections = sections_filter if sections_filter is not None else HIGH_VALUE_SECTIONS

        # Filter to target sections
        if filter_sections:
            filtered = [
                c for c in chunks
                if c.metadata.get("section") in filter_sections
            ]
        else:
            filtered = chunks

        # Skip already-processed chunks
        processed_ids = self._load_progress()
        pending = [
            c for c in filtered
            if c.metadata.get("chunk_id") not in processed_ids
        ]

        already_done = len(filtered) - len(pending)

        print(f"\nEntity extraction plan:")
        print(f"  Chunks in target sections:  {len(filtered)}")
        print(f"  Already processed:          {already_done} (skipping)")
        print(f"  To process now:             {len(pending)}")
        print(f"  Estimated tokens:           ~{len(pending) * 450:,}")
        print(f"  Provider status:            {self.router.status_report()}")

        if not pending:
            print("\nAll chunks already processed.")
            self._print_stats()
            return 0

        processed_this_session = 0
        total_entities = 0
        total_rels = 0

        for i, chunk in enumerate(pending, 1):
            chunk_id = chunk.metadata.get("chunk_id", f"unknown_{i}")

            try:
                data = self.extract_from_chunk(chunk)

                self._append_to_jsonl(chunk_id, data, chunk.metadata)
                processed_ids.add(chunk_id)
                processed_this_session += 1

                e_count = len(data.get("entities", []))
                r_count = len(data.get("relationships", []))
                total_entities += e_count
                total_rels += r_count

                # Save progress every 10 chunks
                if i % 10 == 0:
                    self._save_progress(processed_ids)

                if i % 50 == 0 or i == len(pending):
                    print(
                        f"  [{i}/{len(pending)}] "
                        f"{round(i / len(pending) * 100)}% | "
                        f"Entities: {total_entities} | "
                        f"Relations: {total_rels} | "
                        f"Status: {self.router.status_report()}"
                    )

                time.sleep(self.DELAY_BETWEEN_CALLS)

            except KeyboardInterrupt:
                self._save_progress(processed_ids)
                print(f"\nInterrupted at {i}/{len(pending)}. Progress saved.")
                print("Restart to continue from this point.")
                return processed_this_session

            except Exception as e:
                logger.error(f"Unexpected error on chunk {chunk_id}: {e}")
                continue

        self._save_progress(processed_ids)
        print(f"\nExtraction complete: {processed_this_session} chunks, {total_entities} entities, {total_rels} relationships")
        self._print_stats()
        return processed_this_session

    def _print_stats(self) -> None:
        """Print current extraction statistics."""
        if not ENTITIES_FILE.exists():
            print("No entities extracted yet.")
            return
        entity_count = 0
        rel_count = 0
        companies = set()
        with open(ENTITIES_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    entity_count += len(r.get("entities", []))
                    rel_count += len(r.get("relationships", []))
                    if r.get("company_name"):
                        companies.add(r["company_name"])
                except Exception:
                    continue
        print(f"\nExtraction stats: {entity_count} entities, {rel_count} relationships, {len(companies)} companies")


from typing import Optional  # noqa: E402 — placed here to avoid circular at top