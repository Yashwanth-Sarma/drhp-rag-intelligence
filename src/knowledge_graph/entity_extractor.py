"""
src/knowledge_graph/entity_extractor.py

Extracts financial entities and relationships from document chunks using Groq.

Why LLM-based extraction instead of spaCy?
Financial documents contain entities spaCy has never seen:
- "Adjusted EBITDA", "GMV", "ESOP pool", "Hyperpure", "One97 Communications"
- Relationships: "Zomato owns Hyperpure", "Info Edge holds 18.4% stake in Zomato"
- Time-bound facts: "Revenue grew from Rs.1,994 Cr in FY21 to Rs.7,079 Cr in FY23"

These require understanding financial language, not just pattern matching.

Inputs:  List of Document chunks with metadata
Outputs: List of entity dicts ready for Neo4j ingestion

IMPORTANT: This is Groq-token-heavy.
Run in small batches, ideally after Stage 2 enrichment is complete for the day.
Estimated tokens: ~500 per chunk for extraction. Run on 200 chunks = ~100K tokens.
"""

import json
import time
from pathlib import Path

from groq import Groq
from langchain_core.documents import Document

from src.configuration.config import GROQ_API_KEY, GROQ_MODEL, DATA_DIR
from src.shared.logger import get_logger
from src.shared.exceptions import FinSightError

logger = get_logger(__name__)

# Where we save extracted entities before loading into Neo4j
ENTITIES_DIR = DATA_DIR / "knowledge_graph"
ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
ENTITIES_FILE = ENTITIES_DIR / "extracted_entities.jsonl"
PROGRESS_FILE = ENTITIES_DIR / "extraction_progress.json"

EXTRACTION_PROMPT = """You are a financial document analyst extracting structured information.

From the document chunk below, extract:
1. ENTITIES: companies, people, subsidiaries, financial metrics, products, regulators
2. RELATIONSHIPS: how entities relate to each other

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "entity name", "type": "COMPANY|PERSON|METRIC|PRODUCT|REGULATOR|SUBSIDIARY", "value": "optional numeric value"}}
  ],
  "relationships": [
    {{"from": "entity1 name", "relation": "OWNS|COMPETES_WITH|INVESTED_IN|REGULATED_BY|REPORTED|ACQUIRED", "to": "entity2 name", "context": "brief context"}}
  ]
}}

If nothing meaningful found, return: {{"entities": [], "relationships": []}}

DOCUMENT CHUNK:
Company: {company_name}
Document: {doc_type} {year}
Page: {page_number}
Section: {section}

Text:
{chunk_text}

JSON ONLY:"""


class EntityExtractor:
    """
    Extracts entities and relationships from document chunks using Groq.
    Fully resumable — tracks which chunk_ids have been processed.
    """

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise FinSightError("GROQ_API_KEY missing.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.requests_made = 0
        logger.info("EntityExtractor initialized")

    def _load_progress(self) -> set[str]:
        """Load set of already-processed chunk_ids."""
        if not PROGRESS_FILE.exists():
            return set()
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        return set(data.get("processed_chunk_ids", []))

    def _save_progress(self, processed_ids: set[str]) -> None:
        """Save progress so extraction is resumable."""
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"processed_chunk_ids": list(processed_ids)}, f)

    def _append_entities(self, chunk_id: str, entities_data: dict, metadata: dict) -> None:
        """Append extracted entities to JSONL file — one line per chunk."""
        record = {
            "chunk_id": chunk_id,
            "company_name": metadata.get("company_name"),
            "doc_type": metadata.get("doc_type"),
            "year": metadata.get("year"),
            "page_number": metadata.get("page_number"),
            "entities": entities_data.get("entities", []),
            "relationships": entities_data.get("relationships", []),
        }
        with open(ENTITIES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def extract_from_chunk(self, chunk: Document) -> dict:
        """
        Extract entities and relationships from one chunk.
        Returns dict with 'entities' and 'relationships' lists.
        Falls back to empty dict on failure.
        """
        meta = chunk.metadata
        prompt = EXTRACTION_PROMPT.format(
            company_name=meta.get("company_name", "Unknown"),
            doc_type=meta.get("doc_type", "Unknown"),
            year=meta.get("year", "Unknown"),
            page_number=meta.get("page_number", "Unknown"),
            section=meta.get("section", "Unknown"),
            chunk_text=chunk.page_content[:800],
        )

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                )
                text = response.choices[0].message.content.strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                result = json.loads(text.strip())
                self.requests_made += 1
                return result

            except json.JSONDecodeError:
                logger.warning(f"JSON parse failed for chunk {meta.get('chunk_id')} — skipping.")
                return {"entities": [], "relationships": []}

            except Exception as e:
                error_str = str(e)
                if "tokens per day" in error_str or "TPD" in error_str:
                    logger.error("Groq daily token limit reached. Progress saved.")
                    raise FinSightError("Groq TPD limit reached. Restart tomorrow.") from e

                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Attempt {attempt + 1} failed: {error_str[:80]}. Retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"All attempts failed for chunk {meta.get('chunk_id')} — skipping.")
                    return {"entities": [], "relationships": []}

        return {"entities": [], "relationships": []}

    def extract_from_chunks(
        self,
        chunks: list[Document],
        target_sections: list[str] = None,
        delay_seconds: float = 3.0,
    ) -> int:
        """
        Extract entities from all chunks. Fully resumable.

        Args:
            chunks:          All chunks from load_all_pdfs()
            target_sections: Only process these sections (saves tokens).
                             Recommended: ["Risk Factors", "Business Overview",
                             "Financial Statements", "Capital Structure"]
                             None = process all sections.
            delay_seconds:   Delay between Groq calls (default 3s = 20 RPM)

        Returns:
            Number of chunks processed this session.
        """
        # Default to high-value sections only — saves ~60% of tokens
        if target_sections is None:
            target_sections = [
                "Risk Factors",
                "Business Overview",
                "Financial Statements",
                "Capital Structure",
                "Management",
                "Objects of the Offer",
            ]

        # Filter to target sections
        filtered = [
            c for c in chunks
            if c.metadata.get("section") in target_sections
        ]

        # Load progress — skip already-processed chunks
        processed_ids = self._load_progress()

        pending = [
            c for c in filtered
            if c.metadata.get("chunk_id") not in processed_ids
        ]

        total = len(filtered)
        already_done = len(filtered) - len(pending)

        print(f"\nEntity extraction plan:")
        print(f"  Chunks in target sections: {total}")
        print(f"  Already processed:         {already_done} (skipping)")
        print(f"  To process now:            {len(pending)}")
        print(f"  Estimated Groq tokens:     ~{len(pending) * 500:,}")
        print(f"  Groq daily limit:          100,000 tokens")
        print(f"  Safe batch per day:        ~200 chunks")

        if not pending:
            print("\nAll chunks already processed.")
            return 0

        processed_this_session = 0

        for i, chunk in enumerate(pending, 1):
            chunk_id = chunk.metadata.get("chunk_id", f"unknown_{i}")
            try:
                entities_data = self.extract_from_chunk(chunk)

                # Save to JSONL
                self._append_entities(chunk_id, entities_data, chunk.metadata)

                # Update progress
                processed_ids.add(chunk_id)
                if i % 10 == 0:
                    self._save_progress(processed_ids)

                processed_this_session += 1
                time.sleep(delay_seconds)

                if i % 50 == 0 or i == len(pending):
                    entity_count = sum(
                        len(d.get("entities", []))
                        for d in [entities_data]
                    )
                    print(
                        f"  [{i}/{len(pending)}] "
                        f"{round(i / len(pending) * 100)}% — "
                        f"Groq requests: {self.requests_made}"
                    )

            except FinSightError:
                self._save_progress(processed_ids)
                print(f"\nStopped at chunk {i}/{len(pending)}. Progress saved.")
                print("Run again tomorrow to continue.")
                return processed_this_session

            except KeyboardInterrupt:
                self._save_progress(processed_ids)
                print(f"\nInterrupted. {processed_this_session} chunks processed. Progress saved.")
                return processed_this_session

        self._save_progress(processed_ids)
        print(f"\nExtraction complete! {processed_this_session} chunks processed.")
        return processed_this_session
        