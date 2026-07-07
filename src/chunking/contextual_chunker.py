"""
src/chunking/contextual_chunker.py

Stage 2 Contextual Chunking — Anthropic's Contextual Retrieval method.
Now uses ProviderRouter (Cerebras primary) instead of Groq directly.
This gives 1M tokens/day instead of 100K — Stage 2 finishes in one session.

Fully resumable: checks Stage 2 ChromaDB collection before any API call.
Already-enriched chunks are skipped. Safe to stop and restart anytime.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from src.configuration.config import (
    EMBEDDINGS_STAGE2_DIR,
    COLLECTION_STAGE2,
)
from src.llm.provider_router import get_router, TaskType
from src.shared.exceptions import FinSightError

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """You are a financial document analyst. Given metadata and text from a document chunk, write a concise 2-3 sentence context description.

Include ALL of these:
- Company name and document type (e.g. "This excerpt is from Zomato Limited's DRHP filed April 2021")
- Section this chunk belongs to (e.g. "in the Risk Factors section")
- What specific topic or data point this chunk discusses

Be factual and specific. No preamble. Output only the context sentences.

METADATA:
Company: {company_name}
Document Type: {doc_type}
Year: {year}
Section: {section}
Page: {page_number}

CHUNK TEXT:
{chunk_text}

CONTEXT (2-3 sentences only):"""


class ContextualChunker:
    """
    Enriches document chunks with LLM-generated context before embedding.

    Uses ProviderRouter which assigns this task to Cerebras (1M tokens/day).
    Falls back to Groq if Cerebras is unavailable.

    Fully resumable: at startup, fetches all chunk_ids already present in
    the Stage 2 ChromaDB collection and skips those chunks entirely.
    No Groq/Cerebras calls are made for already-processed chunks.
    """

    # 3 seconds between calls = 20 RPM, stays safely under all provider limits
    DELAY_BETWEEN_CALLS: float = 12.0

    def __init__(self) -> None:
        self.router = get_router()
        logger.info(
            "ContextualChunker initialized",
            extra={"providers": self.router.available_providers()},
        )

    def _get_existing_stage2_ids(self) -> set[str]:
        """
        Fetch all chunk_ids already in the Stage 2 ChromaDB collection.
        Called once at the start — avoids any redundant API calls.
        Returns empty set if collection doesn't exist yet.
        """
        try:
            import chromadb
            from chromadb.config import Settings

            EMBEDDINGS_STAGE2_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(EMBEDDINGS_STAGE2_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection(
                name=COLLECTION_STAGE2,
                metadata={"hnsw:space": "cosine"},
            )
            count = collection.count()
            if count == 0:
                logger.info("Stage 2 collection is empty — all chunks need enrichment")
                return set()

            result = collection.get(limit=count, include=[])
            existing = set(result["ids"])
            logger.info(
                f"Stage 2 collection has {len(existing)} existing chunks — will skip these"
            )
            return existing

        except Exception as e:
            logger.warning(
                f"Could not read Stage 2 collection: {e}. "
                "Proceeding without skip check (will re-enrich all chunks)."
            )
            return set()

    def generate_context(self, chunk: Document) -> str:
        """
        Generate a 2-3 sentence context description for one chunk.
        Uses ProviderRouter — Cerebras by default, Groq as fallback.

        Returns a metadata-based fallback string if all providers fail.
        Never raises an exception — the pipeline continues regardless.
        """
        meta = chunk.metadata
        prompt = CONTEXT_PROMPT.format(
            company_name=meta.get("company_name", "Unknown"),
            doc_type=meta.get("doc_type", "Unknown"),
            year=meta.get("year", "Unknown"),
            section=meta.get("section", "Unknown"),
            page_number=meta.get("page_number", "Unknown"),
            chunk_text=chunk.page_content[:600],
        )

        try:
            context = self.router.generate(
                task=TaskType.ENRICHMENT,
                prompt=prompt,
                temperature=0.1,
                max_tokens=120,
            )
            return context.strip()

        except FinSightError as e:
            # All providers exhausted — use metadata fallback
            logger.warning(
                f"All providers failed for chunk {meta.get('chunk_id')} — "
                f"using metadata fallback. Error: {e}"
            )
            return (
                f"This excerpt is from {meta.get('company_name', 'Unknown')} "
                f"{meta.get('doc_type', 'document')} ({meta.get('year', 'Unknown')}), "
                f"Section: {meta.get('section', 'Unknown')}, "
                f"Page {meta.get('page_number', 'Unknown')}."
            )

        except Exception as e:
            logger.warning(
                f"Unexpected error for chunk {meta.get('chunk_id')}: {e} — "
                "using metadata fallback."
            )
            return (
                f"From {meta.get('company_name', 'Unknown')} "
                f"{meta.get('doc_type', '')} ({meta.get('year', '')})."
            )

    def enrich_chunk(self, chunk: Document) -> Document:
        """
        Enrich a single chunk. Returns new Document where:
          page_content = context_prefix + original_text  (this gets embedded)
          metadata['original_text'] = original text  (shown to user in UI)
          metadata['context'] = just the generated context (for debugging)
          metadata['enriched'] = True
        """
        context = self.generate_context(chunk)
        original_text = chunk.page_content

        enriched_metadata = dict(chunk.metadata)
        enriched_metadata["original_text"] = original_text
        enriched_metadata["context"] = context
        enriched_metadata["enriched"] = True

        return Document(
            page_content=f"{context}\n\n{original_text}",
            metadata=enriched_metadata,
        )

    def enrich_all_chunks(self, chunks: list[Document]) -> list[Document]:
        """
        Enrich all chunks. Fully resumable.

        Workflow:
          1. Fetch existing Stage 2 chunk_ids from ChromaDB (one fast query)
          2. Filter chunks to only those NOT already enriched
          3. Enrich remaining chunks with context via ProviderRouter
          4. Return enriched chunks ready for embedding and storage

        If stopped (Ctrl+C, rate limit, crash):
          - Partially enriched chunks are already stored in ChromaDB
            (by index_documents_stage2.py which stores after each batch)
          - Restart and the already-done chunks are automatically skipped

        Args:
            chunks: All chunks from load_all_pdfs()

        Returns:
            List of enriched Documents for NEW chunks only.
            Already-indexed chunks return empty list (already in ChromaDB).
        """
        # Step 1: Check what's already done
        existing_ids = self._get_existing_stage2_ids()

        # Step 2: Filter to pending chunks only
        pending = [
            c for c in chunks
            if c.metadata.get("chunk_id") not in existing_ids
        ]

        already_done = len(chunks) - len(pending)
        estimated_tokens = len(pending) * 400

        logger.info(
            "Enrichment plan",
            extra={
                "total": len(chunks),
                "already_done": already_done,
                "to_enrich": len(pending),
                "estimated_tokens": estimated_tokens,
            },
        )

        print(f"\nContextual enrichment plan:")
        print(f"  Total chunks in corpus:      {len(chunks)}")
        print(f"  Already in Stage 2 index:    {already_done} (skipping)")
        print(f"  Chunks to enrich now:         {len(pending)}")
        print(f"  Estimated API tokens:         ~{estimated_tokens:,}")
        print(f"  Provider:                     Cerebras (1M/day) → Groq fallback")
        print(f"  Provider status:              {self.router.status_report()}")

        if not pending:
            print("\nAll chunks already enriched. Nothing to do.")
            return []

        print(f"\nStarting enrichment ({self.DELAY_BETWEEN_CALLS}s between calls)...\n")

        enriched = []
        failed_count = 0

        for i, chunk in enumerate(pending, 1):
            try:
                enriched_chunk = self.enrich_chunk(chunk)
                enriched.append(enriched_chunk)
                time.sleep(self.DELAY_BETWEEN_CALLS)

                if i % 25 == 0 or i == len(pending):
                    print(
                        f"  [{i}/{len(pending)}] "
                        f"{round(i / len(pending) * 100, 1)}% | "
                        f"Providers: {self.router.status_report()}"
                    )

            except KeyboardInterrupt:
                print(f"\nInterrupted at chunk {i}. {len(enriched)} chunks enriched.")
                print("Restart anytime — already-enriched chunks will be skipped.")
                return enriched

            except Exception as e:
                failed_count += 1
                logger.error(f"Chunk {i} failed completely: {e}")
                # Continue with next chunk — don't stop the whole run
                continue

        logger.info(
            "Enrichment session complete",
            extra={
                "enriched": len(enriched),
                "failed": failed_count,
            },
        )
        print(f"\nEnrichment complete: {len(enriched)} enriched, {failed_count} failed")
        return enriched