"""
src/chunking/contextual_chunker.py

Stage 2 Contextual Chunking — Anthropic's Contextual Retrieval method.

Key design: fully resumable.
- Before making ANY Groq call, checks which chunk_ids already exist in Stage 2 collection
- Only enriches chunks that are missing
- Safe to interrupt and restart at any time — never duplicates Groq calls
- Groq free tier: 100,000 tokens/day — 3,184 chunks needs ~2-3 days at this limit
"""

import time
from typing import Optional

from groq import Groq
from langchain_core.documents import Document

from src.configuration.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    EMBEDDINGS_STAGE2_DIR,
    COLLECTION_STAGE2,
)
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import FinSightError

logger = get_logger(__name__)

CONTEXT_PROMPT = """You are a financial document analyst.
Given metadata about a document chunk, write a concise 2-3 sentence context.

Include:
- Company name and document type (e.g. Zomato DRHP 2021)
- Section this chunk belongs to
- What specific topic this chunk discusses

Be factual and specific. No preamble. Just the context sentences.

DOCUMENT METADATA:
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
    Enriches chunks with LLM-generated context before embedding.
    Fully resumable — skips chunks already in the Stage 2 collection.

    Groq free tier limits:
        100,000 tokens/day (TPD)
        Each context call uses ~300-500 tokens
        Safe throughput: ~200-300 chunks/day

    Strategy: run once per day, let it process as many as it can,
    restart tomorrow — it picks up exactly where it left off.
    """

    REQUESTS_PER_MINUTE = 20
    DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # 3 seconds

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise FinSightError("GROQ_API_KEY missing from environment.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.requests_made = 0
        logger.info("ContextualChunker initialized", extra={"model": GROQ_MODEL})

    def _get_existing_stage2_ids(self) -> set[str]:
        """
        Fetch all chunk_ids already present in the Stage 2 ChromaDB collection.
        Called once at the start of enrich_all_chunks — avoids redundant Groq calls.
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
                return set()

            result = collection.get(limit=count, include=[])
            existing = set(result["ids"])
            logger.info(
                f"Stage 2 collection has {len(existing)} existing chunks — will skip these."
            )
            return existing

        except Exception as e:
            logger.warning(f"Could not fetch existing Stage 2 IDs: {e}. Proceeding without skip check.")
            return set()

    def generate_context(self, chunk: Document) -> str:
        """
        Generate a 2-3 sentence context description for one chunk.
        Falls back to metadata string if Groq call fails after retries.
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

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=120,
                )
                context = response.choices[0].message.content.strip()
                self.requests_made += 1
                return context

            except Exception as e:
                error_str = str(e)

                # Detect daily token limit (different from RPM limit)
                if "tokens per day" in error_str or "TPD" in error_str:
                    logger.error(
                        "Groq daily token limit (100K TPD) reached. "
                        "Stop and restart tomorrow — progress is saved. "
                        "Chunks already enriched will be skipped on next run."
                    )
                    raise FinSightError(
                        "Groq daily token limit reached. "
                        "Restart tomorrow to continue from where you left off."
                    ) from e

                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        f"Context generation attempt {attempt + 1} failed: "
                        f"{error_str[:100]}. Retrying in {wait}s"
                    )
                    time.sleep(wait)
                else:
                    # Metadata fallback — better than crashing
                    logger.warning(
                        f"LLM failed for chunk {meta.get('chunk_id')} — using metadata fallback."
                    )
                    return (
                        f"This excerpt is from {meta.get('company_name', 'Unknown')} "
                        f"{meta.get('doc_type', 'document')} ({meta.get('year', 'Unknown')}), "
                        f"Section: {meta.get('section', 'Unknown')}, "
                        f"Page {meta.get('page_number', 'Unknown')}."
                    )
        # Should never reach here but satisfies type checker
        return f"From {meta.get('company_name', 'Unknown')} {meta.get('doc_type', '')}."

    def enrich_chunk(self, chunk: Document) -> Document:
        """
        Enrich a single chunk with context prefix.
        Returns new Document where page_content = context + original text.
        Original text preserved in metadata['original_text'].
        """
        context = self.generate_context(chunk)
        original_text = chunk.page_content
        enriched_content = f"{context}\n\n{original_text}"

        enriched_metadata = dict(chunk.metadata)
        enriched_metadata["original_text"] = original_text
        enriched_metadata["context"] = context
        enriched_metadata["enriched"] = True

        return Document(page_content=enriched_content, metadata=enriched_metadata)

    @log_duration("enrich_all_chunks")
    def enrich_all_chunks(self, chunks: list[Document]) -> list[Document]:
        """
        Enrich all chunks with context. Fully resumable.

        Before making any Groq call:
        1. Fetches all chunk_ids already present in Stage 2 ChromaDB collection
        2. Skips those chunks entirely
        3. Only enriches missing chunks

        Safe to interrupt (Ctrl+C, rate limit crash) and restart.
        On restart, already-enriched chunks are detected and skipped.

        Args:
            chunks: All chunks from load_all_pdfs() — including already-done ones.

        Returns:
            List of enriched Document objects for NEW chunks only.
            Already-indexed chunks are not returned (they're already in ChromaDB).
        """
        # Step 1: Check what's already in Stage 2
        existing_ids = self._get_existing_stage2_ids()

        # Step 2: Filter to only chunks that need enrichment
        pending = [
            c for c in chunks
            if c.metadata.get("chunk_id") not in existing_ids
        ]

        total = len(chunks)
        already_done = total - len(pending)

        logger.info(
            "Enrichment plan",
            extra={
                "total_chunks": total,
                "already_in_stage2": already_done,
                "to_enrich": len(pending),
                "estimated_groq_tokens": len(pending) * 400,
            },
        )

        print(f"\nContextual enrichment plan:")
        print(f"  Total chunks in corpus:     {total}")
        print(f"  Already in Stage 2 index:   {already_done} (skipping)")
        print(f"  Chunks to enrich now:        {len(pending)}")
        print(f"  Estimated Groq tokens:       ~{len(pending) * 400:,}")
        print(f"  Groq free daily limit:       100,000 tokens")

        if not pending:
            print("\nAll chunks already enriched. Nothing to do.")
            return []

        if len(pending) * 400 > 90000:
            sessions_needed = (len(pending) * 400) // 90000 + 1
            print(f"\nWARNING: This will likely take {sessions_needed} days at Groq free tier.")
            print("Run the script each day — it will resume automatically from where it stopped.")

        print(f"\nStarting enrichment (3s delay between calls)...\n")

        enriched = []
        for i, chunk in enumerate(pending, 1):
            try:
                enriched_chunk = self.enrich_chunk(chunk)
                enriched.append(enriched_chunk)
                time.sleep(self.DELAY_BETWEEN_REQUESTS)

                if i % 25 == 0 or i == len(pending):
                    logger.info(
                        f"Progress: {i}/{len(pending)} enriched",
                        extra={
                            "percent": round(i / len(pending) * 100, 1),
                            "groq_requests": self.requests_made,
                        },
                    )
                    print(
                        f"  [{i}/{len(pending)}] "
                        f"{round(i / len(pending) * 100, 1)}% complete | "
                        f"Groq requests: {self.requests_made}"
                    )

            except FinSightError as e:
                # Daily limit hit — return what we have so far so it gets stored
                print(f"\nStopping: {e}")
                print(f"Progress saved: {len(enriched)} chunks enriched this session.")
                print("Restart tomorrow — script will skip these and continue.")
                return enriched

            except KeyboardInterrupt:
                print(f"\nInterrupted. {len(enriched)} chunks enriched this session.")
                print("Restart anytime — already-enriched chunks will be skipped.")
                return enriched

        logger.info(
            "Enrichment complete",
            extra={"enriched": len(enriched), "total_requests": self.requests_made},
        )
        return enriched