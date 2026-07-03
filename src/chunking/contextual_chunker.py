"""
src/chunking/contextual_chunker.py

Stage 2 Contextual Chunking — Anthropic's Contextual Retrieval method.

The problem with Stage 1 chunking:
    A chunk saying "revenue grew 23% YoY" has no context about which company,
    which year, or which revenue line. Its embedding is weak because the
    embedding model doesn't know what this chunk is about.

The fix:
    Before embedding each chunk, ask an LLM to prepend a 2-3 sentence
    context description: "This chunk is from Zomato Limited's DRHP filed
    April 2021, Section: Financial Summary. It discusses year-over-year
    revenue growth for food delivery operations."

    Now the embedding is rich, precise, and company-specific.

Result from Anthropic's benchmarks:
    Retrieval failure rate drops from 5.7% to 2.9% — a 49% improvement.

Inputs:  Raw chunks (LangChain Documents from base_chunker / data_loader)
Outputs: Same chunks with context-enriched page_content for embedding
         (original text preserved in metadata for display)
"""

import time
from typing import Optional
from langchain_core.documents import Document
from groq import Groq

from src.configuration.config import GROQ_API_KEY, GROQ_MODEL
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import FinSightError

logger = get_logger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────
CONTEXT_PROMPT = """You are a financial document analyst. 
Given a full document excerpt and a specific chunk from that document, 
write a concise 2-3 sentence context description for the chunk.

The context must include:
- The company name and document type (e.g. Zomato DRHP 2021)
- The section this chunk belongs to (e.g. Risk Factors, Financial Statements)
- What specific topic or data point this chunk discusses

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

    Why Groq for context generation?
    - Free tier with 14,400 requests/day — sufficient for our corpus
    - Llama 3.3 70B understands financial document structure well
    - Fast inference — contextual enrichment of 3,184 chunks takes ~20 minutes

    Cost estimate: 3,184 chunks × ~500 tokens each = ~1.6M tokens
    Groq free tier: 14,400 requests/day × ~6,000 tokens/request = well within limits.
    """

    # Rate limiting for Groq free tier
    REQUESTS_PER_MINUTE = 25       # Groq free tier: 30 RPM, stay conservative
    DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # ~2.4 seconds

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise FinSightError("GROQ_API_KEY missing from environment.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.requests_made = 0
        logger.info("ContextualChunker initialized", extra={"model": GROQ_MODEL})

    def generate_context(self, chunk: Document) -> str:
        """
        Generate a context description for a single chunk.

        Args:
            chunk: LangChain Document with metadata (company_name, doc_type, etc.)

        Returns:
            Context string (2-3 sentences) to prepend to chunk text.
            Falls back to a simple metadata string if LLM call fails.
        """
        meta = chunk.metadata
        prompt = CONTEXT_PROMPT.format(
            company_name=meta.get("company_name", "Unknown Company"),
            doc_type=meta.get("doc_type", "Unknown Document Type"),
            year=meta.get("year", "Unknown Year"),
            section=meta.get("section", "Unknown Section"),
            page_number=meta.get("page_number", "Unknown"),
            chunk_text=chunk.page_content[:800],  # first 800 chars sufficient for context
        )

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150,   # context should be brief
                )
                context = response.choices[0].message.content.strip()
                self.requests_made += 1
                return context

            except Exception as e:
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        f"Context generation attempt {attempt+1} failed: {e}. "
                        f"Retrying in {wait}s"
                    )
                    time.sleep(wait)
                else:
                    # Fallback — construct context from metadata without LLM
                    logger.warning(
                        f"LLM context generation failed for chunk "
                        f"{meta.get('chunk_id')} — using metadata fallback."
                    )
                    return (
                        f"This excerpt is from {meta.get('company_name', 'Unknown')} "
                        f"{meta.get('doc_type', 'document')} ({meta.get('year', 'Unknown')}), "
                        f"Section: {meta.get('section', 'Unknown')}, "
                        f"Page {meta.get('page_number', 'Unknown')}."
                    )

    def enrich_chunk(self, chunk: Document) -> Document:
        """
        Enrich a single chunk with context.
        Returns a new Document where:
        - page_content = context + original text  (this gets embedded)
        - metadata["original_text"] = original text  (this gets displayed to user)
        - metadata["context"] = just the context string  (for debugging)
        """
        context = self.generate_context(chunk)

        # Store original text before modifying
        original_text = chunk.page_content

        # New content = context prefix + original text
        enriched_content = f"{context}\n\n{original_text}"

        # Build enriched document — preserve all original metadata
        enriched_metadata = dict(chunk.metadata)
        enriched_metadata["original_text"] = original_text
        enriched_metadata["context"] = context
        enriched_metadata["enriched"] = True

        return Document(
            page_content=enriched_content,
            metadata=enriched_metadata
        )

    @log_duration("enrich_all_chunks")
    def enrich_all_chunks(
        self,
        chunks: list[Document],
        batch_size: int = 50,
        save_progress_every: int = 100
    ) -> list[Document]:
        """
        Enrich all chunks with context. Handles rate limiting automatically.

        Args:
            chunks:               All chunks to enrich
            batch_size:           Process this many before logging progress
            save_progress_every:  Log progress every N chunks

        Returns:
            List of enriched Document objects ready for embedding.
        """
        enriched = []
        total = len(chunks)
        logger.info(
            f"Starting contextual enrichment",
            extra={"total_chunks": total, "estimated_minutes": round(total * self.DELAY_BETWEEN_REQUESTS / 60, 1)}
        )

        for i, chunk in enumerate(chunks, 1):
            enriched_chunk = self.enrich_chunk(chunk)
            enriched.append(enriched_chunk)

            # Rate limiting — stay under Groq's RPM limit
            time.sleep(self.DELAY_BETWEEN_REQUESTS)

            if i % save_progress_every == 0 or i == total:
                logger.info(
                    f"Progress: {i}/{total} chunks enriched",
                    extra={"percent": round(i/total*100, 1), "requests_made": self.requests_made}
                )

        logger.info(
            "Contextual enrichment complete",
            extra={"enriched": len(enriched), "total_requests": self.requests_made}
        )
        return enriched


if __name__ == "__main__":
    from langchain_core.documents import Document

    # Test with a single chunk
    chunker = ContextualChunker()
    test_chunk = Document(
        page_content="Revenue from operations grew by 23% year over year, driven primarily by increased order volumes and higher average order values across all markets.",
        metadata={
            "company_name": "Zomato",
            "doc_type": "DRHP",
            "year": "2021",
            "section": "Financial Statements",
            "page_number": 187,
            "chunk_id": "zomato_drhp_2021_p187_c2"
        }
    )

    print("Original text:")
    print(test_chunk.page_content)
    print("\nEnriching with context...")

    enriched = chunker.enrich_chunk(test_chunk)

    print("\nEnriched content (what gets embedded):")
    print(enriched.page_content)
    print("\nContext generated:")
    print(enriched.metadata["context"])
    print("\nContextual chunker working correctly.")