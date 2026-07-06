"""
src/retrieval/base_retriever.py

Stage 1 Baseline Retriever.
Updated to use ProviderRouter (Gemini Flash for reasoning).
Returns full evidence chain with per-chunk confidence scores
for the evidence assembler and UI citation panel.
"""

import time
import logging
from typing import Optional

from langchain_core.documents import Document

from src.configuration.config import (
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
    EMBEDDINGS_STAGE1_DIR,
    COLLECTION_STAGE1,
)
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.vector_store.chroma_store import ChromaStore
from src.llm.provider_router import get_router, TaskType
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import RetrievalError

logger = get_logger(__name__)

ANSWER_PROMPT = """You are a senior financial analyst specializing in Indian capital markets.
You have been given excerpts from official financial documents (DRHPs, Annual Reports, Earnings Transcripts).

STRICT RULES:
1. Answer ONLY using information from the provided document excerpts below.
2. If the answer is not in the excerpts, say exactly: "The provided documents do not contain sufficient information to answer this question."
3. Never invent numbers, dates, names, or financial figures.
4. For every factual claim, specify which company and document it comes from.
5. Write in the style of a professional investment research note.
6. If you see conflicting information across documents, explicitly state the conflict.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER (cite every claim as [Company | Document Type Year | Page X]):"""


def format_context(chunks: list[Document]) -> str:
    """Format retrieved chunks into labeled context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        label = (
            f"[{i}] {meta.get('company_name', 'Unknown')} | "
            f"{meta.get('doc_type', 'Unknown')} {meta.get('year', '')} | "
            f"Page {meta.get('page_number', 'Unknown')} | "
            f"Section: {meta.get('section', 'Unknown')}"
        )
        # Use original_text if available (Stage 2 contextual chunks store this)
        # so the LLM sees clean text, not the context-prefixed version
        display_text = meta.get("original_text", chunk.page_content)
        parts.append(f"{label}\n{display_text}")
    return "\n\n---\n\n".join(parts)


def build_citations(chunks: list[Document]) -> list[dict]:
    """
    Build structured citation objects from retrieved chunks.
    Every citation carries enough information to:
    - Display in the evidence panel
    - Locate the exact page in the source PDF
    - Show confidence score
    """
    citations = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata

        # Prefer rerank_score > similarity_score for confidence display
        confidence = meta.get("rerank_score")
        if confidence is None:
            confidence = meta.get("similarity_score", 0.0)

        citations.append({
            "citation_index": i + 1,
            "company_name": meta.get("company_name", "Unknown"),
            "doc_type": meta.get("doc_type", "Unknown"),
            "year": meta.get("year", "Unknown"),
            "page_number": meta.get("page_number", "Unknown"),
            "section": meta.get("section", "Unknown"),
            "source_file": meta.get("source_file", "Unknown"),
            "chunk_id": meta.get("chunk_id", "Unknown"),
            "confidence": round(float(confidence), 4),
            # Show original text (without context prefix) in evidence panel
            "text_excerpt": (
                meta.get("original_text", chunk.page_content)[:400] + "..."
                if len(meta.get("original_text", chunk.page_content)) > 400
                else meta.get("original_text", chunk.page_content)
            ),
            # Full text for PDF highlighting
            "full_text": meta.get("original_text", chunk.page_content),
        })
    return citations


class BaseRetriever:
    """
    Stage 1 Baseline Retriever.
    Uses Stage 1 ChromaDB collection.
    Reasoning routed to Gemini Flash via ProviderRouter.
    """

    def __init__(self) -> None:
        self.embedder = VoyageEmbedder()
        self.store = ChromaStore(
            persist_dir=EMBEDDINGS_STAGE1_DIR,
            collection_name=COLLECTION_STAGE1,
        )
        self.router = get_router()
        logger.info(
            "BaseRetriever initialized",
            extra={
                "stage": 1,
                "collection": COLLECTION_STAGE1,
                "chunks": self.store.collection.count(),
            },
        )

    @log_duration("retrieve_and_answer")
    def query(
        self,
        question: str,
        metadata_filter: Optional[dict] = None,
        top_k: int = RERANK_TOP_N,
        hard_query: bool = False,
    ) -> dict:
        """
        Retrieve relevant chunks and generate a grounded answer.

        Args:
            question:        User's natural language question.
            metadata_filter: ChromaDB filter dict e.g. {"company_name": "Zomato"}
            top_k:           Number of chunks to use for answer generation.
            hard_query:      If True, routes reasoning to Gemini Pro instead of Flash.

        Returns:
            dict with keys:
                answer         — generated answer string
                citations      — list of citation dicts with page/chunk info
                chunks         — raw retrieved Document objects
                latency_ms     — total time in milliseconds
                provider_used  — which LLM provider generated the answer
        """
        start_time = time.time()

        # Health check
        stats = self.store.get_collection_stats()
        if stats["total_chunks"] == 0:
            return {
                "answer": (
                    "No documents are indexed yet. "
                    "Run: python scripts/index_documents.py"
                ),
                "citations": [],
                "chunks": [],
                "latency_ms": 0,
                "provider_used": "none",
            }

        logger.info("Processing query", extra={"question": question[:100]})

        # Embed the question
        query_embedding = self.embedder.embed_query(question)

        # Retrieve from ChromaDB
        retrieved_chunks = self.store.similarity_search(
            query_embedding=query_embedding,
            top_k=RETRIEVAL_TOP_K,
            metadata_filter=metadata_filter,
        )

        if not retrieved_chunks:
            return {
                "answer": (
                    "No relevant documents found for this query. "
                    "Try different search terms or check your filters."
                ),
                "citations": [],
                "chunks": [],
                "latency_ms": round((time.time() - start_time) * 1000),
                "provider_used": "none",
            }

        # Use top_k chunks for generation
        top_chunks = retrieved_chunks[:top_k]
        context = format_context(top_chunks)
        prompt = ANSWER_PROMPT.format(context=context, question=question)

        # Route to appropriate reasoning provider
        try:
            answer = self.router.generate(
                task=TaskType.REASONING,
                prompt=prompt,
                temperature=0.1,
                max_tokens=1500,
                hard_query=hard_query,
            )
            provider_used = "gemini_flash" if not hard_query else "gemini_pro"
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise RetrievalError(f"Answer generation failed: {e}") from e

        latency_ms = round((time.time() - start_time) * 1000)

        logger.info(
            "Query complete",
            extra={
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_used": len(top_chunks),
                "latency_ms": latency_ms,
                "provider": provider_used,
            },
        )

        return {
            "answer": answer,
            "citations": build_citations(top_chunks),
            "chunks": top_chunks,
            "chunks_retrieved_total": len(retrieved_chunks),
            "latency_ms": latency_ms,
            "provider_used": provider_used,
        }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    retriever = BaseRetriever()
    stats = retriever.store.get_collection_stats()
    print(f"Collection: {stats['total_chunks']} chunks")
    print(f"Companies: {stats.get('companies_sampled', [])}")

    if stats["total_chunks"] > 0:
        result = retriever.query(
            question="What are the main risk factors mentioned in Zomato's DRHP?",
            metadata_filter={"company_name": "Zomato"},
        )
        print(f"\nAnswer ({result['latency_ms']}ms via {result['provider_used']}):")
        print(result["answer"])
        print(f"\nCitations ({len(result['citations'])}):")
        for c in result["citations"]:
            print(
                f"  [{c['citation_index']}] {c['company_name']} | "
                f"{c['doc_type']} | Page {c['page_number']} | "
                f"Confidence: {c['confidence']}"
            )
    else:
        print("No documents indexed.")