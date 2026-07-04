"""
src/retrieval/base_retriever.py

Stage 1 Baseline Retriever.
Naive vector similarity search — no BM25, no reranking, no context enrichment.
Exists to produce baseline scores for the ablation study.
"""

import time
from typing import Optional

from groq import Groq
from langchain_core.documents import Document

from src.configuration.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
    EMBEDDINGS_STAGE1_DIR,
    COLLECTION_STAGE1,
)
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.vector_store.chroma_store import ChromaStore
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import RetrievalError

logger = get_logger(__name__)

ANSWER_PROMPT = """You are a senior financial analyst specializing in Indian capital markets.
You have been given excerpts from official financial documents (DRHPs, Annual Reports, Earnings Transcripts).

STRICT RULES:
1. Answer ONLY using information from the provided document excerpts below.
2. If the answer is not in the excerpts, say: "The provided documents do not contain sufficient information to answer this question."
3. Never invent numbers, dates, names, or financial figures.
4. Always specify which company and document you are drawing from.
5. Write in the style of a professional investment research note.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER (cite sources in format: [Company | Document | Page X]):"""


def format_context(chunks: list[Document]) -> str:
    """Format retrieved chunks into a labeled context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        label = (
            f"[{i}] {meta.get('company_name', 'Unknown')} | "
            f"{meta.get('doc_type', 'Unknown')} {meta.get('year', '')} | "
            f"Page {meta.get('page_number', 'Unknown')} | "
            f"Section: {meta.get('section', 'Unknown')}"
        )
        parts.append(f"{label}\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def build_citations(chunks: list[Document]) -> list[dict]:
    """Build structured citation objects from retrieved chunks."""
    citations = []
    for chunk in chunks:
        meta = chunk.metadata
        citations.append(
            {
                "company_name": meta.get("company_name", "Unknown"),
                "doc_type": meta.get("doc_type", "Unknown"),
                "year": meta.get("year", "Unknown"),
                "page_number": meta.get("page_number", "Unknown"),
                "section": meta.get("section", "Unknown"),
                "source_file": meta.get("source_file", "Unknown"),
                "chunk_id": meta.get("chunk_id", "Unknown"),
                "similarity_score": meta.get("similarity_score", 0.0),
                "text_excerpt": (
                    chunk.page_content[:300] + "..."
                    if len(chunk.page_content) > 300
                    else chunk.page_content
                ),
            }
        )
    return citations


class BaseRetriever:
    """
    Stage 1 Baseline RAG Retriever.
    Uses Stage 1 ChromaDB collection by default.
    """

    def __init__(self) -> None:
        self.embedder = VoyageEmbedder()
        self.store = ChromaStore(
            persist_dir=EMBEDDINGS_STAGE1_DIR,
            collection_name=COLLECTION_STAGE1,
        )
        self.llm_client = Groq(api_key=GROQ_API_KEY)
        logger.info("BaseRetriever initialized (Stage 1)")

    @log_duration("retrieve_and_answer")
    def query(
        self,
        question: str,
        metadata_filter: Optional[dict] = None,
        top_k: int = RERANK_TOP_N,
    ) -> dict:
        """
        Takes a question, returns an answer with citations.

        Args:
            question:        User's natural language question.
            metadata_filter: Optional ChromaDB filter dict.
            top_k:           Number of chunks to use for answer generation.

        Returns:
            dict with keys: answer, citations, chunks, latency_ms
        """
        start_time = time.time()

        stats = self.store.get_collection_stats()
        if stats["total_chunks"] == 0:
            return {
                "answer": "No documents indexed yet. Run the indexing pipeline first.",
                "citations": [],
                "chunks": [],
                "latency_ms": 0,
            }

        logger.info("Processing query", extra={"question": question[:100]})

        query_embedding = self.embedder.embed_query(question)

        retrieved_chunks = self.store.similarity_search(
            query_embedding=query_embedding,
            top_k=RETRIEVAL_TOP_K,
            metadata_filter=metadata_filter,
        )

        if not retrieved_chunks:
            return {
                "answer": "No relevant documents found for this query.",
                "citations": [],
                "chunks": [],
                "latency_ms": round((time.time() - start_time) * 1000),
            }

        top_chunks = retrieved_chunks[:top_k]
        context = format_context(top_chunks)
        prompt = ANSWER_PROMPT.format(context=context, question=question)

        try:
            response = self.llm_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RetrievalError(f"Answer generation failed: {e}") from e

        latency_ms = round((time.time() - start_time) * 1000)

        logger.info(
            "Query complete",
            extra={
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_used": len(top_chunks),
                "latency_ms": latency_ms,
            },
        )

        return {
            "answer": answer,
            "citations": build_citations(top_chunks),
            "chunks": top_chunks,
            "chunks_retrieved_total": len(retrieved_chunks),
            "latency_ms": latency_ms,
        }


if __name__ == "__main__":
    retriever = BaseRetriever()
    stats = retriever.store.get_collection_stats()
    print(f"Collection: {stats['total_chunks']} chunks")
    print(f"Companies: {stats.get('companies_sampled', [])}")

    if stats["total_chunks"] > 0:
        result = retriever.query(
            question="What are the main risk factors mentioned in the DRHP?",
        )
        print(f"\nAnswer ({result['latency_ms']}ms):")
        print(result["answer"])
        print(f"\nCitations ({len(result['citations'])}):")
        for c in result["citations"]:
            print(
                f"  - {c['company_name']} | {c['doc_type']} | "
                f"Page {c['page_number']} | Score: {c['similarity_score']}"
            )
    else:
        print("No documents indexed yet.")