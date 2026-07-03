"""
src/retrieval/base_retriever.py

Stage 1 Baseline Retriever — naive vector similarity search.

This is intentionally simple. It exists to:
1. Establish a working end-to-end pipeline quickly
2. Produce baseline RAGAs scores to compare against Stage 2 and Stage 3
3. Give us real failure cases to motivate Stage 2 improvements

Pipeline:
    User query → Voyage AI embed → ChromaDB similarity search → Groq LLM → Answer

Inputs:  User query string, optional metadata filter
Outputs: Answer string + retrieved evidence chunks with citations
"""

import os
import time
from typing import Optional
from langchain_core.documents import Document
from groq import Groq

from src.configuration.config import (
    GROQ_API_KEY, GROQ_MODEL, RETRIEVAL_TOP_K, RERANK_TOP_N
)
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.vector_store.chroma_store import ChromaStore
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import RetrievalError

logger = get_logger(__name__)


# ── Prompt Template ──────────────────────────────────────────────────────────
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
    """
    Format retrieved chunks into a context string for the LLM.
    Each chunk is clearly labeled with its source for citation purposes.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        source_label = (
            f"[{i}] {meta.get('company_name', 'Unknown')} | "
            f"{meta.get('doc_type', 'Unknown')} {meta.get('year', '')} | "
            f"Page {meta.get('page_number', 'Unknown')} | "
            f"Section: {meta.get('section', 'Unknown')}"
        )
        context_parts.append(f"{source_label}\n{chunk.page_content}")

    return "\n\n---\n\n".join(context_parts)


def build_citations(chunks: list[Document]) -> list[dict]:
    """
    Build structured citation objects from retrieved chunks.
    These are returned alongside the answer for the UI evidence panel.
    """
    citations = []
    for chunk in chunks:
        meta = chunk.metadata
        citations.append({
            "company_name": meta.get("company_name", "Unknown"),
            "doc_type": meta.get("doc_type", "Unknown"),
            "year": meta.get("year", "Unknown"),
            "page_number": meta.get("page_number", "Unknown"),
            "section": meta.get("section", "Unknown"),
            "source_file": meta.get("source_file", "Unknown"),
            "chunk_id": meta.get("chunk_id", "Unknown"),
            "similarity_score": meta.get("similarity_score", 0.0),
            "text_excerpt": chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content
        })
    return citations


class BaseRetriever:
    """
    Stage 1 Baseline RAG Retriever.
    
    Simple pipeline: embed query → vector search → LLM answer.
    No BM25, no reranking, no contextual enrichment.
    This is the baseline we will improve in Stage 2.
    """

    def __init__(self) -> None:
        self.embedder = VoyageEmbedder()
        self.store = ChromaStore()
        self.llm_client = Groq(api_key=GROQ_API_KEY)
        logger.info("BaseRetriever initialized (Stage 1)")

    @log_duration("retrieve_and_answer")
    def query(
        self,
        question: str,
        metadata_filter: Optional[dict] = None,
        top_k: int = RERANK_TOP_N
    ) -> dict:
        """
        Main query method — takes a question, returns an answer with citations.
        
        Args:
            question:        User's natural language question.
            metadata_filter: Optional ChromaDB filter.
                             Example: {"company_name": "Zomato"}
                             Example: {"$and": [{"company_name": "Zomato"}, {"year": "2021"}]}
            top_k:           Number of chunks to use for answer generation.
        
        Returns:
            dict with keys:
                - answer:     Generated answer string
                - citations:  List of citation dicts with source info
                - chunks:     Raw retrieved Document objects
                - latency_ms: Total time taken
        
        Example:
            retriever = BaseRetriever()
            result = retriever.query(
                question="What are Zomato's main risk factors?",
                metadata_filter={"company_name": "Zomato"}
            )
            print(result["answer"])
            for cite in result["citations"]:
                print(f"Source: {cite['company_name']} | Page {cite['page_number']}")
        """
        start_time = time.time()

        # Step 1: Check collection has data
        stats = self.store.get_collection_stats()
        if stats["total_chunks"] == 0:
            return {
                "answer": "No documents are indexed yet. Please run the indexing pipeline first.",
                "citations": [],
                "chunks": [],
                "latency_ms": 0
            }

        # Step 2: Embed the question
        logger.info(f"Processing query", extra={"question": question[:100]})
        query_embedding = self.embedder.embed_query(question)

        # Step 3: Retrieve from ChromaDB
        # Retrieve more than needed (RETRIEVAL_TOP_K) so Stage 2 reranker has room to work
        retrieved_chunks = self.store.similarity_search(
            query_embedding=query_embedding,
            top_k=RETRIEVAL_TOP_K,
            metadata_filter=metadata_filter
        )

        if not retrieved_chunks:
            return {
                "answer": "No relevant documents found for this query. Try different search terms or check your metadata filter.",
                "citations": [],
                "chunks": [],
                "latency_ms": round((time.time() - start_time) * 1000)
            }

        # Stage 1: Use top_k directly (no reranking yet)
        top_chunks = retrieved_chunks[:top_k]

        # Step 4: Format context for LLM
        context = format_context(top_chunks)
        prompt = ANSWER_PROMPT.format(context=context, question=question)

        # Step 5: Generate answer with Groq
        try:
            response = self.llm_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,    # Low temperature = more factual, less creative
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
                "question_length": len(question),
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_used": len(top_chunks),
                "answer_length": len(answer),
                "latency_ms": latency_ms
            }
        )

        return {
            "answer": answer,
            "citations": build_citations(top_chunks),
            "chunks": top_chunks,
            "chunks_retrieved_total": len(retrieved_chunks),
            "latency_ms": latency_ms
        }


if __name__ == "__main__":
    retriever = BaseRetriever()
    stats = retriever.store.get_collection_stats()
    print(f"\nCollection: {stats['total_chunks']} chunks indexed")
    print(f"Companies: {stats.get('companies_sampled', [])}")

    if stats["total_chunks"] > 0:
        result = retriever.query(
            question="What are the main risk factors mentioned in the DRHP?",
        )
        print(f"\n{'='*60}")
        print(f"ANSWER ({result['latency_ms']}ms):")
        print(result["answer"])
        print(f"\nCITATIONS ({len(result['citations'])}):")
        for c in result["citations"]:
            print(f"  - {c['company_name']} | {c['doc_type']} | Page {c['page_number']} | Score: {c['similarity_score']}")
    else:
        print("\nNo documents indexed yet. Run the indexing script first.")