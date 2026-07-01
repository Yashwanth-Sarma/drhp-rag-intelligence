"""
src/retrieval/hybrid_retriever.py

Stage 2 Hybrid Retrieval — BM25 + Vector Search + Metadata Filter + Reranker.

Why hybrid?
    Vector search: finds semantically similar chunks ("what is the financial performance")
    BM25:          finds exact keyword matches ("EBITDA", "GMV", "Clause 49", "FY2021")

    Financial documents contain a lot of precise terminology that
    vector search misses. BM25 catches these exact matches.
    Combining both gives recall that neither achieves alone.

Fusion method: Reciprocal Rank Fusion (RRF)
    RRF merges two ranked lists into one without needing score normalization.
    Formula: RRF(d) = Σ 1 / (k + rank(d))  where k=60 is a standard constant.
    Chunks appearing in both lists get higher combined scores.

Full Stage 2 Pipeline:
    Query
      → metadata filter (narrow ChromaDB search space)
      → vector search (top 20 by cosine similarity)
      → BM25 search (top 20 by keyword relevance)
      → RRF fusion (top 20 combined)
      → Cohere reranker (top 5 final)
      → LLM answer generation
"""

from typing import Optional
from langchain.schema import Document
from rank_bm25 import BM25Okapi

from src.vector_store.chroma_store import ChromaStore
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.retrieval.reranker import CohereReranker
from src.retrieval.metadata_filter import build_filter, extract_companies_from_query
from src.configuration.config import RETRIEVAL_TOP_K, RERANK_TOP_N
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import RetrievalError

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    results_list: list[list[Document]],
    k: int = 60
) -> list[Document]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        results_list: List of result lists (e.g. [vector_results, bm25_results])
        k:            RRF constant — 60 is standard, higher = less penalty for low ranks

    Returns:
        Single merged and re-ranked list of Documents, highest RRF score first.
        Duplicate chunks (same chunk_id) are merged — no duplicates in output.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Document] = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            rrf_score = 1.0 / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            chunk_map[chunk_id] = doc  # store/overwrite chunk object

    # Sort by combined RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Add RRF score to metadata for debugging
    merged = []
    for chunk_id in sorted_ids:
        doc = chunk_map[chunk_id]
        updated_meta = dict(doc.metadata)
        updated_meta["rrf_score"] = round(scores[chunk_id], 6)
        merged.append(Document(page_content=doc.page_content, metadata=updated_meta))

    return merged


class HybridRetriever:
    """
    Stage 2 retriever combining vector search, BM25, and reranking.

    The BM25 index is built in memory at query time from the ChromaDB collection.
    This is fine for our corpus size (~3,000-10,000 chunks).
    For larger corpora (100K+ chunks), pre-build and persist the BM25 index.
    """

    def __init__(self) -> None:
        self.store = ChromaStore()
        self.embedder = VoyageEmbedder()
        self.reranker = CohereReranker()
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_chunks: list[Document] = []
        logger.info("HybridRetriever initialized (Stage 2)")

    def _build_bm25_index(
        self,
        metadata_filter: Optional[dict] = None
    ) -> tuple[BM25Okapi, list[Document]]:
        """
        Build BM25 index from ChromaDB collection.
        Filtered by metadata if provided — so BM25 only searches relevant company docs.

        Returns:
            Tuple of (BM25Okapi index, list of Documents in index order)
        """
        logger.info("Building BM25 index from collection...")

        # Get all chunks (optionally filtered)
        results = self.store.collection.get(
            where=metadata_filter if metadata_filter else None,
            include=["documents", "metadatas"]
        )

        if not results["documents"]:
            logger.warning("No documents found for BM25 index — check metadata filter.")
            return None, []

        chunks = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

        # Tokenize for BM25 — simple whitespace tokenization
        # For financial text, this works well because exact term matching is the goal
        tokenized = [doc.split() for doc in results["documents"]]
        index = BM25Okapi(tokenized)

        logger.info(
            "BM25 index built",
            extra={"total_docs": len(chunks)}
        )
        return index, chunks

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[dict] = None
    ) -> list[Document]:
        """
        BM25 keyword search.
        Rebuilds index if metadata filter changes (ensures correct scope).
        """
        bm25, chunks = self._build_bm25_index(metadata_filter)
        if not bm25 or not chunks:
            return []

        # Tokenize query same way as documents
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)

        # Get top_k indices by score
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive BM25 score
                doc = chunks[idx]
                updated_meta = dict(doc.metadata)
                updated_meta["bm25_score"] = round(float(scores[idx]), 4)
                results.append(Document(page_content=doc.page_content, metadata=updated_meta))

        logger.info(
            "BM25 search complete",
            extra={"results": len(results), "top_score": results[0].metadata["bm25_score"] if results else 0}
        )
        return results

    @log_duration("hybrid_query")
    def query(
        self,
        question: str,
        companies: Optional[list[str]] = None,
        years: Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
        top_k: int = RETRIEVAL_TOP_K,
        final_top_n: int = RERANK_TOP_N,
        auto_detect_companies: bool = True
    ) -> dict:
        """
        Full Stage 2 hybrid retrieval pipeline.

        Args:
            question:              User's natural language question.
            companies:             Filter to these companies (e.g. ["Zomato"]).
            years:                 Filter to these years (e.g. ["2021"]).
            doc_types:             Filter to these doc types.
            top_k:                 Chunks per retrieval method before fusion.
            final_top_n:           Chunks after reranking — sent to LLM.
            auto_detect_companies: If True, detect company names from query text.

        Returns:
            dict with: answer, citations, chunks, latency_ms, retrieval_debug
        """
        import time
        from groq import Groq
        from src.retrieval.base_retriever import (
            format_context, build_citations, ANSWER_PROMPT
        )
        from src.configuration.config import GROQ_API_KEY, GROQ_MODEL

        start = time.time()

        # Auto-detect companies from query if not explicitly provided
        if auto_detect_companies and not companies:
            companies = extract_companies_from_query(question) or None

        # Build metadata filter
        metadata_filter = build_filter(
            companies=companies,
            years=years,
            doc_types=doc_types,
        )

        logger.info(
            "Starting hybrid retrieval",
            extra={
                "question": question[:100],
                "companies": companies,
                "metadata_filter": bool(metadata_filter)
            }
        )

        # ── Vector Search ──────────────────────────────────────────────────
        query_embedding = self.embedder.embed_query(question)
        vector_results = self.store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter
        )

        # ── BM25 Search ────────────────────────────────────────────────────
        bm25_results = self._bm25_search(
            query=question,
            top_k=top_k,
            metadata_filter=metadata_filter
        )

        # ── RRF Fusion ─────────────────────────────────────────────────────
        fused_results = reciprocal_rank_fusion([vector_results, bm25_results])
        top_fused = fused_results[:top_k]

        logger.info(
            "Fusion complete",
            extra={
                "vector_results": len(vector_results),
                "bm25_results": len(bm25_results),
                "fused_results": len(fused_results)
            }
        )

        # ── Cohere Reranker ────────────────────────────────────────────────
        if top_fused:
            reranked = self.reranker.rerank(
                query=question,
                chunks=top_fused,
                top_n=final_top_n
            )
        else:
            reranked = []

        if not reranked:
            return {
                "answer": "No relevant evidence found. Try broader search terms or check that documents are indexed.",
                "citations": [], "chunks": [], "latency_ms": round((time.time()-start)*1000),
                "retrieval_debug": {"vector": 0, "bm25": 0, "fused": 0, "reranked": 0}
            }

        # ── LLM Answer Generation ──────────────────────────────────────────
        context = format_context(reranked)
        prompt = ANSWER_PROMPT.format(context=context, question=question)

        llm = Groq(api_key=GROQ_API_KEY)
        response = llm.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,
        )
        answer = response.choices[0].message.content

        latency_ms = round((time.time() - start) * 1000)

        return {
            "answer": answer,
            "citations": build_citations(reranked),
            "chunks": reranked,
            "latency_ms": latency_ms,
            "retrieval_debug": {
                "vector_results": len(vector_results),
                "bm25_results": len(bm25_results),
                "fused_results": len(fused_results),
                "reranked": len(reranked),
                "metadata_filter":