"""
src/retrieval/hybrid_retriever.py

Stage 2 Hybrid Retriever.
Combines BM25 keyword search + vector semantic search + Cohere reranker.
Reads from whichever stage collection is specified via the stage parameter.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.configuration.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
    EMBEDDINGS_STAGE1_DIR,
    EMBEDDINGS_STAGE2_DIR,
    EMBEDDINGS_STAGE3_DIR,
    COLLECTION_STAGE1,
    COLLECTION_STAGE2,
    COLLECTION_STAGE3,
)
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.vector_store.chroma_store import ChromaStore
from src.retrieval.reranker import CohereReranker
from src.retrieval.metadata_filter import build_filter, extract_companies_from_query
from src.retrieval.base_retriever import format_context, build_citations, ANSWER_PROMPT
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import RetrievalError

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    results_list: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    Deduplicates by chunk_id. Returns merged list sorted by RRF score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Document] = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            chunk_map[chunk_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    merged = []
    for chunk_id in sorted_ids:
        doc = chunk_map[chunk_id]
        updated_meta = dict(doc.metadata)
        updated_meta["rrf_score"] = round(scores[chunk_id], 6)
        merged.append(Document(page_content=doc.page_content, metadata=updated_meta))

    return merged


class HybridRetriever:
    """
    Stage 2 retriever: BM25 + vector search + metadata filter + Cohere reranker.

    Args:
        stage: Which pipeline stage to use. Determines which ChromaDB
               collection and embeddings directory to read from.
               1 = Stage 1 naive embeddings
               2 = Stage 2 contextual embeddings (default)
               3 = Stage 3 ColPali embeddings
    """

    STAGE_MAP = {
        1: (EMBEDDINGS_STAGE1_DIR, COLLECTION_STAGE1),
        2: (EMBEDDINGS_STAGE2_DIR, COLLECTION_STAGE2),
        3: (EMBEDDINGS_STAGE3_DIR, COLLECTION_STAGE3),
    }

    def __init__(self, stage: int = 2) -> None:
        persist_dir, collection_name = self.STAGE_MAP.get(stage, self.STAGE_MAP[2])

        if not persist_dir.exists():
            logger.warning(
                f"Stage {stage} embeddings not found at {persist_dir}. "
                "Falling back to Stage 1."
            )
            persist_dir, collection_name = self.STAGE_MAP[1]

        self.store = ChromaStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        self.embedder = VoyageEmbedder()
        self.reranker = CohereReranker()
        self.stage = stage

        logger.info(
            "HybridRetriever initialized",
            extra={
                "stage": stage,
                "collection": collection_name,
                "chunks": self.store.collection.count(),
            },
        )

    def _build_bm25_index(
        self,
        metadata_filter: Optional[dict] = None,
    ) -> tuple[Optional[BM25Okapi], list[Document]]:
        """Build BM25 index from ChromaDB. Filtered by metadata if provided."""
        get_params: dict = {"include": ["documents", "metadatas"]}
        if metadata_filter:
            get_params["where"] = metadata_filter

        results = self.store.collection.get(**get_params)

        if not results["documents"]:
            logger.warning("No documents found for BM25 index.")
            return None, []

        chunks = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        tokenized = [doc.split() for doc in results["documents"]]
        index = BM25Okapi(tokenized)

        logger.info("BM25 index built", extra={"total_docs": len(chunks)})
        return index, chunks

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[dict] = None,
    ) -> list[Document]:
        """BM25 keyword search over the collection."""
        bm25, chunks = self._build_bm25_index(metadata_filter)
        if not bm25 or not chunks:
            return []

        tokenized_query = query.split()
        raw_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(raw_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if raw_scores[idx] > 0:
                doc = chunks[idx]
                updated_meta = dict(doc.metadata)
                updated_meta["bm25_score"] = round(float(raw_scores[idx]), 4)
                results.append(
                    Document(page_content=doc.page_content, metadata=updated_meta)
                )

        logger.info(
            "BM25 search complete",
            extra={
                "results": len(results),
                "top_score": (
                    results[0].metadata["bm25_score"] if results else 0
                ),
            },
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
        auto_detect_companies: bool = True,
    ) -> dict:
        """
        Full Stage 2 hybrid retrieval pipeline.

        Args:
            question:              User's natural language question.
            companies:             Filter to these companies e.g. ["Zomato"].
            years:                 Filter to these years e.g. ["2021"].
            doc_types:             Filter to these doc types e.g. ["DRHP"].
            top_k:                 Chunks per retrieval method before fusion.
            final_top_n:           Chunks after reranking — sent to LLM.
            auto_detect_companies: Detect company names from query text if True.

        Returns:
            dict: answer, citations, chunks, latency_ms, retrieval_debug
        """
        from groq import Groq

        start = time.time()

        if auto_detect_companies and not companies:
            detected = extract_companies_from_query(question)
            if detected:
                companies = detected

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
                "metadata_filter": bool(metadata_filter),
            },
        )

        # Vector search
        query_embedding = self.embedder.embed_query(question)
        vector_results = self.store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        # BM25 search
        bm25_results = self._bm25_search(
            query=question,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        # RRF fusion
        fused = reciprocal_rank_fusion([vector_results, bm25_results])
        top_fused = fused[:top_k]

        logger.info(
            "Fusion complete",
            extra={
                "vector": len(vector_results),
                "bm25": len(bm25_results),
                "fused": len(fused),
            },
        )

        # Rerank
        if top_fused:
            reranked = self.reranker.rerank(
                query=question,
                chunks=top_fused,
                top_n=final_top_n,
            )
        else:
            reranked = []

        if not reranked:
            return {
                "answer": "No relevant evidence found. Try broader search terms.",
                "citations": [],
                "chunks": [],
                "latency_ms": round((time.time() - start) * 1000),
                "retrieval_debug": {
                    "vector": 0,
                    "bm25": 0,
                    "fused": 0,
                    "reranked": 0,
                },
            }

        # LLM answer generation
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
                "fused_results": len(fused),
                "reranked": len(reranked),
                "metadata_filter": metadata_filter,
            },
        }


if __name__ == "__main__":
    retriever = HybridRetriever(stage=2)
    stats = retriever.store.get_collection_stats()
    print(f"Stage 2 collection: {stats['total_chunks']} chunks")
    print(f"Companies: {stats.get('companies_sampled', [])}")

    if stats["total_chunks"] > 0:
        result = retriever.query(
            question="What are the main risk factors for Zomato?",
            companies=["Zomato"],
        )
        print(f"\nAnswer ({result['latency_ms']}ms):")
        print(result["answer"][:500])
        print(f"\nDebug: {result['retrieval_debug']}")