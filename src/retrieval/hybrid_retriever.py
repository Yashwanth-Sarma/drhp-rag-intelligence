"""
src/retrieval/hybrid_retriever.py

Stage 2 Hybrid Retriever.
BM25 + Dense Vector + Metadata Filter + Cohere Reranker.
Integrates RetrievalQualityAssessor for mathematical gap detection.
Routes reasoning to Gemini Flash/Pro via ProviderRouter.
"""

import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.configuration.config import (
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
from src.retrieval.quality_assessor import RetrievalQualityAssessor, compute_difficulty_score
from src.retrieval.base_retriever import format_context, build_citations, ANSWER_PROMPT
from src.llm.provider_router import get_router, TaskType
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import RetrievalError

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    results_list: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    Deduplicates by chunk_id. Higher RRF score = appeared higher in more lists.
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
    Stage 2/3 retriever with BM25, dense vector, reranking,
    mathematical quality assessment, and adaptive Gemini Flash/Pro routing.

    Args:
        stage: 1 = naive embeddings, 2 = contextual embeddings (default), 3 = ColPali
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
            stage = 1

        self.store = ChromaStore(persist_dir=persist_dir, collection_name=collection_name)
        self.embedder = VoyageEmbedder()
        self.reranker = CohereReranker()
        self.quality_assessor = RetrievalQualityAssessor()
        self.router = get_router()
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
        """Build BM25 index from ChromaDB, optionally filtered by metadata."""
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
        bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built", extra={"total_docs": len(chunks)})
        return bm25, chunks

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
                results.append(Document(page_content=doc.page_content, metadata=updated_meta))

        return results

    def _llm_gap_detection(
        self,
        question: str,
        quality_summary: str,
        missing_entities: list[str],
    ) -> tuple[bool, str]:
        """
        Called only when mathematical quality assessment flags LOW confidence.
        Asks the LLM whether to retrieve again and with what refined query.

        Returns:
            (should_retry: bool, refined_query: str)
        """
        gap_prompt = f"""You are evaluating whether retrieved evidence is sufficient to answer a financial question.

ORIGINAL QUESTION: {question}

RETRIEVAL QUALITY SUMMARY: {quality_summary}

MISSING ENTITIES (not found in retrieved chunks): {missing_entities}

Based on the above, answer TWO things:
1. Is the retrieved evidence sufficient? (yes/no)
2. If no, what specific refined search query would find the missing information? (one sentence)

Reply in this exact format:
SUFFICIENT: yes/no
REFINED_QUERY: [refined query or "none"]"""

        try:
            response = self.router.generate(
                task=TaskType.GAP_DETECTION,
                prompt=gap_prompt,
                temperature=0.0,
                max_tokens=100,
            )
            lines = response.strip().split("\n")
            sufficient = "yes" in lines[0].lower() if lines else True
            refined = lines[1].replace("REFINED_QUERY:", "").strip() if len(lines) > 1 else "none"
            return not sufficient, refined if refined != "none" else question
        except Exception as e:
            logger.warning(f"Gap detection LLM call failed: {e}. Proceeding with original results.")
            return False, question

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
        max_retrieval_cycles: int = 2,
    ) -> dict:
        """
        Full Stage 2 hybrid retrieval pipeline with quality assessment.

        Pipeline:
            1. Auto-detect entities from query
            2. Build metadata filter
            3. Dense vector search + BM25 keyword search in parallel
            4. RRF fusion
            5. Cohere reranking
            6. Mathematical quality assessment
            7. If LOW confidence: LLM gap detector → optional second retrieval
            8. Compute difficulty score → route to Flash or Pro
            9. Generate answer with citations

        Args:
            question:              User query.
            companies:             Override company filter.
            years:                 Override year filter.
            doc_types:             Override doc type filter.
            top_k:                 Chunks per retrieval before reranking.
            final_top_n:           Chunks after reranking sent to LLM.
            auto_detect_companies: Detect companies from query text.
            max_retrieval_cycles:  Maximum retrieval attempts (1 or 2).

        Returns:
            dict: answer, citations, chunks, latency_ms, quality_report,
                  provider_used, retrieval_debug
        """
        start = time.time()

        # Step 1: Entity detection
        if auto_detect_companies and not companies:
            detected = extract_companies_from_query(question)
            if detected:
                companies = detected

        # Step 2: Metadata filter
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
                "stage": self.stage,
            },
        )

        # Steps 3-6: Retrieval + quality assessment loop
        current_question = question
        final_reranked = []
        quality_report = None

        for cycle in range(max_retrieval_cycles):
            # Dense vector search
            query_embedding = self.embedder.embed_query(current_question)
            vector_results = self.store.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )

            # BM25 keyword search
            bm25_results = self._bm25_search(
                query=current_question,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )

            # RRF fusion
            fused = reciprocal_rank_fusion([vector_results, bm25_results])
            top_fused = fused[:top_k]

            # Cohere reranking
            if top_fused:
                reranked = self.reranker.rerank(
                    query=current_question,
                    chunks=top_fused,
                    top_n=final_top_n,
                )
            else:
                reranked = []

            # Mathematical quality assessment
            quality_report = self.quality_assessor.assess(
                vector_results=vector_results,
                bm25_results=bm25_results,
                fused_reranked=reranked,
                queried_entities=companies or [],
            )

            logger.info(f"Cycle {cycle + 1} quality: {quality_report.summary}")

            # If high confidence or last cycle, use these results
            if quality_report.is_high_confidence or cycle == max_retrieval_cycles - 1:
                final_reranked = reranked
                break

            # Low confidence: ask LLM gap detector if worth retrying
            should_retry, refined_query = self._llm_gap_detection(
                question=current_question,
                quality_summary=quality_report.summary,
                missing_entities=quality_report.missing_entities,
            )

            if should_retry and refined_query != current_question:
                logger.info(f"Gap detector triggered retry with: '{refined_query[:80]}'")
                current_question = refined_query
                continue
            else:
                final_reranked = reranked
                break

        if not final_reranked:
            return {
                "answer": "No relevant evidence found. Try different search terms.",
                "citations": [],
                "chunks": [],
                "latency_ms": round((time.time() - start) * 1000),
                "quality_report": quality_report,
                "provider_used": "none",
                "retrieval_debug": {"vector": 0, "bm25": 0, "reranked": 0},
            }

        # Step 7: Compute difficulty → decide Flash vs Pro
        is_cross_doc = len(set(
            c.metadata.get("company_name", "") for c in final_reranked
        )) > 1
        has_table = any(
            kw in question.lower()
            for kw in ["table", "breakdown", "quarter", "margin", "ratio", "trend"]
        )
        has_contradiction = (
            quality_report is not None
            and quality_report.score_variance > 0.06
        )

        difficulty = compute_difficulty_score(
            query=question,
            detected_entities=companies or [],
            is_cross_document=is_cross_doc,
            has_table_intent=has_table,
            contradiction_detected=has_contradiction,
        )

        use_hard_query = difficulty >= 0.4
        logger.info(
            f"Difficulty: {difficulty:.2f} → {'Gemini Pro' if use_hard_query else 'Gemini Flash'}"
        )

        # Step 8: Generate answer
        context = format_context(final_reranked)
        prompt = ANSWER_PROMPT.format(context=context, question=question)

        try:
            answer = self.router.generate(
                task=TaskType.REASONING,
                prompt=prompt,
                temperature=0.1,
                max_tokens=1500,
                hard_query=use_hard_query,
            )
            provider_used = "gemini_pro" if use_hard_query else "gemini_flash"
        except Exception as e:
            raise RetrievalError(f"Answer generation failed: {e}") from e

        latency_ms = round((time.time() - start) * 1000)

        return {
            "answer": answer,
            "citations": build_citations(final_reranked),
            "chunks": final_reranked,
            "latency_ms": latency_ms,
            "quality_report": quality_report,
            "provider_used": provider_used,
            "difficulty_score": difficulty,
            "retrieval_debug": {
                "vector_results": len(vector_results),
                "bm25_results": len(bm25_results),
                "fused_results": len(fused),
                "reranked": len(final_reranked),
                "metadata_filter": metadata_filter,
                "quality_confidence": (
                    quality_report.overall_confidence if quality_report else 0.0
                ),
                "retrieval_cycles": cycle + 1,
            },
        }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    retriever = HybridRetriever(stage=2)
    stats = retriever.store.get_collection_stats()
    print(f"Stage 2 collection: {stats['total_chunks']} chunks")

    if stats["total_chunks"] > 0:
        result = retriever.query(
            question="What are the main risk factors for Zomato?",
            companies=["Zomato"],
        )
        print(f"\nAnswer ({result['latency_ms']}ms via {result['provider_used']}):")
        print(result["answer"][:500])
        print(f"\nQuality: {result.get('quality_report', {}).summary if result.get('quality_report') else 'N/A'}")
        print(f"Citations: {len(result['citations'])}")
        print(f"Debug: {result['retrieval_debug']}")
    else:
        print("Stage 2 not indexed yet. Run: python scripts/index_documents_stage2.py")