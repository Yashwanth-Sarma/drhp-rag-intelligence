"""
src/retrieval/reranker.py

Cohere reranker for Stage 2.

Why reranking?
    Vector search finds chunks that are semantically similar to the query.
    But "similar" doesn't always mean "most useful for answering this specific question."

    The reranker is a cross-encoder — it looks at the query AND each chunk
    together (not separately) and scores how well they answer the query.
    This is slower but much more accurate than vector similarity alone.

Pipeline position:
    Hybrid retrieval → top 20 chunks → Reranker → top 5 chunks → LLM

Why Cohere?
    Industry standard. Free tier (1000 calls/month) sufficient for development.
    rerank-english-v3.0 is strong on domain-specific text.

Inputs:  Query string + list of retrieved Documents (top 20)
Outputs: Reranked list of Documents (top N, default 5)
"""

import time
from typing import Optional
import cohere
from langchain_core.documents import Document

from src.configuration.config import COHERE_API_KEY, RERANK_TOP_N, RERANKER_MODEL
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import RerankerError

logger = get_logger(__name__)


class CohereReranker:
    """
    Wraps Cohere's rerank API with error handling and retry logic.
    """

    MAX_RETRIES = 3

    def __init__(self) -> None:
        if not COHERE_API_KEY:
            raise RerankerError("COHERE_API_KEY missing from environment.")
        self.client = cohere.Client(api_key=COHERE_API_KEY)
        logger.info("CohereReranker initialized", extra={"model": RERANKER_MODEL})

    @log_duration("rerank")
    def rerank(
        self,
        query: str,
        chunks: list[Document],
        top_n: int = RERANK_TOP_N
    ) -> list[Document]:
        """
        Rerank retrieved chunks by relevance to the query.

        Args:
            query:  The user's original question.
            chunks: Retrieved chunks from hybrid retrieval (typically 20).
            top_n:  How many to return after reranking (typically 5).

        Returns:
            Top N chunks reranked by relevance, most relevant first.
            Each chunk gets a "rerank_score" added to its metadata.

        Raises:
            RerankerError: If Cohere API fails after all retries.
        """
        if not chunks:
            return []

        # Cohere rerank accepts raw text strings
        documents_text = [chunk.page_content for chunk in chunks]

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.rerank(
                    model=RERANKER_MODEL,
                    query=query,
                    documents=documents_text,
                    top_n=min(top_n, len(chunks)),
                    return_documents=True,
                )
                break
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait = 3 * (attempt + 1)
                    logger.warning(
                        f"Reranker attempt {attempt+1} failed: {e}. Retrying in {wait}s"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Reranker failed after {self.MAX_RETRIES} attempts: {e}")
                    raise RerankerError(f"Reranking failed: {e}") from e

        # Map results back to original Document objects (preserving metadata)
        reranked_chunks = []
        for result in response.results:
            original_chunk = chunks[result.index]
            # Add rerank score to metadata
            updated_metadata = dict(original_chunk.metadata)
            updated_metadata["rerank_score"] = round(result.relevance_score, 4)
            reranked_chunks.append(
                Document(
                    page_content=original_chunk.page_content,
                    metadata=updated_metadata
                )
            )

        logger.info(
            "Reranking complete",
            extra={
                "input_chunks": len(chunks),
                "output_chunks": len(reranked_chunks),
                "top_score": reranked_chunks[0].metadata["rerank_score"] if reranked_chunks else None
            }
        )
        return reranked_chunks


if __name__ == "__main__":
    from langchain_core.documents import Document

    reranker = CohereReranker()
    test_chunks = [
        Document(
            page_content="Zomato reported revenue from operations of Rs. 7,079 crores in FY2023.",
            metadata={"company_name": "Zomato", "page_number": 45, "chunk_id": "z_p45_c1"}
        ),
        Document(
            page_content="The company faces risk from intense competition in the food delivery market.",
            metadata={"company_name": "Zomato", "page_number": 22, "chunk_id": "z_p22_c3"}
        ),
        Document(
            page_content="Paytm's gross merchandise value grew significantly in the period.",
            metadata={"company_name": "Paytm", "page_number": 88, "chunk_id": "p_p88_c2"}
        ),
    ]

    query = "What was Zomato's revenue?"
    results = reranker.rerank(query, test_chunks, top_n=2)

    print(f"Query: {query}")
    print(f"\nReranked results (top 2):")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Score: {doc.metadata['rerank_score']}")
        print(f"   Company: {doc.metadata['company_name']} | Page: {doc.metadata['page_number']}")
        print(f"   Text: {doc.page_content[:150]}")
    print("\nCohereReranker working correctly.")