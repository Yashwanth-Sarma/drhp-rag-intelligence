"""
src/embeddings/voyage_embedder.py

Embeds text chunks using Voyage AI's voyage-finance-2 model.

Why Voyage AI voyage-finance-2?
- Domain-trained on financial text: understands EBITDA, GMV, DRHP, ESOP
- 1024-dimensional embeddings — good precision/storage balance
- 50M free tokens — entire project corpus fits in free tier
- Outperforms OpenAI text-embedding-3-small on financial terminology retrieval

Why not run embeddings locally?
- Voyage AI's financial domain training cannot be replicated locally without GPU training
- API call is fast (~80-120ms) and free for our volume
- Keeps laptop free for other tasks during bulk indexing

Inputs:  List of LangChain Document objects (chunks with metadata)
Outputs: List of (embedding_vector, metadata) pairs, or ChromaDB-ready format
"""

import time
from typing import Optional
import voyageai

from src.configuration.config import VOYAGE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import EmbeddingError

logger = get_logger(__name__)


class VoyageEmbedder:
    """
    Wraps Voyage AI embedding API with retry logic, batching, and error handling.
    
    Voyage AI has a rate limit on the free tier. This class:
    - Batches chunks to minimize API calls
    - Retries on transient failures with exponential backoff
    - Tracks total tokens used (useful for monitoring free tier usage)
    """

    # Voyage AI free tier: 50M tokens. voyage-finance-2 max input: 32K tokens per request
    MAX_BATCH_SIZE = 128    # chunks per API call — keeps request size manageable
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2

    def __init__(self) -> None:
        if not VOYAGE_API_KEY:
            raise EmbeddingError(
                "VOYAGE_API_KEY not found in environment. "
                "Check your .env file."
            )
        self.client = voyageai.Client(api_key=VOYAGE_API_KEY)
        self.total_tokens_used = 0
        logger.info(
            "VoyageEmbedder initialized",
            extra={"model": EMBEDDING_MODEL, "dimension": EMBEDDING_DIMENSION}
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.
        Handles batching automatically — pass any number of texts.
        
        Args:
            texts: List of strings to embed.
        
        Returns:
            List of embedding vectors (each is a list of 1024 floats).
        
        Raises:
            EmbeddingError: If embedding fails after all retries.
        
        Example:
            embedder = VoyageEmbedder()
            vectors = embedder.embed_texts(["Zomato revenue grew 23%", "Risk factors include..."])
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE

        logger.info(
            f"Starting embedding",
            extra={"total_texts": len(texts), "total_batches": total_batches}
        )

        for batch_idx in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[batch_idx: batch_idx + self.MAX_BATCH_SIZE]
            batch_num = (batch_idx // self.MAX_BATCH_SIZE) + 1

            embeddings = self._embed_batch_with_retry(batch, batch_num, total_batches)
            all_embeddings.extend(embeddings)
            time.sleep(0.1)  # Voyage AI recommendation — small pause between batches

        logger.info(
            f"Embedding complete",
            extra={
                "total_embedded": len(all_embeddings),
                "total_tokens_used_session": self.total_tokens_used
            }
        )
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.
        Uses 'query' input type which Voyage AI optimizes differently from document embedding.
        
        Args:
            query: The user's search query.
        
        Returns:
            Single embedding vector (list of 1024 floats).
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self.client.embed(
                    [query],
                    model=EMBEDDING_MODEL,
                    input_type="query"
                )
                return result.embeddings[0]
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_DELAY_SECONDS * (2 ** attempt)
                    logger.warning(f"Query embedding attempt {attempt+1} failed: {e}. Retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise EmbeddingError(f"Query embedding failed after {self.MAX_RETRIES} attempts: {e}") from e

    def _embed_batch_with_retry(
        self,
        batch: list[str],
        batch_num: int,
        total_batches: int
    ) -> list[list[float]]:
        """Internal: embed one batch with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self.client.embed(
                    batch,
                    model=EMBEDDING_MODEL,
                    input_type="document"
                )
                # Track token usage if available
                if hasattr(result, "total_tokens"):
                    self.total_tokens_used += result.total_tokens

                logger.info(
                    f"Batch {batch_num}/{total_batches} embedded",
                    extra={"batch_size": len(batch), "attempt": attempt + 1}
                )
                return result.embeddings

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_DELAY_SECONDS * (2 ** attempt)
                    logger.warning(
                        f"Batch {batch_num} attempt {attempt+1} failed: {e}. Retrying in {wait}s"
                    )
                    time.sleep(wait)
                else:
                    raise EmbeddingError(
                        f"Batch {batch_num} failed after {self.MAX_RETRIES} attempts: {e}"
                    ) from e

    def get_langchain_embeddings(self):
        """
        Returns a LangChain-compatible embeddings object.
        Use this when LangChain components (like ChromaDB) need an embeddings object.
        
        Returns:
            LangChain Embeddings wrapper around Voyage AI.
        """
        from langchain_community.embeddings import VoyageAIEmbeddings
        return VoyageAIEmbeddings(
            voyage_api_key=VOYAGE_API_KEY,
            model=EMBEDDING_MODEL
        )


if __name__ == "__main__":
    embedder = VoyageEmbedder()
    test_texts = [
        "Zomato Limited reported revenue from operations of Rs. 7,079 crores in FY2023.",
        "Risk factors include intense competition from Swiggy and other food delivery platforms.",
        "The company plans to use IPO proceeds for expanding the Hyperpure B2B business.",
    ]
    vectors = embedder.embed_texts(test_texts)
    print(f"Embedded {len(vectors)} texts")
    print(f"Vector dimension: {len(vectors[0])}")
    print(f"Expected dimension: {EMBEDDING_DIMENSION}")
    assert len(vectors[0]) == EMBEDDING_DIMENSION, "Dimension mismatch!"

    query_vec = embedder.embed_query("What is Zomato's revenue?")
    print(f"Query vector dimension: {len(query_vec)}")
    print("VoyageEmbedder working correctly.")