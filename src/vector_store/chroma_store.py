"""
src/vector_store/chroma_store.py

ChromaDB vector store for FinSight.

Why ChromaDB?
- Runs locally with zero infrastructure setup
- Persistent — embeddings survive restarts
- Supports metadata filtering natively (filter by company_name, year, doc_type)
- Free, open-source, actively maintained
- Easy migration path to Qdrant for production

This module handles:
- Adding chunks + embeddings to the store
- Semantic similarity search
- Metadata-filtered search (most important for financial document retrieval)
- Collection management (create, reset, inspect)

Inputs:  Chunks (LangChain Documents) + embeddings from VoyageEmbedder
Outputs: Retrieved chunks with similarity scores for any query
"""

from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
from langchain.schema import Document

from src.configuration.config import EMBEDDINGS_DIR, RETRIEVAL_TOP_K
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import RetrievalError

logger = get_logger(__name__)

COLLECTION_NAME = "finsight_documents"


class ChromaStore:
    """
    Manages the ChromaDB vector store for FinSight.
    
    Collection naming: one collection for all documents, differentiated by metadata.
    Why one collection? Metadata filtering within one collection is faster and simpler
    than managing 20+ collections (one per company). Filtering by company_name at query
    time achieves the same result.
    """

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        self.persist_dir = persist_dir or EMBEDDINGS_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}   # cosine similarity for text embeddings
        )
        count = self.collection.count()
        logger.info(
            "ChromaStore initialized",
            extra={"persist_dir": str(self.persist_dir), "existing_chunks": count}
        )

    @log_duration("add_chunks")
    def add_chunks(
        self,
        chunks: list[Document],
        embeddings: list[list[float]]
    ) -> int:
        """
        Add chunks and their embeddings to the vector store.
        Skips duplicates using chunk_id metadata.
        
        Args:
            chunks:     List of LangChain Documents with metadata.
            embeddings: Corresponding embedding vectors (same length as chunks).
        
        Returns:
            Number of chunks actually added (excluding duplicates).
        
        Raises:
            ValueError: If chunks and embeddings lengths don't match.
            RetrievalError: If ChromaDB operation fails.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be same length."
            )

        # Build unique IDs, documents, embeddings, metadatas for ChromaDB
        ids, docs, vecs, metas = [], [], [], []
        skipped = 0

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.metadata.get("chunk_id")
            if not chunk_id:
                logger.warning(f"Chunk missing chunk_id — generating fallback ID")
                chunk_id = f"chunk_{hash(chunk.page_content)}"

            # Check for existing chunk — skip if already indexed
            try:
                existing = self.collection.get(ids=[chunk_id])
                if existing["ids"]:
                    skipped += 1
                    continue
            except Exception:
                pass  # Not found — proceed to add

            # ChromaDB metadata values must be str, int, float, or bool
            clean_meta = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in chunk.metadata.items()
            }

            ids.append(chunk_id)
            docs.append(chunk.page_content)
            vecs.append(embedding)
            metas.append(clean_meta)

        if not ids:
            logger.info(f"All {skipped} chunks already indexed — nothing added.")
            return 0

        # ChromaDB add in batches of 1000 (API limit)
        BATCH_SIZE = 500
        for i in range(0, len(ids), BATCH_SIZE):
            self.collection.add(
                ids=ids[i:i+BATCH_SIZE],
                documents=docs[i:i+BATCH_SIZE],
                embeddings=vecs[i:i+BATCH_SIZE],
                metadatas=metas[i:i+BATCH_SIZE]
            )

        added = len(ids)
        logger.info(
            "Chunks added to ChromaDB",
            extra={"added": added, "skipped_duplicates": skipped, "total_in_store": self.collection.count()}
        )
        return added

    @log_duration("similarity_search")
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = RETRIEVAL_TOP_K,
        metadata_filter: Optional[dict] = None
    ) -> list[Document]:
        """
        Search the vector store for chunks most similar to the query.
        
        Args:
            query_embedding: Vector from VoyageEmbedder.embed_query()
            top_k:           Number of results to return.
            metadata_filter: ChromaDB where filter. Examples:
                             {"company_name": "Zomato"}
                             {"$and": [{"company_name": "Zomato"}, {"year": "2021"}]}
                             {"doc_type": {"$in": ["DRHP", "Annual_Report"]}}
        
        Returns:
            List of LangChain Documents, ordered by similarity (most similar first).
            Each document has a "similarity_score" key added to its metadata.
        
        Example:
            results = store.similarity_search(
                query_embedding=embedder.embed_query("Zomato revenue FY2021"),
                top_k=10,
                metadata_filter={"company_name": "Zomato"}
            )
        """
        try:
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, self.collection.count()),
                "include": ["documents", "metadatas", "distances"]
            }
            if metadata_filter:
                query_params["where"] = metadata_filter

            results = self.collection.query(**query_params)

        except Exception as e:
            raise RetrievalError(f"ChromaDB query failed: {e}") from e

        # Convert ChromaDB result format to LangChain Documents
        retrieved = []
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 = identical, -1 = opposite
            similarity = 1 - (distance / 2)
            metadata["similarity_score"] = round(similarity, 4)
            retrieved.append(Document(page_content=doc_text, metadata=metadata))

        logger.info(
            "Similarity search complete",
            extra={
                "query_top_k": top_k,
                "returned": len(retrieved),
                "filter_applied": bool(metadata_filter),
                "top_score": retrieved[0].metadata.get("similarity_score") if retrieved else None
            }
        )
        return retrieved

    def get_collection_stats(self) -> dict:
        """Returns statistics about the current collection."""
        count = self.collection.count()
        if count == 0:
            return {"total_chunks": 0, "companies": [], "doc_types": []}

        # Sample first 1000 to get stats (avoid loading entire collection)
        sample = self.collection.get(limit=min(1000, count), include=["metadatas"])
        companies = list(set(m.get("company_name", "Unknown") for m in sample["metadatas"]))
        doc_types = list(set(m.get("doc_type", "Unknown") for m in sample["metadatas"]))

        return {
            "total_chunks": count,
            "companies_sampled": companies,
            "doc_types_sampled": doc_types,
        }

    def reset_collection(self) -> None:
        """
        Deletes and recreates the collection. USE WITH CAUTION.
        Only for development — resets all indexed data.
        """
        logger.warning("Resetting ChromaDB collection — all data will be lost.")
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection reset complete.")


if __name__ == "__main__":
    from src.embeddings.voyage_embedder import VoyageEmbedder

    store = ChromaStore()
    stats = store.get_collection_stats()
    print(f"Collection stats: {stats}")

    embedder = VoyageEmbedder()
    test_query = "What are the risk factors for Zomato?"
    query_vec = embedder.embed_query(test_query)

    if stats["total_chunks"] > 0:
        results = store.similarity_search(query_vec, top_k=3)
        print(f"\nTop 3 results for: '{test_query}'")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {doc.metadata['similarity_score']}) ---")
            print(f"Company: {doc.metadata.get('company_name')} | Page: {doc.metadata.get('page_number')}")
            print(f"Text: {doc.page_content[:200]}...")
    else:
        print("Collection empty — run the indexing pipeline first.")