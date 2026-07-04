"""
src/vector_store/chroma_store.py

ChromaDB vector store for FinSight.
Accepts persist_dir and collection_name so different pipeline stages
can use different collections without any monkey-patching.
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from src.configuration.config import (
    EMBEDDINGS_STAGE1_DIR,
    COLLECTION_STAGE1,
    RETRIEVAL_TOP_K,
)
from src.shared.logger import get_logger, log_duration
from src.shared.exceptions import RetrievalError

logger = get_logger(__name__)


class ChromaStore:
    """
    Manages a ChromaDB vector store collection for FinSight.

    Args:
        persist_dir:     Directory where ChromaDB stores data on disk.
                         Defaults to Stage 1 embeddings directory.
        collection_name: Name of the ChromaDB collection to use.
                         Defaults to Stage 1 collection name.

    Example — Stage 1:
        store = ChromaStore()

    Example — Stage 2:
        from src.configuration.config import EMBEDDINGS_STAGE2_DIR, COLLECTION_STAGE2
        store = ChromaStore(
            persist_dir=EMBEDDINGS_STAGE2_DIR,
            collection_name=COLLECTION_STAGE2
        )
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.persist_dir = persist_dir or EMBEDDINGS_STAGE1_DIR
        self.collection_name = collection_name or COLLECTION_STAGE1
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        count = self.collection.count()
        logger.info(
            "ChromaStore initialized",
            extra={
                "collection": self.collection_name,
                "persist_dir": str(self.persist_dir),
                "existing_chunks": count,
            },
        )

    @log_duration("add_chunks")
    def add_chunks(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add chunks and embeddings to the collection.
        Skips chunks whose chunk_id already exists.

        Returns:
            Number of chunks actually added.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must be the same length."
            )

        # Collect all existing IDs in one query
        existing_count = self.collection.count()
        existing_ids: set[str] = set()
        if existing_count > 0:
            result = self.collection.get(limit=existing_count, include=[])
            existing_ids = set(result["ids"])

        ids, docs, vecs, metas = [], [], [], []
        skipped = 0

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = f"chunk_{hash(chunk.page_content)}"

            if chunk_id in existing_ids:
                skipped += 1
                continue

            clean_meta = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in chunk.metadata.items()
            }

            ids.append(chunk_id)
            docs.append(chunk.page_content)
            vecs.append(embedding)
            metas.append(clean_meta)

        if not ids:
            logger.info(
                f"All {skipped} chunks already indexed — nothing added."
            )
            return 0

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                documents=docs[i : i + batch_size],
                embeddings=vecs[i : i + batch_size],
                metadatas=metas[i : i + batch_size],
            )

        added = len(ids)
        logger.info(
            "Chunks added to ChromaDB",
            extra={
                "added": added,
                "skipped_duplicates": skipped,
                "total_in_store": self.collection.count(),
            },
        )
        return added

    @log_duration("similarity_search")
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = RETRIEVAL_TOP_K,
        metadata_filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Search for chunks most similar to the query embedding.

        Args:
            query_embedding: Vector from VoyageEmbedder.embed_query()
            top_k:           Number of results to return.
            metadata_filter: ChromaDB where-filter dict. Examples:
                             {"company_name": "Zomato"}
                             {"$and": [{"company_name": "Zomato"}, {"year": "2021"}]}

        Returns:
            List of Documents ordered by similarity, most similar first.
        """
        try:
            n = min(top_k, self.collection.count())
            if n == 0:
                return []

            query_params: dict = {
                "query_embeddings": [query_embedding],
                "n_results": n,
                "include": ["documents", "metadatas", "distances"],
            }
            if metadata_filter:
                query_params["where"] = metadata_filter

            results = self.collection.query(**query_params)

        except Exception as e:
            raise RetrievalError(f"ChromaDB query failed: {e}") from e

        retrieved = []
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1 - (distance / 2)
            metadata["similarity_score"] = round(similarity, 4)
            retrieved.append(
                Document(page_content=doc_text, metadata=metadata)
            )

        logger.info(
            "Similarity search complete",
            extra={
                "top_k": top_k,
                "returned": len(retrieved),
                "filter_applied": bool(metadata_filter),
                "top_score": (
                    retrieved[0].metadata.get("similarity_score")
                    if retrieved
                    else None
                ),
            },
        )
        return retrieved

    def get_collection_stats(self) -> dict:
        """Returns statistics about the current collection."""
        count = self.collection.count()
        if count == 0:
            return {
                "total_chunks": 0,
                "companies_sampled": [],
                "doc_types_sampled": [],
            }

        result = self.collection.get(include=["metadatas"])
        companies = sorted(
            set(m.get("company_name", "Unknown") for m in result["metadatas"])
        )
        doc_types = sorted(
            set(m.get("doc_type", "Unknown") for m in result["metadatas"])
        )

        return {
            "total_chunks": count,
            "companies_sampled": companies,
            "doc_types_sampled": doc_types,
        }

    def reset_collection(self) -> None:
        """Delete and recreate the collection. USE WITH CAUTION."""
        logger.warning(
            f"Resetting collection '{self.collection_name}' — all data will be lost."
        )
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
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
            print(
                f"\n--- Result {i} "
                f"(score: {doc.metadata['similarity_score']}) ---"
            )
            print(
                f"Company: {doc.metadata.get('company_name')} "
                f"| Page: {doc.metadata.get('page_number')}"
            )
            print(f"Text: {doc.page_content[:200]}...")
    else:
        print("Collection empty — run the indexing pipeline first.")