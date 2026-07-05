"""
src/retrieval/colpali_retriever.py

Stage 3 ColPali retriever — queries the pre-built vision index
created on Google Colab and downloaded locally.

No GPU needed at query time — only indexing needs GPU.
"""

from pathlib import Path
from typing import Optional

from src.configuration.config import DATA_DIR
from src.shared.logger import get_logger

logger = get_logger(__name__)

COLPALI_INDEX_DIR = DATA_DIR / "embeddings_stage3" / "colpali_index"

COLPALI_MODEL = "vidore/colqwen2-v1.0"
class ColPaliRetriever:
    """
    Wraps the byaldi ColPali index for table-heavy financial document queries.
    Requires the index to be built on Colab and downloaded to COLPALI_INDEX_DIR.
    """

    def __init__(self) -> None:
        try:
            from byaldi import RAGMultiModalModel
        except ImportError:
            raise ImportError(
                "byaldi not installed. Run: pip install byaldi pdf2image"
            )

        if not COLPALI_INDEX_DIR.exists():
            logger.warning(
                f"ColPali index not found at {COLPALI_INDEX_DIR}. "
                "Build it on Google Colab first and download."
            )
            self.model = None
            return

        self.model = RAGMultiModalModel.from_index(
            str(COLPALI_INDEX_DIR.name),
            index_root=str(COLPALI_INDEX_DIR.parent),
        )
        logger.info("ColPaliRetriever initialized", extra={"index_dir": str(COLPALI_INDEX_DIR)})

    def is_available(self) -> bool:
        return self.model is not None

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search the ColPali index for the most visually relevant pages.

        Returns:
            List of dicts: {doc_id, page_num, score}
        """
        if not self.is_available():
            return []

        results = self.model.search(query, k=top_k)
        return [
            {"doc_id": r.doc_id, "page_num": r.page_num, "score": r.score}
            for r in results
        ]


if __name__ == "__main__":
    retriever = ColPaliRetriever()
    if retriever.is_available():
        results = retriever.search("What was the quarterly revenue breakdown?")
        for r in results:
            print(r)
    else:
        print("ColPali index not found. Build it on Colab first.")