"""
Central configuration for the DRHP RAG pipeline.
All constants, model names, and paths live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ── API Keys ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ── LangSmith Tracing ──────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "drhp-rag")

# ── Models ─────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "voyage-finance-2"
RERANKER_MODEL = "rerank-english-v3.0"
EMBEDDING_DIMENSION = 1024

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 50        # overlap between chunks
RETRIEVAL_TOP_K = 20      # how many chunks to retrieve before reranking
RERANK_TOP_N = 5          # how many chunks after reranking

# ── Companies in our corpus ────────────────────────────────────────────────
COMPANY_METADATA = {
    "zomato_drhp_2021": {
        "company_name": "Zomato",
        "year": "2021",
        "doc_type": "DRHP",
        "sector": "Food-tech"
    },
    "ola_electric_drhp_2024": {
        "company_name": "Ola Electric",
        "year": "2024",
        "doc_type": "DRHP",
        "sector": "EV Manufacturing"
    },
    "swiggy_drhp_2024": {
        "company_name": "Swiggy",
        "year": "2024",
        "doc_type": "DRHP",
        "sector": "Food-tech"
    },
    "paytm_drhp_2021": {
        "company_name": "Paytm",
        "year": "2021",
        "doc_type": "DRHP",
        "sector": "Fintech"
    },
}

# ── Validation ─────────────────────────────────────────────────────────────
def validate_env() -> None:
    """Raise early if critical API keys are missing."""
    required = {
        "GROQ_API_KEY": GROQ_API_KEY,
        "VOYAGE_API_KEY": VOYAGE_API_KEY,
        "COHERE_API_KEY": COHERE_API_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            "Check your .env file."
        )

if __name__ == "__main__":
    validate_env()
    print("✓ All required environment variables present")
    print(f"✓ Base directory: {BASE_DIR}")
    print(f"✓ Raw PDFs directory: {RAW_PDFS_DIR}")