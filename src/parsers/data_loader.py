"""
data_loader.py

Loads PDF documents, extracts text page by page using PyMuPDF,
splits into chunks, and attaches rich metadata to every chunk.

Why PyMuPDF over pypdf?
- Faster on large financial PDFs
- Better handling of complex layouts
- More reliable text extraction order

Why 512 token chunk size?
- Large enough to preserve context within a section
- Small enough to be precise during retrieval
- 50 token overlap prevents losing information at chunk boundaries
"""

import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.configuration.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COMPANY_METADATA,
    RAW_PDFS_DIR,
    validate_env,
)

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def auto_detect_metadata(pdf_path: Path, sample_text: str) -> dict:
    """
    Auto-detect company name, document type, and year from PDF filename
    and first few pages of content. Falls back gracefully if detection fails.
    
    This eliminates the need to rename PDFs to match COMPANY_METADATA keys.
    Works on any DRHP or annual report regardless of how it was downloaded.
    """
    filename = pdf_path.stem.lower()
    text_lower = sample_text.lower()[:3000]  # first 3000 chars sufficient
    
    # ── Company detection ──────────────────────────────────────────────────
    company_patterns = {
        "Zomato": ["zomato"],
        "Paytm": ["paytm", "one97 communications", "one 97"],
        "Ola Electric": ["ola electric", "ola mobility"],
        "Swiggy": ["swiggy", "bundl technologies"],
        "Nykaa": ["nykaa", "fsh limited", "fsn e-commerce"],
        "Delhivery": ["delhivery"],
        "PolicyBazaar": ["policybazaar", "pb fintech"],
        "CarTrade": ["cartrade"],
        "Freshworks": ["freshworks"],
        "Nazara": ["nazara"],
        "Mobikwik": ["mobikwik"],
        "Boat": ["imagine marketing", "boat lifestyle"],
    }
    company_name = "Unknown"
    for name, keywords in company_patterns.items():
        if any(k in filename or k in text_lower for k in keywords):
            company_name = name
            break
    
    # ── Document type detection ────────────────────────────────────────────
    doc_type = "Unknown"
    if any(k in text_lower[:500] for k in ["draft red herring prospectus", "drhp"]):
        doc_type = "DRHP"
    elif any(k in text_lower[:500] for k in ["red herring prospectus", "rhp"]):
        doc_type = "RHP"
    elif any(k in text_lower[:500] for k in ["annual report", "annual accounts"]):
        doc_type = "Annual_Report"
    elif any(k in text_lower[:500] for k in ["quarterly results", "q1", "q2", "q3", "q4"]):
        doc_type = "Quarterly_Report"
    elif any(k in text_lower[:500] for k in ["earnings call", "conference call", "transcript"]):
        doc_type = "Earnings_Transcript"
    
    # ── Year detection ─────────────────────────────────────────────────────
    import re
    year = "Unknown"
    # Look for financial year pattern first (FY2024, FY 2024, F.Y. 2024)
    fy_match = re.search(r'f\.?y\.?\s*(\d{4})', text_lower)
    if fy_match:
        year = f"FY{fy_match.group(1)}"
    else:
        # Fall back to calendar year in filename or text
        year_match = re.search(r'20(1[5-9]|2[0-9])', filename + " " + text_lower[:1000])
        if year_match:
            year = year_match.group(0)
    
    # ── Sector detection ──────────────────────────────────────────────────
    sector_patterns = {
        "Food-tech": ["food delivery", "restaurant", "zomato", "swiggy"],
        "Fintech": ["payments", "financial services", "paytm", "lending", "insurance"],
        "EV Manufacturing": ["electric vehicle", "ev", "ola electric"],
        "E-commerce": ["e-commerce", "online retail", "marketplace"],
        "Logistics": ["logistics", "supply chain", "delhivery"],
        "Healthcare": ["healthcare", "pharmaceutical", "hospital"],
        "EdTech": ["education", "edtech", "learning"],
        "SaaS": ["software", "saas", "cloud"],
    }
    sector = "Technology"  # default
    for sec, keywords in sector_patterns.items():
        if any(k in text_lower for k in keywords):
            sector = sec
            break
    
    detected = {
        "company_name": company_name,
        "doc_type": doc_type,
        "year": year,
        "sector": sector,
    }
    
    if company_name == "Unknown":
        logger.warning(
            f"Could not detect company for {pdf_path.name}. "
            f"Add it to company_patterns in auto_detect_metadata()."
        )
    else:
        logger.info(
            f"Auto-detected metadata for {pdf_path.name}",
            extra=detected
        )
    
    return detected
def detect_section(text: str) -> str:
    """
    Heuristic: detect which section of a DRHP a page belongs to
    based on common DRHP headings.
    Returns a section label string.
    """
    text_upper = text.upper()
    section_keywords = {
        "RISK FACTORS": "Risk Factors",
        "BUSINESS OVERVIEW": "Business Overview",
        "FINANCIAL STATEMENTS": "Financial Statements",
        "OBJECTS OF THE OFFER": "Objects of the Offer",
        "CAPITAL STRUCTURE": "Capital Structure",
        "LEGAL PROCEEDINGS": "Legal Proceedings",
        "RELATED PARTY": "Related Party Transactions",
        "MANAGEMENT": "Management",
        "INDUSTRY OVERVIEW": "Industry Overview",
        "SUMMARY": "Summary",
        "DEFINITIONS": "Definitions",
    }
    for keyword, label in section_keywords.items():
        if keyword in text_upper:
            return label
    return "General"


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove excessive whitespace
    - Remove page headers/footers (short repeated lines)
    - Preserve paragraph breaks
    """
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove lines that are just page numbers or very short (likely headers/footers)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just numbers (page numbers) or very short
        if len(stripped) > 3 and not re.match(r'^\d+$', stripped):
            cleaned_lines.append(stripped)
    return '\n'.join(cleaned_lines).strip()


def load_pdf(
    pdf_path: Path,
    company_key: str,
    extra_metadata: Optional[dict] = None
) -> list[Document]:
    """
    Load a single PDF file and return a list of LangChain Documents.
    Each document represents one page with full metadata attached.

    Args:
        pdf_path: Path to the PDF file
        company_key: Key from COMPANY_METADATA dict (e.g. 'zomato_drhp_2021')
        extra_metadata: Any additional metadata to attach

    Returns:
        List of LangChain Document objects, one per page
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    # Get base metadata from config
    # Try config first, then auto-detect from content
    base_meta = COMPANY_METADATA.get(company_key, {})
    if not base_meta:
        logger.info(
            f"'{company_key}' not in COMPANY_METADATA — "
            "will auto-detect from PDF content."
        )

    logger.info(f"Loading PDF: {pdf_path.name} ({company_key})")

    documents = []
    try:
        pdf = fitz.open(str(pdf_path))
        # Auto-detect metadata if not in config
        if not base_meta:
            # Use first 5 pages for detection
            sample_pages = min(5, len(pdf))
            sample_text = " ".join(
                pdf[i].get_text("text") for i in range(sample_pages)
            )
            base_meta = auto_detect_metadata(pdf_path, sample_text)
        total_pages = len(pdf)
        logger.info(f"  Total pages: {total_pages}")

        for page_num in range(total_pages):
            page = pdf[page_num]
            raw_text = page.get_text("text")

            if not raw_text.strip():
                # Skip empty pages (often cover pages, blank separators)
                logger.debug(f"  Skipping empty page {page_num + 1}")
                continue

            cleaned = clean_text(raw_text)
            section = detect_section(cleaned)

            metadata = {
                # Core identification — these are used for metadata filtering
                "company_name": base_meta.get("company_name", "Unknown"),
                "year": base_meta.get("year", "Unknown"),
                "doc_type": base_meta.get("doc_type", "Unknown"),
                "sector": base_meta.get("sector", "Unknown"),
                "source_file": pdf_path.name,
                "company_key": company_key,
                # Location metadata — used for citation in answers
                "page_number": page_num + 1,
                "total_pages": total_pages,
                "section": section,
            }

            if extra_metadata:
                metadata.update(extra_metadata)

            documents.append(Document(page_content=cleaned, metadata=metadata))

        pdf.close()
        logger.info(f"  Loaded {len(documents)} non-empty pages")

    except Exception as e:
        logger.error(f"Failed to load PDF {pdf_path.name}: {e}")
        raise

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split page-level documents into smaller chunks for embedding.

    Why RecursiveCharacterTextSplitter?
    - Tries to split on paragraph breaks first, then sentences, then words
    - Preserves semantic units better than fixed-character splitting
    - Industry standard for document RAG

    Each chunk INHERITS all metadata from its parent page document.
    We also add chunk-level metadata: chunk_index, chunk_total.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,   # *4 because chunk_size is in chars, not tokens
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    for doc in documents:
        splits = splitter.split_documents([doc])
        # Add chunk position metadata to each split
        for i, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(splits)
            # Create a unique chunk ID for deduplication later
            chunk.metadata["chunk_id"] = (
                f"{doc.metadata['company_key']}_"
                f"p{doc.metadata['page_number']}_"
                f"c{i}"
            )
            all_chunks.append(chunk)

    return all_chunks


def load_all_pdfs() -> list[Document]:
    """
    Load all PDFs from the raw_pdfs directory.
    Matches filenames to company_keys using COMPANY_METADATA.

    Expected filename format: {company_key}.pdf
    Example: zomato_drhp_2021.pdf

    Returns all chunks across all documents.
    """
    all_chunks = []
    pdf_files = list(RAW_PDFS_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.warning(
            f"No PDF files found in {RAW_PDFS_DIR}. "
            "Download DRHPs and place them there first."
        )
        return []

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    for pdf_path in pdf_files:
        # Derive company_key from filename (without .pdf extension)
        company_key = pdf_path.stem

        try:
            pages = load_pdf(pdf_path, company_key)
            chunks = chunk_documents(pages)
            all_chunks.extend(chunks)
            logger.info(
                f"  ✓ {pdf_path.name}: "
                f"{len(pages)} pages → {len(chunks)} chunks"
            )
        except Exception as e:
            logger.error(f"  ✗ Failed on {pdf_path.name}: {e}")
            continue

    logger.info(f"\nTotal chunks across all documents: {len(all_chunks)}")
    return all_chunks


def get_corpus_stats(chunks: list[Document]) -> dict:
    """
    Print a summary of the loaded corpus.
    Useful for verifying data loading before embedding.
    """
    from collections import Counter

    companies = Counter(c.metadata["company_name"] for c in chunks)
    doc_types = Counter(c.metadata["doc_type"] for c in chunks)
    sections = Counter(c.metadata["section"] for c in chunks)

    stats = {
        "total_chunks": len(chunks),
        "by_company": dict(companies),
        "by_doc_type": dict(doc_types),
        "by_section": dict(sections.most_common(10)),
    }
    return stats


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    validate_env()

    # Test with a single PDF if available
    test_pdfs = list(RAW_PDFS_DIR.glob("*.pdf"))

    if not test_pdfs:
        print(f"\nNo PDFs in {RAW_PDFS_DIR}")
        print("Download a DRHP and rename it to match a key in COMPANY_METADATA")
        print("Example: zomato_drhp_2021.pdf")
    else:
        # Test first available PDF
        test_pdf = test_pdfs[0]
        company_key = test_pdf.stem

        print(f"\nTesting with: {test_pdf.name}")
        pages = load_pdf(test_pdf, company_key)
        print(f"Pages loaded: {len(pages)}")
        print(f"\nSample page metadata:\n{pages[0].metadata}")
        print(f"\nSample text (first 300 chars):\n{pages[0].page_content[:300]}")

        chunks = chunk_documents(pages)
        print(f"\nChunks created: {len(chunks)}")
        print(f"\nSample chunk metadata:\n{chunks[0].metadata}")
        print(f"\nSample chunk (first 300 chars):\n{chunks[0].page_content[:300]}")

        stats = get_corpus_stats(chunks)
        print(f"\nCorpus stats:\n{stats}")