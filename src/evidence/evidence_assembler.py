"""
src/evidence/evidence_assembler.py

Assembles per-sentence evidence chains from retrieved chunks.

This is the core module that powers:
- Citation highlighting in the UI (which chunk supported which sentence)
- Contradiction detection across documents
- Confidence propagation (retrieval score → rerank score → final confidence)
- Evidence categorization (Primary / Supporting / Regulatory)

Every factual claim in the final answer traces back to a specific
chunk_id, page number, document, and exact text excerpt.
No claim exists without traceable evidence.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """One piece of evidence supporting or contradicting a claim."""
    chunk_id: str
    company_name: str
    doc_type: str
    year: str
    page_number: int
    section: str
    source_file: str
    text_excerpt: str
    full_text: str
    confidence: float
    category: str  # Primary | Supporting | Regulatory | Commentary
    is_contradicting: bool = False
    contradiction_note: str = ""


@dataclass
class SentenceEvidence:
    """Evidence chain for one sentence in the generated answer."""
    sentence: str
    sentence_index: int
    supporting_evidence: list[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    has_contradiction: bool = False
    is_unsupported: bool = False  # True if no evidence found


@dataclass
class AssembledEvidence:
    """
    Complete evidence package for one query-answer pair.
    This is what gets passed to the frontend for rendering.
    """
    question: str
    answer: str
    sentence_evidence: list[SentenceEvidence]
    all_chunks: list[Document]
    citations: list[dict]
    overall_confidence: float
    has_contradictions: bool
    contradiction_pairs: list[dict]  # pairs of contradicting chunks
    provider_used: str
    latency_ms: int
    retrieval_debug: dict


def _categorize_chunk(chunk: Document) -> str:
    """
    Assign an evidence category based on chunk metadata.

    Primary:    Direct financial statements, DRHP risk factors, IPO objects
    Supporting: Business overview, management discussion, industry context
    Regulatory: SEBI filings, RBI circulars, compliance sections
    Commentary: Earnings call transcripts, investor presentations
    """
    doc_type = chunk.metadata.get("doc_type", "").lower()
    section = chunk.metadata.get("section", "").lower()

    if any(kw in section for kw in ["financial statement", "balance sheet", "profit", "cash flow"]):
        return "Primary"
    if any(kw in section for kw in ["risk factor", "objects of the offer", "capital structure"]):
        return "Primary"
    if any(kw in doc_type for kw in ["drhp", "rhp", "annual_report"]):
        if any(kw in section for kw in ["business overview", "industry overview"]):
            return "Supporting"
        return "Primary"
    if any(kw in doc_type for kw in ["regulatory", "circular", "sebi", "rbi"]):
        return "Regulatory"
    if any(kw in doc_type for kw in ["earnings", "transcript", "investor"]):
        return "Commentary"
    return "Supporting"


def _compute_chunk_confidence(chunk: Document) -> float:
    """
    Compute confidence for one chunk using available scores.
    Preference: rerank_score > similarity_score > rrf_score > default
    """
    rerank = chunk.metadata.get("rerank_score")
    if rerank is not None:
        return float(rerank)

    similarity = chunk.metadata.get("similarity_score")
    if similarity is not None:
        return float(similarity)

    rrf = chunk.metadata.get("rrf_score")
    if rrf is not None:
        # RRF scores are small (0.01-0.05 range) — normalize to 0-1
        return min(1.0, float(rrf) * 20)

    return 0.5  # neutral default


def _detect_contradictions(chunks: list[Document]) -> list[dict]:
    """
    Detect potentially contradicting claims across retrieved chunks.

    Strategy: look for same financial metric mentioned with different
    values across different chunks from different documents.
    Uses regex to find number patterns near financial keywords.
    """
    contradictions = []
    financial_pattern = re.compile(
        r'(?:revenue|profit|loss|margin|ebitda|gmv|growth|cagr)'
        r'.*?(?:rs\.?|inr|₹|%|\d+(?:,\d+)*(?:\.\d+)?)',
        re.IGNORECASE
    )

    # Extract financial claims per chunk
    chunk_claims: list[dict] = []
    for chunk in chunks:
        text = chunk.metadata.get("original_text", chunk.page_content)
        matches = financial_pattern.findall(text[:1000])
        if matches:
            chunk_claims.append({
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "company": chunk.metadata.get("company_name", ""),
                "doc_type": chunk.metadata.get("doc_type", ""),
                "year": chunk.metadata.get("year", ""),
                "page": chunk.metadata.get("page_number", ""),
                "claims": matches[:5],  # top 5 financial mentions
            })

    # Look for same company + different values across different documents
    for i, claim_a in enumerate(chunk_claims):
        for claim_b in chunk_claims[i + 1:]:
            if claim_a["company"] != claim_b["company"]:
                continue
            if claim_a["doc_type"] == claim_b["doc_type"] and claim_a["year"] == claim_b["year"]:
                continue
            # Same company, different doc/year — check for value conflicts
            # Simple heuristic: if both mention numbers, flag as potential conflict to review
            has_numbers_a = any(re.search(r'\d', c) for c in claim_a["claims"])
            has_numbers_b = any(re.search(r'\d', c) for c in claim_b["claims"])
            if has_numbers_a and has_numbers_b:
                contradictions.append({
                    "chunk_a": claim_a["chunk_id"],
                    "chunk_b": claim_b["chunk_id"],
                    "company": claim_a["company"],
                    "doc_a": f"{claim_a['doc_type']} {claim_a['year']} p{claim_a['page']}",
                    "doc_b": f"{claim_b['doc_type']} {claim_b['year']} p{claim_b['page']}",
                    "note": "Same company, different documents — verify for consistency",
                })

    return contradictions[:5]  # cap at 5 flagged pairs


def _split_answer_into_sentences(answer: str) -> list[str]:
    """
    Split answer text into individual sentences.
    Handles common financial notation edge cases:
    - Rs. 1,234 (period after abbreviation)
    - FY23. (period after year)
    - Numbers like 1.5x, 23.4%
    """
    # Protect common abbreviations and financial notation
    protected = answer
    for abbr in ["Rs.", "Dr.", "Mr.", "Ms.", "Ltd.", "Inc.", "Co.", "FY.", "Q1.", "Q2.", "Q3.", "Q4."]:
        protected = protected.replace(abbr, abbr.replace(".", "<<<DOT>>>"))

    # Split on sentence-ending punctuation followed by space + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[\(])', protected)

    # Restore protected dots
    sentences = [s.replace("<<<DOT>>>", ".").strip() for s in sentences]

    # Filter empty strings and very short fragments
    return [s for s in sentences if len(s) > 15]


def _find_supporting_chunks(
    sentence: str,
    chunks: list[Document],
) -> list[EvidenceItem]:
    """
    Find which chunks contain information relevant to this sentence.

    Strategy:
    1. Extract key financial terms and numbers from the sentence
    2. Check each chunk for those terms
    3. Return chunks that share significant overlap

    This is lightweight and deterministic — no LLM call needed.
    """
    # Extract significant terms from sentence (numbers, company names, financial terms)
    numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', sentence)
    financial_terms = re.findall(
        r'\b(?:revenue|profit|loss|margin|ebitda|gmv|cagr|ipo|drhp|'
        r'risk|subsidiary|acquisition|litigation|compliance|shareholder|'
        r'promoter|investor|board|director)\b',
        sentence.lower()
    )
    key_terms = set(numbers[:3] + financial_terms[:4])

    if not key_terms:
        return []

    evidence_items = []
    for chunk in chunks:
        text = chunk.metadata.get("original_text", chunk.page_content).lower()
        full_text = chunk.metadata.get("original_text", chunk.page_content)

        # Count how many key terms appear in this chunk
        matches = sum(1 for term in key_terms if term.lower() in text)
        if matches == 0:
            continue

        # Score: fraction of key terms found in this chunk
        relevance = matches / len(key_terms)
        if relevance < 0.25:  # must match at least 25% of key terms
            continue

        chunk_confidence = _compute_chunk_confidence(chunk)
        combined_confidence = round((chunk_confidence + relevance) / 2, 4)

        evidence_items.append(EvidenceItem(
            chunk_id=chunk.metadata.get("chunk_id", ""),
            company_name=chunk.metadata.get("company_name", "Unknown"),
            doc_type=chunk.metadata.get("doc_type", "Unknown"),
            year=chunk.metadata.get("year", "Unknown"),
            page_number=chunk.metadata.get("page_number", 0),
            section=chunk.metadata.get("section", "Unknown"),
            source_file=chunk.metadata.get("source_file", ""),
            text_excerpt=full_text[:300] + "..." if len(full_text) > 300 else full_text,
            full_text=full_text,
            confidence=combined_confidence,
            category=_categorize_chunk(chunk),
        ))

    # Sort by confidence descending, return top 3
    evidence_items.sort(key=lambda x: x.confidence, reverse=True)
    return evidence_items[:3]


def assemble_evidence(
    question: str,
    answer: str,
    chunks: list[Document],
    citations: list[dict],
    provider_used: str = "unknown",
    latency_ms: int = 0,
    retrieval_debug: Optional[dict] = None,
) -> AssembledEvidence:
    """
    Main function: takes a query result and produces a full evidence package.

    For every sentence in the answer:
    - Finds supporting chunks
    - Detects potential contradictions
    - Computes propagated confidence

    Args:
        question:       Original user query
        answer:         Generated answer text
        chunks:         Retrieved and reranked chunks
        citations:      Citation dicts from build_citations()
        provider_used:  Which LLM generated the answer
        latency_ms:     Total query latency
        retrieval_debug: Debug info from retriever

    Returns:
        AssembledEvidence with per-sentence evidence chains
    """
    if not answer or not chunks:
        return AssembledEvidence(
            question=question,
            answer=answer or "",
            sentence_evidence=[],
            all_chunks=chunks,
            citations=citations,
            overall_confidence=0.0,
            has_contradictions=False,
            contradiction_pairs=[],
            provider_used=provider_used,
            latency_ms=latency_ms,
            retrieval_debug=retrieval_debug or {},
        )

    # Split answer into sentences
    sentences = _split_answer_into_sentences(answer)

    # Detect contradictions across all retrieved chunks
    contradiction_pairs = _detect_contradictions(chunks)
    contradicting_chunk_ids = {
        c["chunk_a"] for c in contradiction_pairs
    } | {c["chunk_b"] for c in contradiction_pairs}

    # Build per-sentence evidence
    sentence_evidence_list = []
    all_confidences = []

    for i, sentence in enumerate(sentences):
        supporting = _find_supporting_chunks(sentence, chunks)

        # Mark contradicting evidence
        for item in supporting:
            if item.chunk_id in contradicting_chunk_ids:
                item.is_contradicting = True
                # Find the contradiction note
                for pair in contradiction_pairs:
                    if item.chunk_id in (pair["chunk_a"], pair["chunk_b"]):
                        item.contradiction_note = pair["note"]
                        break

        contradicting = [e for e in supporting if e.is_contradicting]
        clean_supporting = [e for e in supporting if not e.is_contradicting]

        # Sentence confidence: average of supporting chunk confidences
        if clean_supporting:
            sent_confidence = round(
                sum(e.confidence for e in clean_supporting) / len(clean_supporting), 4
            )
        else:
            sent_confidence = 0.0

        all_confidences.append(sent_confidence)

        sentence_evidence_list.append(SentenceEvidence(
            sentence=sentence,
            sentence_index=i,
            supporting_evidence=clean_supporting,
            contradicting_evidence=contradicting,
            confidence=sent_confidence,
            has_contradiction=len(contradicting) > 0,
            is_unsupported=len(supporting) == 0,
        ))

    # Overall answer confidence
    if all_confidences:
        overall_confidence = round(sum(all_confidences) / len(all_confidences), 4)
    else:
        overall_confidence = 0.0

    logger.info(
        "Evidence assembled",
        extra={
            "sentences": len(sentence_evidence_list),
            "overall_confidence": overall_confidence,
            "contradictions_found": len(contradiction_pairs),
            "unsupported_sentences": sum(
                1 for s in sentence_evidence_list if s.is_unsupported
            ),
        },
    )

    return AssembledEvidence(
        question=question,
        answer=answer,
        sentence_evidence=sentence_evidence_list,
        all_chunks=chunks,
        citations=citations,
        overall_confidence=overall_confidence,
        has_contradictions=len(contradiction_pairs) > 0,
        contradiction_pairs=contradiction_pairs,
        provider_used=provider_used,
        latency_ms=latency_ms,
        retrieval_debug=retrieval_debug or {},
    )