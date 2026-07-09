"""
src/agents/financial_extractor.py

Extracts structured financial metrics from retrieved text chunks.

Why this exists:
Retrieved chunks contain prose like "revenue grew from Rs. 1,312 crores
in FY2020 to Rs. 2,605 crores in FY2021, representing 98.5% growth"

The report composer and chart engine need structured data:
{metric: "revenue", value: 2605, unit: "crores", period: "FY2021", company: "Zomato"}

This module does that extraction deterministically where possible
(regex for numbers) and uses LLM only for complex table parsing.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class FinancialMetric:
    """One extracted financial data point."""
    metric_name: str        # revenue, profit, loss, margin, gmv, etc.
    value: float
    unit: str               # crores, millions, percentage, etc.
    period: str             # FY2021, Q3FY24, etc.
    company: str
    source_page: int
    source_doc: str
    raw_text: str           # original sentence for citation


# Patterns for common Indian financial notation
CRORE_PATTERN = re.compile(
    r'(?:rs\.?\s*|inr\s*|₹\s*)?'
    r'(\d+(?:,\d+)*(?:\.\d+)?)'
    r'\s*(?:crores?|cr\.?)',
    re.IGNORECASE
)

MILLION_PATTERN = re.compile(
    r'(?:rs\.?\s*|inr\s*|₹\s*)?'
    r'(\d+(?:,\d+)*(?:\.\d+)?)'
    r'\s*(?:millions?|mn\.?)',
    re.IGNORECASE
)

PERCENT_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)',
    re.IGNORECASE
)

# Financial metric keyword detection
METRIC_KEYWORDS = {
    "revenue": ["revenue", "income from operations", "revenue from operations", "total income"],
    "profit": ["profit after tax", "pat", "net profit", "profit for the year"],
    "loss": ["loss after tax", "net loss", "loss for the year"],
    "ebitda": ["ebitda", "adjusted ebitda", "operating profit"],
    "gmv": ["gross merchandise value", "gmv"],
    "margin": ["margin", "gross margin", "ebitda margin"],
    "growth": ["growth", "increased by", "grew by", "declined by", "decreased by"],
    "cagr": ["cagr", "compound annual growth"],
}

PERIOD_PATTERN = re.compile(
    r'\b(?:fy|financial year)\s*(\d{2,4}(?:-\d{2,4})?)'
    r'|(?:q[1-4])\s*fy\s*(\d{2,4})'
    r'|\b(20\d{2}-\d{2,4}|\d{4})\b',
    re.IGNORECASE
)


def _detect_metric_type(text: str) -> str:
    """Detect which financial metric type a text snippet discusses."""
    text_lower = text.lower()
    for metric, keywords in METRIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return metric
    return "financial_metric"


def _extract_period(text: str) -> str:
    """Extract financial period from text."""
    match = PERIOD_PATTERN.search(text)
    if match:
        for group in match.groups():
            if group:
                period = group.strip()
                if not period.upper().startswith("FY"):
                    period = f"FY{period}"
                return period
    return "Unknown Period"


def extract_metrics_from_chunks(
    chunks: list[Document],
    company_filter: Optional[str] = None,
) -> list[FinancialMetric]:
    """
    Extract structured financial metrics from retrieved chunks.

    Strategy:
    1. Split each chunk into sentences
    2. For each sentence, check if it contains financial numbers
    3. Extract: metric type, value, unit, period
    4. Return structured FinancialMetric objects

    This is deterministic regex-based extraction.
    No LLM call needed for number extraction.

    Args:
        chunks:         Retrieved Document chunks
        company_filter: Only extract for this company if specified

    Returns:
        List of FinancialMetric objects, sorted by period
    """
    metrics = []

    for chunk in chunks:
        meta = chunk.metadata
        company = meta.get("company_name", "Unknown")

        if company_filter and company != company_filter:
            continue

        text = meta.get("original_text", chunk.page_content)
        page = meta.get("page_number", 0)
        doc = f"{meta.get('doc_type', '')} {meta.get('year', '')}"

        # Split into sentences for focused extraction
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            # Check for crore values
            crore_matches = CRORE_PATTERN.findall(sentence)
            for match in crore_matches:
                try:
                    value = float(match.replace(",", ""))
                    metric_type = _detect_metric_type(sentence)
                    period = _extract_period(sentence)

                    metrics.append(FinancialMetric(
                        metric_name=metric_type,
                        value=value,
                        unit="crores",
                        period=period,
                        company=company,
                        source_page=page,
                        source_doc=doc,
                        raw_text=sentence[:200],
                    ))
                except ValueError:
                    continue

            # Check for percentage values
            pct_matches = PERCENT_PATTERN.findall(sentence)
            if pct_matches and any(kw in sentence.lower() for kw in ["growth", "margin", "cagr", "increased", "decreased"]):
                for match in pct_matches:
                    try:
                        value = float(match)
                        metric_type = _detect_metric_type(sentence)
                        period = _extract_period(sentence)

                        metrics.append(FinancialMetric(
                            metric_name=f"{metric_type}_pct",
                            value=value,
                            unit="percent",
                            period=period,
                            company=company,
                            source_page=page,
                            source_doc=doc,
                            raw_text=sentence[:200],
                        ))
                    except ValueError:
                        continue

    # Remove duplicates (same metric + value + period + company)
    seen = set()
    unique_metrics = []
    for m in metrics:
        key = (m.metric_name, m.value, m.period, m.company)
        if key not in seen:
            seen.add(key)
            unique_metrics.append(m)

    logger.info(
        f"Extracted {len(unique_metrics)} financial metrics from {len(chunks)} chunks"
    )
    return unique_metrics


def metrics_to_chart_data(
    metrics: list[FinancialMetric],
    metric_name: str,
    companies: Optional[list[str]] = None,
) -> dict:
    """
    Convert extracted metrics into chart-ready data structure.

    Args:
        metrics:     All extracted metrics
        metric_name: Which metric to chart (e.g. "revenue")
        companies:   Companies to include (None = all)

    Returns:
        Dict with 'periods', 'series' (company → values), 'unit', 'citations'
    """
    # Filter to requested metric
    filtered = [
        m for m in metrics
        if m.metric_name == metric_name
        and (companies is None or m.company in companies)
        and m.period != "Unknown Period"
        and m.value > 0
    ]

    if not filtered:
        return {}

    # Sort by period
    filtered.sort(key=lambda x: x.period)

    # Build per-company series
    all_periods = sorted(set(m.period for m in filtered))
    series = {}
    citations = {}

    for m in filtered:
        if m.company not in series:
            series[m.company] = {}
            citations[m.company] = {}
        series[m.company][m.period] = m.value
        citations[m.company][m.period] = f"{m.source_doc} p{m.source_page}"

    # Fill missing periods with None
    company_series = {
        company: [data.get(p) for p in all_periods]
        for company, data in series.items()
    }

    unit = filtered[0].unit if filtered else "crores"

    return {
        "periods": all_periods,
        "series": company_series,
        "unit": unit,
        "citations": citations,
        "metric": metric_name,
    }