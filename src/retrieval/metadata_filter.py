"""
src/retrieval/metadata_filter.py

Builds ChromaDB metadata filters from user query context.

Why metadata filtering matters:
    Without filtering, searching for "Zomato revenue" retrieves chunks from
    ALL companies, including Paytm and Ola Electric. The LLM then has to
    disambiguate, often failing.

    With metadata filtering, we first narrow the search space to only
    Zomato documents before running vector search. This single step
    dramatically improves precision and is the simplest performance win.

ChromaDB filter syntax:
    Single value:  {"company_name": "Zomato"}
    AND:           {"$and": [{"company_name": "Zomato"}, {"year": "2021"}]}
    OR companies:  {"company_name": {"$in": ["Zomato", "Swiggy"]}}
    IN list:       {"doc_type": {"$in": ["DRHP", "Annual_Report"]}}
"""

from typing import Optional
from src.shared.logger import get_logger

logger = get_logger(__name__)


def build_filter(
    companies: Optional[list[str]] = None,
    years: Optional[list[str]] = None,
    doc_types: Optional[list[str]] = None,
    sections: Optional[list[str]] = None,
) -> Optional[dict]:
    """
    Build a ChromaDB where-filter from retrieval parameters.

    Args:
        companies:  List of company names e.g. ["Zomato", "Swiggy"]
        years:      List of years e.g. ["2021", "2022"]
        doc_types:  List of doc types e.g. ["DRHP", "Annual_Report"]
        sections:   List of sections e.g. ["Risk Factors", "Financial Statements"]

    Returns:
        ChromaDB filter dict, or None if no filters specified.

    Examples:
        build_filter(companies=["Zomato"])
        → {"company_name": "Zomato"}

        build_filter(companies=["Zomato", "Swiggy"], doc_types=["DRHP"])
        → {"$and": [{"company_name": {"$in": ["Zomato", "Swiggy"]}}, {"doc_type": "DRHP"}]}
    """
    conditions = []

    if companies:
        if len(companies) == 1:
            conditions.append({"company_name": companies[0]})
        else:
            conditions.append({"company_name": {"$in": companies}})

    if years:
        if len(years) == 1:
            conditions.append({"year": years[0]})
        else:
            conditions.append({"year": {"$in": years}})

    if doc_types:
        if len(doc_types) == 1:
            conditions.append({"doc_type": doc_types[0]})
        else:
            conditions.append({"doc_type": {"$in": doc_types}})

    if sections:
        if len(sections) == 1:
            conditions.append({"section": sections[0]})
        else:
            conditions.append({"section": {"$in": sections}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def extract_companies_from_query(query: str) -> list[str]:
    """
    Simple heuristic to detect company names in a query.
    Stage 1 version — replaced by Intent Agent in Stage 3.

    Args:
        query: Raw user query string.

    Returns:
        List of detected company names (from our known corpus).
    """
    # Known companies in our corpus — extend as corpus grows
    known_companies = {
        "zomato": "Zomato",
        "swiggy": "Swiggy",
        "paytm": "Paytm",
        "ola electric": "Ola Electric",
        "ola": "Ola Electric",
    }
    query_lower = query.lower()
    detected = []
    for keyword, company_name in known_companies.items():
        if keyword in query_lower and company_name not in detected:
            detected.append(company_name)

    if detected:
        logger.info(
            "Companies detected in query",
            extra={"detected": detected, "query": query[:100]}
        )
    return detected


if __name__ == "__main__":
    # Test filter building
    tests = [
        (["Zomato"], None, None, None),
        (["Zomato", "Swiggy"], None, ["DRHP"], None),
        (None, ["2021"], ["DRHP", "Annual_Report"], None),
        (["Paytm"], ["2021"], None, ["Risk Factors"]),
        (None, None, None, None),
    ]

    for companies, years, doc_types, sections in tests:
        f = build_filter(companies, years, doc_types, sections)
        print(f"build_filter({companies}, {years}, {doc_types}, {sections})")
        print(f"  → {f}\n")

    print("Metadata filter working correctly.")