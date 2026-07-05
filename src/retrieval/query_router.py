"""
src/retrieval/query_router.py

Routes queries to the right retrieval path:
- Table/numerical questions -> ColPali
- Relationship/comparison questions -> GraphRAG (built later)
- Everything else -> Hybrid retriever (Stage 2)

Uses simple keyword heuristics — cheap, no LLM call needed for routing.
Upgrade to an LLM-based intent agent later if needed.
"""

from src.shared.logger import get_logger

logger = get_logger(__name__)

TABLE_KEYWORDS = [
    "table", "breakdown", "quarter", "q1", "q2", "q3", "q4",
    "margin", "ratio", "percentage", "trend", "over time",
    "year over year", "compare financials", "balance sheet",
]

RELATIONSHIP_KEYWORDS = [
    "compare", "versus", "vs", "relationship", "subsidiary",
    "related to", "how does", "connection between",
]


def route_query(query: str) -> str:
    """
    Returns one of: "colpali", "graphrag", "hybrid"
    """
    query_lower = query.lower()

    if any(kw in query_lower for kw in TABLE_KEYWORDS):
        logger.info("Query routed to ColPali", extra={"query": query[:80]})
        return "colpali"

    if any(kw in query_lower for kw in RELATIONSHIP_KEYWORDS):
        logger.info("Query routed to GraphRAG", extra={"query": query[:80]})
        return "graphrag"

    logger.info("Query routed to Hybrid (default)", extra={"query": query[:80]})
    return "hybrid"


if __name__ == "__main__":
    test_queries = [
        "What was the quarterly revenue breakdown for Ola Electric?",
        "Compare Zomato and Paytm's risk factors",
        "What are the main risk factors for Zomato?",
    ]
    for q in test_queries:
        route = route_query(q)
        print(f"'{q[:50]}...' -> {route}")