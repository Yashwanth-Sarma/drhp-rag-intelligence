"""
src/agents/query_decomposer.py

Decomposes a complex user query into multiple focused retrieval tasks.

Why this matters:
"Tell me about Zomato revenue, risks, and compare with Paytm"
is actually 5 separate retrieval tasks, each needing different
metadata filters and section targeting.

Without decomposition: one vague retrieval → 5 mediocre chunks → bad report
With decomposition: 5 focused retrievals → 25 targeted chunks → good report

Output is a list of RetrievalTask objects, each with:
- A focused sub-question
- Company filter
- Section filter
- Doc type filter
- Expected output type (narrative / financial_metrics / risk_list / comparison)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.llm.provider_router import get_router, TaskType

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTask:
    """One focused retrieval task derived from query decomposition."""
    task_id: str
    sub_question: str
    companies: Optional[list[str]]
    doc_types: Optional[list[str]]
    sections: Optional[list[str]]
    output_type: str  # narrative | financial_metrics | risk_list | comparison | table
    priority: int     # 1 = highest
    depends_on: list[str] = field(default_factory=list)


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""
    original_query: str
    tasks: list[RetrievalTask]
    detected_companies: list[str]
    detected_intents: list[str]
    is_comparison: bool
    requires_charts: bool
    complexity: str  # simple | moderate | complex


DECOMPOSE_PROMPT = """You are a financial research query analyzer for an Indian capital markets platform.

Analyze this user query and decompose it into specific retrieval tasks.

USER QUERY: {query}

AVAILABLE COMPANIES IN DATABASE: {available_companies}

Return ONLY valid JSON — no explanation, no markdown:
{{
  "detected_companies": ["company1", "company2"],
  "detected_intents": ["revenue_analysis", "risk_analysis", "comparison", "profit_analysis"],
  "is_comparison": true/false,
  "requires_charts": true/false,
  "complexity": "simple/moderate/complex",
  "tasks": [
    {{
      "task_id": "task_1",
      "sub_question": "specific focused question",
      "companies": ["Zomato"],
      "doc_types": ["DRHP", "Annual_Report"],
      "sections": ["Financial Statements"],
      "output_type": "financial_metrics",
      "priority": 1
    }}
  ]
}}

Rules:
- Break complex queries into 2-6 focused tasks
- Each task targets ONE company and ONE topic
- output_type must be: narrative | financial_metrics | risk_list | comparison | table
- sections must be from: Financial Statements | Risk Factors | Business Overview | Management | Capital Structure | Objects of the Offer | Industry Overview | General
- Only include companies that appear in AVAILABLE COMPANIES
- Priority 1 = most important, higher number = less critical

JSON:"""


class QueryDecomposer:
    """
    Decomposes complex financial queries into focused retrieval tasks.
    Uses Groq for fast classification — this runs on every query.
    """

    def __init__(self) -> None:
        self.router = get_router()

    def decompose(
        self,
        query: str,
        available_companies: list[str],
        analysis_types: Optional[list[str]] = None,
    ) -> DecomposedQuery:
        """
        Decompose a query into focused retrieval tasks.

        Args:
            query:              Original user query.
            available_companies: Companies indexed in ChromaDB.
            analysis_types:     User-selected analysis types from blueprint.
                                If provided, guides decomposition.

        Returns:
            DecomposedQuery with list of RetrievalTask objects.
        """
        # Enhance query with blueprint analysis types if provided
        enhanced_query = query
        if analysis_types:
            enhanced_query = f"{query}\n[User wants: {', '.join(analysis_types)}]"

        prompt = DECOMPOSE_PROMPT.format(
            query=enhanced_query,
            available_companies=", ".join(available_companies),
        )

        try:
            raw = self.router.generate(
                task=TaskType.ROUTING,
                prompt=prompt,
                temperature=0.0,
                max_tokens=800,
                system="You are a precise financial query analyzer. Return only valid JSON."
            )

            # Strip markdown fences
            text = raw.strip()
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else parts[0]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            data = json.loads(text)

            tasks = []
            for i, t in enumerate(data.get("tasks", []), 1):
                tasks.append(RetrievalTask(
                    task_id=t.get("task_id", f"task_{i}"),
                    sub_question=t.get("sub_question", ""),
                    companies=t.get("companies"),
                    doc_types=t.get("doc_types"),
                    sections=t.get("sections"),
                    output_type=t.get("output_type", "narrative"),
                    priority=t.get("priority", i),
                ))

            # Sort by priority
            tasks.sort(key=lambda x: x.priority)

            decomposed = DecomposedQuery(
                original_query=query,
                tasks=tasks,
                detected_companies=data.get("detected_companies", []),
                detected_intents=data.get("detected_intents", []),
                is_comparison=data.get("is_comparison", False),
                requires_charts=data.get("requires_charts", False),
                complexity=data.get("complexity", "moderate"),
            )

            logger.info(
                "Query decomposed",
                extra={
                    "tasks": len(tasks),
                    "companies": decomposed.detected_companies,
                    "complexity": decomposed.complexity,
                    "requires_charts": decomposed.requires_charts,
                }
            )
            return decomposed

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Decomposition failed: {e}. Using simple single-task fallback.")
            # Fallback: single task covering the whole query
            return DecomposedQuery(
                original_query=query,
                tasks=[RetrievalTask(
                    task_id="task_1",
                    sub_question=query,
                    companies=None,
                    doc_types=None,
                    sections=None,
                    output_type="narrative",
                    priority=1,
                )],
                detected_companies=[],
                detected_intents=["general"],
                is_comparison=False,
                requires_charts=False,
                complexity="simple",
            )