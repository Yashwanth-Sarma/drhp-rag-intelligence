"""
src/agents/report_composer.py

Composes structured multi-section financial reports.

Takes:
- Decomposed query tasks
- Retrieved + verified evidence per task
- Extracted financial metrics
- Analysis type selections

Produces:
- Structured report with sections
- Executive summary
- Per-section evidence links
- Chart specifications
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from src.agents.financial_extractor import FinancialMetric, metrics_to_chart_data
from src.agents.query_decomposer import DecomposedQuery, RetrievalTask
from src.llm.provider_router import get_router, TaskType
from src.retrieval.base_retriever import format_context, build_citations

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """One section of the generated report."""
    title: str
    content: str
    citations: list[dict]
    chart_specs: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    task_id: str = ""


@dataclass
class ComposedReport:
    """Complete structured report ready for rendering."""
    query: str
    executive_summary: str
    sections: list[ReportSection]
    all_citations: list[dict]
    chart_specs: list[dict]
    overall_confidence: float
    companies_covered: list[str]
    provider_used: str
    total_latency_ms: int


SECTION_PROMPT = """You are a senior investment research analyst. Write a clear, professional report section.

SECTION: {section_title}
ANALYSIS TYPE: {analysis_types}

EVIDENCE:
{context}

INSTRUCTIONS:
- Use ONLY information from the evidence above
- Write in professional investment research style
- Include specific numbers, percentages, and dates when available
- Every claim must be traceable to the evidence
- For financial metrics: state exact figures (e.g. "Revenue of Rs. 2,605 Cr in FY2021")
- Keep to 200-400 words for this section
- Do NOT invent any numbers or facts not in the evidence
- End with 1-2 key takeaways for this section

REPORT SECTION:"""

EXECUTIVE_SUMMARY_PROMPT = """You are a senior investment research analyst.
Write a concise executive summary (150-200 words) covering the key findings.

COMPANIES ANALYZED: {companies}
ANALYSIS TYPES: {analysis_types}

KEY FINDINGS FROM SECTIONS:
{section_summaries}

INSTRUCTIONS:
- Lead with the most important finding
- State 3-4 specific facts with numbers
- Mention key risks if any risk analysis was done
- Professional tone, no filler phrases
- End with one-line investment relevance statement

EXECUTIVE SUMMARY:"""


class ReportComposer:
    """
    Composes multi-section financial reports from evidence.

    One LLM call per section, plus one for executive summary.
    Uses Gemini Flash for sections, Gemini Pro for complex comparisons.
    """

    def __init__(self) -> None:
        self.router = get_router()

    def compose_section(
        self,
        task: RetrievalTask,
        chunks: list[Document],
        analysis_types: list[str],
        section_title: str,
    ) -> ReportSection:
        """Generate one report section from a retrieval task's evidence."""
        if not chunks:
            return ReportSection(
                title=section_title,
                content=(
                    f"*Insufficient evidence found for {section_title}. "
                    "The indexed documents may not contain this information.*"
                ),
                citations=[],
                confidence=0.0,
                task_id=task.task_id,
            )

        context = format_context(chunks[:5])
        prompt = SECTION_PROMPT.format(
            section_title=section_title,
            analysis_types=", ".join(analysis_types),
            context=context,
        )

        is_hard = task.is_comparison if hasattr(task, "is_comparison") else False
        is_hard = is_hard or "Peer Comparison" in analysis_types

        try:
            content = self.router.generate(
                task=TaskType.REASONING,
                prompt=prompt,
                temperature=0.1,
                max_tokens=600,
                hard_query=is_hard,
            )
        except Exception as e:
            logger.error(f"Section generation failed for {section_title}: {e}")
            content = f"*Section generation failed: {str(e)[:100]}*"

        citations = build_citations(chunks[:5])

        # Compute section confidence from chunk scores
        scores = [
            c.metadata.get("rerank_score") or c.metadata.get("similarity_score") or 0.5
            for c in chunks[:5]
        ]
        confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        return ReportSection(
            title=section_title,
            content=content,
            citations=citations,
            confidence=confidence,
            task_id=task.task_id,
        )

    def compose_report(
        self,
        query: str,
        decomposed: DecomposedQuery,
        chunks_per_task: dict[str, list[Document]],
        metrics: list[FinancialMetric],
        analysis_types: list[str],
        total_latency_ms: int = 0,
    ) -> ComposedReport:
        """
        Compose the full report from all task results.

        Args:
            query:           Original user query
            decomposed:      Decomposed query with task definitions
            chunks_per_task: Retrieved chunks keyed by task_id
            metrics:         Extracted financial metrics
            analysis_types:  User-selected analysis types
            total_latency_ms: Total retrieval latency

        Returns:
            ComposedReport ready for rendering
        """
        sections = []
        all_citations = []
        all_chart_specs = []
        total_confidence = []

        # Map task output_type to section title
        SECTION_TITLES = {
            "financial_metrics": "Financial Performance",
            "risk_list": "Risk Analysis",
            "narrative": "Overview",
            "comparison": "Comparative Analysis",
            "table": "Data Tables",
        }

        for task in decomposed.tasks:
            chunks = chunks_per_task.get(task.task_id, [])
            companies_str = (
                ", ".join(task.companies) if task.companies else "All companies"
            )
            title = f"{SECTION_TITLES.get(task.output_type, 'Analysis')} — {companies_str}"

            section = self.compose_section(
                task=task,
                chunks=chunks,
                analysis_types=analysis_types,
                section_title=title,
            )
            sections.append(section)
            all_citations.extend(section.citations)
            if section.confidence > 0:
                total_confidence.append(section.confidence)

            # Generate chart specs for financial metric tasks
            if task.output_type == "financial_metrics" and metrics:
                company = task.companies[0] if task.companies else None
                for metric_name in ["revenue", "profit", "loss", "ebitda"]:
                    chart_data = metrics_to_chart_data(
                        metrics,
                        metric_name=metric_name,
                        companies=task.companies,
                    )
                    if chart_data and chart_data.get("series"):
                        all_chart_specs.append({
                            "chart_type": "bar" if len(chart_data["periods"]) > 1 else "metric",
                            "title": f"{company or 'Company'} {metric_name.title()} (₹ {chart_data['unit']})",
                            "data": chart_data,
                            "task_id": task.task_id,
                        })

        # Executive summary
        section_summaries = "\n\n".join(
            f"{s.title}:\n{s.content[:300]}..."
            for s in sections if s.content and not s.content.startswith("*")
        )

        if section_summaries:
            exec_prompt = EXECUTIVE_SUMMARY_PROMPT.format(
                companies=", ".join(decomposed.detected_companies) or "Multiple companies",
                analysis_types=", ".join(analysis_types),
                section_summaries=section_summaries,
            )
            try:
                exec_summary = self.router.generate(
                    task=TaskType.REASONING,
                    prompt=exec_prompt,
                    temperature=0.1,
                    max_tokens=300,
                    hard_query=decomposed.is_comparison,
                )
            except Exception as e:
                exec_summary = f"Executive summary unavailable: {str(e)[:80]}"
        else:
            exec_summary = (
                "Insufficient evidence retrieved to generate an executive summary. "
                "Please check that the requested companies are indexed and try again."
            )

        overall_confidence = (
            round(sum(total_confidence) / len(total_confidence), 4)
            if total_confidence else 0.0
        )

        # Deduplicate citations by chunk_id
        seen_ids = set()
        unique_citations = []
        for c in all_citations:
            if c["chunk_id"] not in seen_ids:
                seen_ids.add(c["chunk_id"])
                unique_citations.append(c)

        # Re-number citations
        for i, c in enumerate(unique_citations, 1):
            c["citation_index"] = i

        return ComposedReport(
            query=query,
            executive_summary=exec_summary,
            sections=sections,
            all_citations=unique_citations,
            chart_specs=all_chart_specs,
            overall_confidence=overall_confidence,
            companies_covered=decomposed.detected_companies,
            provider_used="gemini_flash",
            total_latency_ms=total_latency_ms,
        )