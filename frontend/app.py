"""
frontend/app.py

FinSight — AI Financial Intelligence Platform
Streamlit frontend with Research Blueprint + Evidence Explorer.

Features:
- Research Blueprint: editable execution plan before generation
- Evidence Explorer: per-sentence citations with source text
- Contradiction detection panel
- PDF page display with highlighted evidence text
- Confidence scores at sentence and overall level
- Multi-company support with metadata filtering
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="FinSight — Financial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports after page config ──────────────────────────────────────────────
from src.configuration.config import (
    COMPANY_METADATA,
    EMBEDDINGS_STAGE1_DIR,
    EMBEDDINGS_STAGE2_DIR,
    COLLECTION_STAGE1,
    COLLECTION_STAGE2,
)
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.metadata_filter import build_filter
from src.evidence.evidence_assembler import assemble_evidence, AssembledEvidence
from src.vector_store.chroma_store import ChromaStore


# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark financial-grade color palette */
:root {
    --bg-primary: #0D1117;
    --bg-secondary: #161B22;
    --bg-card: #1C2128;
    --accent-blue: #2F81F7;
    --accent-green: #3FB950;
    --accent-amber: #D29922;
    --accent-red: #F85149;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --border: #30363D;
}

.stApp { background-color: var(--bg-primary); }

/* Confidence badge */
.confidence-high {
    background: #1a4a1a; color: #3FB950;
    padding: 2px 8px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
}
.confidence-med {
    background: #3d2e00; color: #D29922;
    padding: 2px 8px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
}
.confidence-low {
    background: #3d1a1a; color: #F85149;
    padding: 2px 8px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
}

/* Evidence card */
.evidence-card {
    background: #1C2128;
    border: 1px solid #30363D;
    border-left: 4px solid #2F81F7;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
}
.evidence-card.contradiction {
    border-left-color: #F85149;
}
.evidence-card.regulatory {
    border-left-color: #D29922;
}
.evidence-card.supporting {
    border-left-color: #3FB950;
}

/* Highlighted text */
.highlight {
    background: #2d4a1e;
    border-radius: 3px;
    padding: 1px 4px;
    font-style: italic;
}

/* Citation pill */
.citation-pill {
    display: inline-block;
    background: #1f3a5f;
    color: #79c0ff;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 11px;
    margin: 2px;
    cursor: pointer;
}

/* Blueprint panel */
.blueprint-panel {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 20px;
}

/* Sentence with evidence */
.sentence-block {
    padding: 8px 0;
    border-bottom: 1px solid #21262D;
    line-height: 1.7;
}
.unsupported {
    color: #8B949E;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────

def get_confidence_badge(score: float) -> str:
    if score >= 0.70:
        return f'<span class="confidence-high">● {score:.0%}</span>'
    elif score >= 0.45:
        return f'<span class="confidence-med">● {score:.0%}</span>'
    else:
        return f'<span class="confidence-low">● {score:.0%}</span>'


def get_category_color(category: str) -> str:
    return {
        "Primary": "#2F81F7",
        "Supporting": "#3FB950",
        "Regulatory": "#D29922",
        "Commentary": "#8B949E",
    }.get(category, "#8B949E")


def render_evidence_card(item, index: int) -> None:
    """Render one evidence item as a styled card."""
    card_class = "contradiction" if item.is_contradicting else item.category.lower()
    category_color = get_category_color(item.category)

    st.markdown(f"""
    <div class="evidence-card {card_class}">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="font-weight:600; color:#E6EDF3;">
                [{index}] {item.company_name} — {item.doc_type} {item.year}
            </span>
            <span style="color:#8B949E; font-size:12px;">
                Page {item.page_number} · {item.section}
            </span>
        </div>
        <div style="color:#8B949E; font-size:11px; margin-bottom:6px;">
            <span style="color:{category_color}">■</span> {item.category}
            · Confidence: {item.confidence:.0%}
            {'· ⚠️ Potential conflict' if item.is_contradicting else ''}
        </div>
        <div class="highlight" style="color:#CDD9E5; font-size:13px; line-height:1.6;">
            "{item.text_excerpt}"
        </div>
        {'<div style="color:#F85149; font-size:11px; margin-top:6px;">⚠ ' + item.contradiction_note + '</div>' if item.contradiction_note else ''}
    </div>
    """, unsafe_allow_html=True)


def render_research_blueprint(
    question: str,
    available_companies: list[str],
    stage2_available: bool,
) -> dict:
    """
    Render the editable Research Blueprint panel.
    Returns the user's configuration choices.
    """
    st.markdown("### 📋 Research Blueprint")
    st.markdown(
        "*Review and edit the research plan before generation. "
        "Changes here directly affect what evidence is retrieved.*"
    )

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Companies to Research**")
            selected_companies = st.multiselect(
                "Select companies",
                options=available_companies,
                default=[],
                label_visibility="collapsed",
            )

            st.markdown("**Document Sources**")
            use_drhp = st.checkbox("DRHPs (IPO Prospectuses)", value=True)
            use_annual = st.checkbox("Annual Reports", value=True)
            use_earnings = st.checkbox("Earnings Call Transcripts", value=False)

        with col2:
            st.markdown("**Analysis Type**")
            analysis_types = st.multiselect(
                "What kind of analysis? (select all that apply)",
                ["Risk Analysis", "Financial Performance", "Company Overview",
                "Peer Comparison", "IPO Analysis", "Executive Summary",
                "Litigation & Legal", "Management Background", "Custom"],
                default=["Financial Performance"],
                label_visibility="collapsed",
            )

            st.markdown("**Evidence Requirements**")
            require_citations = st.checkbox("Every claim must cite source page", value=True)
            show_contradictions = st.checkbox("Flag contradictory evidence", value=True)
            show_confidence = st.checkbox("Show confidence scores", value=True)

        st.markdown("**Pipeline Stage**")
        if stage2_available:
            stage = st.radio(
                "Retrieval quality",
                ["Stage 2 — Contextual Hybrid (Recommended)", "Stage 1 — Baseline"],
                label_visibility="collapsed",
                horizontal=True,
            )
            use_stage2 = "Stage 2" in stage
        else:
            st.info("Stage 2 not indexed yet. Using Stage 1 baseline.")
            use_stage2 = False

        st.markdown("**Report Detail Level**")
        detail_level = st.select_slider(
            "Detail",
            options=["Brief Summary", "Standard Analysis", "Detailed Report"],
            value="Standard Analysis",
            label_visibility="collapsed",
        )

    # Determine doc_types from checkboxes
    doc_types = []
    if use_drhp:
        doc_types.append("DRHP")
    if use_annual:
        doc_types.append("Annual_Report")
    if use_earnings:
        doc_types.append("Earnings_Transcript")

    return {
        "companies": selected_companies if selected_companies else None,
        "doc_types": doc_types if doc_types else None,
        "analysis_type": analysis_type,
        "use_stage2": use_stage2,
        "require_citations": require_citations,
        "show_contradictions": show_contradictions,
        "show_confidence": show_confidence,
        "detail_level": detail_level,
        "hard_query": analysis_type in ["Peer Comparison", "IPO Analysis"],
    }


def render_answer_with_evidence(assembled: AssembledEvidence, blueprint: dict) -> None:
    """Render the answer with inline evidence and the evidence explorer panel."""

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("### 📄 Analysis Report")

        # Overall confidence bar
        if blueprint["show_confidence"]:
            conf = assembled.overall_confidence
            badge = get_confidence_badge(conf)
            st.markdown(
                f"**Overall Evidence Confidence:** {badge} "
                f"· {len(assembled.all_chunks)} chunks retrieved "
                f"· {assembled.latency_ms}ms "
                f"· via {assembled.provider_used}",
                unsafe_allow_html=True,
            )

        # Contradiction warning
        if assembled.has_contradictions and blueprint["show_contradictions"]:
            st.warning(
                f"⚠️ **{len(assembled.contradiction_pairs)} potential contradiction(s) detected** "
                "across retrieved documents. Check the Evidence Explorer for details."
            )

        st.divider()

        # Render answer sentence by sentence
        for sent_ev in assembled.sentence_evidence:
            cols = st.columns([20, 1])
            with cols[0]:
                if sent_ev.is_unsupported:
                    st.markdown(
                        f'<div class="sentence-block unsupported">'
                        f'{sent_ev.sentence}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Show sentence with confidence badge
                    badge = get_confidence_badge(sent_ev.confidence) if blueprint["show_confidence"] else ""
                    contradiction_flag = " ⚠️" if sent_ev.has_contradiction else ""

                    st.markdown(
                        f'<div class="sentence-block">'
                        f'{sent_ev.sentence} {badge}{contradiction_flag}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Show supporting evidence inline if citations required
                    if blueprint["require_citations"] and sent_ev.supporting_evidence:
                        pills = " ".join(
                            f'<span class="citation-pill">'
                            f'[{e.company_name} · {e.doc_type} · p{e.page_number}]'
                            f'</span>'
                            for e in sent_ev.supporting_evidence[:2]
                        )
                        st.markdown(pills, unsafe_allow_html=True)

    with right_col:
        st.markdown("### 🔍 Evidence Explorer")

        # Evidence tabs
        tab_labels = ["All Evidence"]
        if assembled.has_contradictions and blueprint["show_contradictions"]:
            tab_labels.append("⚠️ Contradictions")
        tab_labels.append("Debug Info")

        tabs = st.tabs(tab_labels)

        with tabs[0]:
            # Group evidence by company
            company_chunks: dict[str, list] = {}
            for chunk in assembled.all_chunks:
                company = chunk.metadata.get("company_name", "Unknown")
                if company not in company_chunks:
                    company_chunks[company] = []
                company_chunks[company].append(chunk)

            for company, company_chunk_list in company_chunks.items():
                with st.expander(f"**{company}** ({len(company_chunk_list)} chunks)", expanded=True):
                    from src.evidence.evidence_assembler import (
                        EvidenceItem, _compute_chunk_confidence, _categorize_chunk
                    )
                    for i, chunk in enumerate(company_chunk_list[:5], 1):
                        full_text = chunk.metadata.get("original_text", chunk.page_content)
                        item = EvidenceItem(
                            chunk_id=chunk.metadata.get("chunk_id", ""),
                            company_name=chunk.metadata.get("company_name", "Unknown"),
                            doc_type=chunk.metadata.get("doc_type", "Unknown"),
                            year=chunk.metadata.get("year", "Unknown"),
                            page_number=chunk.metadata.get("page_number", 0),
                            section=chunk.metadata.get("section", "Unknown"),
                            source_file=chunk.metadata.get("source_file", ""),
                            text_excerpt=full_text[:300] + "..." if len(full_text) > 300 else full_text,
                            full_text=full_text,
                            confidence=_compute_chunk_confidence(chunk),
                            category=_categorize_chunk(chunk),
                        )
                        render_evidence_card(item, i)

        if assembled.has_contradictions and blueprint["show_contradictions"] and len(tabs) > 2:
            with tabs[1]:
                st.markdown("**Potential contradictions found across documents:**")
                for pair in assembled.contradiction_pairs:
                    st.markdown(f"""
                    **Company:** {pair['company']}
                    - Document A: `{pair['doc_a']}`
                    - Document B: `{pair['doc_b']}`
                    - Note: {pair['note']}
                    """)
                    st.divider()

        with tabs[-1]:
            st.markdown("**Retrieval Debug Info**")
            debug = assembled.retrieval_debug
            if debug:
                for key, val in debug.items():
                    st.markdown(f"- **{key}:** `{val}`")
            st.markdown(f"**Provider:** `{assembled.provider_used}`")
            st.markdown(f"**Latency:** `{assembled.latency_ms}ms`")
            st.markdown(f"**Sentences parsed:** `{len(assembled.sentence_evidence)}`")
            st.markdown(f"**Unsupported sentences:** `{sum(1 for s in assembled.sentence_evidence if s.is_unsupported)}`")


# ── Main App ───────────────────────────────────────────────────────────────

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# 📊 FinSight")
        st.markdown("*AI Financial Intelligence Platform*")
        st.divider()

        # Check which stages are available
        stage1_available = EMBEDDINGS_STAGE1_DIR.exists() and ChromaStore(
            persist_dir=EMBEDDINGS_STAGE1_DIR,
            collection_name=COLLECTION_STAGE1
        ).collection.count() > 0

        stage2_available = EMBEDDINGS_STAGE2_DIR.exists() and ChromaStore(
            persist_dir=EMBEDDINGS_STAGE2_DIR,
            collection_name=COLLECTION_STAGE2
        ).collection.count() > 0

        st.markdown("**System Status**")
        st.markdown(f"Stage 1 (Baseline): {'✅' if stage1_available else '❌ Not indexed'}")
        st.markdown(f"Stage 2 (Contextual): {'✅' if stage2_available else '⏳ Indexing in progress'}")

        st.divider()

        # Get available companies from indexed corpus
        available_companies = []
        if stage1_available:
            try:
                store = ChromaStore(
                    persist_dir=EMBEDDINGS_STAGE1_DIR,
                    collection_name=COLLECTION_STAGE1,
                )
                stats = store.get_collection_stats()
                available_companies = stats.get("companies_sampled", [])
            except Exception:
                available_companies = list({
                    v["company_name"] for v in COMPANY_METADATA.values()
                })

        if available_companies:
            st.markdown(f"**Indexed Companies:** {len(available_companies)}")
            for c in available_companies:
                st.markdown(f"  • {c}")

        st.divider()
        st.markdown("*Built with LangChain, Voyage AI,*")
        st.markdown("*Gemini, Cerebras, ChromaDB*")

    # Main content
    st.markdown("# FinSight Financial Intelligence")
    st.markdown("*Evidence-backed analysis of Indian IPO and financial documents*")

    if not stage1_available:
        st.error(
            "No documents indexed yet. "
            "Run: `python scripts/index_documents.py` to get started."
        )
        return

    # Query input
    st.markdown("### 💬 Research Query")
    question = st.text_area(
        "Enter your financial research question",
        placeholder=(
            "Examples:\n"
            "• What are the main risk factors for Zomato?\n"
            "• Compare Zomato and Paytm's revenue growth\n"
            "• What were the objects of Ola Electric's IPO?\n"
            "• Summarize Paytm's business model from their DRHP"
        ),
        height=100,
        label_visibility="collapsed",
    )

    if not question:
        st.info("Enter a question above to begin your financial research.")
        return

    # Research Blueprint
    st.divider()
    blueprint = render_research_blueprint(
        question=question,
        available_companies=available_companies,
        stage2_available=stage2_available,
    )

    # Generate button
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate = st.button(
            "🚀 Generate Research Report",
            use_container_width=True,
            type="primary",
        )

    if not generate:
        return

    # Show decomposition plan first
    progress_bar = st.progress(0, text="Analyzing query...")

    from src.agents.query_decomposer import QueryDecomposer
    from src.agents.financial_extractor import extract_metrics_from_chunks, metrics_to_chart_data
    from src.agents.report_composer import ReportComposer
    import plotly.graph_objects as go

    decomposer = QueryDecomposer()
    decomposed = decomposer.decompose(
        query=question,
        available_companies=available_companies,
        analysis_types=blueprint.get("analysis_types", ["Financial Performance"]),
    )

    # Show execution plan
    with st.expander("📋 Research Blueprint — Execution Plan", expanded=True):
        st.markdown(f"**Detected intent:** {', '.join(decomposed.detected_intents)}")
        st.markdown(f"**Companies identified:** {', '.join(decomposed.detected_companies) or 'None detected'}")
        st.markdown(f"**Query complexity:** {decomposed.complexity}")
        st.markdown(f"**Requires charts:** {'Yes' if decomposed.requires_charts else 'No'}")
        st.markdown(f"**Comparison query:** {'Yes' if decomposed.is_comparison else 'No'}")
        st.divider()
        st.markdown(f"**{len(decomposed.tasks)} retrieval tasks planned:**")
        for task in decomposed.tasks:
            st.markdown(
                f"- **{task.task_id}** [{task.output_type}]: {task.sub_question} "
                f"| Companies: {task.companies} | Sections: {task.sections}"
            )

    progress_bar.progress(15, text=f"Running {len(decomposed.tasks)} retrieval tasks...")

    # Run retrieval for each task
    chunks_per_task = {}
    all_chunks = []
    all_citations = []
    start_time = time.time()

    try:
        for i, task in enumerate(decomposed.tasks):
            progress_pct = 15 + int((i / len(decomposed.tasks)) * 40)
            progress_bar.progress(progress_pct, text=f"Retrieving: {task.sub_question[:60]}...")

            from src.retrieval.metadata_filter import build_filter
            task_filter = build_filter(
                companies=task.companies,
                doc_types=task.doc_types,
                sections=task.sections,
            )

            if blueprint["use_stage2"] and stage2_available:
                retriever = HybridRetriever(stage=2)
                result = retriever.query(
                    question=task.sub_question,
                    companies=task.companies,
                    doc_types=task.doc_types,
                    hard_query=decomposed.is_comparison,
                )
            else:
                retriever = BaseRetriever()
                result = retriever.query(
                    question=task.sub_question,
                    metadata_filter=task_filter,
                )

            chunks_per_task[task.task_id] = result["chunks"]
            all_chunks.extend(result["chunks"])
            all_citations.extend(result["citations"])

        progress_bar.progress(60, text="Extracting financial metrics...")

        # Extract structured metrics
        metrics = extract_metrics_from_chunks(all_chunks)

        progress_bar.progress(70, text="Composing report sections...")

        # Compose full report
        composer = ReportComposer()
        report = composer.compose_report(
            query=question,
            decomposed=decomposed,
            chunks_per_task=chunks_per_task,
            metrics=metrics,
            analysis_types=blueprint.get("analysis_types", ["Financial Performance"]),
            total_latency_ms=round((time.time() - start_time) * 1000),
        )

        progress_bar.progress(90, text="Assembling evidence and rendering...")

        assembled = assemble_evidence(
            question=question,
            answer=report.executive_summary + "\n\n" + "\n\n".join(
                s.content for s in report.sections
            ),
            chunks=all_chunks[:10],
            citations=report.all_citations,
            provider_used=report.provider_used,
            latency_ms=report.total_latency_ms,
        )

        progress_bar.progress(100, text="Complete!")
        time.sleep(0.2)
        progress_bar.empty()

        # Render report
        st.divider()
        st.markdown(f"## 📊 Research Report: {', '.join(report.companies_covered) or 'Financial Analysis'}")

        conf_badge = get_confidence_badge(report.overall_confidence)
        st.markdown(
            f"**Confidence:** {conf_badge} · "
            f"**Sections:** {len(report.sections)} · "
            f"**Citations:** {len(report.all_citations)} · "
            f"**Latency:** {report.total_latency_ms}ms",
            unsafe_allow_html=True,
        )

        st.divider()

        # Executive summary
        st.markdown("### 📋 Executive Summary")
        st.markdown(report.executive_summary)

        # Charts from extracted metrics
        if report.chart_specs and decomposed.requires_charts:
            st.divider()
            st.markdown("### 📈 Financial Charts")
            chart_cols = st.columns(min(2, len(report.chart_specs)))
            for idx, spec in enumerate(report.chart_specs[:4]):
                with chart_cols[idx % 2]:
                    data = spec["data"]
                    if data and data.get("series"):
                        fig = go.Figure()
                        for company, values in data["series"].items():
                            fig.add_trace(go.Bar(
                                name=company,
                                x=data["periods"],
                                y=values,
                                text=[f"₹{v:.0f}Cr" if v else "N/A" for v in values],
                                textposition="outside",
                            ))
                        fig.update_layout(
                            title=spec["title"],
                            xaxis_title="Period",
                            yaxis_title=f"₹ {data['unit']}",
                            template="plotly_dark",
                            height=350,
                            showlegend=len(data["series"]) > 1,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"*Data sourced from retrieved documents*")

        # Sections
        st.divider()
        for section in report.sections:
            st.markdown(f"### {section.title}")
            st.markdown(section.content)
            if section.citations:
                with st.expander(f"📎 {len(section.citations)} citations", expanded=False):
                    for c in section.citations:
                        st.markdown(
                            f"**[{c['citation_index']}]** {c['company_name']} | "
                            f"{c['doc_type']} {c['year']} | "
                            f"Page {c['page_number']} | "
                            f"Section: {c['section']} | "
                            f"Confidence: {c['confidence']:.0%}"
                        )
                        st.markdown(f"> {c['text_excerpt'][:200]}...")
                        st.divider()
            st.divider()

        # Evidence explorer
        render_answer_with_evidence(assembled, blueprint)

        # Download
        full_report_text = f"# FinSight Research Report\n\n"
        full_report_text += f"**Query:** {question}\n\n"
        full_report_text += f"**Companies:** {', '.join(report.companies_covered)}\n"
        full_report_text += f"**Confidence:** {report.overall_confidence:.0%}\n\n---\n\n"
        full_report_text += f"## Executive Summary\n\n{report.executive_summary}\n\n"
        for section in report.sections:
            full_report_text += f"## {section.title}\n\n{section.content}\n\n"
        full_report_text += "## Citations\n\n"
        for c in report.all_citations:
            full_report_text += (
                f"[{c['citation_index']}] {c['company_name']} | "
                f"{c['doc_type']} {c['year']} | "
                f"Page {c['page_number']}\n"
            )

        st.download_button(
            "⬇️ Download Full Report",
            data=full_report_text,
            file_name=f"finsight_{int(time.time())}.md",
            mime="text/markdown",
        )

    except Exception as e:
        progress_bar.empty()
        st.error(f"Report generation failed: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()