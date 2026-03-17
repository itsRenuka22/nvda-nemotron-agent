#!/usr/bin/env python3
"""
PaperGap Streamlit Application
Identifies research gaps in academic literature and generates novel research questions
"""

import sys
import os
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent/papergap')
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent')

import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Import PaperGap pipeline
from agents import run_pipeline
from models import AgentTrace

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperGap",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Title & Description
# ─────────────────────────────────────────────────────────────────────────────

st.title("PaperGap — Research Gap Identification Agent")
st.caption("Powered by NVIDIA Nemotron · Real papers from OpenAlex (250M+ works) · No API key required")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Research Parameters")

    # Topic input
    topic = st.text_input(
        "Research Topic",
        value="federated learning rare disease diagnosis",
        placeholder="Enter your research topic..."
    )

    # Year range selector
    year_range = st.selectbox(
        "Publication Years",
        ["2022-2025", "2021-2025", "2023-2025"],
        index=0
    )

    # Demo topic quick-picks
    st.markdown("**Quick Pick Demo Topics:**")
    cols = st.columns(3)

    demo_topics = [
        "federated learning\nrare disease\ndiagnosis",
        "large language\nmodels clinical\ndecision support",
        "graph neural\nnetworks drug\ninteraction"
    ]
    demo_full = [
        "federated learning rare disease diagnosis",
        "large language models clinical decision support",
        "graph neural networks drug interaction prediction"
    ]

    for idx, col in enumerate(cols):
        if col.button(demo_topics[idx], key=f"demo_{idx}", use_container_width=True):
            topic = demo_full[idx]
            # Clear old results when topic changes
            st.session_state.pipeline_result = None
            st.rerun()

    # Main action button
    st.divider()
    run_pipeline_btn = st.button(
        "🔍 Find Research Gaps",
        key="run_pipeline",
        type="primary",
        disabled=not topic.strip(),
        use_container_width=True
    )

    st.divider()
    st.info(
        "**OpenAlex Data Source**\n"
        "• 250M+ scholarly works\n"
        "• CC0 license\n"
        "• No authentication required"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Session State
# ─────────────────────────────────────────────────────────────────────────────

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "last_topic_analyzed" not in st.session_state:
    st.session_state.last_topic_analyzed = None

# Clear results if user changed the topic
if topic.strip() != st.session_state.last_topic_analyzed:
    st.session_state.pipeline_result = None

# ─────────────────────────────────────────────────────────────────────────────
# Main Content Area
# ─────────────────────────────────────────────────────────────────────────────

# Handle pipeline execution
if run_pipeline_btn:
    st.session_state.is_running = True

    with st.spinner(f"🔄 Analyzing research gaps in '{topic}' ({year_range})..."):
        try:
            result = run_pipeline(topic)
            st.session_state.pipeline_result = result
            st.session_state.last_topic_analyzed = topic.strip()  # Track analyzed topic
            st.session_state.is_running = False
            st.success("✓ Pipeline completed successfully!")
        except Exception as e:
            st.session_state.is_running = False
            st.error(f"Error running pipeline: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: AGENT TRACE (Execution Log)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result

    with st.expander("📋 Agent Trace — Execution Log", expanded=False):
        trace = result.get('trace')
        if trace and hasattr(trace, 'steps'):
            for log in trace.steps:
                st.text(log)
        else:
            st.text("No trace logs available")

    # ─────────────────────────────────────────────────────────────────────────────
    # Section 2: Publication Landscape & Identified Gaps (Two Columns)
    # ─────────────────────────────────────────────────────────────────────────────

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Publication Landscape")

        # Bar chart: subtopics by paper count
        subtopics = result.get('subtopics', [])
        if subtopics:
            subtopic_names = [s.name for s in subtopics[:10]]
            paper_counts = [s.paper_count for s in subtopics[:10]]

            fig = go.Figure(data=[
                go.Bar(
                    x=subtopic_names,
                    y=paper_counts,
                    marker=dict(color='#1f77b4'),
                    text=paper_counts,
                    textposition='auto',
                )
            ])

            fig.update_layout(
                title="Papers by Subtopic (Top 10)",
                xaxis_title="Subtopic",
                yaxis_title="Number of Papers",
                height=400,
                showlegend=False,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Subtopic details table
            st.caption("Subtopic Statistics")
            subtopic_data = []
            for s in subtopics[:5]:
                subtopic_data.append({
                    "Subtopic": s.name,
                    "Papers": s.paper_count,
                    "Avg Citations": f"{s.avg_citations:.1f}",
                    "% of Total": f"{s.pct_of_total:.1f}%"
                })
            st.dataframe(subtopic_data, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("🎯 Identified Research Gaps")

        gaps = result.get('gaps', [])
        if gaps:
            for i, gap in enumerate(gaps, 1):
                with st.container(border=True):
                    st.markdown(f"**Gap {i}: {gap.subtopic}**")
                    st.metric(
                        "Citation Demand",
                        f"{gap.citation_demand:.0f}",
                        delta=None
                    )
                    st.metric(
                        "Publication Supply",
                        f"{gap.publication_supply}",
                        delta=None
                    )
                    st.write(f"**Why it's a gap:**")
                    st.caption(gap.why_its_a_gap)

                    if gap.shared_assumption:
                        st.write(f"**Shared Assumption:**")
                        st.caption(gap.shared_assumption)
        else:
            st.info("No gaps identified yet. Try a different topic.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Section 3: Research Questions
    # ─────────────────────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("💡 Generated Research Questions")

    questions = result.get('questions', [])
    if questions:
        for i, q in enumerate(questions, 1):
            with st.expander(
                f"**Q{i}:** {q.question[:80]}...",
                expanded=(i == 1)
            ):
                st.markdown(f"**Research Gap:** {q.gap}")
                st.markdown(f"**Full Question:** {q.question}")
                st.markdown(f"**Proposed Methodology:** {q.methodology}")
                st.markdown(f"**Why it's Novel:** {q.novelty_reason}")

                if q.foundational_papers:
                    st.markdown("**Foundational Papers:**")
                    for paper in q.foundational_papers[:3]:
                        st.caption(f"• {paper}")
    else:
        st.info("No research questions generated yet.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Section 4: Footer Metrics
    # ─────────────────────────────────────────────────────────────────────────────

    st.divider()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Papers Analyzed", result.get('paper_count', 0))

    with col2:
        st.metric("Subtopics Found", len(result.get('subtopics', [])))

    with col3:
        st.metric("Gaps Identified", len(result.get('gaps', [])))

    with col4:
        st.metric("Questions Generated", len(result.get('questions', [])))

    with col5:
        drilled = "Yes" if result.get('drilled_deeper', False) else "No"
        st.metric("Autonomous Drill-Down", drilled)

    if result.get('drilled_deeper') and result.get('drill_reason'):
        st.info(f"**Autonomous Analysis:** {result.get('drill_reason')}")

else:
    # Initial placeholder state
    st.info(
        "👋 Welcome to PaperGap!\n\n"
        "**How it works:**\n"
        "1. Enter a research topic in the sidebar\n"
        "2. Click 'Find Research Gaps' to analyze\n"
        "3. Review identified gaps and generated research questions\n\n"
        "Use the 'Quick Pick' buttons to try demo topics on healthcare + AI themes."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    PaperGap v1.0 · NVIDIA Nemotron LLM · OpenAlex API · Hackathon 2025
    </div>
    """,
    unsafe_allow_html=True
)
