#!/usr/bin/env python3
"""
PaperGap — Research Gap Identification Agent
"""

import sys
import os

# Add papergap/ to path — all core modules (agents, tools, models, client) live there
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'papergap'))

import streamlit as st
import plotly.graph_objects as go

from agents import run_pipeline
from models import AgentTrace

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperGap",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — fix sidebar width, card styles, thinking panel
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Fix sidebar to a constant width */
[data-testid="stSidebar"] {
    min-width: 380px !important;
    max-width: 380px !important;
}

/* Gap cards equal height feel */
.gap-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

/* Thinking panel */
.thinking-step {
    font-size: 0.92rem;
    color: #aaa;
    padding: 2px 0;
}
.thinking-step.active {
    color: #fff;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Live-progress trace — updates Streamlit placeholders as phases run
# ─────────────────────────────────────────────────────────────────────────────

class _StreamlitTrace(AgentTrace):
    """AgentTrace that mirrors phase transitions to Streamlit UI elements."""

    _PHASES = [
        ("Phase 1: Fetching",   "📡 Fetching papers from OpenAlex...",           10),
        ("Phase 1c",            "🔬 Analysing cluster signals...",                 28),
        ("Phase 2",             "📈 Loading publication trends...",               42),
        ("Phase 3",             "🧠 Building semantic clusters...",               56),
        ("Phase 4",             "🎯 Asking AI to identify research gaps...",      72),
        ("Phase 5: Generating", "💡 Generating research questions...",            86),
        ("Phase 5b",            "✨ Rewriting questions in plain English...",      95),
        ("=== PaperGap complete", "✅ Done!",                                    100),
    ]

    def __init__(self, status_el, bar_el):
        super().__init__()
        self._status = status_el
        self._bar = bar_el

    def log(self, msg: str):
        super().log(msg)
        for prefix, label, pct in self._PHASES:
            if prefix in msg:
                self._status.markdown(f"*{label}*")
                self._bar.progress(pct)
                break

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────

if "topic_input" not in st.session_state:
    st.session_state.topic_input = "federated learning rare disease diagnosis"
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "last_topic_analyzed" not in st.session_state:
    st.session_state.last_topic_analyzed = None

# Apply any pending demo-button selection BEFORE the text_input widget is
# rendered — Streamlit forbids writing to a keyed widget's state after it
# has been instantiated in the same script run.
if "_pending_topic" in st.session_state:
    st.session_state.topic_input = st.session_state.pop("_pending_topic")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Research Parameters")

    # Topic input — keyed to session state so demo buttons can update it
    topic = st.text_input(
        "Research Topic",
        key="topic_input",
        placeholder="Enter your research topic...",
    )

    year_range = st.selectbox(
        "Publication Years",
        ["2022-2025", "2021-2025", "2023-2025"],
        index=0,
    )

    # Demo quick-picks — clicking fills the topic box
    st.markdown("**Quick Pick Demo Topics:**")
    demo_labels = [
        "federated\nlearning rare\ndisease diagnosis",
        "large language\nmodels clinical\ndecision support",
        "graph neural\nnetworks drug\ninteraction",
    ]
    demo_values = [
        "federated learning rare disease diagnosis",
        "large language models clinical decision support",
        "graph neural networks drug interaction prediction",
    ]
    cols = st.columns(3)
    for idx, col in enumerate(cols):
        if col.button(demo_labels[idx], key=f"demo_{idx}", use_container_width=True):
            st.session_state._pending_topic = demo_values[idx]
            st.session_state.pipeline_result = None
            st.session_state.last_topic_analyzed = None
            st.rerun()

    st.divider()
    run_btn = st.button(
        "🔍 Find Research Gaps",
        key="run_pipeline",
        type="primary",
        disabled=not topic.strip(),
        use_container_width=True,
    )

    st.divider()
    st.info(
        "**OpenAlex Data Source**\n"
        "• 250M+ scholarly works\n"
        "• CC0 license\n"
        "• No authentication required"
    )

# Clear stale results if topic changed since last run
if topic.strip() != st.session_state.last_topic_analyzed:
    st.session_state.pipeline_result = None

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("PaperGap — Research Gap Identification Agent")
st.caption("Powered by NVIDIA Nemotron · Real papers from OpenAlex (250M+ works)")

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution with live thinking display
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    thinking_box = st.container(border=True)
    with thinking_box:
        st.markdown("**🤔 Thinking...**")
        status_el  = st.empty()
        bar_el     = st.empty()
        bar_el.progress(0)
        status_el.markdown("*Starting up...*")

    trace = _StreamlitTrace(status_el, bar_el)

    try:
        result = run_pipeline(topic, trace=trace)
        st.session_state.pipeline_result = result
        st.session_state.last_topic_analyzed = topic.strip()
        # Replace the thinking box with a success message
        thinking_box.empty()
        st.success("✓ Analysis complete!")
    except Exception as e:
        thinking_box.empty()
        st.error(f"Pipeline error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result

    # ── Agent trace (collapsed by default) ───────────────────────────────────
    with st.expander("📋 Agent Trace — Execution Log", expanded=False):
        trace = result.get("trace")
        if trace and hasattr(trace, "steps"):
            for log in trace.steps:
                st.text(log)
        else:
            st.text("No trace logs available.")

    # ── Publication Landscape ─────────────────────────────────────────────────
    st.subheader("📊 Publication Landscape")
    subtopics = result.get("subtopics", [])
    if subtopics:
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            names  = [s.name for s in subtopics[:10]]
            counts = [s.paper_count for s in subtopics[:10]]
            fig = go.Figure(data=[go.Bar(
                x=names, y=counts,
                marker_color="#4C9BE8",
                text=counts, textposition="auto",
            )])
            fig.update_layout(
                title="Papers by Subtopic (Top 10)",
                xaxis_title="Subtopic",
                yaxis_title="Number of Papers",
                height=360,
                showlegend=False,
                margin=dict(t=40, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.caption("Subtopic breakdown")
            st.dataframe(
                [{"Subtopic": s.name, "Papers": s.paper_count,
                  "Avg Citations": f"{s.avg_citations:.1f}",
                  "% of Total": f"{s.pct_of_total:.1f}%"}
                 for s in subtopics[:6]],
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # ── Research Gaps — horizontal cards ─────────────────────────────────────
    st.subheader("🎯 Identified Research Gaps")

    with st.expander("ℹ️ What do these numbers mean?", expanded=False):
        st.markdown(
            "**Avg Citations per Paper** — how often papers in this area are cited by other "
            "researchers. High = the community finds this topic important.\n\n"
            "**Papers in this area** — total papers found. When citations are high but paper "
            "count is low (and few published recently), it signals a gap: demand exists but "
            "supply hasn't caught up."
        )

    gaps = result.get("gaps", [])
    if gaps:
        gap_cols = st.columns(len(gaps))
        for i, (gap, col) in enumerate(zip(gaps, gap_cols), 1):
            with col:
                with st.container(border=True):
                    st.markdown(f"**Gap {i}**")
                    st.markdown(f"##### {gap.subtopic}")
                    st.divider()

                    m1, m2 = st.columns(2)
                    m1.metric("Avg Citations", f"{gap.citation_demand:.0f}")
                    m2.metric("Papers", f"{gap.publication_supply}")

                    st.markdown("**Why it's a gap:**")
                    st.write(gap.why_its_a_gap)

                    if gap.top_papers:
                        st.markdown("**Key papers:**")
                        for t in gap.top_papers[:2]:
                            st.caption(f"• {t}")
    else:
        st.info("No gaps identified. Try a different or more specific topic.")

    st.divider()

    # ── Research Questions ────────────────────────────────────────────────────
    st.subheader("💡 Generated Research Questions")

    questions = result.get("questions", [])
    if questions:
        for i, q in enumerate(questions, 1):
            with st.expander(f"**Q{i}:** {q.question[:90]}{'...' if len(q.question) > 90 else ''}", expanded=(i == 1)):
                st.markdown(f"**Research Gap:** {q.gap}")
                st.markdown(f"**Full Question:** {q.question}")
                st.markdown(f"**Suggested Approach:** {q.methodology}")
                st.markdown(f"**Why it's new:** {q.novelty_reason}")
                if q.foundational_papers:
                    st.markdown("**Foundational Papers:**")
                    for p in q.foundational_papers[:3]:
                        st.caption(f"• {p}")
    else:
        st.info("No research questions generated yet.")

    st.divider()

    # ── Footer metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Papers Analyzed",    result.get("paper_count", 0))
    c2.metric("Subtopics Found",    len(result.get("subtopics", [])))
    c3.metric("Gaps Identified",    len(result.get("gaps", [])))
    c4.metric("Questions Generated",len(result.get("questions", [])))
    c5.metric("Drill-Down",         "Yes" if result.get("drilled_deeper") else "No")

    if result.get("drilled_deeper") and result.get("drill_reason"):
        st.info(f"**Autonomous drill-down:** {result.get('drill_reason')}")

else:
    st.info(
        "👋 **Welcome to PaperGap!**\n\n"
        "1. Enter a research topic in the sidebar (or click a Quick Pick)\n"
        "2. Click **Find Research Gaps**\n"
        "3. Watch the AI analyse papers and surface what's missing\n\n"
        "Results include identified gaps, the papers behind them, and concrete research questions."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.85em;'>"
    "PaperGap · NVIDIA Nemotron · OpenAlex API · NVIDIA GTC Hackathon 2025"
    "</div>",
    unsafe_allow_html=True,
)
