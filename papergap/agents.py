import json
from typing import List, Dict, Any, Optional
from client import ask
from models import Gap, Subtopic, TrendPoint, AgentTrace, Paper, ResearchQuestion


def gap_detection_agent(
    subtopics: List[Subtopic],
    trends: List[TrendPoint],
    semantic_result: Dict[str, Any],
    trace: AgentTrace,
    topic: str = ""
) -> List[Gap]:
    """Detect research gaps using Nemotron analysis.

    Args:
        subtopics: List of Subtopic objects with paper counts and citation stats
        trends: List of TrendPoint objects showing publication trends
        semantic_result: Dict with "orphans" key containing list of Paper objects
        trace: AgentTrace object for logging
        topic: Original search topic for domain-relevant gap analysis

    Returns:
        List of Gap objects identified by Nemotron analysis
    """

    # ── Step 1: Build subtopic summary ──────────────────────────────
    trace.log("Building subtopic summary...")

    # Filter out single-paper outliers that would skew gap detection.
    # A subtopic with only 1 paper but 3000+ citations is just a stray
    # high-impact paper from another field, not a real gap.
    filtered_subtopics = [
        s for s in subtopics
        if s.paper_count >= 2 or s.avg_citations < 500
    ]

    # Fall back to all subtopics if filtering removed too many
    if len(filtered_subtopics) < 5:
        filtered_subtopics = subtopics

    # Sort by paper_count descending to surface the most represented areas
    sorted_subtopics = sorted(filtered_subtopics, key=lambda s: s.paper_count, reverse=True)

    subtopic_summary = "\n".join(
        f"- {s.name}: {s.paper_count} papers, avg {s.avg_citations:.0f} citations ({s.pct_of_total:.1f}%)"
        for s in sorted_subtopics[:15]  # Top 15 by paper count
    )

    # ── Step 2: Build trend summary ────────────────────────────────
    trace.log("Building trend summary...")

    # Group TrendPoints by subtopic
    trends_by_subtopic: Dict[str, Dict[int, int]] = {}
    for tp in trends:
        if tp.subtopic not in trends_by_subtopic:
            trends_by_subtopic[tp.subtopic] = {}
        trends_by_subtopic[tp.subtopic][tp.year] = tp.count

    trend_summary = ""
    for subtopic_name in list(trends_by_subtopic.keys())[:5]:  # Top 5 by frequency
        years = sorted(trends_by_subtopic[subtopic_name].keys())
        trend_line = f"- {subtopic_name}: " + ", ".join(
            f"{year}({trends_by_subtopic[subtopic_name][year]})"
            for year in years
        )
        trend_summary += trend_line + "\n"

    # ── Step 3: List orphan paper titles ───────────────────────────
    trace.log("Collecting orphan papers...")

    orphans = semantic_result.get("orphans", [])
    orphan_titles = "\n".join(f"- {p.title}" for p in orphans[:10])  # Top 10 orphans

    if not orphan_titles:
        orphan_titles = "(No semantically isolated papers detected)"

    # ── Step 4: Prompt Nemotron ───────────────────────────────────
    trace.log("Calling Nemotron for gap analysis...")

    topic_context = f"Research domain: '{topic}'\n\n" if topic else ""

    prompt = f"""You are a research gap analyst specializing in '{topic}'.

{topic_context}These are the subtopics found in papers about '{topic}':
{subtopic_summary}

Year-by-year publication trends:
{trend_summary}

Semantically isolated papers (emerging directions within '{topic}'):
{orphan_titles}

IMPORTANT: Identify the top 3 research gaps WITHIN the domain of '{topic}'.
- Only report gaps directly relevant to '{topic}'
- Ignore subtopics that are clearly from unrelated fields
- A gap = high citation demand (many papers cite this area) but low publication supply (few papers exist)
- For each gap explain WHY it is under-researched in 2 sentences

Return ONLY valid JSON (no markdown):
[{{"subtopic":"name","why_its_a_gap":"explanation","citation_demand":0.0,"publication_supply":0}}]"""

    try:
        response = ask(prompt, reasoning=True)
        trace.log(f"Nemotron response received ({len(response)} chars)")
    except Exception as e:
        trace.log(f"Error calling Nemotron: {e}")
        response = None

    # ── Step 5: Parse JSON response ────────────────────────────────
    trace.log("Parsing gap analysis response...")

    gaps = []

    if response:
        try:
            # Strip markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            # If still not valid JSON, try to extract JSON array from response
            if not cleaned.startswith("["):
                # Look for [ ... ] pattern in response
                start_idx = cleaned.find("[")
                end_idx = cleaned.rfind("]")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned = cleaned[start_idx:end_idx + 1]

            # Parse JSON
            gaps_data = json.loads(cleaned)

            # Create Gap objects
            for gap_data in gaps_data:
                # Validate gap_data is a dict before processing
                if not isinstance(gap_data, dict):
                    trace.log(f"Skipping invalid gap entry (not a dict): {type(gap_data)}")
                    continue

                gap = Gap(
                    subtopic=gap_data.get("subtopic", "Unknown"),
                    why_its_a_gap=gap_data.get("why_its_a_gap", ""),
                    citation_demand=float(gap_data.get("citation_demand", 0)),
                    publication_supply=int(gap_data.get("publication_supply", 0)),
                    orphan_papers=[p.title for p in orphans],
                )
                gaps.append(gap)

            trace.log(f"GAP DETECTION: found {len(gaps)} gaps")
            return gaps

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            trace.log(f"JSON parsing failed: {e}, using fallback")

    # ── Step 6: Fallback logic ────────────────────────────────────
    trace.log("Using fallback gap detection from citation/supply ratio...")

    # Create gaps from top subtopics by citation/paper ratio
    fallback_gaps = []

    for subtopic in sorted_subtopics[:3]:
        if subtopic.paper_count > 0:
            ratio = subtopic.avg_citations / subtopic.paper_count
            gap = Gap(
                subtopic=subtopic.name,
                why_its_a_gap=f"High citation demand ({subtopic.avg_citations:.0f} avg) with limited publications ({subtopic.paper_count} papers). Research in this area is valued but under-explored.",
                citation_demand=subtopic.avg_citations,
                publication_supply=subtopic.paper_count,
                orphan_papers=[p.title for p in orphans],
            )
            fallback_gaps.append(gap)

    trace.log(f"GAP DETECTION: found {len(fallback_gaps)} gaps (fallback)")
    return fallback_gaps


def question_generation_agent(
    gaps: List[Gap],
    all_papers: List[Paper],
    trace: AgentTrace,
    topic: str = ""
) -> List[ResearchQuestion]:
    """Generate research questions for identified gaps.

    Args:
        gaps: List of Gap objects from gap_detection_agent
        all_papers: All Paper objects fetched from OpenAlex
        trace: AgentTrace object for logging
        topic: Original search topic for context-aware question generation

    Returns:
        List of ResearchQuestion objects
    """

    all_questions = []

    for gap in gaps:
        trace.log(f"Generating questions for gap: {gap.subtopic}")

        # ── Step 1: Find papers matching gap.subtopic ──────────────
        matching_papers = [
            p for p in all_papers
            if p.topics and p.topics[0] == gap.subtopic
        ]

        # Sort by citations descending, take top 8
        matching_papers.sort(key=lambda p: p.citations, reverse=True)
        top_papers = matching_papers[:8]

        trace.log(f"  Found {len(matching_papers)} papers, using top {len(top_papers)}")

        # ── Step 2: Build paper list for prompt ────────────────────
        paper_list = ""
        for i, paper in enumerate(top_papers, 1):
            # First 300 chars of abstract
            abstract_snippet = paper.abstract[:300] if paper.abstract else "(No abstract)"
            paper_list += f"{i}. {paper.title}\n   {abstract_snippet}...\n\n"

        if not paper_list:
            paper_list = "(No papers found in this subtopic)"

        # ── Step 3: Prompt Nemotron ───────────────────────────────
        trace.log(f"  Calling Nemotron for question generation...")

        topic_context = f"Overall research domain: '{topic}'\n" if topic else ""

        prompt = f"""You are a research strategist analyzing a gap in the field of '{topic}'.
{topic_context}
Specific gap identified: '{gap.subtopic}'
Why this is a gap: {gap.why_its_a_gap}

Existing papers in this gap area:
{paper_list}

Reason step by step:
1. What assumption do ALL papers share that nobody has questioned?
2. What would be TRUE if this assumption is WRONG?
3. Generate 2 specific, falsifiable research questions that:
   - Are directly relevant to '{topic}' and the gap '{gap.subtopic}'
   - Are NOT variations of existing work
   - Address the shared assumption you found
   - Name a specific methodology
   - Are completable in a PhD thesis scope

Cite WHICH paper reveals this assumption.

Return ONLY valid JSON:
[{{"question":"str","methodology":"str","foundational_papers":["title"],"novelty_reason":"str","shared_assumption":"str"}}]"""

        try:
            response = ask(prompt, reasoning=True)
            trace.log(f"  Nemotron response received ({len(response)} chars)")
        except Exception as e:
            trace.log(f"  Error calling Nemotron: {e}")
            response = None

        # ── Step 4: Parse JSON response ────────────────────────────
        trace.log(f"  Parsing question generation response...")

        questions_for_gap = []

        if response:
            try:
                # Strip markdown code blocks if present
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()

                # If still not valid JSON, try to extract JSON array from response
                if not cleaned.startswith("["):
                    # Look for [ ... ] pattern in response
                    start_idx = cleaned.find("[")
                    end_idx = cleaned.rfind("]")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        cleaned = cleaned[start_idx:end_idx + 1]

                # Parse JSON
                questions_data = json.loads(cleaned)

                # Create ResearchQuestion objects
                for i, q_data in enumerate(questions_data):
                    # Validate q_data is a dict before processing
                    if not isinstance(q_data, dict):
                        trace.log(f"  Skipping invalid question entry (not a dict): {type(q_data)}")
                        continue

                    question = ResearchQuestion(
                        gap=gap.subtopic,
                        question=q_data.get("question", ""),
                        methodology=q_data.get("methodology", ""),
                        foundational_papers=q_data.get("foundational_papers", []),
                        novelty_reason=q_data.get("novelty_reason", ""),
                    )
                    questions_for_gap.append(question)

                    # Update gap.shared_assumption from first question
                    if i == 0 and q_data.get("shared_assumption"):
                        gap.shared_assumption = q_data.get("shared_assumption")

                trace.log(f"QUESTIONS: generated {len(questions_for_gap)} questions for '{gap.subtopic}'")
                all_questions.extend(questions_for_gap)

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                trace.log(f"  JSON parsing failed: {e}, using fallback")

                # Fallback: create basic questions from gap info
                fallback_question = ResearchQuestion(
                    gap=gap.subtopic,
                    question=f"How can we improve outcomes in {gap.subtopic} by challenging existing assumptions about {gap.subtopic.lower()}?",
                    methodology="Systematic literature review followed by empirical validation",
                    foundational_papers=[p.title for p in top_papers[:3]],
                    novelty_reason=f"Addresses the gap where citation demand ({gap.citation_demand:.0f}) far exceeds publication supply ({gap.publication_supply})",
                )
                all_questions.append(fallback_question)
                trace.log(f"QUESTIONS: generated 1 question for '{gap.subtopic}' (fallback)")

    return all_questions


def run_pipeline(topic: str, trace: Optional[AgentTrace] = None) -> Dict[str, Any]:
    """Run the complete PaperGap research gap analysis pipeline.

    Orchestrates: fetch → cluster → (optional drill-down) → trends →
    semantic clustering → gap detection → question generation

    Args:
        topic: Research topic to analyze
        trace: Optional AgentTrace for logging (creates new if not provided)

    Returns:
        Dictionary with keys:
            - subtopics: List of Subtopic objects
            - trends: List of TrendPoint objects
            - gaps: List of Gap objects
            - questions: List of ResearchQuestion objects
            - trace: AgentTrace with all logs
            - drilled_deeper: Boolean indicating if autonomous drill-down occurred
            - drill_reason: String explaining why drill-down happened
            - paper_count: Number of papers analyzed
            - topic_used: Final topic used (may differ from input if drilled deeper)
    """

    # ── Step 1: Create trace if not provided ────────────────────
    if trace is None:
        trace = AgentTrace()

    # ── Step 2: Log pipeline start ──────────────────────────────
    trace.log(f"=== PaperGap starting for '{topic}' ===")

    # ── Step 3: Import tools inside function (avoid circular imports)
    from tools import fetch_papers, fetch_trends, cluster_by_topic, semantic_cluster

    # ── Step 4: Fetch papers and cluster by topic ───────────────
    trace.log(f"Phase 1: Fetching papers for '{topic}'...")
    papers = fetch_papers(topic, trace)
    trace.log(f"Phase 1: Clustering {len(papers)} papers by topic...")
    subtopics = cluster_by_topic(papers, trace)

    # ── Step 6: Autonomous drill-down (if one subtopic dominates) ─
    drilled_deeper = False
    drill_reason = ""
    topic_used = topic

    total_papers = sum(s.paper_count for s in subtopics)
    if total_papers > 0:
        dominant = max(subtopics, key=lambda s: s.paper_count)
        dominant_pct = (dominant.paper_count / total_papers) * 100

        if dominant_pct > 50:
            trace.log(
                f"AUTONOMOUS: '{dominant.name}' dominates {dominant_pct:.0f}% — drilling deeper..."
            )

            # Drill down with more specific topic
            specific_topic = f"{topic} {dominant.name.split()[0]}"
            trace.log(f"Phase 1b: Fetching papers for '{specific_topic}'...")
            sub_papers = fetch_papers(specific_topic, trace, limit=100)

            trace.log(f"Phase 1b: Clustering {len(sub_papers)} papers by topic...")
            sub_subtopics = cluster_by_topic(sub_papers, trace)

            # Use drilled results
            subtopics = sub_subtopics
            papers = sub_papers
            drilled_deeper = True
            drill_reason = f"'{dominant.name}' dominated {dominant_pct:.0f}%"
            topic_used = specific_topic

            trace.log(f"AUTONOMOUS: Drill-down complete. Now have {len(papers)} papers in {len(subtopics)} subtopics")

    # ── Step 7: Fetch trends ────────────────────────────────────
    trace.log("Phase 2: Fetching publication trends...")
    trends = fetch_trends(topic_used, trace)

    # ── Step 8: Semantic clustering ─────────────────────────────
    trace.log("Phase 3: Performing semantic clustering...")
    semantic_result = semantic_cluster(papers, trace=trace)

    # ── Step 9: Gap detection ───────────────────────────────────
    trace.log("Phase 4: Detecting research gaps...")
    gaps = gap_detection_agent(subtopics, trends, semantic_result, trace, topic=topic_used)

    # ── Step 10: Question generation ────────────────────────────
    trace.log("Phase 5: Generating research questions...")
    questions = question_generation_agent(gaps, papers, trace, topic=topic_used)

    # ── Step 11: Log completion ─────────────────────────────────
    trace.log(f"=== PaperGap complete ===")
    trace.log(
        f"Results: {len(papers)} papers → {len(subtopics)} subtopics → "
        f"{len(gaps)} gaps → {len(questions)} questions"
    )

    # ── Step 12: Return results ─────────────────────────────────
    return {
        "subtopics": subtopics,
        "trends": trends,
        "gaps": gaps,
        "questions": questions,
        "trace": trace,
        "drilled_deeper": drilled_deeper,
        "drill_reason": drill_reason,
        "paper_count": len(papers),
        "topic_used": topic_used,
    }
