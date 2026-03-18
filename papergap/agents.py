import json
import re
from typing import List, Dict, Any, Optional
from client import ask
from models import Gap, Subtopic, TrendPoint, AgentTrace, Paper, ResearchQuestion


def _extract_json_array(text: str) -> str:
    """Robustly extract the first valid JSON array from an LLM response.

    Handles: markdown code blocks, explanatory text before/after JSON,
    and nested brackets. Uses bracket-counting instead of fragile rfind.
    """
    # Strip markdown fences
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Find the opening bracket
    start = text.find('[')
    if start == -1:
        return '[]'

    # Walk forward counting brackets to find the matching close
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return '[]'


def _gap_score(subtopic: Subtopic) -> float:
    """Compute gap score: citation intensity per paper × inverse of recent supply.

    High score = many citations per paper (demand) but few recent papers (supply not growing).
    This is more meaningful than raw avg_citations which rewards already-established areas.
    """
    if not subtopic.papers:
        return 0.0
    recent_count = sum(1 for p in subtopic.papers if p.year >= 2024)
    citation_intensity = subtopic.avg_citations / max(subtopic.paper_count, 1)
    # Penalise subtopics with growing recent output (supply increasing → less of a gap)
    supply_gap_factor = 1.0 / (1 + recent_count)
    return citation_intensity * supply_gap_factor


_STOP_WORDS = {'and', 'or', 'the', 'in', 'of', 'for', 'to', 'a', 'an', 'with', 'using',
               'based', 'via', 'through', 'by', 'on', 'at', 'from', 'its', 'their'}


def _meaningful_words(text: str) -> set:
    return {w.lower() for w in text.split() if w.lower() not in _STOP_WORDS and len(w) > 2}


def _topic_overlap(gap_subtopic: str, paper_topic: str, threshold: float = 0.4) -> bool:
    """True if gap subtopic and paper topic share enough meaningful words.

    Used as fuzzy fallback when Nemotron generates a gap name that differs
    slightly from the exact OpenAlex topic string.
    """
    gap_words = _meaningful_words(gap_subtopic)
    topic_words = _meaningful_words(paper_topic)
    if not gap_words or not topic_words:
        return False
    overlap = len(gap_words & topic_words)
    return overlap / min(len(gap_words), len(topic_words)) >= threshold


def _is_relevant_to_search(subtopic_name: str, search_topic: str) -> bool:
    """True if subtopic shares at least one meaningful word with the search topic.

    Filters out subtopics from completely unrelated domains (e.g. 'Pulmonary
    Hypertension Research' when searching 'large language models').
    """
    return len(_meaningful_words(subtopic_name) & _meaningful_words(search_topic)) >= 1


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

    # Require at least 3 papers — single/dual-paper outliers inflate gap_score
    # without representing a real research cluster.
    filtered_subtopics = [s for s in subtopics if s.paper_count >= 3]
    if len(filtered_subtopics) < 5:
        filtered_subtopics = [s for s in subtopics if s.paper_count >= 2]
    if len(filtered_subtopics) < 5:
        filtered_subtopics = subtopics

    # Additionally filter by keyword relevance to the original search topic.
    # This removes subtopics from completely unrelated domains (e.g. "Pulmonary
    # Hypertension" when searching "large language models").
    if topic:
        relevant = [s for s in filtered_subtopics if _is_relevant_to_search(s.name, topic)]
        # Only apply relevance filter if it leaves enough candidates
        if len(relevant) >= 5:
            filtered_subtopics = relevant
            trace.log(f"Relevance filter: kept {len(relevant)}/{len(subtopics)} subtopics related to '{topic}'")

    # FIX 2: Sort by gap_score (citation intensity ÷ recent supply) instead of raw citations
    sorted_subtopics = sorted(filtered_subtopics, key=_gap_score, reverse=True)
    top_subtopics = sorted_subtopics[:15]

    # Build summary with gap_score so Nemotron can reason about demand vs supply
    subtopic_summary = "\n".join(
        f"- {s.name}: {s.paper_count} papers, "
        f"avg_citations={s.avg_citations:.0f}, "
        f"recent_2024={sum(1 for p in s.papers if p.year >= 2024)}, "
        f"gap_score={_gap_score(s):.1f}"
        for s in top_subtopics
    )

    # FIX 1: Build exact name list Nemotron must choose from
    allowed_names = "\n".join(f"  - {s.name}" for s in top_subtopics)

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

    prompt = f"""You are a research gap analyst. Your job is to find under-researched areas within a SPECIFIC domain.

DOMAIN: '{topic}'

gap_score = avg_citations_per_paper ÷ (1 + recent_2024_papers).
Higher gap_score = high citation demand but low recent publication supply = stronger gap.

Subtopics from papers about '{topic}' sorted by gap_score:
{subtopic_summary}

Year-by-year publication trends:
{trend_summary}

Semantically isolated papers (emerging directions):
{orphan_titles}

TASK: Choose the top 3 research gaps within '{topic}'.

STRICT RULES — violations will invalidate your response:
1. subtopic field MUST be copied EXACTLY (character for character) from this list:
{allowed_names}
2. Do NOT invent names, abbreviate, or rephrase subtopic names.
3. Every gap MUST be directly relevant to '{topic}'. If a subtopic is about an unrelated field, SKIP it.
4. Rank by gap_score (higher = better gap candidate).
5. Return ONLY a JSON array. No explanation text before or after.

Output format (replace values, keep keys exact):
[{{"subtopic":"exact name from list above","why_its_a_gap":"2 sentences explaining why demand exceeds supply in context of {topic}","citation_demand":0.0,"publication_supply":0}}]"""

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
            # Robust bracket-matching extraction
            cleaned = _extract_json_array(response)
            gaps_data = json.loads(cleaned)

            for gap_data in gaps_data:
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

            if gaps:
                trace.log(f"GAP DETECTION: found {len(gaps)} gaps")
                return gaps

            # JSON parsed but all entries were non-dicts — fall through to fallback
            trace.log("JSON valid but no dict entries found — using fallback")

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            trace.log(f"JSON parsing failed: {e}, using fallback")

    # ── Step 6: Fallback logic ────────────────────────────────────
    trace.log("Using fallback gap detection from gap_score ranking...")

    fallback_gaps = []
    for subtopic in top_subtopics[:3]:
        if subtopic.paper_count > 0:
            recent = sum(1 for p in subtopic.papers if p.year >= 2024)
            gap = Gap(
                subtopic=subtopic.name,
                why_its_a_gap=(
                    f"High citation intensity ({subtopic.avg_citations:.0f} avg citations, "
                    f"{subtopic.paper_count} papers) with only {recent} papers published in 2024. "
                    f"Demand outpaces recent supply, indicating an underserved research need."
                ),
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
        # Try exact match first (works when Nemotron followed the constraint)
        matching_papers = [
            p for p in all_papers
            if p.topics and p.topics[0] == gap.subtopic
        ]

        # FIX 1: Fuzzy word-overlap fallback across top 3 topics per paper
        if not matching_papers:
            matching_papers = [
                p for p in all_papers
                if p.topics and any(
                    _topic_overlap(gap.subtopic, t) for t in p.topics[:3]
                )
            ]
            if matching_papers:
                trace.log(f"  Fuzzy match found {len(matching_papers)} papers for '{gap.subtopic}'")

        # Last resort: use all papers — Nemotron still has full topic context in prompt
        if not matching_papers:
            matching_papers = all_papers
            trace.log(f"  No topic match — using all {len(all_papers)} papers as context")

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
                # FIX 6: Robust bracket-matching extraction
                cleaned = _extract_json_array(response)
                questions_data = json.loads(cleaned)

                # Create ResearchQuestion objects
                for i, q_data in enumerate(questions_data):
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
                    if i == 0 and q_data.get("shared_assumption"):
                        gap.shared_assumption = q_data.get("shared_assumption")

                if questions_for_gap:
                    trace.log(f"QUESTIONS: generated {len(questions_for_gap)} questions for '{gap.subtopic}'")
                    all_questions.extend(questions_for_gap)
                    continue  # skip fallback for this gap

                # JSON parsed but all entries were non-dicts — fall through to fallback
                trace.log(f"  No valid question dicts — using fallback for '{gap.subtopic}'")

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                trace.log(f"  JSON parsing failed: {e}, using fallback")

        # Fallback: derive a specific question from gap stats and paper titles
        paper_hint = top_papers[0].title if top_papers else gap.subtopic
        fallback_question = ResearchQuestion(
            gap=gap.subtopic,
            question=(
                f"Within the field of {topic}, how does {gap.subtopic.lower()} "
                f"affect outcomes — and what methodological approaches remain untested "
                f"given the current citation demand ({gap.citation_demand:.0f}) "
                f"versus publication supply ({gap.publication_supply})?"
            ),
            methodology="Systematic literature review followed by empirical validation study",
            foundational_papers=[p.title for p in top_papers[:3]],
            novelty_reason=(
                f"High citation intensity on '{paper_hint[:60]}' signals strong community "
                f"interest in {gap.subtopic} within {topic}, yet recent output is low."
            ),
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

            # FIX 5: Use meaningful keywords from dominant subtopic, not just first word
            _stop = {'and', 'or', 'the', 'in', 'of', 'for', 'to', 'a', 'an', 'with', 'using'}
            key_words = [w for w in dominant.name.split() if w.lower() not in _stop][:3]
            specific_topic = f"{topic} {' '.join(key_words)}"
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
