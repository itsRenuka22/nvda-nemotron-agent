import json
import re
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
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

    Filters out subtopics from completely unrelated domains (e.g. 'RAFT in
    Distributed Systems' when searching 'liposomes nanocarriers biomedical').
    """
    return len(_meaningful_words(subtopic_name) & _meaningful_words(search_topic)) >= 1


def _subtopic_papers_match_domain(subtopic: "Subtopic", search_topic: str) -> bool:
    """Check if a subtopic's PAPERS are actually about the search domain.

    Uses paper titles as ground truth — an OpenAlex topic label like
    'RAFT in Distributed Systems' might be assigned to a chemistry paper by
    mistake, but the paper title will reveal the true domain.

    Returns True if at least 1 paper title shares a meaningful word (prefix match)
    with the search topic.
    """
    search_words = _meaningful_words(search_topic)
    if not search_words or not subtopic.papers:
        return True  # can't tell → keep it

    for paper in subtopic.papers[:10]:  # check up to 10 papers
        title_words = _meaningful_words(paper.title)
        # Prefix match: "liposome" matches search word "liposomes", etc.
        for tw in title_words:
            for sw in search_words:
                if tw.startswith(sw[:6]) or sw.startswith(tw[:6]):
                    return True
    return False


def gap_detection_agent(
    subtopics: List[Subtopic],
    trends: List[TrendPoint],
    semantic_result: Dict[str, Any],
    trace: AgentTrace,
    topic: str = "",
    enrichment: Optional[Dict[str, Any]] = None,
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

    # Filter by domain relevance to the original search topic.
    # Two-pass: first try label-based keyword match, then fall back to checking
    # whether the subtopic's actual PAPER TITLES match the search domain.
    # This catches cases like "RAFT in Distributed Systems" (label looks wrong)
    # being assigned to liposome chemistry papers by OpenAlex.
    if topic:
        # Pass 1: subtopic label shares a keyword with the search topic
        label_match = [s for s in filtered_subtopics if _is_relevant_to_search(s.name, topic)]
        # Pass 2: for subtopics whose label didn't match, check their paper titles
        label_no_match = [s for s in filtered_subtopics if s not in label_match]
        paper_match = [s for s in label_no_match if _subtopic_papers_match_domain(s, topic)]
        relevant = label_match + paper_match

        # Remove subtopics where NEITHER label nor papers match the domain
        off_domain = [s for s in label_no_match if s not in paper_match]
        if off_domain:
            trace.log(
                f"Relevance filter: removed {len(off_domain)} off-domain subtopics: "
                + ", ".join(s.name for s in off_domain[:5])
            )

        if len(relevant) >= 2:
            filtered_subtopics = relevant
            trace.log(f"Relevance filter: kept {len(relevant)}/{len(subtopics)} subtopics related to '{topic}'")
        elif len(relevant) == 1:
            others = [s for s in filtered_subtopics if s not in relevant]
            filtered_subtopics = relevant + sorted(others, key=_gap_score, reverse=True)[:3]
            trace.log(f"Relevance filter: only 1 match — padded with {len(filtered_subtopics)-1} by gap_score")
        else:
            trace.log(f"Relevance filter: 0 matches for '{topic}' — using gap_score ranking only")

    # FIX 2: Sort by gap_score (citation intensity ÷ recent supply) instead of raw citations
    sorted_subtopics = sorted(filtered_subtopics, key=_gap_score, reverse=True)
    top_subtopics = sorted_subtopics[:15]

    _enrich = enrichment or {}

    # Build per-subtopic summary block including enrichment signals when available
    subtopic_blocks: List[str] = []
    for s in top_subtopics:
        score = _gap_score(s)
        recent_2024 = sum(1 for p in s.papers if p.year >= 2024)
        sig = _enrich.get(s.name, {})

        lines = [
            f"Subtopic: {s.name}",
            f"  papers={s.paper_count}, avg_citations={s.avg_citations:.0f}, "
            f"recent_2024={recent_2024}, gap_score={score:.2f} "
            f"(citations ÷ recent papers — higher = more demand)",
        ]

        # Signal 1 — explicit gap sentences from abstracts
        gap_count = sig.get("explicit_gap_count", 0)
        gap_sentences = sig.get("explicit_gap_sentences", [])
        lines.append(
            f"  explicit_gap_count: {gap_count} papers in this cluster state it as unsolved"
        )
        for qs in gap_sentences[:2]:
            lines.append(f"    \"{qs[:120]}\"")

        # Signal 2 — citation frontier
        frontier_flag = sig.get("citation_frontier_flag", False)
        frontier_papers = sig.get("frontier_papers", [])
        frontier_label = "Yes" if frontier_flag else "No"
        lines.append(
            f"  citation_frontier: {frontier_label} — "
            f"{len(frontier_papers)} foundational papers with no follow-up since 2022"
        )
        for fp in frontier_papers[:2]:
            lines.append(f"    \"{fp['title'][:80]}\" ({fp['year']})")

        # Signal 3 — concept isolation
        iso_score = sig.get("concept_isolation_score", 0.0)
        iso_concepts = sig.get("isolated_concepts", [])
        concept_str = ", ".join(iso_concepts[:5]) if iso_concepts else "none"
        lines.append(
            f"  concept_isolation: {iso_score:.2f} — "
            f"concepts unique to this cluster: {concept_str}"
        )

        subtopic_blocks.append("\n".join(lines))

    subtopic_summary = "\n\n".join(subtopic_blocks)

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

For why_its_a_gap write 2-3 plain sentences that:
- Name what researchers ARE currently studying in this area (reference actual paper titles or themes from the data)
- Identify the SPECIFIC aspect that is missing or underexplored
- Explain in simple terms why that gap matters to the field
Do NOT use jargon like "citation intensity", "demand outpaces supply", or "underserved research need".
Write as if explaining to a curious student, not a statistics report.

Output format (replace values, keep keys exact):
[{{"subtopic":"exact name from list above","why_its_a_gap":"plain analytical explanation","citation_demand":0.0,"publication_supply":0}}]"""

    try:
        response = ask(prompt, reasoning=False, timeout=20.0)
        trace.log(f"Nemotron response received ({len(response)} chars)")
    except Exception as e:
        trace.log(f"Error calling Nemotron: {e}")
        response = None

    # ── Step 5: Parse JSON response ────────────────────────────────
    trace.log("Parsing gap analysis response...")

    # Build a lookup: subtopic name → top 3 paper titles (by citations)
    subtopic_top_papers: Dict[str, List[str]] = {}
    for s in top_subtopics:
        top3 = sorted(s.papers, key=lambda p: p.citations, reverse=True)[:3]
        subtopic_top_papers[s.name] = [p.title for p in top3]

    gaps = []

    if response:
        try:
            # Robust bracket-matching extraction
            cleaned = _extract_json_array(response)
            gaps_data = json.loads(cleaned)

            # Template placeholder values Nemotron sometimes returns verbatim
            _PLACEHOLDERS = {
                'exact name', 'unknown', 'str', 'name', 'subtopic name',
                'subtopic', 'gap name', 'example', 'placeholder',
            }

            for gap_data in gaps_data:
                if not isinstance(gap_data, dict):
                    trace.log(f"Skipping invalid gap entry (not a dict): {type(gap_data)}")
                    continue

                subtopic_val = gap_data.get("subtopic", "").strip()
                why_val = gap_data.get("why_its_a_gap", "").strip()

                # Detect template echoes (Nemotron returned the example values)
                if subtopic_val.lower() in _PLACEHOLDERS:
                    trace.log(f"Skipping template placeholder gap: '{subtopic_val}'")
                    continue
                if why_val.lower() in {'2 sentences', '2 sentence explanation', 'explanation', 'str'}:
                    trace.log(f"Skipping template placeholder why_its_a_gap")
                    continue

                gap = Gap(
                    subtopic=subtopic_val,
                    why_its_a_gap=why_val,
                    citation_demand=float(gap_data.get("citation_demand", 0)),
                    publication_supply=int(gap_data.get("publication_supply", 0)),
                    orphan_papers=[p.title for p in orphans],
                    top_papers=subtopic_top_papers.get(subtopic_val, []),
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
            # Pick top 2 paper titles by citations to make the explanation concrete
            top_papers_fb = sorted(subtopic.papers, key=lambda p: p.citations, reverse=True)[:2]
            paper_refs = " and ".join(f'"{p.title[:70]}"' for p in top_papers_fb)
            recent_note = (
                "but no new studies have appeared in 2024"
                if recent == 0
                else f"but only {recent} new {'study' if recent == 1 else 'studies'} appeared in 2024"
            )
            gap = Gap(
                subtopic=subtopic.name,
                top_papers=[p.title for p in top_papers_fb],
                why_its_a_gap=(
                    f"Researchers have published {subtopic.paper_count} papers in this area "
                    f"— including {paper_refs} — which together average {subtopic.avg_citations:.0f} "
                    f"citations each, showing strong interest from the community. However, {recent_note}, "
                    f"suggesting that while the community finds this topic important, new work addressing "
                    f"open questions here is still missing."
                ),
                citation_demand=subtopic.avg_citations,
                publication_supply=subtopic.paper_count,
                orphan_papers=[p.title for p in orphans],
            )
            fallback_gaps.append(gap)

    trace.log(f"GAP DETECTION: found {len(fallback_gaps)} gaps (fallback)")
    return fallback_gaps


def _questions_for_one_gap(
    gap: Gap,
    all_papers: List[Paper],
    topic: str,
) -> List[ResearchQuestion]:
    """Generate research questions for a single gap (runs in a thread)."""

    # ── Find matching papers ───────────────────────────────────────
    matching = [p for p in all_papers if p.topics and p.topics[0] == gap.subtopic]
    if not matching:
        matching = [
            p for p in all_papers
            if p.topics and any(_topic_overlap(gap.subtopic, t) for t in p.topics[:3])
        ]
    if not matching:
        matching = all_papers

    matching.sort(key=lambda p: p.citations, reverse=True)
    top_papers = matching[:4]  # 4 papers max (was 8) — shorter prompt

    # ── Build prompt (compact — titles + 80-char abstract snippet) ─
    paper_list = "\n".join(
        f"{i}. {p.title} ({p.citations} citations)"
        for i, p in enumerate(top_papers, 1)
    ) or "(No papers)"

    # Compact prompt — fewer tokens = faster response
    prompt = f"""Field: '{topic}'. Gap: '{gap.subtopic}'.
Gap reason: {gap.why_its_a_gap[:200]}

Top papers in this gap:
{paper_list}

Generate 1 specific falsifiable research question relevant to '{topic}' and '{gap.subtopic}'.
Name the methodology. Keep response short.

Return ONLY JSON (no markdown):
[{{"question":"str","methodology":"str","novelty_reason":"str"}}]"""

    # ── Call Nemotron (reasoning=False, 15s timeout) ──────────────
    try:
        response = ask(prompt, reasoning=False, timeout=15.0)
    except Exception:
        response = None

    # ── Parse response ─────────────────────────────────────────────
    questions = []
    if response:
        try:
            data = json.loads(_extract_json_array(response))
            for q in data:
                if not isinstance(q, dict):
                    continue
                questions.append(ResearchQuestion(
                    gap=gap.subtopic,
                    question=q.get("question", ""),
                    methodology=q.get("methodology", ""),
                    foundational_papers=[p.title for p in top_papers[:2]],
                    novelty_reason=q.get("novelty_reason", ""),
                ))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    # ── Fallback if nothing parsed ─────────────────────────────────
    if not questions:
        paper_hint = top_papers[0].title if top_papers else gap.subtopic
        questions.append(ResearchQuestion(
            gap=gap.subtopic,
            question=(
                f"What aspects of {gap.subtopic.lower()} within {topic} "
                f"have not been explored yet, and what would it take to study them?"
            ),
            methodology="Systematic literature review",
            foundational_papers=[p.title for p in top_papers[:2]],
            novelty_reason=(
                f"Papers like '{paper_hint[:70]}' have attracted significant attention "
                f"but the area of {gap.subtopic.lower()} still has open questions "
                f"that newer studies haven't addressed."
            ),
        ))

    return questions


def question_generation_agent(
    gaps: List[Gap],
    all_papers: List[Paper],
    trace: AgentTrace,
    topic: str = ""
) -> List[ResearchQuestion]:
    """Generate research questions for all gaps in parallel (3× faster).

    Spawns one thread per gap so all Nemotron calls run simultaneously.
    Uses reasoning=False for speed — gap detection already did the deep thinking.
    """
    PARALLEL_TIMEOUT = 20  # hard wall: abandon slow calls after this many seconds
    trace.log(f"Generating questions for {len(gaps)} gaps in parallel (max {PARALLEL_TIMEOUT}s)...")

    results: Dict[int, List[ResearchQuestion]] = {}

    def _fallback_q(gap: Gap) -> ResearchQuestion:
        return ResearchQuestion(
            gap=gap.subtopic,
            question=(
                f"What aspects of {gap.subtopic.lower()} within {topic} "
                f"have not been explored yet, and what would it take to study them?"
            ),
            methodology="Systematic literature review",
            foundational_papers=[],
            novelty_reason=(
                f"This area of {gap.subtopic.lower()} has attracted strong research interest "
                f"but recent publications have not kept up, leaving open questions unanswered."
            ),
        )

    with ThreadPoolExecutor(max_workers=len(gaps)) as executor:
        future_to_idx = {
            executor.submit(_questions_for_one_gap, gap, all_papers, topic): idx
            for idx, gap in enumerate(gaps)
        }

        # Wait up to PARALLEL_TIMEOUT seconds — don't block forever on slow calls
        done, not_done = futures_wait(future_to_idx, timeout=PARALLEL_TIMEOUT)

        # Collect results from futures that finished in time
        for future in done:
            idx = future_to_idx[future]
            gap = gaps[idx]
            try:
                qs = future.result()
                results[idx] = qs
                tag = "fallback" if "untested" in qs[0].question else "specific"
                trace.log(f"QUESTIONS: {len(qs)} [{tag}] for '{gap.subtopic}'")
            except Exception as e:
                trace.log(f"QUESTIONS: error for '{gap.subtopic}': {e}")
                results[idx] = [_fallback_q(gap)]

        # Generate instant fallbacks for any calls that timed out
        for future in not_done:
            future.cancel()
            idx = future_to_idx[future]
            gap = gaps[idx]
            trace.log(f"QUESTIONS: timeout — fallback for '{gap.subtopic}'")
            results[idx] = [_fallback_q(gap)]

    # Reassemble in gap order
    all_questions: List[ResearchQuestion] = []
    for idx in range(len(gaps)):
        all_questions.extend(results.get(idx, []))

    return all_questions


def question_clarity_agent(
    questions: List[ResearchQuestion],
    trace: AgentTrace,
    topic: str = ""
) -> List[ResearchQuestion]:
    """Rewrite research questions in plain, student-accessible English.

    Strips academic jargon ("methodological approaches", "empirical validation",
    "citation demand", "publication supply") and rewrites questions so a curious
    student can understand what is being asked and why it matters.

    Makes a single Nemotron call to rewrite all questions at once.
    Falls back to the originals on timeout or parse failure.
    """
    if not questions:
        return questions

    # Build a compact numbered list of the original questions
    q_list = "\n".join(
        f"{i}. {q.question}" for i, q in enumerate(questions, 1)
    )

    prompt = f"""You are a science communicator helping undergraduate students understand research gaps in '{topic}'.

Rewrite each research question below so that:
- A curious university student with no prior background can understand it
- It sounds like a genuine, interesting question — not a template
- It avoids jargon like "methodological approaches", "empirical validation", "citation demand", "publication supply", "falsifiable", "longitudinal"
- It is specific to the actual topic, not generic
- It stays as a question (ends with ?)

Original questions:
{q_list}

Return ONLY a JSON array of rewritten question strings, one per original question, in the same order.
Example format: ["Why do...", "What happens when...", "Can we..."]
No markdown, no explanation."""

    try:
        response = ask(prompt, reasoning=False, timeout=20.0)
        raw = _extract_json_array(response)
        rewritten = json.loads(raw)

        if not isinstance(rewritten, list) or len(rewritten) != len(questions):
            trace.log(f"CLARITY: parse mismatch — using originals")
            return questions

        # Replace question text in-place, keep all other fields
        clarified = []
        for orig, new_text in zip(questions, rewritten):
            if isinstance(new_text, str) and new_text.strip():
                clarified.append(ResearchQuestion(
                    gap=orig.gap,
                    question=new_text.strip(),
                    methodology=orig.methodology,
                    foundational_papers=orig.foundational_papers,
                    novelty_reason=orig.novelty_reason,
                ))
            else:
                clarified.append(orig)

        trace.log(f"CLARITY: rewrote {len(clarified)} questions in plain English")
        return clarified

    except Exception as e:
        trace.log(f"CLARITY: failed ({e}) — using originals")
        return questions


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
    from tools import fetch_papers, fetch_trends, cluster_by_topic, semantic_cluster, enrich_clusters

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

    # ── Step 6b: Enrich clusters with additional gap signals ─────
    trace.log("Phase 1c: Enriching clusters with gap signals...")
    enrichment: Dict[str, Any] = {}
    try:
        enrichment = enrich_clusters(subtopics, papers)
        trace.log(f"  Enrichment complete for {len(enrichment)} subtopics")
    except Exception as e:
        trace.log(f"  Enrichment failed ({e}) — continuing without enrichment")

    # ── Step 7: Fetch trends ────────────────────────────────────
    trace.log("Phase 2: Fetching publication trends...")
    trends = fetch_trends(topic_used, trace)

    # ── Step 8: Semantic clustering (top 100 by citations for speed) ─
    trace.log("Phase 3: Performing semantic clustering...")
    papers_for_sem = sorted(papers, key=lambda p: p.citations, reverse=True)[:100]
    semantic_result = semantic_cluster(papers_for_sem, trace=trace)

    # ── Step 9: Gap detection ───────────────────────────────────
    trace.log("Phase 4: Detecting research gaps...")
    gaps = gap_detection_agent(subtopics, trends, semantic_result, trace, topic=topic_used, enrichment=enrichment)

    # ── Step 10: Question generation ────────────────────────────
    trace.log("Phase 5: Generating research questions...")
    questions = question_generation_agent(gaps, papers, trace, topic=topic_used)

    # ── Step 10b: Clarity rewrite ────────────────────────────────
    trace.log("Phase 5b: Rewriting questions in plain English...")
    questions = question_clarity_agent(questions, trace, topic=topic_used)

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
