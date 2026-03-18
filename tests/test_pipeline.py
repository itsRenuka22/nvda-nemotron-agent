#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'papergap'))

from agents import run_pipeline

print("\n" + "="*70)
print("END-TO-END PIPELINE TEST")
print("="*70 + "\n")

# Run the complete pipeline
result = run_pipeline("federated learning healthcare")

# Display results
print("\n" + "="*70)
print("PIPELINE RESULTS")
print("="*70)

print(f"\nTopic: {result['topic_used']}")
print(f"Papers Analyzed: {result['paper_count']}")
print(f"Subtopics Found: {len(result['subtopics'])}")
print(f"Gaps Identified: {len(result['gaps'])}")
print(f"Questions Generated: {len(result['questions'])}")
print(f"Autonomous Drill-Down: {result['drilled_deeper']}")
if result['drill_reason']:
    print(f"  Reason: {result['drill_reason']}")

print(f"\n{'='*70}")
print("TOP 3 RESEARCH GAPS")
print(f"{'='*70}")

for i, gap in enumerate(result['gaps'][:3], 1):
    print(f"\n{i}. {gap.subtopic}")
    print(f"   Citation Demand: {gap.citation_demand:.0f}")
    print(f"   Publication Supply: {gap.publication_supply}")
    print(f"   Gap Description: {gap.why_its_a_gap[:80]}...")

    # Find corresponding questions
    gap_questions = [q for q in result['questions'] if q.gap == gap.subtopic]
    print(f"   Research Questions: {len(gap_questions)}")
    for j, q in enumerate(gap_questions[:2], 1):
        print(f"     {j}. {q.question[:70]}...")

print(f"\n{'='*70}")
print("TRACE LOG (last 10 entries)")
print(f"{'='*70}")

trace = result['trace']
for log in trace.steps[-10:]:
    print(f"  {log}")

print(f"\n{'='*70}")
print(f"✓ SUCCESS: Pipeline completed with {len(result['questions'])} actionable research questions")
print(f"{'='*70}\n")

# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TEST: enrich_clusters on cached corpus data
# ─────────────────────────────────────────────────────────────────────────────
from models import AgentTrace
from tools import fetch_papers, cluster_by_topic, enrich_clusters

print("=" * 70)
print("INTEGRATION TEST: enrich_clusters on cached corpus")
print("=" * 70)

_trace = AgentTrace()
_papers = fetch_papers("federated learning healthcare", _trace, limit=100)
_subtopics = cluster_by_topic(_papers, _trace)

print(f"\n  Loaded {len(_papers)} papers, {len(_subtopics)} subtopics")
print("  Running enrich_clusters...")

_enrichment = enrich_clusters(_subtopics, _papers)

# Every subtopic must be present in the enrichment dict
assert len(_enrichment) == len(_subtopics), \
    f"FAIL: enrichment has {len(_enrichment)} entries, expected {len(_subtopics)}"

# Every entry must contain all six expected keys with correct types
_required_keys = {
    "explicit_gap_count": int,
    "explicit_gap_sentences": list,
    "citation_frontier_flag": bool,
    "frontier_papers": list,
    "concept_isolation_score": float,
    "isolated_concepts": list,
}
for subtopic_name, signals in _enrichment.items():
    for key, expected_type in _required_keys.items():
        assert key in signals, \
            f"FAIL: '{subtopic_name}' missing key '{key}'"
        assert signals[key] is not None, \
            f"FAIL: '{subtopic_name}'.{key} is None"
        assert isinstance(signals[key], expected_type), \
            f"FAIL: '{subtopic_name}'.{key} expected {expected_type}, got {type(signals[key])}"

# explicit_gap_count must be a non-negative integer for every subtopic
for subtopic_name, signals in _enrichment.items():
    assert signals["explicit_gap_count"] >= 0, \
        f"FAIL: '{subtopic_name}'.explicit_gap_count is negative"

# concept_isolation_score must be in [0.0, 1.0]
for subtopic_name, signals in _enrichment.items():
    score = signals["concept_isolation_score"]
    assert 0.0 <= score <= 1.0, \
        f"FAIL: '{subtopic_name}'.concept_isolation_score={score} out of range"

print(f"\n  Results for top 3 subtopics:")
for name in list(_enrichment.keys())[:3]:
    sig = _enrichment[name]
    print(f"    {name[:50]}")
    print(f"      explicit_gap_count={sig['explicit_gap_count']}, "
          f"frontier={sig['citation_frontier_flag']}, "
          f"isolation={sig['concept_isolation_score']:.2f}")

print("\n" + "=" * 70)
print("✓ INTEGRATION TEST PASSED: all three signal fields present on every cluster")
print("=" * 70 + "\n")
