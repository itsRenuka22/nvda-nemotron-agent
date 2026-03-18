#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'papergap'))

from models import AgentTrace
from tools import fetch_papers, fetch_trends, cluster_by_topic, semantic_cluster
from agents import gap_detection_agent

# Create trace object
trace = AgentTrace()

print("\n" + "="*70)
print("INTEGRATED TEST: Gap Detection Agent")
print("="*70)

# Step 1: Fetch papers
print("\n[1/5] Fetching papers...")
papers = fetch_papers('federated learning healthcare', trace, limit=100)
papers = papers[:100]
print(f"  ✓ Loaded {len(papers)} papers\n")

# Step 2: Get trends
print("[2/5] Fetching trends...")
trends = fetch_trends('federated learning healthcare', trace)
print(f"  ✓ Got {len(trends)} trend points\n")

# Step 3: Cluster by topic
print("[3/5] Clustering by topic...")
trace = AgentTrace()
subtopics = cluster_by_topic(papers, trace)
print(f"  ✓ Found {len(subtopics)} subtopics\n")

# Step 4: Semantic clustering
print("[4/5] Semantic clustering...")
trace = AgentTrace()
semantic_result = semantic_cluster(papers, n_clusters=7, trace=trace)
print(f"  ✓ Found {len(semantic_result['orphans'])} orphan papers\n")

# Step 5: Gap detection
print("[5/5] Detecting research gaps...")
trace = AgentTrace()
gaps = gap_detection_agent(subtopics, trends, semantic_result, trace)

print(f"\n" + "="*70)
print("RESULTS: Research Gaps Detected")
print("="*70)

for i, gap in enumerate(gaps, 1):
    print(f"\n{i}. {gap.subtopic}")
    print(f"   Citation Demand: {gap.citation_demand:.0f}")
    print(f"   Publication Supply: {gap.publication_supply}")
    print(f"   Why: {gap.why_its_a_gap}")
    print(f"   Orphan Papers: {len(gap.orphan_papers)}")

print(f"\n" + "="*70)
print(f"✓ SUCCESS: Gap detection completed ({len(gaps)} gaps found)")
print("="*70 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS: Enrichment signals (synthetic fixtures — no network calls)
# ─────────────────────────────────────────────────────────────────────────────
from models import Paper, Subtopic
from tools import (
    enrich_with_explicit_signals,
    enrich_with_citation_frontier,
    enrich_with_concept_isolation,
)

print("=" * 70)
print("UNIT TEST: enrich_with_explicit_signals")
print("=" * 70)

# 5 papers: 2 contain future-work / limitation sentences, 3 do not
_papers_sig1 = [
    Paper(id="p1", title="Paper 1", year=2023, citations=10, topics=["T"],
          abstract="We propose a new method. In future work, we plan to extend this to larger datasets."),
    Paper(id="p2", title="Paper 2", year=2023, citations=5, topics=["T"],
          abstract="Results are promising. This work has a limitation: we do not handle missing values."),
    Paper(id="p3", title="Paper 3", year=2022, citations=8, topics=["T"],
          abstract="Our approach achieves state-of-the-art results on all benchmarks."),
    Paper(id="p4", title="Paper 4", year=2022, citations=3, topics=["T"],
          abstract="The model performs well across diverse settings."),
    Paper(id="p5", title="Paper 5", year=2021, citations=12, topics=["T"],
          abstract="Extensive experiments confirm the effectiveness of our approach."),
]
_subtopic_sig1 = Subtopic(
    name="Test Subtopic A", paper_count=5, avg_citations=7.6,
    pct_of_total=100.0, papers=_papers_sig1,
)

result_sig1 = enrich_with_explicit_signals([_subtopic_sig1], _papers_sig1)

assert "Test Subtopic A" in result_sig1,                       "FAIL: subtopic key missing"
assert result_sig1["Test Subtopic A"]["explicit_gap_count"] == 2, \
    f"FAIL: expected 2, got {result_sig1['Test Subtopic A']['explicit_gap_count']}"
assert isinstance(result_sig1["Test Subtopic A"]["explicit_gap_sentences"], list), \
    "FAIL: explicit_gap_sentences not a list"
assert len(result_sig1["Test Subtopic A"]["explicit_gap_sentences"]) <= 3, \
    "FAIL: more than 3 quotes returned"
print(f"  PASS — explicit_gap_count={result_sig1['Test Subtopic A']['explicit_gap_count']}, "
      f"quotes={len(result_sig1['Test Subtopic A']['explicit_gap_sentences'])}")

print("\n" + "=" * 70)
print("UNIT TEST: enrich_with_citation_frontier")
print("=" * 70)

# Paper B (year=2020) is cited by both Paper A (year=2021) and Paper C (year=2022).
# In-degree of B = 2, most_recent_citing_year = 2022 <= 2022 → frontier_flag = True
_paper_b = Paper(id="pb", title="Foundational Paper B", abstract="", year=2020, citations=50,
                 topics=["T"], referenced_works=[])
_paper_a = Paper(id="pa", title="Paper A builds on B", abstract="", year=2021, citations=20,
                 topics=["T"], referenced_works=["pb"])
_paper_c = Paper(id="pc", title="Paper C also cites B", abstract="", year=2022, citations=15,
                 topics=["T"], referenced_works=["pb"])
_paper_d = Paper(id="pd", title="Paper D unrelated", abstract="", year=2023, citations=5,
                 topics=["T"], referenced_works=[])

_subtopic_sig2 = Subtopic(
    name="Test Subtopic B", paper_count=4, avg_citations=22.5,
    pct_of_total=100.0, papers=[_paper_a, _paper_b, _paper_c, _paper_d],
)

result_sig2 = enrich_with_citation_frontier([_subtopic_sig2], [])

assert "Test Subtopic B" in result_sig2,                       "FAIL: subtopic key missing"
assert result_sig2["Test Subtopic B"]["citation_frontier_flag"] is True, \
    "FAIL: expected citation_frontier_flag=True"
assert len(result_sig2["Test Subtopic B"]["frontier_papers"]) >= 1, \
    "FAIL: expected at least 1 frontier paper"
assert result_sig2["Test Subtopic B"]["frontier_papers"][0]["title"] == "Foundational Paper B", \
    "FAIL: wrong frontier paper identified"
print(f"  PASS — frontier_flag=True, frontier_paper='{result_sig2['Test Subtopic B']['frontier_papers'][0]['title']}'")

print("\n" + "=" * 70)
print("UNIT TEST: enrich_with_concept_isolation")
print("=" * 70)

# Cluster X: 3 concepts unique to it (not in any other paper in the corpus)
# Cluster Y: concepts shared with cluster X → X should have high isolation score
_papers_x = [
    Paper(id="px1", title="X Paper 1", abstract="", year=2023, citations=10,
          topics=["UniqueConceptA", "UniqueConceptB", "UniqueConceptC"]),
    Paper(id="px2", title="X Paper 2", abstract="", year=2022, citations=8,
          topics=["UniqueConceptA", "UniqueConceptB"]),
]
_papers_y = [
    Paper(id="py1", title="Y Paper 1", abstract="", year=2023, citations=5,
          topics=["SharedConceptZ", "SharedConceptW"]),
    Paper(id="py2", title="Y Paper 2", abstract="", year=2022, citations=3,
          topics=["SharedConceptZ"]),
]
_subtopic_x = Subtopic(
    name="Cluster X", paper_count=2, avg_citations=9.0,
    pct_of_total=50.0, papers=_papers_x,
)
_subtopic_y = Subtopic(
    name="Cluster Y", paper_count=2, avg_citations=4.0,
    pct_of_total=50.0, papers=_papers_y,
)
_all_papers_sig3 = _papers_x + _papers_y

result_sig3 = enrich_with_concept_isolation(
    [_subtopic_x, _subtopic_y], _all_papers_sig3
)

assert "Cluster X" in result_sig3,                             "FAIL: Cluster X key missing"
assert result_sig3["Cluster X"]["concept_isolation_score"] > 0.5, \
    f"FAIL: expected isolation > 0.5, got {result_sig3['Cluster X']['concept_isolation_score']}"
assert isinstance(result_sig3["Cluster X"]["isolated_concepts"], list), \
    "FAIL: isolated_concepts not a list"
print(f"  PASS — Cluster X isolation={result_sig3['Cluster X']['concept_isolation_score']:.2f}, "
      f"isolated_concepts={result_sig3['Cluster X']['isolated_concepts']}")

print("\n" + "=" * 70)
print("✓ ALL ENRICHMENT UNIT TESTS PASSED")
print("=" * 70 + "\n")
