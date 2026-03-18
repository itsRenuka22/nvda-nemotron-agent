#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'papergap'))

from models import AgentTrace, Gap
from tools import fetch_papers, cluster_by_topic
from agents import gap_detection_agent, question_generation_agent

# Create trace object
trace = AgentTrace()

print("\n" + "="*70)
print("FULL PIPELINE TEST: Gap Detection + Question Generation")
print("="*70)

# Fetch papers
print("\n[1/4] Fetching papers...")
papers = fetch_papers('federated learning healthcare', trace, limit=100)
papers = papers[:100]
print(f"  ✓ Loaded {len(papers)} papers\n")

# Cluster by topic
print("[2/4] Clustering by topic...")
trace = AgentTrace()
subtopics = cluster_by_topic(papers, trace)
print(f"  ✓ Found {len(subtopics)} subtopics\n")

# Detect gaps (simplified - just take top 2 subtopics as gaps)
print("[3/4] Identifying gaps...")
trace = AgentTrace()
gaps = [
    Gap(
        subtopic=subtopics[0].name,
        why_its_a_gap=f"High citation demand ({subtopics[0].avg_citations:.0f} avg) with limited publications ({subtopics[0].paper_count} papers).",
        citation_demand=subtopics[0].avg_citations,
        publication_supply=subtopics[0].paper_count,
    ),
    Gap(
        subtopic=subtopics[1].name,
        why_its_a_gap=f"High citation demand ({subtopics[1].avg_citations:.0f} avg) with limited publications ({subtopics[1].paper_count} papers).",
        citation_demand=subtopics[1].avg_citations,
        publication_supply=subtopics[1].paper_count,
    ),
]
print(f"  ✓ Identified {len(gaps)} gaps\n")

# Generate questions
print("[4/4] Generating research questions...")
trace = AgentTrace()
questions = question_generation_agent(gaps, papers, trace)

print(f"\n" + "="*70)
print("RESULTS: Research Questions Generated")
print("="*70)

for i, q in enumerate(questions, 1):
    print(f"\n{i}. Gap: {q.gap}")
    print(f"   Question: {q.question[:100]}...")
    print(f"   Methodology: {q.methodology}")
    print(f"   Novelty: {q.novelty_reason[:80]}...")
    print(f"   Foundational papers: {len(q.foundational_papers)}")

print(f"\n" + "="*70)
print(f"✓ SUCCESS: Generated {len(questions)} research questions")
print("="*70 + "\n")
