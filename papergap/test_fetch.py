#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent/papergap')

from models import AgentTrace
from tools import fetch_papers, fetch_trends, cluster_by_topic

# Create trace object
trace = AgentTrace()

print("\n" + "="*70)
print("TEST 1: fetch_papers()")
print("="*70)

# Fetch papers
papers = fetch_papers('federated learning healthcare', trace)

# Check results
print(f"\nTotal papers returned: {len(papers)}")
print(f"Papers with non-empty abstracts: {sum(1 for p in papers if p.abstract.strip())}")

# Show some stats
if papers:
    print(f"\nSample papers:")
    for i, paper in enumerate(papers[:2]):
        print(f"\n{i+1}. Title: {paper.title[:60]}...")
        print(f"   Year: {paper.year}, Citations: {paper.citations}")
        print(f"   Topics: {paper.topics[:2]}")

print("\n" + "="*70)
print("TEST 2: fetch_trends()")
print("="*70)

# Test fetch_trends
trace = AgentTrace()
trends = fetch_trends('federated learning healthcare', trace)

print(f"\nYear trends returned: {len(trends)}")
print(f"\nYear-by-year publication counts:")
for trend in trends:
    print(f"  {trend.year}: {trend.count} papers")

print("\n" + "="*70)
print("TEST 3: cluster_by_topic()")
print("="*70)

# Test cluster_by_topic
trace = AgentTrace()
subtopics = cluster_by_topic(papers, trace)

print(f"\nSubtopics found: {len(subtopics)}")
print(f"\nAll subtopics with details:")
for i, subtopic in enumerate(subtopics, 1):
    print(f"{i}. {subtopic.name}")
    print(f"   Papers: {subtopic.paper_count}")
    print(f"   Avg Citations: {subtopic.avg_citations:.1f}")
    print(f"   % of Total: {subtopic.pct_of_total:.1f}%")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
checks = []
checks.append(("fetch_papers: 150+ papers", len(papers) >= 150))
checks.append(("fetch_trends: Returns year counts", len(trends) > 0 and all(hasattr(t, 'count') for t in trends)))
checks.append(("cluster_by_topic: 5+ subtopics", len(subtopics) >= 5))

for check_name, result in checks:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status}: {check_name}")

print("="*70 + "\n")
