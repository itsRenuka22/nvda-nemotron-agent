#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent/papergap')
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent')

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
