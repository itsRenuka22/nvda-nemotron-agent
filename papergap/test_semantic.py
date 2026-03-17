#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent/papergap')

from models import AgentTrace
from tools import fetch_papers, semantic_cluster

# Create trace object
trace = AgentTrace()

print("\n" + "="*70)
print("TEST: semantic_cluster()")
print("="*70)

# Fetch papers
papers = fetch_papers('federated learning healthcare', trace, limit=100)
print(f"\nLoaded {len(papers)} papers\n")

# Run semantic clustering
trace = AgentTrace()
result = semantic_cluster(papers, n_clusters=7, trace=trace)

# Display results
print(f"\nClustering Results:")
print(f"  Total papers: {len(papers)}")
print(f"  Orphan papers found: {len(result['orphans'])}")
print(f"  Cluster distribution: {result['cluster_sizes']}")

print(f"\nCluster sizes:")
for cluster_id, size in sorted(result['cluster_sizes'].items()):
    print(f"  Cluster {cluster_id}: {size} papers")

if result['orphans']:
    print(f"\nTop 3 orphan papers (semantic outliers):")
    for i, paper in enumerate(result['orphans'][:3], 1):
        print(f"{i}. {paper.title[:60]}...")
        print(f"   Citations: {paper.citations}, Year: {paper.year}")

print("\n" + "="*70)
if len(result['labels']) > 0 and len(result['orphans']) > 0:
    print("✓ SUCCESS: semantic_cluster() working")
else:
    print("✗ FAIL: semantic_cluster() returned empty results")
print("="*70 + "\n")
