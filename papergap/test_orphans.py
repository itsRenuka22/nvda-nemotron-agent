#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent/papergap')

from models import AgentTrace
from tools import fetch_papers, semantic_cluster

# Create trace object
trace = AgentTrace()

print("\n" + "="*70)
print("TEST: semantic_cluster() orphan detection on 100 papers")
print("="*70)

# Fetch papers and use only 100
all_papers = fetch_papers('federated learning healthcare', trace, limit=100)
papers = all_papers[:100]  # Ensure exactly 100 papers
print(f"\nUsing {len(papers)} papers")

# Run semantic clustering
trace = AgentTrace()
result = semantic_cluster(papers, n_clusters=7, trace=trace)

# Display results
orphan_count = len(result['orphans'])
orphan_percentage = (orphan_count / len(papers)) * 100

print(f"\n{'='*70}")
print(f"Results:")
print(f"  Total papers: {len(papers)}")
print(f"  Orphan papers found: {orphan_count}")
print(f"  Orphan percentage: {orphan_percentage:.1f}%")
print(f"  Expected range (8-15%): 8-15 orphans")
print(f"{'='*70}")

# Check if orphans are in expected range
expected_min = 8
expected_max = 15
if expected_min <= orphan_count <= expected_max:
    print(f"✓ SUCCESS: Found {orphan_count} orphans (within 8-15 range)")
else:
    print(f"✗ FAIL: Found {orphan_count} orphans (expected 8-15)")

print(f"\nCluster distribution: {result['cluster_sizes']}")
print(f"\nTop 5 orphan papers:")
for i, paper in enumerate(result['orphans'][:5], 1):
    print(f"{i}. {paper.title[:60]}...")
    print(f"   Citations: {paper.citations}, Year: {paper.year}\n")

print("="*70 + "\n")
