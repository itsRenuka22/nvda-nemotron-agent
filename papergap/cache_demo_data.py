#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent/papergap')

from pathlib import Path
from models import AgentTrace
from tools import fetch_papers, fetch_trends

trace = AgentTrace()

DEMO_TOPICS = [
    "federated learning rare disease diagnosis",
    "large language models clinical decision support",
    "graph neural networks drug interaction prediction"
]

print("\n" + "="*70)
print("PRE-CACHING DEMO DATA")
print("="*70)

cache_dir = Path("cache")
cache_files = {}

for topic in DEMO_TOPICS:
    print(f"\nCaching: {topic}")

    # Fetch papers
    papers = fetch_papers(topic, trace, limit=200)
    print(f"  ✓ Fetched {len(papers)} papers")

    # Fetch trends
    trends = fetch_trends(topic, trace)
    print(f"  ✓ Fetched {len(trends)} trend points")

    # Record cache file path
    cache_file = cache_dir / f"{topic.replace(' ', '_')}_2022-2025.json"
    if cache_file.exists():
        file_size = cache_file.stat().st_size / 1024  # KB
        cache_files[topic] = (cache_file, file_size)
        print(f"  ✓ Cache file: {cache_file.name} ({file_size:.1f} KB)")

print("\n" + "="*70)
print("CACHE STATUS")
print("="*70)

all_ok = True
for topic, (cache_file, size_kb) in cache_files.items():
    status = "✓" if size_kb >= 500 else "✗"
    size_status = "OK (>500KB)" if size_kb >= 500 else f"SMALL ({size_kb:.1f}KB)"
    print(f"{status} {cache_file.name}: {size_status}")
    if size_kb < 500:
        all_ok = False

print("\n" + "="*70)
if all_ok and len(cache_files) == 3:
    print("✓ SUCCESS: All demo data cached and ready for offline demo")
else:
    print("✗ WARNING: Some cache files are too small or missing")
print("="*70 + "\n")
