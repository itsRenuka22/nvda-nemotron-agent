#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent/papergap')
sys.path.insert(0, '/Users/renuka/Documents/Hackathon/NvidiaGTC/nvda-nemotron-agent')

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
