from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    year: int
    citations: int
    topics: List[str]
    referenced_works: List[str] = field(default_factory=list)  # OpenAlex work IDs this paper cites

@dataclass
class Subtopic:
    name: str
    paper_count: int
    avg_citations: float
    pct_of_total: float
    papers: List[Paper] = field(default_factory=list)

@dataclass
class TrendPoint:
    subtopic: str
    year: int
    count: int

@dataclass
class Gap:
    subtopic: str
    why_its_a_gap: str
    citation_demand: float   # avg citations of papers in this area
    publication_supply: int  # number of papers
    orphan_papers: List[str] = field(default_factory=list)
    shared_assumption: Optional[str] = None   # filled by Nemotron
    top_papers: List[str] = field(default_factory=list)  # titles of top papers in this gap

@dataclass
class ResearchQuestion:
    gap: str
    question: str
    methodology: str
    foundational_papers: List[str]
    novelty_reason: str

@dataclass
class AgentTrace:
    steps: List[str] = field(default_factory=list)
    def log(self, msg: str):
        self.steps.append(msg)
        print(f"[TRACE] {msg}")
