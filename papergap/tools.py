import hashlib
import json
import re
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
from models import Paper, AgentTrace, TrendPoint, Subtopic

# FIX 7: Cache TTL — re-fetch if data is older than this many days
CACHE_TTL_DAYS = 30

# FIX 4: OpenAlex topic names that are too generic to be useful as subtopic labels.
# When topics[0] is one of these, we look deeper in the topic list for something specific.
_GENERIC_TOPICS = {
    "Artificial Intelligence", "Machine Learning", "Computer Science",
    "Deep Learning", "Natural Language Processing", "Neural Networks",
    "Data Science", "Medicine", "Biology", "Physics", "Mathematics",
    "Engineering", "Technology", "Science", "Education", "Psychology",
    "Economics", "Social Science", "Chemistry", "Environmental Science",
}


def _decode_abstract(inv_index: Dict[str, List[int]]) -> str:
    """Rebuild abstract text from OpenAlex inverted index format.

    OpenAlex stores abstracts as {word: [positions]}, need to reconstruct
    the original text by ordering words by their position.
    """
    if not inv_index:
        return ""

    # Find max position to know the length
    max_pos = max(max(positions) for positions in inv_index.values() if positions)

    # Create array of words at each position
    words = [''] * (max_pos + 1)
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word

    return ' '.join(words).strip()


def _fetch_openalex(
    topic: str,
    years: str,
    sort: str = "cited_by_count:desc",
    limit: int = 100
) -> List[Dict]:
    """Make a single OpenAlex API request and return raw result items."""
    params = {
        "search": topic,
        "filter": f"publication_year:{years},has_abstract:true",
        "sort": sort,
        "per_page": min(limit, 200),
        "select": "id,title,abstract_inverted_index,publication_year,cited_by_count,topics"
    }
    response = requests.get("https://api.openalex.org/works", params=params)
    response.raise_for_status()
    return response.json().get('results', [])


def fetch_papers(
    topic: str,
    trace: AgentTrace,
    years: str = "2022-2025",
    limit: int = 200
) -> List[Paper]:
    """Fetch papers from OpenAlex API using dual-query strategy and cache results.

    Makes two queries — top-cited (established research) + most-recent (emerging
    directions) — then merges and deduplicates. This ensures emerging research
    gaps are visible alongside established areas.

    Args:
        topic: Search query for papers
        trace: AgentTrace object for logging progress
        years: Year range filter (e.g., "2022-2025")
        limit: Maximum number of papers to fetch

    Returns:
        List of Paper objects with abstracts decoded from inverted index
    """
    # Use absolute path to cache directory (relative to this file's location)
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"{topic.replace(' ', '_')}_{years}.json"

    # FIX 7: Load cache only if it exists AND is not expired
    def _cache_is_valid(path: Path) -> bool:
        if not path.exists():
            return False
        with open(path, 'r') as f:
            d = json.load(f)
        cached_at = d.get('_cached_at', 0)
        age_days = (time.time() - cached_at) / 86400
        if age_days > CACHE_TTL_DAYS:
            trace.log(f"Cache expired ({age_days:.0f} days old) — re-fetching")
            return False
        return True

    if _cache_is_valid(cache_file):
        trace.log(f"Loading cached papers for '{topic}'")
        with open(cache_file, 'r') as f:
            data = json.load(f)
        all_results = data.get('results', [])
    else:
        # FIX 3: Dual-query strategy — top cited + most recent
        half = limit // 2

        # Query 1: Most cited papers (established research baseline)
        trace.log(f"Fetching top-cited papers for '{topic}' from OpenAlex API")
        cited_results = _fetch_openalex(topic, years, sort="cited_by_count:desc", limit=half)
        trace.log(f"  Got {len(cited_results)} top-cited papers")

        # Query 2: Most recent papers (emerging research directions — 2024+)
        trace.log(f"Fetching recent papers for '{topic}' from OpenAlex API")
        recent_results = _fetch_openalex(topic, years, sort="publication_date:desc", limit=half)
        trace.log(f"  Got {len(recent_results)} recent papers")

        # Merge and deduplicate by paper ID
        seen_ids: set = set()
        all_results = []
        for item in cited_results + recent_results:
            pid = item.get('id', '')
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_results.append(item)

        n_dupes = len(cited_results) + len(recent_results) - len(all_results)
        trace.log(f"Merged to {len(all_results)} unique papers ({n_dupes} duplicates removed)")

        # Cache combined result with timestamp (Fix 7)
        with open(cache_file, 'w') as f:
            json.dump({'results': all_results, '_cached_at': time.time()}, f, indent=2)
        trace.log(f"Cached {len(all_results)} papers to {cache_file}")

    # Convert to Paper objects
    papers = []
    results = all_results
    trace.log(f"Processing {len(results)} papers")

    for item in results:
        # Skip papers with empty abstracts
        abstract_inv_index = item.get('abstract_inverted_index')
        if not abstract_inv_index:
            continue

        # Decode abstract from inverted index
        abstract = _decode_abstract(abstract_inv_index)
        if not abstract:
            continue

        # Extract topics
        topics = []
        if item.get('topics'):
            topics = [t.get('display_name', '') for t in item['topics']]

        # Create Paper object
        paper = Paper(
            id=item.get('id', ''),
            title=item.get('title', ''),
            abstract=abstract,
            year=item.get('publication_year', 0),
            citations=item.get('cited_by_count', 0),
            topics=topics
        )
        papers.append(paper)

    trace.log(f"Successfully created {len(papers)} Paper objects")

    # ── Semantic relevance filter ────────────────────────────────────────
    # Use SentenceTransformer cosine similarity to drop papers that are
    # not actually about the search topic. Reuses the cached model so no
    # extra load time after the first call.
    papers = _filter_by_semantic_similarity(papers, topic, trace)

    return papers


def _filter_by_semantic_similarity(
    papers: List[Paper],
    topic: str,
    trace: AgentTrace,
    threshold: float = 0.20,
    min_papers: int = 10,
) -> List[Paper]:
    """Keep only papers whose abstract is semantically close to the search topic.

    Computes cosine similarity between the topic embedding and each paper's
    abstract embedding using the cached SentenceTransformer model.
    Papers below `threshold` are dropped as off-domain.

    Falls back to returning all papers if the model is unavailable or if
    filtering would leave fewer than `min_papers`.
    """
    try:
        import numpy as np
        model = _get_sentence_model()
    except ImportError:
        trace.log("Semantic filter: skipped (sentence_transformers not installed)")
        return papers

    if not papers:
        return papers

    trace.log(f"Semantic filter: scoring {len(papers)} papers against '{topic[:60]}'...")

    # Encode the topic once, then all abstracts in one batch (fast)
    topic_emb = model.encode([topic], show_progress_bar=False)[0]
    texts = [p.abstract if p.abstract else p.title for p in papers]
    paper_embs = model.encode(texts, show_progress_bar=False)

    # Cosine similarity = dot product of unit vectors (cast to float64 to avoid overflow)
    topic_emb = topic_emb.astype(np.float64)
    paper_embs = paper_embs.astype(np.float64)
    topic_norm = topic_emb / (np.linalg.norm(topic_emb) + 1e-9)
    paper_norms = paper_embs / (np.linalg.norm(paper_embs, axis=1, keepdims=True) + 1e-9)
    scores = paper_norms @ topic_norm
    # Replace any NaN/inf (bad embeddings) with 0 so they get filtered out
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Pair each paper with its score for easier manipulation
    scored = sorted(zip(scores.tolist(), papers), key=lambda x: x[0], reverse=True)

    filtered = [(s, p) for s, p in scored if s >= threshold]

    # Safety net: never return fewer than min_papers
    if len(filtered) < min_papers:
        filtered = scored[:min_papers]
        trace.log(
            f"Semantic filter: threshold too strict — keeping top {min_papers} "
            f"by similarity (best={scored[0][0]:.2f}, worst kept={filtered[-1][0]:.2f})"
        )
    else:
        trace.log(
            f"Semantic filter: kept {len(filtered)}/{len(papers)} papers "
            f"(threshold={threshold}, best={filtered[0][0]:.2f}, worst={filtered[-1][0]:.2f})"
        )

    return [p for _, p in filtered]


def fetch_trends(
    topic: str,
    trace: AgentTrace,
    years: str = "2022-2025"
) -> List[TrendPoint]:
    """Fetch publication trends from OpenAlex API.

    Groups papers by publication year to show trend of publications over time.

    Args:
        topic: Search query for papers
        trace: AgentTrace object for logging progress
        years: Year range filter (e.g., "2022-2025")

    Returns:
        List of TrendPoint objects with year and publication count
    """
    trace.log(f"Fetching publication trends for '{topic}'")

    params = {
        "search": topic,
        "filter": f"publication_year:{years}",
        "group_by": "publication_year"
    }

    response = requests.get("https://api.openalex.org/works", params=params)
    response.raise_for_status()

    data = response.json()
    group_by = data.get('group_by', [])

    trace.log(f"Got trends for {len(group_by)} years")

    # Convert to TrendPoint objects
    trend_points = []
    for item in group_by:
        trend = TrendPoint(
            subtopic=topic,
            year=int(item.get('key')),
            count=item.get('count', 0)
        )
        trend_points.append(trend)

    # Sort by year
    trend_points.sort(key=lambda x: x.year)
    return trend_points


def cluster_by_topic(papers: List[Paper], trace: AgentTrace) -> List[Subtopic]:
    """Group papers by their first topic.

    Args:
        papers: List of Paper objects to cluster
        trace: AgentTrace object for logging progress

    Returns:
        List of Subtopic objects sorted by paper count (descending)
    """
    trace.log(f"Clustering {len(papers)} papers by topic")

    # Group papers by first topic
    topic_groups: Dict[str, List[Paper]] = {}

    for paper in papers:
        # FIX 4: Prefer a specific topic over a generic one.
        # OpenAlex topics[0] is often too broad (e.g. "Artificial Intelligence").
        # Walk the top 3 topics and pick the first non-generic one.
        topic_name = "Uncategorized"
        for t in paper.topics[:3]:
            if t and t not in _GENERIC_TOPICS:
                topic_name = t
                break
        else:
            # All top-3 are generic — fall back to topics[0]
            if paper.topics:
                topic_name = paper.topics[0]

        if topic_name not in topic_groups:
            topic_groups[topic_name] = []
        topic_groups[topic_name].append(paper)

    # Calculate stats and create Subtopic objects
    total_papers = len(papers)
    subtopics = []

    for topic_name, topic_papers in topic_groups.items():
        avg_citations = (
            sum(p.citations for p in topic_papers) / len(topic_papers)
            if topic_papers else 0
        )
        pct_of_total = (len(topic_papers) / total_papers * 100) if total_papers > 0 else 0

        subtopic = Subtopic(
            name=topic_name,
            paper_count=len(topic_papers),
            avg_citations=avg_citations,
            pct_of_total=pct_of_total,
            papers=topic_papers
        )
        subtopics.append(subtopic)

    # Sort by paper count descending
    subtopics.sort(key=lambda x: x.paper_count, reverse=True)

    # Log top 5 subtopics
    trace.log("Top subtopics:")
    for i, subtopic in enumerate(subtopics[:5]):
        trace.log(f"  {i+1}. {subtopic.name}: {subtopic.paper_count} papers ({subtopic.pct_of_total:.1f}%)")

    return subtopics


# Module-level model cache — loaded once, reused across all pipeline runs
_sentence_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model


def semantic_cluster(
    papers: List[Paper],
    n_clusters: int = 7,
    trace: Optional[AgentTrace] = None
) -> Dict[str, Any]:
    """Cluster papers using semantic embeddings and identify outliers.

    Uses SentenceTransformer to encode paper abstracts/titles, then applies
    KMeans clustering. Identifies "orphan" papers that are far from any cluster.

    Args:
        papers: List of Paper objects to cluster
        n_clusters: Number of clusters (default 7)
        trace: Optional AgentTrace object for logging

    Returns:
        Dictionary with:
            - "labels": List of cluster labels for each paper
            - "orphans": List of Paper objects identified as outliers
            - "cluster_sizes": Dict mapping cluster label to paper count
    """
    try:
        from sklearn.cluster import KMeans
        import numpy as np
        model = _get_sentence_model()
    except ImportError as e:
        warning_msg = f"Missing dependencies for semantic clustering: {e}"
        warnings.warn(warning_msg)
        if trace:
            trace.log(f"Warning: {warning_msg}")
        return {
            "labels": [],
            "orphans": [],
            "cluster_sizes": {}
        }

    if trace:
        trace.log("Loading SentenceTransformer model (cached)...")

    # Build text list: use abstract if exists, else title
    texts = [paper.abstract if paper.abstract else paper.title for paper in papers]

    if trace:
        trace.log(f"Encoding {len(texts)} papers into embeddings...")

    # Encode texts
    embeddings = model.encode(texts, show_progress_bar=False)

    # Cap clusters to number of papers (KMeans requires n_samples >= n_clusters)
    n_clusters = min(n_clusters, len(papers))
    if n_clusters < 2:
        if trace:
            trace.log("Too few papers for clustering — skipping semantic cluster")
        return {"labels": [], "orphans": [], "cluster_sizes": {}}

    if trace:
        trace.log(f"Running KMeans clustering with {n_clusters} clusters...")

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Find orphans: papers far from cluster centers
    if trace:
        trace.log("Computing distances to identify orphan papers...")

    # Compute distance from each point to its nearest cluster center
    distances = np.min(
        np.linalg.norm(embeddings[:, np.newaxis, :] - kmeans.cluster_centers_, axis=2),
        axis=1
    )

    # 90th percentile as threshold
    threshold = np.percentile(distances, 90)
    orphan_indices = np.where(distances > threshold)[0]
    orphans = [papers[i] for i in orphan_indices]

    # Count cluster sizes
    cluster_sizes = {}
    for label in range(n_clusters):
        cluster_sizes[label] = int(np.sum(labels == label))

    if trace:
        trace.log(f"Encoded {len(texts)} papers, found {len(orphans)} orphan papers")
        trace.log(f"Cluster distribution: {cluster_sizes}")

    return {
        "labels": labels.tolist(),
        "orphans": orphans,
        "cluster_sizes": cluster_sizes
    }

# ── Phrase lists for Signal 1 (explicit gap / limitation detection) ─────────
_FUTURE_WORK_PHRASES = [
    "future work", "will investigate", "we leave", "remains to be",
    "open problem", "future research", "we plan to", "next step",
    "an interesting direction", "left for future", "beyond the scope",
    "remains open", "has not been studied", "not yet addressed",
]

_LIMITATION_PHRASES = [
    "limitation", "we do not", "cannot handle", "does not support",
    "we assume", "restricted to", "only applicable",
    "does not consider", "out of scope", "we focus only",
    "cannot be applied", "is not suitable", "we exclude",
]


def enrich_with_explicit_signals(
    subtopics: List[Subtopic],
    papers: List[Paper],
) -> Dict[str, Any]:
    """Count abstract sentences that explicitly state a limitation or future gap.

    For each subtopic, scans its papers' abstracts for sentences matching
    future-work or limitation phrases. Records the total match count and up
    to 3 representative quote sentences as evidence for the Nemotron prompt.

    Args:
        subtopics: List of Subtopic objects from cluster_by_topic
        papers: Full paper list (unused; subtopic.papers are used directly)

    Returns:
        Dict keyed by subtopic name:
            {name: {"explicit_gap_count": int,
                    "explicit_gap_sentences": list[str]}}
    """
    try:
        result: Dict[str, Any] = {}

        for subtopic in subtopics:
            count = 0
            quotes: List[str] = []

            for paper in subtopic.papers:
                if not paper.abstract:
                    continue

                # Split into sentences on period, exclamation mark, or question mark
                sentences = re.split(r'[.!?]', paper.abstract)

                for sentence in sentences:
                    s_lower = sentence.lower().strip()
                    if not s_lower:
                        continue

                    matched = (
                        any(phrase in s_lower for phrase in _FUTURE_WORK_PHRASES)
                        or any(phrase in s_lower for phrase in _LIMITATION_PHRASES)
                    )

                    if matched:
                        count += 1
                        if len(quotes) < 3:
                            quotes.append(sentence.strip())

            result[subtopic.name] = {
                "explicit_gap_count": count,
                "explicit_gap_sentences": quotes,
            }

        return result

    except Exception as e:
        warnings.warn(f"enrich_with_explicit_signals failed: {e}")
        return {}


def enrich_with_citation_frontier(
    subtopics: List[Subtopic],
    papers: List[Paper],
) -> Dict[str, Any]:
    """Detect foundational papers within each subtopic that have no follow-up after 2022.

    Builds an intra-cluster directed citation graph using Paper.referenced_works.
    Papers with in-degree >= 2 (cited by at least 2 cluster peers) whose most
    recent citing paper was published <= 2022 are flagged as "frontier" papers —
    influential work that appears to have stalled without follow-up.

    Note: meaningful results require Paper.referenced_works to be populated.
    When the field is empty (the current default from OpenAlex fetch), the graph
    has no edges and citation_frontier_flag will be False for all subtopics.

    Args:
        subtopics: List of Subtopic objects from cluster_by_topic
        papers: Full paper list (reserved for cross-cluster lookups)

    Returns:
        Dict keyed by subtopic name:
            {name: {"citation_frontier_flag": bool,
                    "frontier_papers": list[dict]}}
    """
    try:
        import networkx as nx

        result: Dict[str, Any] = {}

        for subtopic in subtopics:
            cluster_papers = subtopic.papers
            # Build paper ID → Paper lookup for this cluster
            id_to_paper: Dict[str, Paper] = {p.id: p for p in cluster_papers}

            # Directed graph: edge A → B means paper A cites paper B
            # Only edges where both A and B are in this cluster are included
            G = nx.DiGraph()
            G.add_nodes_from(id_to_paper.keys())

            for paper in cluster_papers:
                for ref_id in paper.referenced_works:
                    if ref_id in id_to_paper:
                        G.add_edge(paper.id, ref_id)

            # Identify frontier papers: cited by >= 2 cluster papers,
            # most recent citing paper published <= 2022
            frontier_papers: List[Dict[str, Any]] = []

            for node_id, in_deg in G.in_degree():
                if in_deg < 2:
                    continue

                citing_years = [
                    id_to_paper[pred].year
                    for pred in G.predecessors(node_id)
                    if pred in id_to_paper
                ]
                if not citing_years:
                    continue

                most_recent_citing = max(citing_years)
                if most_recent_citing <= 2022:
                    node_paper = id_to_paper[node_id]
                    frontier_papers.append({
                        "title": node_paper.title,
                        "year": node_paper.year,
                        "in_degree": in_deg,
                    })

            result[subtopic.name] = {
                "citation_frontier_flag": len(frontier_papers) > 0,
                "frontier_papers": frontier_papers,
            }

        return result

    except ImportError:
        warnings.warn("networkx not installed — skipping citation frontier enrichment")
        return {}
    except Exception as e:
        warnings.warn(f"enrich_with_citation_frontier failed: {e}")
        return {}


def enrich_with_concept_isolation(
    subtopics: List[Subtopic],
    papers: List[Paper],
) -> Dict[str, Any]:
    """Measure how conceptually isolated each subtopic is from the rest of the corpus.

    For each subtopic, computes the fraction of its concept set (OpenAlex topic
    display names) that does not appear in papers outside that subtopic.
    A score near 1.0 means this subtopic uses almost entirely unique concepts;
    a score near 0.0 means its concepts are shared throughout the corpus.

    Results are cached to disk keyed on an MD5 hash of all cluster paper IDs
    to avoid repeating the O(n²) concept comparison on subsequent runs.

    Args:
        subtopics: List of Subtopic objects from cluster_by_topic
        papers: Full paper list (used to compute corpus-wide concept frequencies)

    Returns:
        Dict keyed by subtopic name:
            {name: {"concept_isolation_score": float,
                    "isolated_concepts": list[str]}}
    """
    try:
        # Build a stable cache key from the sorted union of all cluster paper IDs
        all_ids = sorted(p.id for s in subtopics for p in s.papers)
        cache_key = hashlib.md5("|".join(all_ids).encode()).hexdigest()
        cache_dir = Path(__file__).parent / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"concept_isolation_{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Corpus-wide frequency: how many papers in the full list use each concept
        global_concept_freq: Dict[str, int] = {}
        for paper in papers:
            for concept in paper.topics:
                if concept:
                    global_concept_freq[concept] = global_concept_freq.get(concept, 0) + 1

        result: Dict[str, Any] = {}

        for subtopic in subtopics:
            cluster_papers = subtopic.papers

            # Frequency of each concept within this cluster
            cluster_concept_freq: Dict[str, int] = {}
            for paper in cluster_papers:
                for concept in paper.topics:
                    if concept:
                        cluster_concept_freq[concept] = (
                            cluster_concept_freq.get(concept, 0) + 1
                        )

            cluster_concepts = set(cluster_concept_freq.keys())

            if not cluster_concepts:
                result[subtopic.name] = {
                    "concept_isolation_score": 0.0,
                    "isolated_concepts": [],
                }
                continue

            # corpus_concepts = concepts that appear in >= 1 paper OUTSIDE this cluster
            corpus_concepts: set = set()
            for concept in cluster_concepts:
                outside_freq = (
                    global_concept_freq.get(concept, 0)
                    - cluster_concept_freq.get(concept, 0)
                )
                if outside_freq > 0:
                    corpus_concepts.add(concept)

            # overlap = concepts shared between this cluster and the rest of the corpus
            overlap = cluster_concepts & corpus_concepts
            isolation_score = 1.0 - (len(overlap) / len(cluster_concepts))

            # isolated_concepts = concepts appearing in fewer than 2 papers outside this cluster
            isolated_concepts = [
                c for c in cluster_concepts
                if (global_concept_freq.get(c, 0) - cluster_concept_freq.get(c, 0)) < 2
            ]

            result[subtopic.name] = {
                "concept_isolation_score": round(isolation_score, 4),
                "isolated_concepts": isolated_concepts[:10],  # cap for readability
            }

        # Persist to cache
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        warnings.warn(f"enrich_with_concept_isolation failed: {e}")
        return {}


def enrich_clusters(
    subtopics: List[Subtopic],
    papers: List[Paper],
    adjacent_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Orchestrate all three enrichment signals for a set of subtopics.

    Calls enrich_with_explicit_signals, enrich_with_citation_frontier, and
    enrich_with_concept_isolation in sequence. Each function is independently
    fault-tolerant — a failure in one does not prevent the others from running.
    Initialises every subtopic with safe zero-valued defaults before merging
    results so callers always see all six signal keys regardless of failures.

    Args:
        subtopics: List of Subtopic objects from cluster_by_topic
        papers: Full paper list from fetch_papers
        adjacent_fields: Reserved for future cross-domain concept comparison

    Returns:
        Dict keyed by subtopic name with all three signal groups merged:
            {name: {explicit_gap_count, explicit_gap_sentences,
                    citation_frontier_flag, frontier_papers,
                    concept_isolation_score, isolated_concepts}}
    """
    # Initialise safe defaults so every key is always present
    result: Dict[str, Any] = {
        s.name: {
            "explicit_gap_count": 0,
            "explicit_gap_sentences": [],
            "citation_frontier_flag": False,
            "frontier_papers": [],
            "concept_isolation_score": 0.0,
            "isolated_concepts": [],
        }
        for s in subtopics
    }

    for enrich_fn in (
        enrich_with_explicit_signals,
        enrich_with_citation_frontier,
        enrich_with_concept_isolation,
    ):
        try:
            signal_data = enrich_fn(subtopics, papers)
            for name, data in signal_data.items():
                if name in result:
                    result[name].update(data)
        except Exception as e:
            warnings.warn(f"{enrich_fn.__name__} raised unexpectedly: {e}")

    return result


if __name__ == "__main__":
    from models import AgentTrace, Paper, Subtopic, TrendPoint
    trace = AgentTrace()

    # ── Test fetch_papers ──────────────────────────────────────
    print("Testing fetch_papers...")
    papers = fetch_papers("federated learning healthcare", trace, limit=10)

    assert isinstance(papers, list),           "FAIL: not a list"
    assert len(papers) > 0,                    "FAIL: empty list"
    assert isinstance(papers[0], Paper),       "FAIL: items not Paper objects"
    assert papers[0].title != "",              "FAIL: title is empty"
    assert papers[0].abstract != "",           "FAIL: abstract is empty"
    assert papers[0].citations >= 0,           "FAIL: negative citations"
    assert isinstance(papers[0].topics, list), "FAIL: topics not a list"
    print(f"  PASS — {len(papers)} papers, first: '{papers[0].title[:50]}'")

    # ── Test cluster_by_topic ──────────────────────────────────
    print("Testing cluster_by_topic...")
    subtopics = cluster_by_topic(papers, trace)

    assert isinstance(subtopics, list),              "FAIL: not a list"
    assert len(subtopics) > 0,                       "FAIL: empty list"
    assert isinstance(subtopics[0], Subtopic),       "FAIL: items not Subtopic objects"
    assert subtopics[0].paper_count > 0,             "FAIL: zero paper count"
    assert subtopics[0].avg_citations >= 0,          "FAIL: negative citations"
    assert 0 < subtopics[0].pct_of_total <= 100,     "FAIL: pct out of range"
    assert sum(s.paper_count for s in subtopics) == len(papers), "FAIL: counts don't add up"
    print(f"  PASS — {len(subtopics)} subtopics, top: '{subtopics[0].name}' ({subtopics[0].paper_count} papers)")

    # ── Test fetch_trends ──────────────────────────────────────
    print("Testing fetch_trends...")
    trends = fetch_trends("federated learning healthcare", trace)

    assert isinstance(trends, list),               "FAIL: not a list"
    assert len(trends) > 0,                        "FAIL: empty list"
    assert isinstance(trends[0], TrendPoint),      "FAIL: items not TrendPoint objects"
    assert trends[0].count > 0,                    "FAIL: zero count"
    assert 2000 < trends[0].year < 2030,           "FAIL: year looks wrong"
    print(f"  PASS — {len(trends)} years of trend data")

    # ── Test semantic_cluster ──────────────────────────────────
    print("Testing semantic_cluster...")
    result = semantic_cluster(papers, trace=trace)

    assert isinstance(result, dict),               "FAIL: not a dict"
    assert "labels" in result,                     "FAIL: missing 'labels' key"
    assert "orphans" in result,                    "FAIL: missing 'orphans' key"
    assert "cluster_sizes" in result,              "FAIL: missing 'cluster_sizes' key"
    assert isinstance(result["orphans"], list),    "FAIL: orphans not a list"
    # Each orphan must be a Paper object
    for o in result["orphans"]:
        assert isinstance(o, Paper),               "FAIL: orphan is not a Paper object"
    print(f"  PASS — {len(result['orphans'])} orphan papers found")

    # ── Final check: confirm Person 2 can use these ───────────
    print("\nConfirming Person 2 compatibility...")
    assert hasattr(subtopics[0], 'name'),          "FAIL: Subtopic missing .name"
    assert hasattr(subtopics[0], 'papers'),        "FAIL: Subtopic missing .papers"
    assert hasattr(subtopics[0], 'avg_citations'), "FAIL: Subtopic missing .avg_citations"
    assert hasattr(subtopics[0], 'pct_of_total'),  "FAIL: Subtopic missing .pct_of_total"
    assert hasattr(papers[0], 'abstract'),         "FAIL: Paper missing .abstract"
    assert hasattr(papers[0], 'citations'),        "FAIL: Paper missing .citations"
    print("  PASS — all fields Person 2 needs are present")

    print("\nALL TESTS PASSED — tools.py is ready for Person 2")
