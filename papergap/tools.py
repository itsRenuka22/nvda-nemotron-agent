import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
from models import Paper, AgentTrace, TrendPoint, Subtopic


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


def fetch_papers(
    topic: str,
    trace: AgentTrace,
    years: str = "2022-2025",
    limit: int = 200
) -> List[Paper]:
    """Fetch papers from OpenAlex API and cache results.

    Args:
        topic: Search query for papers
        trace: AgentTrace object for logging progress
        years: Year range filter (e.g., "2022-2025")
        limit: Maximum number of papers to fetch

    Returns:
        List of Paper objects with abstracts decoded from inverted index
    """
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"{topic.replace(' ', '_')}_{years}.json"

    # Try to load from cache
    if cache_file.exists():
        trace.log(f"Loading cached papers for '{topic}'")
        with open(cache_file, 'r') as f:
            data = json.load(f)
    else:
        # Fetch from OpenAlex API
        trace.log(f"Fetching papers for '{topic}' from OpenAlex API")

        params = {
            "search": topic,
            "filter": f"publication_year:{years},has_abstract:true",
            "sort": "cited_by_count:desc",
            "per_page": limit,
            "select": "id,title,abstract_inverted_index,publication_year,cited_by_count,topics"
        }

        response = requests.get("https://api.openalex.org/works", params=params)
        response.raise_for_status()

        data = response.json()
        trace.log(f"Got {len(data.get('results', []))} results from API")

        # Cache the raw JSON
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        trace.log(f"Cached results to {cache_file}")

    # Convert to Paper objects
    papers = []
    results = data.get('results', [])
    trace.log(f"Processing {len(results)} papers")

    for i, item in enumerate(results):
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
    return papers


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
        # Use first topic or "Uncategorized"
        topic_name = paper.topics[0] if paper.topics else "Uncategorized"

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
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        import numpy as np
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
        trace.log(f"Loading SentenceTransformer model...")

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Build text list: use abstract if exists, else title
    texts = [paper.abstract if paper.abstract else paper.title for paper in papers]

    if trace:
        trace.log(f"Encoding {len(texts)} papers into embeddings...")

    # Encode texts
    embeddings = model.encode(texts, show_progress_bar=False)

    if trace:
        trace.log(f"Running KMeans clustering with {n_clusters} clusters...")

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Find orphans: papers far from cluster centers
    if trace:
        trace.log(f"Computing distances to identify orphan papers...")

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
