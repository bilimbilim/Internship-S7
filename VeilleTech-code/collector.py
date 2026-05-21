import requests
import xml.etree.ElementTree as ET
import json
import os


_DEFAULT_CLUSTERS = [
    {
        "id": 0,
        "label": "Large Language Models",
        "keywords": [
            "large language model", "LLM", "GPT", "transformer",
            "instruction tuning", "RLHF", "fine-tuning", "prompt", "tokenizer",
        ],
        "query_str": (
            "large language model OR LLM OR GPT OR transformer "
            "OR instruction tuning OR RLHF OR fine-tuning"
        ),
    },
    {
        "id": 1,
        "label": "Generative Models & Diffusion",
        "keywords": [
            "diffusion model", "generative model", "GAN", "VAE",
            "image synthesis", "text-to-image", "latent diffusion", "score matching",
        ],
        "query_str": (
            "diffusion model OR generative model OR GAN OR VAE "
            "OR image synthesis OR text-to-image OR latent diffusion"
        ),
    },
    {
        "id": 2,
        "label": "Reinforcement Learning",
        "keywords": [
            "reinforcement learning", "reward", "policy gradient", "Q-learning",
            "actor critic", "multi-agent", "offline RL", "model-based RL",
        ],
        "query_str": (
            "reinforcement learning OR reward shaping OR policy gradient "
            "OR Q-learning OR actor critic OR multi-agent reinforcement"
        ),
    },
    {
        "id": 3,
        "label": "Graph & Geometric Learning",
        "keywords": [
            "graph neural network", "GNN", "graph transformer", "geometric deep learning",
            "node classification", "link prediction", "knowledge graph", "message passing",
        ],
        "query_str": (
            "graph neural network OR GNN OR graph transformer "
            "OR geometric deep learning OR knowledge graph OR message passing"
        ),
    },
]

CLUSTERS = _DEFAULT_CLUSTERS


def _load_clusters():
    """Charge depuis clusters.json si présent, sinon retourne les défauts."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clusters.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return _DEFAULT_CLUSTERS


#  FETCHERS

def _fetch_crossref(query_str, rows=30):
    url = (
        "https://api.crossref.org/works"
        f"?query={query_str}"
        "&filter=type:journal-article"
        f"&rows={rows}"
        "&sort=published&order=desc"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        return []

    papers = []
    for item in data["message"]["items"]:
        title = item.get("title", [""])[0]
        if "Title Pending" in title:
            continue
        papers.append({
            "title":    title,
            "abstract": item.get("abstract", ""),
            "url":      item.get("URL", ""),
            "source":   "Crossref",
        })
    return papers


def _fetch_arxiv(query_str, rows=30):
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{query_str}"
        f"&start=0&max_results={rows}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return []

    root = ET.fromstring(r.text)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title_el   = entry.find("{http://www.w3.org/2005/Atom}title")
        summary_el = entry.find("{http://www.w3.org/2005/Atom}summary")
        link_el    = entry.find("{http://www.w3.org/2005/Atom}id")
        papers.append({
            "title":    title_el.text.strip()   if title_el   is not None else "",
            "abstract": summary_el.text.strip() if summary_el is not None else "",
            "url":      link_el.text.strip()    if link_el    is not None else "",
            "source":   "arXiv",
        })
    return papers


#  FONCTION PRINCIPALE

def fetch_papers_by_cluster(rows_per_source=25, clusters=None):
    """
    Pour chaque cluster, récupère des articles depuis arXiv + Crossref.
    Si `clusters` n'est pas fourni, charge depuis clusters.json (ou défauts).
    """
    if clusters is None:
        clusters = _load_clusters()

    result = {}
    for cluster in clusters:
        print(f"  → Cluster [{cluster['label']}] : collecte en cours...")

        crossref = _fetch_crossref(cluster["query_str"], rows=rows_per_source)
        arxiv    = _fetch_arxiv(cluster["query_str"],    rows=rows_per_source)

        papers = crossref + arxiv

        # Dédoublonner
        seen = {}
        for p in papers:
            if p["url"] and p["url"] not in seen:
                seen[p["url"]] = p
        papers = list(seen.values())

        for p in papers:
            p["cluster"]       = cluster["id"]
            p["cluster_title"] = cluster["label"]

        result[cluster["id"]] = {"cluster": cluster, "papers": papers}
        print(f"     {len(papers)} articles récupérés")

    return result