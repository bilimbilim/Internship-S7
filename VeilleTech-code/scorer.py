from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")


def _embed_keywords(keywords: list[str]):
    """
    Projette chaque mot-clé individuellement dans l'espace latent
    et retourne la liste des vecteurs (un par mot-clé).
    """
    return model.encode(keywords, convert_to_tensor=True)   # shape: (n_kw, dim)


def score_papers_for_cluster(papers: list[dict], cluster: dict, top_k: int = 10) -> list[dict]:
    """
    Score les articles d'un cluster en projetant CHAQUE mot-clé individuellement,
    puis en calculant la similarité de chaque article avec CHACUN des mots-clés.

    Score final = max de toutes les similarités mot-clé × article
    (on prend le max car un article pertinent n'a pas besoin de couvrir
    tous les mots-clés, juste d'en frapper un fort).

    Retourne les top_k articles les mieux scorés.
    """
    keywords = cluster["keywords"]

    if not papers:
        return []

    # --- 1. Projeter chaque mot-clé individuellement ---
    kw_vecs = _embed_keywords(keywords)           

    # --- 2. Projeter chaque article (titre + abstract) ---
    texts = [p["title"] + " " + p["abstract"] for p in papers]
    paper_vecs = model.encode(texts, convert_to_tensor=True)  

    # --- 3. Matrice de similarité : (n_papers, n_kw) ---
    sim_matrix = util.cos_sim(paper_vecs, kw_vecs)            

    scored = []
    for i, paper in enumerate(papers):
        
        score = sim_matrix[i].max().item()
        paper = dict(paper)   
        paper["score"] = round(score, 7)
        scored.append(paper)

    # --- 4. Trier et garder le top_k ---
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]