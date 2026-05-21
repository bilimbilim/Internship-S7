import warnings
import logging
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

import json
import threading
import webbrowser
import time
_HERE = os.path.dirname(os.path.abspath(__file__))

from collector  import fetch_papers_by_cluster, _DEFAULT_CLUSTERS
from scorer     import score_papers_for_cluster
from clustering import rename_clusters
from resum      import summarize_cluster
from rss        import generate_rss
from app        import app
from pyngrok import ngrok



def load_clusters():
    """Charge clusters.json s'il existe, sinon utilise _DEFAULT_CLUSTERS."""
    path = os.path.join(_HERE, "clusters.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            clusters = json.load(f)
        print(f"     clusters.json chargé ({len(clusters)} clusters)")
        return clusters
    print(f"     clusters.json absent → utilisation des {len(_DEFAULT_CLUSTERS)} clusters par défaut")
    return _DEFAULT_CLUSTERS


def run_pipeline():
    print("=" * 55)
    print("VEILLE TECHNOLOGIQUE IA — Lancement du pipeline")
    print("=" * 55)

    print("\n[0] Chargement des clusters...")
    clusters = load_clusters()
    for c in clusters:
        print(f"  · [{c['id']}] {c['label']} — {len(c.get('keywords', []))} mots-clés")


    print("\n[1/5] Collecte des articles par cluster...")
    cluster_data = fetch_papers_by_cluster(rows_per_source=25, clusters=clusters)
    total = sum(len(d["papers"]) for d in cluster_data.values())
    print(f"     {total} articles récupérés au total")

    if total == 0:
        print(" Aucun article récupéré")
        return {}

    print("\n[2/5] Scoring par mots-clés individuels (top 10 par cluster)...")
    for cid, data in cluster_data.items():
        cluster = data["cluster"]
        papers  = data["papers"]
        top10   = score_papers_for_cluster(papers, cluster, top_k=10)
        data["papers"] = top10
        print(f"  → Cluster [{cluster['label']}] : {len(top10)} articles retenus")

    print("\n[3/5] Renommage des clusters par Llama...")
    cluster_data = rename_clusters(cluster_data)

    print("\n[4/5] Résumé des articles par cluster (1 appel Llama par cluster)...")
    for cid, data in cluster_data.items():
        label  = data["cluster"]["label"]
        papers = data["papers"]
        print(f"  → Cluster [{label}] : résumé de {len(papers)} articles en 1 appel...")
        try:
            data["papers"] = summarize_cluster(papers)
        except Exception as e:
            print(f"     Erreur Llama pour ce cluster : {e}")
            for p in papers:
                p.setdefault("summary", "Résumé non disponible.")

    
    print("\n[5/5] Sauvegarde...")
    new_papers = [p for data in cluster_data.values() for p in data["papers"]]

    papers_path = os.path.join(_HERE, "papers.json")
    old_papers = []
    if os.path.exists(papers_path):
        with open(papers_path, encoding="utf-8") as f:
            old_papers = json.load(f)

    unique = {}
    for p in old_papers + new_papers:
        if p.get("url"):
            unique[p["url"]] = p
    papers = sorted(unique.values(), key=lambda x: -x.get("score", 0))[:400]

    with open(papers_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    generate_rss(papers)

    print(f"\nPipeline terminé — {len(new_papers)} nouveaux articles sauvegardés.")
    return cluster_data



NGROK_DOMAIN = "liberty-unarrogant-alaya.ngrok-free.dev"


def open_browser():
    time.sleep(1.5)
    webbrowser.open(f"https://{NGROK_DOMAIN}")


def main():
    run_pipeline()

    # Tunnel ngrok
    try:
        public_url = ngrok.connect(5000, domain=NGROK_DOMAIN)
        print(f"\n URL publique : {public_url}")
        print("   Mets cette URL en favori — elle ne changera jamais.")
    except Exception as e:
        print(f"\n ngrok non disponible : {e}")
        print("   Accès local uniquement : http://127.0.0.1:5000")

    threading.Thread(target=open_browser, daemon=True).start()
    print("\nServeur web lancé.")
    app.run(debug=False)


if __name__ == "__main__":
    main()