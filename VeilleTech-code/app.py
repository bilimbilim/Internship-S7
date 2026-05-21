from flask import Flask, render_template, request, jsonify
from collections import defaultdict
import threading
import json
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from collector  import fetch_papers_by_cluster, _DEFAULT_CLUSTERS
from scorer     import score_papers_for_cluster
from clustering import rename_clusters
from resum      import summarize_cluster
from rss        import generate_rss

app = Flask(__name__)

CLUSTERS_FILE = os.path.join(_HERE, "clusters.json")
PAPERS_FILE   = os.path.join(_HERE, "papers.json")

_pipeline_status = {"running": False, "log": [], "done": False, "error": False}


#  CLUSTERS — lecture / écriture

def load_clusters():
    """Charge clusters.json. Si absent, crée le fichier depuis les défauts."""
    if os.path.exists(CLUSTERS_FILE):
        with open(CLUSTERS_FILE, encoding="utf-8") as f:
            return json.load(f)

    save_clusters(_DEFAULT_CLUSTERS)
    return _DEFAULT_CLUSTERS


def save_clusters(clusters: list):
    with open(CLUSTERS_FILE, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)


#  ROUTES

@app.route("/")
def home():
    papers = []
    if os.path.exists(PAPERS_FILE):
        with open(PAPERS_FILE, encoding="utf-8") as f:
            papers = json.load(f)

    grouped = defaultdict(list)
    for p in papers:
        grouped[p.get("cluster", 0)].append(p)

    cluster_list = []
    seen_ids = set()
    for p in papers:
        cid = p.get("cluster", 0)
        if cid not in seen_ids:
            seen_ids.add(cid)
            cluster_list.append({
                "id":     cid,
                "title":  p.get("cluster_title", f"Cluster {cid}"),
                "papers": grouped[cid],
            })

    return render_template("index.html", cluster_list=cluster_list, total=len(papers))


#  API clusters 

@app.route("/api/clusters", methods=["GET"])
def api_get_clusters():
    return jsonify(load_clusters())


@app.route("/api/clusters", methods=["POST"])
def api_save_clusters():
    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({"error": "Expected a list"}), 400
    for i, c in enumerate(data):
        c["id"] = i
    save_clusters(data)
    return jsonify({"ok": True})


# API papers 

@app.route("/api/papers", methods=["GET"])
def api_papers():
    if not os.path.exists(PAPERS_FILE):
        return jsonify({})
    with open(PAPERS_FILE, encoding="utf-8") as f:
        papers = json.load(f)
    grouped = defaultdict(list)
    for p in papers:
        grouped[str(p.get("cluster", 0))].append(p)
    return jsonify(dict(grouped))




def _run_pipeline_thread():
    global _pipeline_status
    _pipeline_status = {"running": True, "log": [], "done": False, "error": False}

    def log(msg):
        print(msg)
        _pipeline_status["log"].append(msg)

    try:
        clusters = load_clusters()
        log(f" Démarrage du pipeline avec {len(clusters)} cluster(s)...")

        log(" [1/5] Collecte des articles par cluster...")
        cluster_data = fetch_papers_by_cluster(rows_per_source=25, clusters=clusters)
        total = sum(len(d["papers"]) for d in cluster_data.values())
        log(f"   {total} articles récupérés")

        log(" [2/5] Scoring mot-à-mot (top 10 par cluster)...")
        for cid, data in cluster_data.items():
            top10 = score_papers_for_cluster(data["papers"], data["cluster"], top_k=10)
            data["papers"] = top10
            log(f"   [{data['cluster']['label']}] → {len(top10)} articles retenus")

        log("  [3/5] Renommage des clusters par Llama...")
        cluster_data = rename_clusters(cluster_data)
        for d in cluster_data.values():
            log(f"   → « {d['cluster']['label']} »")

        log("  [4/5] Résumés (1 appel Llama par cluster)...")
        for cid, data in cluster_data.items():
            label  = data["cluster"]["label"]
            papers = data["papers"]
            log(f"   [{label}] : {len(papers)} articles...")
            try:
                data["papers"] = summarize_cluster(papers)
            except Exception as e:
                log(f"    Erreur Llama : {e}")
                for p in papers:
                    p.setdefault("summary", "Résumé non disponible.")

        log(" [5/5] Sauvegarde...")
        new_papers = [p for d in cluster_data.values() for p in d["papers"]]

        old_papers = []
        if os.path.exists(PAPERS_FILE):
            with open(PAPERS_FILE, encoding="utf-8") as f:
                old_papers = json.load(f)

        unique = {}
        for p in old_papers + new_papers:
            if p.get("url"):
                unique[p["url"]] = p
        papers = sorted(unique.values(), key=lambda x: -x.get("score", 0))[:400]

        with open(PAPERS_FILE, "w", encoding="utf-8") as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)

        generate_rss(papers)
        log(f" Pipeline terminé — {len(new_papers)} nouveaux articles sauvegardés.")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        _pipeline_status["error"] = True
        _pipeline_status["log"].append(f" Erreur fatale : {e}")
    finally:
        _pipeline_status["running"] = False
        _pipeline_status["done"]    = True


@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    if _pipeline_status["running"]:
        return jsonify({"error": "Pipeline déjà en cours"}), 409
    threading.Thread(target=_run_pipeline_thread, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/pipeline/status", methods=["GET"])
def api_pipeline_status():
    return jsonify({
        "running": _pipeline_status["running"],
        "done":    _pipeline_status["done"],
        "error":   _pipeline_status["error"],
        "log":     _pipeline_status["log"],
    })


if __name__ == "__main__":
    app.run(debug=False)