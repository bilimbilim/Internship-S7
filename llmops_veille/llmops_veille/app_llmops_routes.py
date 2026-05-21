# app_llmops_routes.py
# ==========================================================
# Routes Flask LLMOps (version propre avec Blueprint)
# ==========================================================

from flask import Blueprint, request, jsonify, send_file
import json
from pathlib import Path
import subprocess

#  Blueprint
llmops_bp = Blueprint("llmops", __name__)

#  Fichiers
LLMOPS_LOG  = Path("logs/llmops_runs.jsonl")
LLMOPS_HTML = Path("reports/llmops_report.html")


#  Dashboard HTML
@llmops_bp.route("/llmops")
def llmops_dashboard():
    """Page HTML du rapport LLMOps."""
    
    # Génère le rapport
    subprocess.run(
        ["python", "llmops_report.py", "--output", str(LLMOPS_HTML)],
        check=False
    )

    if LLMOPS_HTML.exists():
        return send_file(str(LLMOPS_HTML))

    return "Aucun rapport disponible. Lance le pipeline.", 404


# Stats JSON
@llmops_bp.route("/api/llmops/stats")
def api_llmops_stats():
    from llmops_report import load_runs, compute_stats

    runs  = load_runs(LLMOPS_LOG)
    stats = compute_stats(runs)

    return jsonify(stats)


#  Runs JSON
@llmops_bp.route("/api/llmops/runs")
def api_llmops_runs():
    n = int(request.args.get("n", 50))
    runs = []

    if LLMOPS_LOG.exists():
        with open(LLMOPS_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except Exception:
                        pass

    return jsonify(runs[-n:])