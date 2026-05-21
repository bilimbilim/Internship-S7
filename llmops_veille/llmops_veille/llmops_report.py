"""
llmops_report.py — Rapport d'exécution LLMOps
==============================================
Lit logs/llmops_runs.jsonl et génère un rapport HTML + stats console.

Usage :
    python llmops_report.py                     # rapport du dernier run
    python llmops_report.py --all               # tous les runs
    python llmops_report.py --output report.html
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

LOG_FILE    = Path("logs/llmops_runs.jsonl")
HTML_OUTPUT = Path("reports/llmops_report.html")


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[!] Fichier de log introuvable : {path}")
        return []
    runs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return runs


def compute_stats(runs: list[dict]) -> dict:
    if not runs:
        return {}

    total      = len(runs)
    successes  = sum(1 for r in runs if r.get("success"))
    failures   = total - successes
    total_lat  = sum(r.get("latency_ms", 0) for r in runs)
    total_in   = sum(r.get("prompt_tokens", 0) for r in runs)
    total_out  = sum(r.get("output_tokens", 0) for r in runs)
    qualities  = [r.get("quality_score", 0) for r in runs if r.get("success")]
    avg_q      = sum(qualities) / len(qualities) if qualities else 0.0
    avg_lat    = total_lat / total if total else 0.0

    by_step = defaultdict(lambda: {"count": 0, "success": 0, "latency": [], "quality": []})
    for r in runs:
        s = r.get("step", "unknown")
        by_step[s]["count"]   += 1
        by_step[s]["success"] += int(r.get("success", False))
        by_step[s]["latency"].append(r.get("latency_ms", 0))
        if r.get("success"):
            by_step[s]["quality"].append(r.get("quality_score", 0))

    step_stats = {}
    for step, d in by_step.items():
        lats = d["latency"]
        qs   = d["quality"]
        step_stats[step] = {
            "count":      d["count"],
            "success":    d["success"],
            "fail_rate":  round((d["count"] - d["success"]) / max(d["count"], 1) * 100, 1),
            "avg_lat_ms": round(sum(lats) / len(lats), 1) if lats else 0,
            "max_lat_ms": round(max(lats), 1) if lats else 0,
            "avg_quality":round(sum(qs) / len(qs), 3) if qs else 0.0,
        }

    return {
        "total":        total,
        "successes":    successes,
        "failures":     failures,
        "success_rate": round(successes / total * 100, 1) if total else 0,
        "avg_lat_ms":   round(avg_lat, 1),
        "total_tokens_in":  total_in,
        "total_tokens_out": total_out,
        "avg_quality":  round(avg_q, 3),
        "by_step":      step_stats,
    }


def print_stats(stats: dict):
    if not stats:
        print("Aucun run trouvé.")
        return
    print("\n" + "=" * 55)
    print("RAPPORT LLMOps — VeilleTech")
    print("=" * 55)
    print(f"  Runs totaux    : {stats['total']}")
    print(f"  Succès         : {stats['successes']}  ({stats['success_rate']}%)")
    print(f"  Échecs         : {stats['failures']}")
    print(f"  Latence moy.   : {stats['avg_lat_ms']} ms")
    print(f"  Tokens (in)    : {stats['total_tokens_in']}")
    print(f"  Tokens (out)   : {stats['total_tokens_out']}")
    print(f"  Qualité moy.   : {stats['avg_quality']:.3f} / 1.0")
    print()
    for step, s in stats["by_step"].items():
        print(f"  [{step}]")
        print(f"    appels       : {s['count']}  |  échecs : {s['fail_rate']}%")
        print(f"    latence moy  : {s['avg_lat_ms']} ms  (max {s['max_lat_ms']} ms)")
        print(f"    qualité moy  : {s['avg_quality']:.3f}")
    print("=" * 55 + "\n")


def generate_html(runs: list[dict], stats: dict, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = ""
    for r in reversed(runs[-200:]):   # les 200 derniers, plus récents en haut
        ok    = r.get("success", False)
        color = "#d4edda" if ok else "#f8d7da"
        status = "✓" if ok else "✗"
        rows += f"""
        <tr style="background:{color}">
          <td>{r.get('timestamp','')[:19]}</td>
          <td>{status}</td>
          <td>{r.get('step','')}</td>
          <td>{r.get('cluster_id','')}</td>
          <td>{r.get('model','')}</td>
          <td>{r.get('latency_ms','')} ms</td>
          <td>{r.get('prompt_tokens','')} + {r.get('output_tokens','')}</td>
          <td>{r.get('retries','')}</td>
          <td>{r.get('quality_score','')}</td>
          <td title="{r.get('quality_notes','')}">{r.get('quality_notes','')[:40]}</td>
          <td style="color:#c00">{r.get('error','')[:60]}</td>
        </tr>"""

    step_cards = ""
    for step, s in stats.get("by_step", {}).items():
        q_color = "#28a745" if s["avg_quality"] >= 0.7 else ("#fd7e14" if s["avg_quality"] >= 0.4 else "#dc3545")
        step_cards += f"""
        <div class="card">
          <h3>{step}</h3>
          <p>Appels : <b>{s['count']}</b></p>
          <p>Taux d'échec : <b>{s['fail_rate']}%</b></p>
          <p>Latence moy : <b>{s['avg_lat_ms']} ms</b></p>
          <p>Qualité : <b style="color:{q_color}">{s['avg_quality']:.3f}</b></p>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>LLMOps — VeilleTech</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2rem; color: #333; }}
  h1   {{ color: #2c3e50; }}
  .summary {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .kpi {{ background: #f0f4ff; border-radius: 8px; padding: 1rem 1.5rem; min-width: 140px; }}
  .kpi .val {{ font-size: 2rem; font-weight: bold; color: #2c3e50; }}
  .kpi .lbl {{ font-size: 0.85rem; color: #666; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.5rem; min-width: 180px; }}
  .card h3 {{ margin: 0 0 .5rem; color: #555; font-size: 1rem; }}
  .card p  {{ margin: .2rem 0; font-size: .9rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
  th {{ background: #2c3e50; color: #fff; padding: 6px 8px; text-align: left; }}
  td {{ padding: 4px 8px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h1>LLMOps — VeilleTech</h1>
<p>Généré le : <b>{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b>
   &nbsp;|&nbsp; {stats.get('total', 0)} runs au total</p>

<div class="summary">
  <div class="kpi"><div class="val">{stats.get('success_rate', 0)}%</div><div class="lbl">Taux de succès</div></div>
  <div class="kpi"><div class="val">{stats.get('avg_lat_ms', 0)} ms</div><div class="lbl">Latence moyenne</div></div>
  <div class="kpi"><div class="val">{stats.get('total_tokens_in', 0) + stats.get('total_tokens_out', 0)}</div><div class="lbl">Tokens totaux</div></div>
  <div class="kpi"><div class="val">{stats.get('avg_quality', 0):.2f}</div><div class="lbl">Qualité moyenne</div></div>
</div>

<h2>Par étape</h2>
<div class="cards">{step_cards}</div>

<h2>Détail des runs</h2>
<table>
<tr>
  <th>Timestamp</th><th>OK</th><th>Étape</th><th>Cluster</th><th>Modèle</th>
  <th>Latence</th><th>Tokens</th><th>Retries</th><th>Qualité</th><th>Notes</th><th>Erreur</th>
</tr>
{rows}
</table>
</body>
</html>"""

    output.write_text(html, encoding="utf-8")
    print(f"[✓] Rapport HTML généré : {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    default=str(LOG_FILE))
    parser.add_argument("--output", default=str(HTML_OUTPUT))
    parser.add_argument("--all",    action="store_true", help="Afficher tous les runs (pas seulement le dernier)")
    args = parser.parse_args()

    runs  = load_runs(Path(args.log))
    stats = compute_stats(runs)
    print_stats(stats)
    generate_html(runs, stats, Path(args.output))


if __name__ == "__main__":
    main()
