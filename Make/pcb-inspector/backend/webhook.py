"""
webhook.py — Envoi du rapport complet vers Make (ex-Integromat).

Payload envoyé :
  - métadonnées (timestamp, batch, chaîne)
  - résultats YOLO (REF + INSP)
  - anomalies OpenCV (géométrique)
  - rapport Gemini (explication + enrichissement)
  - score AOI
  - images annotées (base64)
  - logs qualité
"""

import httpx
import json
from datetime import datetime
from typing import Optional


async def envoyer_rapport(
    webhook_url:      str,
    batch_id:         str,
    fichier_name:     str,
    chain_context:    str,
    detections_ref:   list[dict],
    detections_insp:  list[dict],
    anomalies_opencv: list[dict],
    rapport_gemini:   dict,
    score_aoi:        int,
    image_insp_b64:   str,
    image_ref_b64:    Optional[str] = None,
) -> dict:
    """
    Construit et envoie le payload complet à Make.
    Retourne {"ok": True} ou {"ok": False, "error": str}
    """

    nb_anomalies = len(rapport_gemini.get("anomalies_enrichies", anomalies_opencv))
    critiques    = sum(
        1 for a in rapport_gemini.get("anomalies_enrichies", anomalies_opencv)
        if a.get("severite") == "critical"
    )

    payload = {
        # ── Métadonnées ──────────────────────────────────────────────────────
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "batch_id":        batch_id,
        "fichier":         fichier_name,
        "chaine_contexte": chain_context or None,

        # ── Verdict global ───────────────────────────────────────────────────
        "statut":          rapport_gemini.get("statut", "nok"),
        "score_aoi":       score_aoi,
        "score_gemini":    rapport_gemini.get("score_gemini", score_aoi),
        "nb_anomalies":    nb_anomalies,
        "nb_critiques":    critiques,

        # ── Message et recommandation ─────────────────────────────────────────
        "message":         rapport_gemini.get("message", ""),
        "recommandation":  rapport_gemini.get("recommandation", ""),
        "description_visuelle": rapport_gemini.get("description_visuelle", ""),

        # ── Détections YOLO ──────────────────────────────────────────────────
        "yolo": {
            "nb_composants_ref":  len(detections_ref),
            "nb_composants_insp": len(detections_insp),
            "detections_ref":     detections_ref,
            "detections_insp":    detections_insp
        },

        # ── Anomalies OpenCV (brutes) ─────────────────────────────────────────
        "anomalies_opencv": anomalies_opencv,

        # ── Anomalies enrichies Gemini ────────────────────────────────────────
        "anomalies": rapport_gemini.get("anomalies_enrichies", anomalies_opencv),

        # ── Images annotées (base64 JPEG) ─────────────────────────────────────
        "images": {
            "inspection_annotee": image_insp_b64,
            "reference_annotee":  image_ref_b64
        },

        # ── Log qualité ───────────────────────────────────────────────────────
        "log_qualite": _build_log(
            batch_id, fichier_name, len(detections_ref),
            len(detections_insp), nb_anomalies, critiques,
            score_aoi, rapport_gemini.get("statut", "nok")
        )
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                webhook_url,
                content=json.dumps(payload, ensure_ascii=False),
                headers={"Content-Type": "application/json; charset=utf-8"}
            )

        if resp.status_code in (200, 201, 202, 204):
            return {"ok": True, "status_code": resp.status_code}
        else:
            return {
                "ok": False,
                "error": f"HTTP {resp.status_code}",
                "body": resp.text[:200]
            }

    except httpx.TimeoutException:
        return {"ok": False, "error": "Timeout (30s) — Make ne répond pas"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _build_log(
    batch_id, fichier, nb_ref, nb_insp,
    nb_anomalies, critiques, score, statut
) -> str:
    lines = [
        f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC]",
        f"Batch       : {batch_id}",
        f"Fichier     : {fichier}",
        f"Composants  : {nb_ref} REF / {nb_insp} INSP",
        f"Anomalies   : {nb_anomalies} dont {critiques} critique(s)",
        f"Score AOI   : {score}/100",
        f"Statut      : {statut.upper()}",
        "─" * 40
    ]
    return "\n".join(lines)
