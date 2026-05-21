"""
gemini_explainer.py — Explication en langage naturel des anomalies détectées.

2 passes :
  1. Gemini décrit librement les zones suspectes sur l'image
  2. Gemini structure son analyse en JSON à partir de cette description + résultats OpenCV
"""

import json
import base64
import httpx
import numpy as np
import cv2
from typing import Optional


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


def _encode_image(image: np.ndarray) -> tuple[str, str]:
    """Encode une image numpy en base64 pour l'API Gemini."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode(), "image/jpeg"


async def expliquer(
    api_key:          str,
    image_ref:        Optional[np.ndarray],
    image_insp:       np.ndarray,
    anomalies_opencv: list[dict],
    detections_ref:   list[dict],
    detections_insp:  list[dict],
    chain_context:    str = ""
) -> dict:
    """
    Passe 1 : description visuelle libre par Gemini.
    Passe 2 : enrichissement JSON des anomalies OpenCV.
    """

    description = await _passe1_description(
        api_key, image_ref, image_insp, chain_context
    )

    result = await _passe2_structuration(
        api_key, description, anomalies_opencv,
        detections_ref, detections_insp, chain_context
    )

    result["description_visuelle"] = description
    return result


async def _passe1_description(
    api_key: str,
    image_ref: Optional[np.ndarray],
    image_insp: np.ndarray,
    chain_context: str
) -> str:
    context_line = f"\nContexte de la chaîne : {chain_context}" if chain_context else ""

    if image_ref is not None:
        prompt = f"""Tu es un expert en inspection visuelle de cartes électroniques PCB.{context_line}

Je te fournis deux images :
- IMAGE 1 : carte de RÉFÉRENCE (conforme, sans défaut)
- IMAGE 2 : carte à INSPECTER

Examine attentivement les deux images.

Décris en détail tout ce que tu observes sur la carte inspectée comparée à la référence :
- Quels composants sont présents sur la référence mais semblent manquants ou déplacés ?
- Y a-t-il des différences de position, d'orientation, de taille ?
- Y a-t-il des signes visuels de défauts de soudure (couleur, forme, brillance anormale) ?
- Y a-t-il des zones endommagées (brûlures, traces, corrosion) ?

Sois précis et factuel. Décris les positions avec des repères (haut-gauche, centre, etc.).
Réponse en texte libre, langue française."""

        parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": _encode_image(image_ref)[0]}},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": _encode_image(image_insp)[0]}}
        ]
    else:
        prompt = f"""Tu es un expert en inspection visuelle de cartes électroniques PCB.{context_line}

Examine attentivement cette carte PCB.

ÉTAPE 1 — CARTOGRAPHIE :
Liste tous les composants que tu identifies sur la carte avec leur position.
Utilise les marquages sérigraphiques visibles, la forme et la couleur.

ÉTAPE 2 — ANALYSE DES ANOMALIES :
Pour chaque zone suspecte, décris précisément ce que tu observes :
- Pads vides ou empreintes sans composant
- Composants mal positionnés ou orientés
- Défauts de soudure visibles
- Zones endommagées

Sois précis et factuel. Décris les positions avec des repères (haut-gauche, centre, etc.).
Réponse en texte libre, langue française."""

        parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": _encode_image(image_insp)[0]}}
        ]

    response = await _call_gemini(api_key, parts, temperature=0.2, max_tokens=1500)
    return response


async def _passe2_structuration(
    api_key:          str,
    description:      str,
    anomalies_opencv: list[dict],
    detections_ref:   list[dict],
    detections_insp:  list[dict],
    chain_context:    str
) -> dict:
    opencv_summary = json.dumps(anomalies_opencv, ensure_ascii=False, indent=2)
    ref_labels  = [d["label"] for d in detections_ref]
    insp_labels = [d["label"] for d in detections_insp]

    context_line = f"Contexte chaîne : {chain_context}\n" if chain_context else ""

    prompt = f"""Tu es un expert en inspection PCB.
{context_line}
Tu as déjà analysé visuellement la carte et produit cette description :
---
{description}
---

L'algorithme de comparaison géométrique (OpenCV) a détecté ces anomalies :
{opencv_summary}

Composants détectés sur la RÉFÉRENCE : {ref_labels}
Composants détectés sur l'INSPECTION : {insp_labels}

En combinant ton analyse visuelle et les données algorithmiques :
1. Confirme ou nuance chaque anomalie OpenCV avec ton observation visuelle
2. Ajoute les anomalies visuelles que tu as détectées et qui ne sont pas dans OpenCV
3. Attribue une sévérité finale : critical (bloque la production), medium (à vérifier), low (mineur)

Réponds UNIQUEMENT avec un objet JSON valide, sans markdown, sans backtick, sans commentaire :
{{
  "statut": "ok" ou "nok",
  "score_gemini": entier 0-100,
  "message": "résumé global en 1 phrase",
  "recommandation": "action corrective principale",
  "anomalies_enrichies": [
    {{
      "type": "ABSENT|DÉCALÉ|ROTATION|TAILLE|SOUDURE|PCB|INCONNU",
      "severite": "critical|medium|low",
      "composant": "nom du composant",
      "description": "explication détaillée combinant vision + algo",
      "source": "opencv|gemini|both",
      "bbox_insp": [x1,y1,x2,y2] ou null
    }}
  ]
}}"""

    response_text = await _call_gemini(api_key, [{"text": prompt}], temperature=0.1, max_tokens=2000)

    try:
        clean = response_text.strip()
        clean = clean.replace("```json", "").replace("```", "").strip()
        start = clean.find('{')
        end = clean.rfind('}') + 1
        if start >= 0 and end > start:
            clean = clean[start:end]
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        return {
            "statut":             "nok",
            "score_gemini":       50,
            "message":            "Analyse partielle — erreur de parsing JSON",
            "recommandation":     "Vérification manuelle requise",
            "anomalies_enrichies": anomalies_opencv
        }


async def _call_gemini(
    api_key: str,
    parts:   list,
    temperature: float = 0.1,
    max_tokens:  int   = 2000
) -> str:
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature":     temperature,
            "maxOutputTokens": max_tokens
        }
    }

    # Force JSON uniquement pour la passe 2
    if len(parts) == 1 and parts[0].get("text", "").startswith("Tu es un expert en inspection PCB"):
        payload["generationConfig"]["response_mime_type"] = "application/json"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{GEMINI_API_URL}?key={api_key}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

    if resp.status_code != 200:
        err = resp.json()
        raise ValueError(f"Gemini API error {resp.status_code}: {err.get('error', {}).get('message', 'unknown')}")

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]