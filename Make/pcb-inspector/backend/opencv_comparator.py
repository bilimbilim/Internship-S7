"""
opencv_comparator.py — Comparaison géométrique REF vs INSPECTION.

Reçoit deux listes de détections YOLO et retourne :
  - composants absents
  - composants décalés
  - composants mal orientés
  - zones suspectes (défauts soudure potentiels)
"""

import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass, field
from typing import Optional


# ── Seuils de tolérance (ajuste selon résolution de tes images) ─────────────
SEUIL_DISTANCE_MATCH   = 0.08   # Distance max (normalisée) pour matcher deux composants
SEUIL_DECALAGE_WARN    = 0.02   # Décalage modéré (orange)
SEUIL_DECALAGE_CRIT    = 0.05   # Décalage critique (rouge)
SEUIL_TAILLE_WARN      = 0.20   # Variation taille 20% → warning
SEUIL_TAILLE_CRIT      = 0.40   # Variation taille 40% → critique
SEUIL_RATIO_ROTATION   = 0.25   # Variation ratio W/H → suspicion rotation


@dataclass
class Anomalie:
    type:        str             # ABSENT | DÉCALÉ | ROTATION | TAILLE | INCONNU
    severite:    str             # critical | medium | low
    composant:   str             # label du composant
    description: str
    position_ref:   Optional[list] = None   # [cx, cy] normalisé REF
    position_insp:  Optional[list] = None   # [cx, cy] normalisé INSP
    bbox_insp:      Optional[list] = None   # [x1,y1,x2,y2] pixels INSP
    bbox_ref:       Optional[list] = None
    decalage:       float = 0.0             # distance normalisée
    extra:          dict = field(default_factory=dict)


def comparer(
    detections_ref:  list[dict],
    detections_insp: list[dict],
    img_w: int,
    img_h: int
) -> dict:
    """
    Compare les détections REF et INSPECTION.

    Returns:
        {
          "anomalies": [Anomalie, ...],
          "matched":   [(det_ref, det_insp), ...],
          "absents":   [det_ref, ...],
          "inconnus":  [det_insp, ...],
          "score_aoi": int (0-100)
        }
    """

    if not detections_ref:
        return {
            "anomalies": [],
            "matched":   [],
            "absents":   [],
            "inconnus":  [],
            "score_aoi": 100,
            "note": "Pas de référence — comparaison impossible"
        }

    anomalies  = []
    matched    = []   # paires (ref, insp) matchées
    absents    = []
    inconnus   = list(detections_insp)  # tout est inconnu au départ

    # ── 1. MATCHING par classe + distance euclidienne ────────────────────────
    # Grouper par classe
    ref_by_class  = _group_by_class(detections_ref)
    insp_by_class = _group_by_class(detections_insp)

    matched_insp_ids = set()

    for cls_id, ref_dets in ref_by_class.items():
        insp_dets = insp_by_class.get(cls_id, [])

        if not insp_dets:
            # Tous les composants de cette classe sont absents
            for rd in ref_dets:
                absents.append(rd)
                anomalies.append(Anomalie(
                    type        = "ABSENT",
                    severite    = "critical",
                    composant   = rd["label"],
                    description = (
                        f"{rd['label']} présent sur la référence "
                        f"à la position ({rd['center'][0]:.0f}, {rd['center'][1]:.0f}) "
                        f"— non trouvé sur la carte inspectée."
                    ),
                    position_ref = rd["bbox_rel"][:2],
                    bbox_ref     = rd["bbox"]
                ))
            continue

        # Matrice de distance entre centres (normalisés)
        ref_centers  = np.array([d["bbox_rel"][:2] for d in ref_dets])
        insp_centers = np.array([d["bbox_rel"][:2] for d in insp_dets])
        dist_matrix  = cdist(ref_centers, insp_centers)

        assigned_insp = set()

        for ri, rd in enumerate(ref_dets):
            # Meilleur match disponible
            row = dist_matrix[ri].copy()
            for ai in assigned_insp:
                row[ai] = np.inf

            best_idx  = int(np.argmin(row))
            best_dist = float(row[best_idx])

            if best_dist > SEUIL_DISTANCE_MATCH:
                # Trop loin — composant absent
                absents.append(rd)
                anomalies.append(Anomalie(
                    type        = "ABSENT",
                    severite    = "critical",
                    composant   = rd["label"],
                    description = (
                        f"{rd['label']} attendu en ({rd['center'][0]:.0f}, {rd['center'][1]:.0f}) "
                        f"— le plus proche est à {best_dist:.1%} de distance, trop éloigné."
                    ),
                    position_ref = rd["bbox_rel"][:2],
                    bbox_ref     = rd["bbox"]
                ))
                continue

            id_ = insp_dets[best_idx]
            assigned_insp.add(best_idx)
            matched_insp_ids.add(id_["id"])
            matched.append((rd, id_))

            # ── 2. ANALYSE du composant matché ──────────────────────────────
            _analyser_composant(rd, id_, best_dist, anomalies)

    # ── 3. Composants inconnus (présents en INSP mais pas en REF) ────────────
    for d in detections_insp:
        if d["id"] not in matched_insp_ids:
            anomalies.append(Anomalie(
                type        = "INCONNU",
                severite    = "medium",
                composant   = d["label"],
                description = (
                    f"{d['label']} détecté en ({d['center'][0]:.0f}, {d['center'][1]:.0f}) "
                    f"— absent sur la référence (composant ajouté ou erreur de détection)."
                ),
                position_insp = d["bbox_rel"][:2],
                bbox_insp     = d["bbox"]
            ))
            inconnus.append(d)

    # ── 4. Score AOI ─────────────────────────────────────────────────────────
    score = _calculer_score(anomalies, len(detections_ref))

    return {
        "anomalies":  anomalies,
        "matched":    matched,
        "absents":    absents,
        "inconnus":   [d for d in detections_insp if d["id"] not in matched_insp_ids],
        "score_aoi":  score
    }


def _analyser_composant(rd: dict, id_: dict, distance: float, anomalies: list):
    """Analyse un composant matché : décalage, rotation, taille."""
    label = rd["label"]

    # Décalage
    if distance > SEUIL_DECALAGE_CRIT:
        sev = "critical"
        desc = f"Décalage important de {distance:.1%} par rapport à la position de référence."
    elif distance > SEUIL_DECALAGE_WARN:
        sev = "medium"
        desc = f"Décalage modéré de {distance:.1%} par rapport à la position de référence."
    else:
        sev = None

    if sev:
        anomalies.append(Anomalie(
            type          = "DÉCALÉ",
            severite      = sev,
            composant     = label,
            description   = desc,
            position_ref  = rd["bbox_rel"][:2],
            position_insp = id_["bbox_rel"][:2],
            bbox_insp     = id_["bbox"],
            bbox_ref      = rd["bbox"],
            decalage      = round(distance, 4)
        ))

    # Variation de taille (peut indiquer mauvais composant ou rotation)
    w_ref,  h_ref  = rd["bbox_rel"][2],  rd["bbox_rel"][3]
    w_insp, h_insp = id_["bbox_rel"][2], id_["bbox_rel"][3]

    delta_w = abs(w_insp - w_ref) / (w_ref + 1e-6)
    delta_h = abs(h_insp - h_ref) / (h_ref + 1e-6)
    max_delta = max(delta_w, delta_h)

    if max_delta > SEUIL_TAILLE_CRIT:
        anomalies.append(Anomalie(
            type        = "TAILLE",
            severite    = "critical",
            composant   = label,
            description = (
                f"Taille très différente de la référence "
                f"(largeur ×{w_insp/w_ref:.1f}, hauteur ×{h_insp/h_ref:.1f}). "
                f"Composant incorrect ou mal orienté."
            ),
            bbox_insp = id_["bbox"],
            bbox_ref  = rd["bbox"]
        ))
    elif max_delta > SEUIL_TAILLE_WARN:
        anomalies.append(Anomalie(
            type        = "TAILLE",
            severite    = "medium",
            composant   = label,
            description = (
                f"Taille légèrement différente de la référence "
                f"(variation {max_delta:.0%}). Vérifier l'orientation."
            ),
            bbox_insp = id_["bbox"],
            bbox_ref  = rd["bbox"]
        ))

    # Suspicion de rotation (ratio W/H inversé)
    ratio_ref  = w_ref  / (h_ref  + 1e-6)
    ratio_insp = w_insp / (h_insp + 1e-6)
    delta_ratio = abs(ratio_insp - ratio_ref) / (ratio_ref + 1e-6)

    if delta_ratio > SEUIL_RATIO_ROTATION and max_delta < SEUIL_TAILLE_WARN:
        anomalies.append(Anomalie(
            type        = "ROTATION",
            severite    = "critical",
            composant   = label,
            description = (
                f"Ratio largeur/hauteur inversé — "
                f"REF={ratio_ref:.2f} vs INSP={ratio_insp:.2f}. "
                f"Composant probablement retourné à 90° ou 180°."
            ),
            bbox_insp = id_["bbox"],
            bbox_ref  = rd["bbox"]
        ))


def _group_by_class(detections: list[dict]) -> dict:
    groups = {}
    for d in detections:
        groups.setdefault(d["class_id"], []).append(d)
    return groups


def _calculer_score(anomalies: list[Anomalie], nb_ref: int) -> int:
    if nb_ref == 0:
        return 100

    penalite = 0
    for a in anomalies:
        if a.severite == "critical":
            penalite += 20
        elif a.severite == "medium":
            penalite += 8
        else:
            penalite += 3

    score = max(0, 100 - penalite)
    return score


def anomalies_to_dict(anomalies: list[Anomalie]) -> list[dict]:
    """Sérialise les anomalies pour JSON / API."""
    return [
        {
            "type":           a.type,
            "severite":       a.severite,
            "composant":      a.composant,
            "description":    a.description,
            "position_ref":   a.position_ref,
            "position_insp":  a.position_insp,
            "bbox_insp":      a.bbox_insp,
            "bbox_ref":       a.bbox_ref,
            "decalage":       a.decalage
        }
        for a in anomalies
    ]
