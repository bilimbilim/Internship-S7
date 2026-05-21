"""
annotator.py — Dessine les annotations sur les images PCB.
Produit l'image annotée envoyée à Make / affichée dans le frontend.
"""

import cv2
import numpy as np
import base64
from typing import Optional


# Palette de couleurs par type d'anomalie
COLORS = {
    "ABSENT":   (59,  78, 255),    # Rouge
    "DÉCALÉ":   (11, 158, 245),    # Ambre
    "ROTATION": (237, 58, 124),    # Violet
    "TAILLE":   (212, 182,   6),   # Cyan
    "SOUDURE":  (24,  24, 190),    # Rouge sombre
    "PCB":      (5,  150, 105),    # Vert
    "INCONNU":  (99,  99,  99),    # Gris
    "YOLO_REF": (160, 229,   0),   # Vert clair
    "YOLO_INSP":(212, 182,   6),   # Cyan
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def annoter_image(
    image:     np.ndarray,
    anomalies: list[dict],
    detections_yolo: Optional[list[dict]] = None,
    show_yolo: bool = True
) -> np.ndarray:
    """
    Dessine sur l'image inspectée :
      - Bboxes YOLO (vert) si show_yolo=True
      - Bboxes anomalies avec couleur par type
      - Labels avec type + sévérité
    """
    out = image.copy()
    H, W = out.shape[:2]

    # ── YOLO detections (fond) ───────────────────────────────────────────────
    if show_yolo and detections_yolo:
        for det in detections_yolo:
            x1, y1, x2, y2 = det["bbox"]
            color = COLORS["YOLO_INSP"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            label = f"{det['label']} {det['confidence']:.0%}"
            _draw_label(out, label, x1, y2 + 14, color, font_scale=0.35)

    # ── Anomalies (premier plan) ─────────────────────────────────────────────
    for i, a in enumerate(anomalies):
        bbox = a.get("bbox_insp") or a.get("bbox_ref")
        if not bbox:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = COLORS.get(a.get("type", "INCONNU"), COLORS["INCONNU"])
        sev   = a.get("severite", "medium")

        # Épaisseur selon sévérité
        thickness = 3 if sev == "critical" else 2

        # Box principale
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Coins renforcés (style industriel)
        _draw_corners(out, x1, y1, x2, y2, color, size=10)

        # Halo pour critique
        if sev == "critical":
            overlay = out.copy()
            cv2.rectangle(overlay, (x1-2, y1-2), (x2+2, y2+2), color, 4)
            cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)

        # Label
        label_type = a.get("type", "?")
        label_sev  = sev.upper()
        label      = f"{label_type} [{label_sev}]"
        _draw_label(out, label, x1, y1 - 6, color)

        # Numéro d'anomalie
        num_x = (x1 + x2) // 2 - 8
        num_y = (y1 + y2) // 2 + 5
        cv2.putText(out, str(i + 1), (num_x, num_y),
                    FONT, 0.5, color, 2, cv2.LINE_AA)

    # ── Légende ──────────────────────────────────────────────────────────────
    _draw_legend(out, anomalies)

    return out


def annoter_reference(
    image:      np.ndarray,
    detections: list[dict]
) -> np.ndarray:
    """Dessine les détections YOLO sur l'image de référence."""
    out = image.copy()
    color = COLORS["YOLO_REF"]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        _draw_label(out, det["label"], x1, y1 - 4, color, font_scale=0.38)

    return out


def image_to_base64(image: np.ndarray) -> str:
    """Encode l'image annotée en base64 JPEG pour envoi HTTP."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()


def _draw_label(img, text, x, y, color, font_scale=0.45):
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, 1)
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = max(th + 4, y)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 2), color, -1)
    cv2.putText(img, text, (x, y - 1), FONT, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_corners(img, x1, y1, x2, y2, color, size=12, thickness=2):
    """Dessine des coins en L aux 4 angles de la bbox."""
    pts = [
        ((x1, y1), (x1 + size, y1), (x1, y1 + size)),
        ((x2, y1), (x2 - size, y1), (x2, y1 + size)),
        ((x1, y2), (x1 + size, y2), (x1, y2 - size)),
        ((x2, y2), (x2 - size, y2), (x2, y2 - size)),
    ]
    for corner, p1, p2 in pts:
        cv2.line(img, corner, p1, color, thickness)
        cv2.line(img, corner, p2, color, thickness)


def _draw_legend(img, anomalies):
    """Légende en bas à gauche."""
    if not anomalies:
        return

    counts = {}
    for a in anomalies:
        sev = a.get("severite", "medium")
        counts[sev] = counts.get(sev, 0) + 1

    lines = [f"ANOMALIES: {len(anomalies)}"]
    if counts.get("critical"): lines.append(f"CRITIQUE: {counts['critical']}")
    if counts.get("medium"):   lines.append(f"MOYEN: {counts['medium']}")
    if counts.get("low"):      lines.append(f"FAIBLE: {counts['low']}")

    H, W = img.shape[:2]
    x, y = 8, H - 8 - len(lines) * 16
    for line in lines:
        color = (59, 78, 255) if "CRITIQUE" in line else (11, 158, 245)
        cv2.putText(img, line, (x, y), FONT, 0.38, color, 1, cv2.LINE_AA)
        y += 16
