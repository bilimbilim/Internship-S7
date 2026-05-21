"""
yolo_detector.py — Chargement YOLO et inférence sur image PCB.
"""

from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import yaml

MODEL_PATH   = Path(__file__).parent / "models" / "pcb_detector.pt"
DATASET_YAML = Path(__file__).parent.parent / "yolo_training" / "dataset.yaml"

_model = None

def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle YOLO non trouvé : {MODEL_PATH}\n"
                "Lance d'abord : cd yolo_training && python scripts/4_export.py"
            )
        _model = YOLO(str(MODEL_PATH))
        print(f"[YOLO] Modèle chargé : {MODEL_PATH}")
    return _model

def load_class_names() -> dict:
    if DATASET_YAML.exists():
        with open(DATASET_YAML) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("names", {})
    return {}

CLASS_NAMES = load_class_names()

def detect(image: np.ndarray, conf: float = 0.25) -> list[dict]:
    """
    Détecte les composants sur une image PCB.

    Args:
        image : image BGR (numpy array)
        conf  : seuil de confiance minimum

    Returns:
        Liste de détections :
        [
          {
            "id": int,
            "label": str,
            "class_id": int,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],       # pixels absolus
            "bbox_rel": [cx, cy, w, h],       # normalisé [0-1]
            "center": [cx_px, cy_px]          # centre en pixels
          },
          ...
        ]
    """
    model = get_model()
    H, W  = image.shape[:2]
    results = model(image, conf=conf, verbose=False)[0]

    detections = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls)
        conf_  = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2

        detections.append({
            "id":         i,
            "label":      CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            "class_id":   cls_id,
            "confidence": round(conf_, 3),
            "bbox":       [x1, y1, x2, y2],
            "bbox_rel":   [
                round(cx_px / W, 4),
                round(cy_px / H, 4),
                round((x2 - x1) / W, 4),
                round((y2 - y1) / H, 4)
            ],
            "center": [round(cx_px, 1), round(cy_px, 1)]
        })

    return detections


def detect_from_bytes(image_bytes: bytes, conf: float = 0.25) -> list[dict]:
    """Détecte depuis des bytes image (upload HTTP)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image invalide ou format non supporté")
    return detect(img, conf)
