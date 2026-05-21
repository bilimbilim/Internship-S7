#!/usr/bin/env python3
"""
4_export.py — Exporte le modèle entraîné vers différents formats de déploiement.
Lance depuis : pcb-inspector/yolo_training/
"""

from ultralytics import YOLO
from pathlib import Path
import shutil, sys

DEFAULT_MODEL = Path("runs/train/pcb_detector/weights/best.pt")
BACKEND_MODEL = Path("../backend/models/pcb_detector.pt")

def export(model_path: str = None, formats: list = None):
    path = Path(model_path) if model_path else DEFAULT_MODEL
    if not path.exists():
        print(f"[ERREUR] Modèle non trouvé : {path}")
        return

    model = YOLO(str(path))
    formats = formats or ["onnx"]

    print(f"\n[MODELE] {path}")
    print(f"[FORMATS] {formats}\n")

    for fmt in formats:
        print(f"── Export {fmt.upper()} ──")
        try:
            exported = model.export(format=fmt, imgsz=640, optimize=True)
            print(f"[OK] {exported}")
        except Exception as e:
            print(f"[ERREUR] {e}")

    # Copie du .pt vers le backend
    BACKEND_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKEND_MODEL)
    print(f"\n[OK] Modèle copié vers le backend : {BACKEND_MODEL}")
    print("     Tu peux maintenant démarrer le backend.")
    print("     Lance : cd ../backend && python main.py")

if __name__ == "__main__":
    mdl = None
    fmts = ["onnx"]   # formats par défaut

    if "--model" in sys.argv:
        mdl = sys.argv[sys.argv.index("--model") + 1]
    if "--formats" in sys.argv:
        fmts = sys.argv[sys.argv.index("--formats") + 1].split(",")

    export(mdl, fmts)
