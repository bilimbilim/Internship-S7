#!/usr/bin/env python3
"""
3_evaluate.py — Évalue le modèle entraîné sur le jeu de validation.
Lance depuis : pcb-inspector/yolo_training/
"""

from ultralytics import YOLO
from pathlib import Path
import yaml, cv2, numpy as np, sys

DATASET_YAML = Path("dataset.yaml")
DEFAULT_MODEL = Path("runs/train/pcb_detector/weights/best.pt")

def load_class_names():
    with open(DATASET_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg["names"]

def evaluate(model_path: str = None):
    path = Path(model_path) if model_path else DEFAULT_MODEL
    if not path.exists():
        print(f"[ERREUR] Modèle non trouvé : {path}")
        print("         Lance d'abord : python 2_train.py")
        return

    print(f"\n[MODELE] {path}")
    model = YOLO(str(path))

    print("[EVAL] Calcul des métriques sur le jeu de validation...")
    metrics = model.val(data=str(DATASET_YAML), verbose=False)

    class_names = load_class_names()

    print("\n══ RÉSULTATS D'ÉVALUATION ══════════════════════════")
    print(f"  mAP50      : {metrics.box.map50:.3f}   (objectif: > 0.80)")
    print(f"  mAP50-95   : {metrics.box.map:.3f}   (objectif: > 0.60)")
    print(f"  Précision  : {metrics.box.mp:.3f}")
    print(f"  Rappel     : {metrics.box.mr:.3f}")

    # Par classe
    if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
        print("\n  Précision par classe :")
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            name = class_names.get(int(cls_idx), str(cls_idx))
            ap   = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
            bar  = "█" * int(ap * 20)
            flag = "" if ap >= 0.70 else "  <- insuffisant"
            print(f"    {name:15s}  AP50={ap:.3f}  {bar}{flag}")

    print("═══════════════════════════════════════════════════")

    # Conseil selon mAP
    map50 = metrics.box.map50
    print("\n── Diagnostic ──")
    if map50 >= 0.85:
        print("[EXCELLENT] Modèle prêt pour la production.")
        print("            Lance : python 4_export.py")
    elif map50 >= 0.70:
        print("[BON] Modèle fonctionnel. Pour améliorer :")
        print("  - Ajouter 20% d'images supplémentaires")
        print("  - Vérifier les annotations des classes faibles")
        print("  - Relancer avec epochs=150")
    elif map50 >= 0.50:
        print("[MOYEN] Dataset probablement insuffisant :")
        print("  - Viser 200+ images par classe")
        print("  - Vérifier la qualité des annotations")
        print("  - Utiliser yolov8l.pt (modèle plus grand)")
    else:
        print("[INSUFFISANT] Problème fondamental :")
        print("  - Vérifier les labels (python 1_prepare_dataset.py --preview)")
        print("  - Augmenter fortement le dataset")
        print("  - Vérifier dataset.yaml (classes correctes)")

    return metrics

def test_single_image(image_path: str, model_path: str = None, conf: float = 0.25):
    """Test le modèle sur une image unique avec visualisation."""
    path = Path(model_path) if model_path else DEFAULT_MODEL
    model = YOLO(str(path))
    class_names = load_class_names()

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERREUR] Image non trouvée : {image_path}")
        return

    results = model(img, conf=conf)[0]

    colors = [
        (0,229,160),(255,59,78),(245,158,11),(59,130,246),
        (6,182,212),(124,58,237),(190,24,93),(5,150,105)
    ]

    print(f"\n[DÉTECTION] {len(results.boxes)} composant(s) trouvé(s)")
    for box in results.boxes:
        cls   = int(box.cls)
        conf_ = float(box.conf)
        name  = class_names.get(cls, str(cls))
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        color = colors[cls % len(colors)]

        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label = f"{name} {conf_:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        print(f"  {name:15s}  conf={conf_:.2f}  bbox=[{x1},{y1},{x2},{y2}]")

    out = "test_result.jpg"
    cv2.imwrite(out, img)
    print(f"\n[OK] Image annotée sauvegardée : {out}")

if __name__ == "__main__":
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        img = sys.argv[idx+1]
        mdl = sys.argv[idx+2] if idx+2 < len(sys.argv) else None
        test_single_image(img, mdl)
    else:
        mdl = sys.argv[1] if len(sys.argv) > 1 else None
        evaluate(mdl)
