#!/usr/bin/env python3
"""
2_train.py — Lance l'entraînement YOLO sur le dataset PCB.
Lance depuis : pcb-inspector/yolo_training/
"""

from ultralytics import YOLO
from pathlib import Path
import yaml, torch, time, sys

DATASET_YAML  = Path("dataset.yaml")
TRAIN_CONFIG  = Path("train_config.yaml")
OUTPUT_DIR    = Path("runs/train")

def check_gpu():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {name} — {vram:.1f} GB VRAM")
        return "cuda"
    else:
        print("[CPU] Pas de GPU détecté — entraînement lent mais fonctionnel")
        print("      Conseil : utilise Google Colab (GPU gratuit) pour accélérer")
        return "cpu"

def train():
    print("\n══ PCB YOLO TRAINING ══════════════════════════════")

    device = check_gpu()

    with open(TRAIN_CONFIG) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.pop("model", "yolov8m.pt")
    print(f"[MODELE] {model_name} (transfer learning depuis COCO)")

    model = YOLO(model_name)

    print(f"[DATASET] {DATASET_YAML}")
    print(f"[EPOCHS] {cfg.get('epochs', 100)}")
    print(f"[BATCH] {cfg.get('batch', 16)}")
    print("═══════════════════════════════════════════════════\n")

    t0 = time.time()

    results = model.train(
        data    = str(DATASET_YAML),
        device  = device,
        project = str(OUTPUT_DIR),
        **cfg
    )

    elapsed = time.time() - t0
    print(f"\n[OK] Entraînement terminé en {elapsed/60:.1f} minutes")

    # Chemin du meilleur modèle
    best = OUTPUT_DIR / cfg.get("name", "pcb_detector") / "weights" / "best.pt"
    if best.exists():
        print(f"[MODELE] Meilleur modèle sauvegardé : {best}")
        print(f"\n  Prochaine étape : python 3_evaluate.py")
    else:
        print("[ERREUR] Modèle best.pt non trouvé — vérifier les logs")

    return results

def train_colab():
    """
    Code à copier dans Google Colab pour entraîner sur GPU gratuit.
    """
    code = '''
# ── COLAB SETUP ──────────────────────────────────────
!pip install ultralytics -q
from google.colab import drive
drive.mount('/content/drive')

# Copie ton dataset depuis Drive
!cp -r "/content/drive/MyDrive/pcb-inspector/yolo_training" /content/

import os
os.chdir("/content/yolo_training")

# Lance l'entraînement
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.train(
    data    = "dataset.yaml",
    epochs  = 150,
    imgsz   = 640,
    batch   = 32,      # Colab T4 supporte batch=32
    device  = 0,       # GPU
    project = "/content/drive/MyDrive/pcb-inspector/runs",
    name    = "pcb_detector"
)
# Le modèle best.pt est sauvegardé directement dans ton Drive
'''
    print("── CODE GOOGLE COLAB ──────────────────────────────")
    print(code)
    print("───────────────────────────────────────────────────")

if __name__ == "__main__":
    if "--colab" in sys.argv:
        train_colab()
    else:
        train()
