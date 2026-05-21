#!/usr/bin/env python3
"""
1_prepare_dataset.py
Vérifie et prépare le dataset avant entraînement.
Convertit les annotations COCO JSON → format YOLO si nécessaire.
Lance depuis : pcb-inspector/yolo_training/
"""

import os
import json
import shutil
import random
from pathlib import Path
import cv2
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL   = DATASET_DIR / "images" / "val"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
LABELS_VAL   = DATASET_DIR / "labels" / "val"
VAL_SPLIT    = 0.2   # 20% des images pour la validation


def create_dirs():
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)
    print("[OK] Dossiers créés")


def verify_dataset():
    """Vérifie que chaque image a un fichier label correspondant."""
    errors = 0
    total = 0

    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"\n[{split.upper()}] {len(images)} images trouvées")

        for img_path in images:
            total += 1
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"  MANQUANT label : {img_path.name}")
                errors += 1
            else:
                # Vérifier format du label
                with open(label_path) as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"  FORMAT INVALIDE {label_path.name} ligne {i+1}: '{line.strip()}'")
                        errors += 1
                    else:
                        cls, cx, cy, w, h = map(float, parts)
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            print(f"  BBOX INVALIDE {label_path.name} ligne {i+1}: coords hors [0,1]")
                            errors += 1

    print(f"\n{'[OK]' if errors == 0 else '[ERREUR]'} {total} images vérifiées — {errors} problème(s)")
    return errors == 0


def split_dataset(source_images_dir: str, source_labels_dir: str):
    """
    Divise un dossier source en train/val.
    source_images_dir : dossier contenant toutes les images
    source_labels_dir : dossier contenant tous les labels YOLO (.txt)
    """
    src_imgs = Path(source_images_dir)
    src_lbls = Path(source_labels_dir)

    images = list(src_imgs.glob("*.jpg")) + list(src_imgs.glob("*.png"))
    random.seed(42)
    random.shuffle(images)

    val_count = int(len(images) * VAL_SPLIT)
    val_imgs  = images[:val_count]
    train_imgs = images[val_count:]

    def copy_pair(img_list, img_dst, lbl_dst):
        copied = 0
        for img in img_list:
            lbl = src_lbls / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(img, img_dst / img.name)
                shutil.copy2(lbl, lbl_dst / (img.stem + ".txt"))
                copied += 1
            else:
                print(f"  [SKIP] Pas de label pour {img.name}")
        return copied

    n_train = copy_pair(train_imgs, IMAGES_TRAIN, LABELS_TRAIN)
    n_val   = copy_pair(val_imgs,   IMAGES_VAL,   LABELS_VAL)
    print(f"[OK] Split terminé — {n_train} train / {n_val} val")


def convert_coco_to_yolo(coco_json_path: str, images_dir: str, output_labels_dir: str):
    """
    Convertit annotations COCO JSON → format YOLO .txt
    Utile si tu as annoté avec CVAT ou Label Studio en format COCO.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    out_dir = Path(output_labels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index des images
    img_index = {img["id"]: img for img in coco["images"]}

    # Regrouper annotations par image
    ann_by_image = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        ann_by_image.setdefault(iid, []).append(ann)

    for img_id, img_info in img_index.items():
        W, H = img_info["width"], img_info["height"]
        stem = Path(img_info["file_name"]).stem
        lines = []

        for ann in ann_by_image.get(img_id, []):
            cls_id = ann["category_id"] - 1  # COCO commence à 1
            x, y, w, h = ann["bbox"]         # COCO: x_min, y_min, w, h
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (out_dir / f"{stem}.txt").write_text("\n".join(lines))

    print(f"[OK] {len(img_index)} fichiers labels YOLO générés dans {output_labels_dir}")


def visualize_annotations(n=5):
    """Affiche N images avec leurs bboxes pour vérification visuelle."""
    import random

    # Charger la config des classes
    import yaml
    with open("dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    class_names = cfg["names"]

    images = list(IMAGES_TRAIN.glob("*.jpg")) + list(IMAGES_TRAIN.glob("*.png"))
    sample = random.sample(images, min(n, len(images)))

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    ]

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        label_path = LABELS_TRAIN / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = map(float, parts)
                cls = int(cls)
                x1 = int((cx - w/2) * W)
                y1 = int((cy - h/2) * H)
                x2 = int((cx + w/2) * W)
                y2 = int((cy + h/2) * H)
                color = colors[cls % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = class_names.get(cls, str(cls))
                cv2.putText(img, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_path = f"preview_{img_path.stem}.jpg"
        cv2.imwrite(out_path, img)
        print(f"  Preview sauvegardé : {out_path}")


def stats():
    """Affiche les statistiques du dataset par classe."""
    import yaml
    with open("dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    class_names = cfg["names"]

    counts = {v: 0 for v in class_names.values()}

    for split in ["train", "val"]:
        lbl_dir = DATASET_DIR / "labels" / split
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        name = class_names.get(cls, str(cls))
                        counts[name] = counts.get(name, 0) + 1

    print("\n── Statistiques dataset ──")
    total = sum(counts.values())
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, int(count / max(counts.values()) * 40))
        print(f"  {name:15s} {count:5d}  {bar}")
    print(f"\n  Total annotations : {total}")
    min_recommended = 100
    for name, count in counts.items():
        if 0 < count < min_recommended:
            print(f"  [AVERTISSEMENT] '{name}' n'a que {count} instances (recommandé: {min_recommended}+)")


if __name__ == "__main__":
    import sys
    create_dirs()

    if "--split" in sys.argv:
        # Usage: python 1_prepare_dataset.py --split /chemin/images /chemin/labels
        idx = sys.argv.index("--split")
        split_dataset(sys.argv[idx+1], sys.argv[idx+2])

    elif "--coco" in sys.argv:
        # Usage: python 1_prepare_dataset.py --coco annotations.json images/ labels/
        idx = sys.argv.index("--coco")
        convert_coco_to_yolo(sys.argv[idx+1], sys.argv[idx+2], sys.argv[idx+3])

    elif "--preview" in sys.argv:
        visualize_annotations(n=5)

    else:
        print("\n── Vérification du dataset ──")
        ok = verify_dataset()
        if ok:
            stats()
            print("\n[OK] Dataset prêt pour l'entraînement")
            print("     Lance : python 2_train.py")
