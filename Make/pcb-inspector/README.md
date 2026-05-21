# PCB Inspector — Architecture complète

## Pipeline

```
Image REF  ──→  YOLO (détection)  ──→  composants REF
                                              ↓
Image INSP ──→  YOLO (détection)  ──→  composants INSP
                                              ↓
                               OpenCV (comparaison géométrique)
                                              ↓
                    absent / décalé / rotation / défaut soudure
                                              ↓
                               Gemini (explication naturelle)
                                              ↓
                               Make (webhook → Sheets / Mail)
```

## Structure du projet

```
pcb-inspector/
├── backend/                  FastAPI — moteur d'analyse
│   ├── main.py               Point d'entrée API
│   ├── yolo_detector.py      Chargement et inférence YOLO
│   ├── opencv_comparator.py  Comparaison géométrique
│   ├── gemini_explainer.py   Explication IA
│   ├── annotator.py          Dessin des bboxes sur image
│   ├── webhook.py            Push Make
│   └── requirements.txt
│
├── yolo_training/            Entraînement du modèle
│   ├── dataset/              Images + labels (format YOLO)
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── labels/train/
│   │   └── labels/val/
│   ├── scripts/
│   │   ├── 1_prepare_dataset.py   Convertit annotations → YOLO
│   │   ├── 2_train.py             Lance l'entraînement
│   │   ├── 3_evaluate.py          Métriques mAP, précision
│   │   └── 4_export.py            Export ONNX / TorchScript
│   ├── dataset.yaml          Config classes YOLO
│   └── train_config.yaml     Hyperparamètres
│
├── frontend/                 Interface HTML
│   └── index.html
│
├── make_webhooks/            Schémas payload pour Make
│   └── payload_schema.json
│
└── docs/
    ├── SETUP.md              Installation pas à pas
    └── LABELING.md           Guide d'annotation des images
```

## Étapes dans l'ordre

1. **Annoter les images** → voir `docs/LABELING.md`
2. **Entraîner YOLO** → `yolo_training/scripts/`
3. **Démarrer le backend** → `backend/`
4. **Ouvrir le frontend** → `frontend/index.html`
5. **Connecter Make** → `make_webhooks/`
