# Internship S7 — IA & MLOps

Travaux réalisés durant mon stage de 7ème semestre, axés sur l'IA, le MLOps et l'automatisation de workflows.

## Projets

### 1. MLops — Déploiement de modèle DPE
Entraînement, versioning et déploiement d'un modèle Random Forest pour la prédiction de la performance énergétique (DPE).
- **Stack :** Python, Scikit-learn, MLflow, DVC, Flask, Docker
- **Fichier principal :** `MLops/DPE.py`, `MLops/api.py`

### 2. PCB Inspector — Détection de défauts sur circuits imprimés
Système de détection de défauts sur PCB combinant YOLOv8, OpenCV et l'API Gemini, intégré via Make.com.
- **Stack :** Python, YOLOv8, OpenCV, Gemini API, Make.com webhooks
- **Fichier principal :** `Make/pcb-inspector/backend/main.py`

### 3. Multi-Agents System — Assistant IA multi-agents
Système multi-agents orchestrant un agent email, un agent calendrier et un système RAG (Retrieval-Augmented Generation) sur ChromaDB.
- **Stack :** Python, Flask, LangChain, ChromaDB, Gmail API, Google Calendar API
- **Fichier principal :** `Multi-Agents-system/Multi-Agents-system/main_agent.py`

### 4. N8N — Automatisation de workflows
Workflows d'automatisation construits avec N8N pour plusieurs cas d'usage métier.
- **Workflows :** Agent CV, Agent Médical, Agent Recrutement, Veille Newsletter
- **Stack :** N8N, JSON workflows

### 5. VeilleTech — Système de veille technologique
Pipeline de veille technologique automatisé : collecte RSS, clustering sémantique, résumé et visualisation.
- **Stack :** Python, Flask, Elasticsearch, RSS, Clustering
- **Fichier principal :** `VeilleTech-code/Run_pipeline.py`

### 6. LLMOps Veille — Monitoring LLM
Module de monitoring des runs LLM avec génération de rapports HTML et clustering des résultats.
- **Stack :** Python, Flask
- **Fichier principal :** `llmops_veille/llmops_veille/llmops.py`

## Structure

```
Internship-S7/
├── MLops/                  # Modèle DPE + API Flask + Docker
├── Make/                   # PCB Inspector (YOLO + Gemini + Make.com)
├── Multi-Agents-system/    # Système multi-agents (Email, Calendar, RAG)
├── N8N/                    # Workflows N8N (CV, Médical, Recrutement, Newsletter)
├── VeilleTech-code/        # Pipeline de veille technologique
└── llmops_veille/          # Monitoring LLMOps
```
