# LLMOps — VeilleTech

Instrumentation complète du pipeline de veille technologique IA.

---

## Fichiers livrés

| Fichier | Rôle |
|---------|------|
| `llmops.py` | Module core : `LLMTracer`, `LLMEvaluator`, `RetryHandler`, `ModelRouter` |
| `resum.py` | Résumé de cluster instrumenté (remplace l'original) |
| `clustering.py` | Renommage de cluster instrumenté (remplace l'original) |
| `llmops_report.py` | Génère rapport HTML + stats console depuis les logs |
| `app_llmops_routes.py` | Routes Flask `/llmops` et `/api/llmops/stats` à intégrer dans `app.py` |
| `logs/llmops_runs.jsonl` | Log des runs (créé automatiquement) |
| `reports/llmops_report.html` | Rapport HTML (généré par `llmops_report.py`) |

---

## Installation rapide

```bash
# 1. Copier les fichiers dans ton projet VeilleTech
cp llmops.py resum.py clustering.py llmops_report.py /ton/projet/

# 2. (Optionnel) Ajouter les routes dans app.py
#    Copier le contenu de app_llmops_routes.py dans app.py

# 3. Créer les dossiers de sortie
mkdir -p logs reports
```

---

## Configuration par variables d'environnement

```bash
# Provider LLM (défaut : ollama)
export LLMOPS_PROVIDER=ollama        # ou "anthropic"
export LLMOPS_MODEL=llama3           # ou "claude-haiku-4-5-20251001"

# Clé API Anthropic (si LLMOPS_PROVIDER=anthropic)
export ANTHROPIC_API_KEY=sk-ant-...

# Comportement du retry
export LLMOPS_MAX_RETRIES=3
export LLMOPS_RETRY_DELAY=2.0        # secondes (backoff exponentiel)

# Chemin du fichier de log
export LLMOPS_LOG_FILE=logs/llmops_runs.jsonl
```

---

## Passer de Ollama à Claude (Anthropic)

```bash
pip install anthropic
export LLMOPS_PROVIDER=anthropic
export LLMOPS_MODEL=claude-haiku-4-5-20251001
export ANTHROPIC_API_KEY=sk-ant-...
python Run_pipeline.py
```

Aucune modification du code nécessaire.

---

## Consulter les métriques

### Console (après un run)
```bash
python llmops_report.py
```

### Rapport HTML
```bash
python llmops_report.py --output reports/llmops_report.html
# Ouvrir reports/llmops_report.html dans un navigateur
```

### Via l'UI Flask
```
http://localhost:5000/llmops          → rapport HTML
http://localhost:5000/api/llmops/stats → JSON stats
http://localhost:5000/api/llmops/runs  → 50 derniers runs
```

---

## Format du log (llmops_runs.jsonl)

Chaque ligne est un JSON :

```json
{
  "run_id": "a3f1b2c4",
  "timestamp": "2026-04-22T10:35:01+00:00",
  "step": "summarize",
  "cluster_id": 0,
  "provider": "ollama",
  "model": "llama3",
  "prompt_tokens": 1842,
  "output_tokens": 312,
  "latency_ms": 4521.3,
  "success": true,
  "retries": 0,
  "error": "",
  "quality_score": 0.87,
  "quality_notes": "json=OK | avg_summary_score=0.87"
}
```

---

## Métriques de qualité

### Résumés (`score_summary`)
| Critère | Pénalité |
|---------|---------|
| Moins de 20 mots | -0.30 |
| Plus de 120 mots | -0.15 |
| Pas en français (heuristique) | -0.20 |
| Beaucoup de répétitions (< 50% mots uniques) | -0.20 |
| Peu de mots en commun avec l'abstract | -0.10 |

### Labels de cluster (`score_cluster_label`)
| Critère | Pénalité |
|---------|---------|
| Label vide | -0.50 |
| Plus de 5 mots | -0.30 |
| Caractères suspects | -0.20 |

---

## Exemple d'utilisation directe de llmops.py

```python
from llmops import ModelRouter, LLMTracer, RetryHandler, evaluator

router = ModelRouter()  # utilise LLMOPS_PROVIDER et LLMOPS_MODEL
retry  = RetryHandler(max_retries=3)

def _call():
    return router.chat(messages=[{"role": "user", "content": "Résume ceci..."}])

def _validate(response):
    text = router.extract_text(response)
    if not text:
        raise ValueError("Réponse vide")
    return text, response

with LLMTracer("mon_etape", cluster_id=42) as tracer:
    result, retries, error = retry.call(_call, validate=_validate)
    if result:
        text, raw = result
        score, notes = evaluator.score_summary(text)
        tracer.record(raw, retries=retries, quality_score=score, quality_notes=notes)
    else:
        tracer.fail(error)
```
