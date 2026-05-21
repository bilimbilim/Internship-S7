"""
test_llmops.py — Script de test autonome pour LLMOps VeilleTech
================================================================
Lance depuis le dossier llmops_veille :
    python test_llmops.py

Necessite : Ollama installe + modele llama3 disponible
    ollama pull llama3
"""

import sys
import os
import json
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llmops import ModelRouter, LLMTracer, RetryHandler, evaluator

print("=" * 55)
print("TEST LLMOps — VeilleTech")
print("=" * 55)

router = ModelRouter()
retry  = RetryHandler(max_retries=2)


def extract_json_array(raw: str) -> list:
    """
    Extrait le premier tableau JSON valide trouvé dans le texte brut,
    même si llama3 ajoute du texte avant/après (ex: "Here is the JSON array:").
    """
    # 1. Nettoyage balises markdown
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw).strip()

    if not raw:
        raise ValueError("Réponse vide du LLM")

    # 2. Tentative directe (cas idéal : le modèle a bien suivi l'instruction)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 3. Extraction du premier [...] valide dans le texte (cas fréquent avec llama3)
    match = re.search(r'\[[\s\S]*?\]', raw)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # 4. Extraction étendue (tableau multi-lignes avec guillemets imbriqués)
    match = re.search(r'\[[\s\S]*\]', raw)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Aucun tableau JSON trouvé dans : {repr(raw[:300])}")


# ── TEST 1 : Résumé (tâche summarize) ────────────────────────────────────────
print("\n[1/2] Test tache : SUMMARIZE")
print("   Envoi d'un abstract a llama3...")

abstracts = [
    "[0] Large language models (LLMs) have shown remarkable capabilities in natural language processing tasks, including text generation, summarization, and question answering.",
    "[1] Retrieval-Augmented Generation (RAG) combines parametric memory of LLMs with non-parametric retrieval to improve factual accuracy in generated text.",
]
prompt_summarize = f"""You will receive 2 scientific abstracts, each prefixed by its index [0], [1].
For each abstract, write a concise summary of 2-3 sentences in French.
Return ONLY a valid JSON array with exactly 2 string elements, nothing else.
No introduction, no explanation, no markdown.
Example output: ["Résumé 0.", "Résumé 1."]

Abstracts:
{chr(10).join(abstracts)}"""


def call_summarize():
    return router.chat(messages=[{"role": "user", "content": prompt_summarize}])


def validate_summarize(response):
    raw = router.extract_text(response)
    print(f"   [debug] RAW = {repr(raw[:200])}")
    data = extract_json_array(raw)
    if len(data) == 0:
        raise ValueError("Tableau JSON vide")
    return data, response, raw


with LLMTracer("summarize", cluster_id=0) as tracer:
    result, retries, error = retry.call(call_summarize, validate=validate_summarize, step="summarize")

    if result is None:
        tracer.fail(error)
        print(f"  ECHEC : {error}")
    else:
        summaries, raw_response, raw_text = result

        json_score, json_notes = evaluator.score_json_output(raw_text, 2)
        summary_scores = []
        for i, s in enumerate(summaries):
            sc, notes = evaluator.score_summary(s, abstracts[i])
            summary_scores.append(sc)
            print(f"  Resume [{i}] : {s[:80]}...")
            print(f"    Score : {sc:.2f} | {notes}")

        avg_q = sum(summary_scores) / len(summary_scores)
        tracer.record(raw_response, retries=retries,
                      quality_score=avg_q,
                      quality_notes=f"json={json_notes} | avg={avg_q:.2f}")
        print(f"  Score moyen : {avg_q:.2f}")


# ── TEST 2 : Renommage cluster (tâche rename) ─────────────────────────────────
print("\n[2/2] Test tache : RENAME CLUSTER")
print("   Envoi de titres d'articles a llama3...")

titles = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-4 Technical Report",
    "LLaMA: Open and Efficient Foundation Language Models",
]
prompt_rename = f"""These research paper titles all belong to the same topic cluster.

Titles:
{chr(10).join(titles)}

Give a short topic label (2-4 words maximum) that best describes this cluster.
Reply with ONLY the topic label, nothing else, no explanation, no punctuation."""


def call_rename():
    return router.chat(messages=[{"role": "user", "content": prompt_rename}])


def validate_rename(response):
    text = router.extract_text(response)
    if not text.strip():
        raise ValueError("Label vide")
    label = text.strip().splitlines()[0].strip()
    if not label:
        raise ValueError("Label vide après nettoyage")
    return label, response


with LLMTracer("rename", cluster_id=0) as tracer:
    result, retries, error = retry.call(call_rename, validate=validate_rename, step="rename")

    if result is None:
        tracer.fail(error)
        print(f"  ECHEC : {error}")
    else:
        label, raw_response = result
        score, notes = evaluator.score_cluster_label(label)
        tracer.record(raw_response, retries=retries,
                      quality_score=score, quality_notes=notes)
        print(f"  Label produit : '{label}'")
        print(f"  Score : {score:.2f} | {notes}")


# ── Rapport console ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("RAPPORT — logs/llmops_runs.jsonl")
print("=" * 55)

from pathlib import Path

log_file = Path("logs/llmops_runs.jsonl")
if log_file.exists():
    runs = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  Total runs loggues : {len(runs)}")
    for r in runs[-4:]:
        status = "OK " if r["success"] else "ERR"
        print(f"  [{status}] step={r['step']:<12} latency={r['latency_ms']:>8.1f}ms  "
              f"tokens={r['prompt_tokens']}+{r['output_tokens']}  "
              f"quality={r['quality_score']:.2f}  retries={r['retries']}")
    print(f"\n  Rapport HTML : python llmops_report.py")
else:
    print("  Fichier log non trouve (normal si premiere execution)")

print("\nTEST TERMINE.")