"""
clustering.py — Renommage de clusters avec instrumentation LLMOps
==================================================================
Remplace l'original. Changements :
  - ModelRouter au lieu d'ollama direct
  - RetryHandler sur l'appel LLM
  - LLMTracer + LLMEvaluator sur chaque cluster
"""

from llmops import ModelRouter, LLMTracer, RetryHandler, evaluator

_router = ModelRouter()
_retry  = RetryHandler()


def rename_cluster_with_llm(papers: list[dict], cluster_id=None) -> str:
    """
    Demande au LLM un nom court (2-4 mots) basé sur les titres des articles.
    Retourne le label ou un fallback si echec.
    """
    titles_text = "\n".join(p["title"] for p in papers[:8])

    prompt = f"""These research paper titles all belong to the same topic cluster.

Titles:
{titles_text}

Give a short topic label (2-4 words maximum) that best describes this cluster.
Reply with ONLY the topic label, nothing else, no explanation, no punctuation."""

    messages = [{"role": "user", "content": prompt}]

    def _call():
        return _router.chat(messages=messages)

    def _validate(response):
        raw = _router.extract_text(response)
        label = raw.splitlines()[0].strip()
        if not label:
            raise ValueError("Label vide retourné par le LLM")
        return label, response

    with LLMTracer("rename", cluster_id=cluster_id) as tracer:
        result, retries, error = _retry.call(_call, validate=_validate, step="rename")

        if result is None:
            tracer.fail(error)
            return f"Cluster {cluster_id}"

        label, raw_response = result
        q_score, q_notes = evaluator.score_cluster_label(label)
        tracer.record(raw_response, retries=retries, quality_score=q_score, quality_notes=q_notes)

    return label


def rename_clusters(cluster_data: dict) -> dict:
    """
    Pour chaque cluster, appelle le LLM pour renommer, puis met à jour
    cluster_title sur chaque article.
    """
    for cid, data in cluster_data.items():
        papers = data["papers"]
        if not papers:
            continue

        print(f"  → Renommage cluster [{data['cluster']['label']}]...")
        new_title = rename_cluster_with_llm(papers, cluster_id=cid)

        for p in papers:
            p["cluster_title"] = new_title

        data["cluster"]["label"] = new_title
        print(f"     Nouveau nom : « {new_title} »")

    return cluster_data
