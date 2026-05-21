

import json
import re

from llmops import ModelRouter, LLMTracer, RetryHandler, evaluator

_router = ModelRouter()
_retry  = RetryHandler()


def extract_json_array(raw: str, expected: int) -> list:
    """
    Extrait le premier tableau JSON valide dans le texte brut.
    Tolère le texte parasite ajouté par llama3 avant/après le tableau.
    En dernier recours, utilise le fallback ligne par ligne.
    """
    # 1. Nettoyage balises markdown
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw).strip()

    if not raw:
        raise ValueError("Réponse vide du LLM")

    # 2. Tentative directe
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 3. Extraction du premier [...] valide (cas llama3 avec intro textuelle)
    match = re.search(r'\[[\s\S]*?\]', raw)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # 4. Extraction étendue (tableau multi-lignes)
    match = re.search(r'\[[\s\S]*\]', raw)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # 5. Fallback : extraction par guillemets / lignes
    print(f"   [warn] Fallback utilisé. raw={repr(raw[:200])}")
    return _fallback_parse(raw, expected)


def summarize_cluster(papers: list[dict], cluster_id=None) -> list[dict]:
    if not papers:
        return papers

    blocks = []
    for i, p in enumerate(papers):
        abstract = p.get("abstract", "").strip() or p.get("title", "No abstract available.")
        blocks.append(f"[{i}] {abstract}")

    numbered_abstracts = "\n\n".join(blocks)

    # Troncature pour rester dans le contexte de llama3
    MAX_CHARS = 8000
    if len(numbered_abstracts) > MAX_CHARS:
        numbered_abstracts = numbered_abstracts[:MAX_CHARS] + "\n...[tronqué]"

    prompt = f"""You will receive {len(papers)} scientific abstracts.
For each abstract, write a concise summary of 2 sentences in French.
Return ONLY a valid JSON array with exactly {len(papers)} string elements.
No introduction, no explanation, no markdown, no code block.
Example output: ["Résumé 1.", "Résumé 2."]

Abstracts:
{numbered_abstracts}"""

    messages = [{"role": "user", "content": prompt}]

    def _call():
        return _router.chat(messages=messages)

    def _validate(response):
        raw = _router.extract_text(response)

        # FIX : si la réponse est vide, on lève une exception
        # pour que RetryHandler déclenche un retry
        if not raw or not raw.strip():
            raise ValueError("Réponse vide retournée par le LLM")

        data = extract_json_array(raw, expected=len(papers))

        if not isinstance(data, list):
            raise ValueError(f"Réponse inattendue : pas une liste ({type(data)})")

        return data, response, raw

    with LLMTracer("summarize", cluster_id=cluster_id) as tracer:
        result, retries, error = _retry.call(_call, validate=_validate, step="summarize")

        if result is None:
            tracer.fail(error)
            for p in papers:
                p.setdefault("summary", "Résumé non disponible.")
            return papers

        summaries, raw_response, raw_text = result

        json_score, json_notes = evaluator.score_json_output(raw_text, len(papers))

        summary_scores = []
        for i, paper in enumerate(papers):
            if i < len(summaries) and summaries[i]:
                paper["summary"] = str(summaries[i]).strip()
                score, _ = evaluator.score_summary(paper["summary"], paper.get("abstract", ""))
                summary_scores.append(score)
            else:
                paper["summary"] = "Résumé non disponible."
                summary_scores.append(0.0)

        avg_quality = sum(summary_scores) / max(len(summary_scores), 1)
        notes = f"json={json_notes} | avg_summary_score={avg_quality:.2f}"

        tracer.record(
            raw_response,
            retries=retries,
            quality_score=avg_quality,
            quality_notes=notes
        )

    return papers


def _fallback_parse(raw: str, expected: int) -> list[str]:
    """Tente d'extraire des chaînes depuis un JSON malformé."""
    matches = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
    if len(matches) >= expected:
        return matches[:expected]

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines[:expected] + ["Résumé non disponible."] * max(0, expected - len(lines))