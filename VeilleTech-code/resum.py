import json
import re
import ollama


def summarize_cluster(papers: list[dict]) -> list[dict]:
    """
    Résume tous les articles d'un cluster en UN SEUL appel Llama.
    Llama reçoit tous les abstracts numérotés et retourne un JSON
    avec un résumé par article.

    Retourne la liste des papers avec le champ "summary" renseigné.
    """
    if not papers:
        return papers

    # Construire le bloc d'entrée numéroté
    blocks = []
    for i, p in enumerate(papers):
        abstract = p.get("abstract", "").strip()
        if not abstract:
            abstract = p.get("title", "No abstract available.")
        blocks.append(f"[{i}] {abstract}")

    numbered_abstracts = "\n\n".join(blocks)

    prompt = f"""You will receive {len(papers)} scientific abstracts, each prefixed by its index [0], [1], etc.

For each abstract, write a concise summary of 2-3 sentences in French.

Return ONLY a valid JSON array — no explanation, no markdown, no extra text.
The array must have exactly {len(papers)} elements, in the same order as the input.
Each element is a plain string (the summary).

Example output format:
["Résumé de l'article 0.", "Résumé de l'article 1.", ...]

Abstracts:
{numbered_abstracts}"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response["message"]["content"].strip()
    
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        summaries = json.loads(raw)
        if not isinstance(summaries, list):
            raise ValueError("Réponse inattendue : pas une liste")
    except (json.JSONDecodeError, ValueError):
        
        summaries = _fallback_parse(raw, len(papers))

    for i, paper in enumerate(papers):
        if i < len(summaries) and summaries[i]:
            paper["summary"] = str(summaries[i]).strip()
        else:
            paper["summary"] = "Résumé non disponible."

    return papers


def _fallback_parse(raw: str, expected: int) -> list[str]:
    """
    Si le JSON est malformé, on tente d'extraire les strings entre guillemets.
    """
    matches = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
    if len(matches) >= expected:
        return matches[:expected]
    # Dernier recours : une ligne par résumé
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines[:expected] + ["Résumé non disponible."] * max(0, expected - len(lines))