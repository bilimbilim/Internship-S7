import ollama


def rename_cluster_with_llama(papers: list[dict]) -> str:
    """
    Demande à Llama de donner un nom court (2-4 mots) au cluster
    en se basant sur les titres des articles qu'il contient.
    """
    titles_text = "\n".join(p["title"] for p in papers[:8])

    prompt = f"""These research paper titles all belong to the same topic cluster.

Titles:
{titles_text}

Give a short topic label (2-4 words maximum) that best describes this cluster.
Reply with ONLY the topic label, nothing else, no explanation, no punctuation."""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"].strip()
    return raw.splitlines()[0].strip()


def rename_clusters(cluster_data: dict) -> dict:
    """
    Pour chaque cluster, appelle Llama pour renommer le cluster,
    puis met à jour cluster_title sur chaque article.

    cluster_data : { cluster_id: { "cluster": {...}, "papers": [...] } }
    Retourne le même dict mis à jour.
    """
    for cid, data in cluster_data.items():
        papers = data["papers"]
        if not papers:
            continue

        print(f"  → Renommage cluster [{data['cluster']['label']}] par Llama...")
        new_title = rename_cluster_with_llama(papers)

        for p in papers:
            p["cluster_title"] = new_title

        data["cluster"]["label"] = new_title
        print(f"     Nouveau nom : « {new_title} »")

    return cluster_data