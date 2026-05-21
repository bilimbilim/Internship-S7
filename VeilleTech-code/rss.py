from feedgen.feed import FeedGenerator
import re


def generate_rss(papers, filename="veille_ai.xml"):
    fg = FeedGenerator()
    fg.title("AI Method Watch")
    fg.description("Veille IA méthodes")
    fg.link(href="https://crossref.org")

    for paper in papers[:50]: 
        fe = fg.add_entry()
        fe.title(paper["title"])
        fe.link(href=paper["url"])

        abstract = paper.get("abstract", "")

        # Nettoyage HTML éventuel
        abstract = re.sub("<.*?>", "", abstract)

        summary = abstract[:300] + "..." if abstract else "Pas de résumé"

        fe.description(
            f"Source : {paper.get('source', 'Inconnue')}\n"
            f"Score pertinence : {paper['score']}\n\n"
            f"Résumé :\n{summary}"
        )

    fg.rss_file(filename)
