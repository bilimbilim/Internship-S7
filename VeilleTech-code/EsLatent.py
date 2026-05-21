"""
Visualisation de l'espace latent — PCA / t-SNE / UMAP
Lance : python visualize_latent.py
Ouvre une fenêtre interactive matplotlib avec les 3 projections.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print(" UMAP non installé — pip install umap-learn")
    print("  PCA et t-SNE seront affichés quand même.\n")

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_clusters():
    path = os.path.join(_HERE, "clusters.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    from collector import _DEFAULT_CLUSTERS
    return _DEFAULT_CLUSTERS


def load_papers():
    path = os.path.join(_HERE, "papers.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)



def build_embeddings(clusters, papers):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts  = []   # texte brut
    labels = []   # label affiché au hover
    types  = []   # "keyword" ou "article"
    cids   = []   # cluster id

    for c in clusters:
        for kw in c["keywords"]:
            texts.append(kw)
            labels.append(kw)
            types.append("keyword")
            cids.append(c["id"])

    papers_by_cluster = {c["id"]: [] for c in clusters}
    for p in papers:
        cid = p.get("cluster")
        if cid in papers_by_cluster:
            papers_by_cluster[cid].append(p)

    for c in clusters:
        for p in papers_by_cluster[c["id"]]:
            text = (p.get("title", "") + " " + p.get("abstract", ""))[:512]
            texts.append(text)
            labels.append(p.get("title", "")[:60])
            types.append("article")
            cids.append(c["id"])

    print(f"Encodage de {len(texts)} points ({sum(t=='keyword' for t in types)} mots-clés, {sum(t=='article' for t in types)} articles)...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings), labels, types, cids



def project(embeddings):
    results = {}

    print("PCA...")
    results["PCA"] = PCA(n_components=2).fit_transform(embeddings)

    print("t-SNE...")
    perplexity = min(30, len(embeddings) - 1)
    results["t-SNE"] = TSNE(n_components=2, perplexity=perplexity,
                             random_state=42, max_iter=1000).fit_transform(embeddings)

    if HAS_UMAP:
        print("UMAP...")
        n_neighbors = min(15, len(embeddings) - 1)
        results["UMAP"] = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                     random_state=42).fit_transform(embeddings)

    return results



CLUSTER_COLORS = ["#378ADD", "#1D9E75", "#BA7517", "#D4537E",
                  "#7F77DD", "#D85A30", "#639922", "#D4537E"]

MARKERS = {"keyword": "o", "article": "s"}
SIZES   = {"keyword": 80, "article": 50}
ALPHAS  = {"keyword": 0.95, "article": 0.65}


def plot_projection(ax, coords, labels, types, cids, clusters, title):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top","right","left","bottom"]].set_visible(False)

    cluster_names = {c["id"]: c["label"] for c in clusters}

    for i, (x, y) in enumerate(coords):
        color  = CLUSTER_COLORS[cids[i] % len(CLUSTER_COLORS)]
        marker = MARKERS[types[i]]
        size   = SIZES[types[i]]
        alpha  = ALPHAS[types[i]]
        ax.scatter(x, y, c=color, marker=marker, s=size,
                   alpha=alpha, edgecolors="white", linewidths=0.5, zorder=3)

    # Légende clusters
    cluster_patches = [
        mpatches.Patch(color=CLUSTER_COLORS[c["id"] % len(CLUSTER_COLORS)],
                       label=c["label"])
        for c in clusters
    ]
    # Légende types
    kw_patch  = plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
                            markersize=8, label="mot-clé")
    art_patch = plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="gray",
                            markersize=8, label="article")

    ax.legend(handles=cluster_patches + [kw_patch, art_patch],
              fontsize=8, loc="best", framealpha=0.85)


def annotate_on_hover(fig, axes, all_coords, labels, types, cids):
    """Affiche le label au survol de la souris."""
    annot = axes[0].annotate("", xy=(0,0), xytext=(10,10),
                              textcoords="offset points",
                              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                        ec="gray", alpha=0.9),
                              fontsize=8)
    annot.set_visible(False)

    def on_move(event):
        for ax_idx, ax in enumerate(axes):
            if event.inaxes != ax:
                continue
            coords = all_coords[ax_idx]
            for i, (x, y) in enumerate(coords):
                tx, ty = ax.transData.transform((x, y))
                dist = ((event.x - tx)**2 + (event.y - ty)**2)**0.5
                if dist < 8:
                    annot.axes = ax
                    annot.xy = (x, y)
                    annot.set_text(f"[{types[i]}]\n{labels[i]}")
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)




def main():
    clusters = load_clusters()
    papers   = load_papers()

    print(f"{len(clusters)} clusters chargés, {len(papers)} articles dans papers.json\n")

    embeddings, labels, types, cids = build_embeddings(clusters, papers)
    projections = project(embeddings)

    n_plots = len(projections)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle("Espace latent — all-MiniLM-L6-v2", fontsize=15, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#0f0f14")
    for ax in axes:
        ax.set_facecolor("#0f0f14")
        ax.title.set_color("white")
        ax.legend_ and ax.get_legend().get_frame().set_facecolor("#1e1e2e")

    all_coords = []
    for ax, (method, coords) in zip(axes, projections.items()):
        plot_projection(ax, coords, labels, types, cids, clusters, method)
        all_coords.append(coords)

    annotate_on_hover(fig, axes, all_coords, labels, types, cids)

    plt.tight_layout()
    plt.savefig(os.path.join(_HERE, "latent_space.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\nImage sauvegardée : latent_space.png")
    plt.show()


if __name__ == "__main__":
    main()