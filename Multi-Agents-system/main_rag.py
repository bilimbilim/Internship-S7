"""
main_rag.py — Point d'entrée du système RAG

Usage :
    python main_rag.py index        → indexe tous les docs du dossier documents/
    python main_rag.py watch        → surveille le dossier et indexe automatiquement
    python main_rag.py chat         → démarre le chat avec l'agent RAG
    python main_rag.py              → affiche l'aide
"""

import sys
import time
import threading
from rag_space.index_documents import (
    index_all_documents,
    index_single_file,
    load_tracking,
    save_tracking,
    get_file_hash,
    DOCUMENTS_DIR,
    PERSIST_DIR,
    LocalEmbeddings
)
from agent_rag import rag_agent
from langchain_community.vectorstores import Chroma
from rag_space.index_documents import LocalEmbeddings

# watchdog pour surveiller le dossier
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# ─────────────────────────────────────────
#  Handler watchdog
# ─────────────────────────────────────────

class DocumentHandler(FileSystemEventHandler):
    """Détecte les nouveaux fichiers ou fichiers modifiés dans documents/"""

    def __init__(self):
        self.embeddings = LocalEmbeddings()
        self.vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings
        )
        # Petit délai pour éviter de traiter un fichier encore en cours de copie
        self._pending = {}
        self._lock = threading.Lock()

    def _process(self, file_path: str):
        """Indexe un fichier si c'est un PDF ou TXT nouveau/modifié."""
        if not (file_path.endswith(".pdf") or file_path.endswith(".txt")):
            return

        import os
        filename = os.path.basename(file_path)
        tracking = load_tracking()
        current_hash = get_file_hash(file_path)

        # Déjà indexé et non modifié → on ignore
        if filename in tracking and tracking[filename]["hash"] == current_hash:
            return

        raison = "MODIFIÉ" if filename in tracking else "NOUVEAU"
        print(f"\n[WATCH] Fichier {raison} détecté : {filename}")
        print("[WATCH] Indexation en cours...")

        if index_single_file(file_path, self.vectordb):
            tracking[filename] = {"hash": current_hash, "path": file_path}
            save_tracking(tracking)
            print(f"[WATCH] ✓ {filename} indexé avec succès\n")
        else:
            print(f"[WATCH] ✗ Échec de l'indexation de {filename}\n")

    def on_created(self, event):
        if not event.is_directory:
            # Petit délai pour s'assurer que le fichier est bien écrit
            time.sleep(1)
            self._process(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            time.sleep(1)
            self._process(event.src_path)


# ─────────────────────────────────────────
#  Modes
# ─────────────────────────────────────────

def mode_index():
    """Indexe tous les documents du dossier rag_space/documents/"""
    print("=" * 50)
    print("   MODE INDEXATION")
    print("=" * 50)
    index_all_documents()


def mode_watch():
    """Surveille le dossier documents/ et indexe automatiquement les nouveaux fichiers."""
    import os
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    print("=" * 50)
    print("   MODE SURVEILLANCE AUTOMATIQUE")
    print(f"   Dossier surveillé : {DOCUMENTS_DIR}")
    print("   Ctrl+C pour arrêter")
    print("=" * 50)

    # Indexation initiale des fichiers déjà présents
    print("\nIndexation initiale des fichiers existants...")
    index_all_documents()

    # Démarrage du watcher
    handler = DocumentHandler()
    observer = Observer()
    observer.schedule(handler, path=DOCUMENTS_DIR, recursive=False)
    observer.start()

    print(f"\n[WATCH] En attente de nouveaux fichiers dans {DOCUMENTS_DIR}...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[WATCH] Surveillance arrêtée.")

    observer.join()


def mode_chat():
    """Démarre une session de chat interactive avec l'agent RAG"""
    print("=" * 50)
    print("   MODE CHAT — Agent RAG")
    print("   Tape 'exit' ou 'quit' pour quitter")
    print("=" * 50)
    print()

    while True:
        try:
            question = input("Toi : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Au revoir !")
            break

        print("\nAgent : ", end="", flush=True)
        reponse = rag_agent(question)
        print(reponse)
        print()


def afficher_aide():
    print(__doc__)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        afficher_aide()
        sys.exit(0)

    commande = sys.argv[1].lower()

    if commande == "index":
        mode_index()

    elif commande == "watch":
        mode_watch()

    elif commande == "chat":
        mode_chat()

    else:
        print(f"Commande inconnue : '{commande}'")
        afficher_aide()
        sys.exit(1)