from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import os
import json
import hashlib

PERSIST_DIR = "rag_space/chroma_db"
DOCUMENTS_DIR = "rag_space/documents"
TRACKING_FILE = "rag_space/indexed_files.json"

# Formats supportés
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".csv", ".xlsx", ".xls", ".docx", ".json")


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


def get_file_hash(file_path: str) -> str:
    """Calcule le hash MD5 d'un fichier pour détecter les modifications."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_tracking() -> dict:
    """Charge le fichier de tracking JSON. Retourne un dict vide si inexistant."""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_tracking(tracking: dict) -> None:
    """Sauvegarde le fichier de tracking JSON."""
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(tracking, f, indent=2, ensure_ascii=False)


def load_file(file_path: str) -> list:
    """
    Charge un fichier selon son extension.
    Retourne une liste de documents LangChain.
    """
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()

    elif file_path.endswith(".txt"):
        return TextLoader(file_path, encoding="utf-8").load()

    elif file_path.endswith(".csv"):
        return CSVLoader(file_path, encoding="utf-8").load()

    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return UnstructuredExcelLoader(file_path).load()

    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()

    elif file_path.endswith(".json"):
        return JSONLoader(
            file_path=file_path,
            jq_schema=".",
            text_content=False
        ).load()

    else:
        return []


def index_single_file(file_path: str, vectordb: Chroma) -> bool:
    """
    Indexe un fichier dans la base vectorielle.
    Supporte : PDF, TXT, CSV, Excel, Word, JSON
    Retourne True si succès, False sinon.
    """
    try:
        documents = load_file(file_path)

        if not documents:
            print(f"  [Vide] Aucun contenu extrait de : {file_path}")
            return False

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)

        vectordb.add_documents(chunks)
        print(f"  [OK] {os.path.basename(file_path)} — {len(chunks)} chunks indexés")
        return True

    except Exception as e:
        print(f"  [Erreur] {os.path.basename(file_path)} : {e}")
        return False


def index_all_documents() -> None:
    """
    Parcourt le dossier documents/ et indexe uniquement les fichiers
    nouveaux ou modifiés depuis la dernière indexation.
    """
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"Dossier créé : {DOCUMENTS_DIR}")
        print(f"Ajoute des fichiers {SUPPORTED_EXTENSIONS} dans ce dossier puis relance.")
        return

    files = [
        f for f in os.listdir(DOCUMENTS_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not files:
        print(f"Aucun fichier supporté trouvé dans {DOCUMENTS_DIR}")
        print(f"Formats acceptés : {SUPPORTED_EXTENSIONS}")
        return

    # Chargement du tracking
    tracking = load_tracking()

    # Détection des fichiers nouveaux ou modifiés
    to_index = []
    for filename in files:
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        current_hash = get_file_hash(file_path)

        if filename not in tracking:
            to_index.append((filename, file_path, current_hash, "nouveau"))
        elif tracking[filename]["hash"] != current_hash:
            to_index.append((filename, file_path, current_hash, "modifié"))
        else:
            print(f"  [Ignoré] {filename} — déjà indexé")

    if not to_index:
        print("\nTous les fichiers sont déjà indexés. Rien à faire.")
        return

    print(f"\n{len(to_index)} fichier(s) à indexer...\n")

    embeddings = LocalEmbeddings()
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    success = 0
    for filename, file_path, file_hash, raison in to_index:
        print(f"  [{raison.upper()}] {filename}")
        if index_single_file(file_path, vectordb):
            tracking[filename] = {
                "hash": file_hash,
                "path": file_path
            }
            success += 1

    # Sauvegarde du tracking mis à jour
    save_tracking(tracking)

    print(f"\nIndexation terminée : {success}/{len(to_index)} fichier(s) indexé(s)")
    print(f"Base vectorielle : {PERSIST_DIR}")
    print(f"Tracking sauvegardé : {TRACKING_FILE}")