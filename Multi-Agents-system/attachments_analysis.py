
import logging
import shutil
from pathlib import Path

import fitz  # pip install pymupdf

logger = logging.getLogger("agent")


# EXTRACTION TEXTE PDF
def extract_text_from_pdf(file_path: str) -> str:
    text = ""

    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()

    except Exception as e:
        logger.warning(f"Erreur lecture PDF {file_path}: {e}")

    return text[:3000]


# CLASSIFICATION DOCUMENT
def classify_attachment(text: str) -> str:

    text = text.lower()

    if any(k in text for k in ["curriculum", "compétence", "expérience"]):
        return "cv"

    if any(k in text for k in ["facture", "montant", "tva"]):
        return "facture"

    if "contrat" in text:
        return "contrat"

    return "document"


# TRI AUTOMATIQUE FICHIERS
def move_file_by_category(file_path: str, category: str) -> str:

    base_dir = Path("emails/attachments")
    target_dir = base_dir / category
    target_dir.mkdir(parents=True, exist_ok=True)

    new_path = target_dir / Path(file_path).name

    shutil.move(file_path, new_path)

    return str(new_path)


# ANALYSE GLOBALE ATTACHMENTS
def process_attachments(attachments: list[str]):

    analyzed = []

    for file_path in attachments:

        if not file_path.lower().endswith(".pdf"):
            continue

      
        text = extract_text_from_pdf(file_path)

        category = classify_attachment(text)

        new_path = move_file_by_category(file_path, category)

        analyzed.append({
            "file": new_path,
            "type": category
        })

    logger.info(f"Analyse pièces jointes : {analyzed}")

    return analyzed
