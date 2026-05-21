import logging
import json
import uuid
from datetime import datetime, timedelta
import re

from gmail_auth import get_calendar_service
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from state import AgentState

# ======================================================
# LOGGING
# ======================================================
logger = logging.getLogger("agent")

# ======================================================
# LLM LOCAL
# ======================================================
llm = OllamaLLM(model="llama3", temperature=0)

# ======================================================
# CHROMADB — MÉMOIRE RAG
# ======================================================
try:
    embeddings = OllamaEmbeddings(model="llama3")
    chroma_db = Chroma(
        collection_name="meeting_proposals",
        embedding_function=embeddings,
        persist_directory="./chroma_calendar_db"
    )
    logger.info("ChromaDB initialisé")
except Exception as e:
    logger.warning(f"Erreur ChromaDB : {e}")
    chroma_db = None

# ======================================================
# PROMPTS
# ======================================================

extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Extrait les informations de cet email de confirmation de rendez-vous.

Email :
{text}

Réponds UNIQUEMENT en JSON (sans ```json ni ```) :
{{
    "creneau_choisi": 1,
    "date_mentionnee": "texte brut de la date si présente, sinon null",
    "heure_mentionnee": "texte brut de l'heure si présente, sinon null"
}}

creneau_choisi doit être 1 ou 2 selon le créneau mentionné. Si on ne peut pas déterminer, mettre 1.
"""
)

# ======================================================
# FONCTIONS UTILITAIRES
# ======================================================

def get_agenda():
    """Récupère les événements à venir du calendrier Google."""
    try:
        service = get_calendar_service()
        now = datetime.utcnow().isoformat() + "Z"
        events_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=20,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        events = events_result.get("items", [])
        logger.info(f"{len(events)} événements récupérés depuis Google Calendar")
        return events
    except Exception as e:
        logger.error(f"Erreur récupération agenda : {e}")
        return []


def get_busy_slots(events):
    """Retourne l'ensemble des créneaux occupés sous forme de set (date, heure)."""
    busy = set()
    for event in events:
        start_raw = event["start"].get("dateTime", event["start"].get("date"))
        try:
            dt = datetime.fromisoformat(start_raw.replace("Z", ""))
            busy.add((dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")))
        except Exception:
            continue
    return busy


def find_next_available_slots(busy_slots, count=2):
    """
    Cherche les N prochains créneaux libres en semaine entre 09h00 et 16h00.
    Retourne une liste de datetime.
    """
    slots = []
    candidate = datetime.now() + timedelta(days=1)
    candidate = candidate.replace(minute=0, second=0, microsecond=0)

    # On part toujours à 09h00 du matin
    if candidate.hour < 9:
        candidate = candidate.replace(hour=9)
    elif candidate.hour >= 16:
        candidate = candidate + timedelta(days=1)
        candidate = candidate.replace(hour=9)
    else:
        # Arrondir à la prochaine heure entière
        candidate = candidate.replace(hour=candidate.hour + 1)

    hours_to_try = [9, 10, 11, 14, 15]

    while len(slots) < count:
        # Sauter le week-end
        if candidate.weekday() >= 5:
            candidate = candidate + timedelta(days=(7 - candidate.weekday()))
            candidate = candidate.replace(hour=9, minute=0)
            continue

        for h in hours_to_try:
            if len(slots) >= count:
                break
            test = candidate.replace(hour=h, minute=0, second=0, microsecond=0)
            key = (test.strftime("%Y-%m-%d"), test.strftime("%H:%M"))
            if key not in busy_slots:
                slots.append(test)

        candidate += timedelta(days=1)
        candidate = candidate.replace(hour=9, minute=0)

    return slots


def format_date_fr(dt):
    """Formate une datetime en français lisible."""
    jours = {
        'Monday': 'Lundi', 'Tuesday': 'Mardi', 'Wednesday': 'Mercredi',
        'Thursday': 'Jeudi', 'Friday': 'Vendredi', 'Saturday': 'Samedi', 'Sunday': 'Dimanche'
    }
    mois = {
        'January': 'janvier', 'February': 'février', 'March': 'mars',
        'April': 'avril', 'May': 'mai', 'June': 'juin',
        'July': 'juillet', 'August': 'août', 'September': 'septembre',
        'October': 'octobre', 'November': 'novembre', 'December': 'décembre'
    }
    jour_en = dt.strftime("%A")
    mois_en = dt.strftime("%B")
    return dt.strftime(f"{jours[jour_en]} %d {mois[mois_en]} %Y à %H:%M")


def detect_email_type(email_text):
    """Détecte si l'email est une demande de RDV, une confirmation, ou autre."""
    email_lower = email_text.lower()

    confirm_words = [
        "confirme", "je confirme", "ok", "okay",
        "d'accord", "daccord", "je suis d'accord",
        "je valide", "validé", "parfait",
        "ça me va", "ca me va", "c'est bon", "cest bon",
        "cela me convient", "ça convient", "convient",
        "je prends", "je choisis", "c'est noté",
        "marché conclu", "cela fonctionne",
        "cela me va très bien", "je suis disponible",
        "cela me convient parfaitement", "très bien pour moi",
        "impeccable", "c'est parfait", "entendu"
    ]

    has_confirm = any(word in email_lower for word in confirm_words)
    days = ["lundi", "mardi", "mercredi", "jeudi", "vendredi"]
    has_day = any(day in email_lower for day in days)
    has_time = re.search(r"\b\d{1,2}h\d{0,2}\b|\b\d{1,2}:\d{2}\b", email_lower)

    creneau = None
    if "créneau 1" in email_lower or "creneau 1" in email_lower or "option 1" in email_lower:
        creneau = 1
    elif "créneau 2" in email_lower or "creneau 2" in email_lower or "option 2" in email_lower:
        creneau = 2

    if has_confirm and (has_day or has_time or creneau):
        email_type = "confirmation_rdv"
    elif any(word in email_lower for word in [
        "rendez-vous", "rdv", "rencontrer", "disponible",
        "créneau", "réunion", "meeting", "discuter"
    ]):
        email_type = "demande_rdv"
    else:
        email_type = "autre"

    result = {
        "type": email_type,
        "creneau_choisie": creneau
    }
    logger.info(f"Détection email : {result}")
    return result


# ======================================================
# RAG — SAUVEGARDE D'UNE PROPOSITION
# ======================================================

def save_proposal_to_rag(proposal_data: dict) -> str:
    """
    Sauvegarde une proposition de rendez-vous dans ChromaDB.
    Le texte embarqué est une description naturelle lisible par le LLM.
    Retourne le proposal_id.
    """
    proposal_id = proposal_data["proposal_id"]

    # Texte sémantique pour la recherche RAG
    text_for_embedding = (
        f"Proposition de rendez-vous pour {proposal_data['sender_name']} "
        f"({proposal_data['sender_email']}). "
        f"Créneau 1 : {proposal_data['creneau_1_label']}. "
        f"Créneau 2 : {proposal_data['creneau_2_label']}. "
        f"Statut : {proposal_data['status']}. "
        f"Sujet : {proposal_data['sujet']}."
    )

    chroma_db.add_documents(
        documents=[
            Document(
                page_content=text_for_embedding,
                metadata={
                    "proposal_id": proposal_id,
                    "sender_email": proposal_data["sender_email"],
                    "sender_name": proposal_data["sender_name"],
                    "status": proposal_data["status"],
                    "raw_json": json.dumps(proposal_data)
                }
            )
        ],
        ids=[proposal_id]
    )
    logger.info(f"Proposition {proposal_id} sauvegardée dans ChromaDB")
    return proposal_id


# ======================================================
# RAG — RECHERCHE DE LA PROPOSITION EN ATTENTE
# ======================================================

def search_pending_proposal(sender_email: str, email_content: str) -> dict | None:
    """
    Recherche dans ChromaDB la proposition en attente pour cet expéditeur.

    Stratégie RAG en 2 passes :
    1. Recherche sémantique sur le contenu de l'email de confirmation
    2. Filtrage par sender_email + status=pending parmi les résultats
    """
    if not chroma_db:
        return None

    # --- Passe 1 : recherche sémantique ---
    query = f"confirmation rendez-vous {sender_email} {email_content[:200]}"

    try:
        results = chroma_db.similarity_search_with_score(query, k=10)
        logger.info(f"RAG : {len(results)} résultats sémantiques trouvés")
    except Exception as e:
        logger.warning(f"Erreur recherche sémantique : {e}")
        results = []

    # --- Passe 2 : filtrage strict ---
    for doc, score in results:
        meta = doc.metadata
        logger.debug(f"  → {meta.get('sender_email')} | status={meta.get('status')} | score={score:.3f}")

        if (
            meta.get("sender_email") == sender_email
            and meta.get("status") == "pending"
        ):
            raw = meta.get("raw_json")
            if raw:
                proposal = json.loads(raw)
                logger.info(f"Proposition trouvée via RAG : {proposal['proposal_id']} (score={score:.3f})")
                return proposal

    # --- Fallback : scan complet si RAG ne trouve rien ---
    logger.warning("RAG sémantique insuffisant, fallback scan complet ChromaDB")
    all_results = chroma_db.get()
    for idx, doc_id in enumerate(all_results.get("ids", [])):
        meta = all_results["metadatas"][idx]
        if (
            meta.get("sender_email") == sender_email
            and meta.get("status") == "pending"
        ):
            raw = meta.get("raw_json")
            if raw:
                logger.info(f"Proposition trouvée via fallback scan : {doc_id}")
                return json.loads(raw)

    logger.warning(f"Aucune proposition pending trouvée pour {sender_email}")
    return None


def update_proposal_status(proposal_data: dict, new_status: str, calendar_event_id: str = None):
    """Met à jour le statut d'une proposition dans ChromaDB."""
    proposal_id = proposal_data["proposal_id"]
    proposal_data["status"] = new_status
    if calendar_event_id:
        proposal_data["calendar_event_id"] = calendar_event_id

    # Supprimer l'ancien document
    chroma_db.delete(ids=[proposal_id])

    # Ré-insérer avec le nouveau statut
    save_proposal_to_rag(proposal_data)
    logger.info(f"Proposition {proposal_id} mise à jour → status={new_status}")


# ======================================================
# PROPOSITION DE RENDEZ-VOUS (2 créneaux via agenda réel)
# ======================================================

def propose_meeting(state: AgentState) -> str:
    """
    1. Lit l'agenda Google Calendar
    2. Trouve 2 créneaux libres
    3. Sauvegarde dans ChromaDB (RAG)
    4. Retourne l'email de réponse
    """
    sender_email = state.email.sender_email
    sender_name = state.email.sender_name

    logger.info(f"[propose_meeting] Proposition pour {sender_name} <{sender_email}>")

    #  Récupération de l'agenda 
    events = get_agenda()
    busy_slots = get_busy_slots(events)
    logger.info(f"Créneaux occupés : {busy_slots}")

    #  Recherche de 2 créneaux libres 
    available = find_next_available_slots(busy_slots, count=2)

    if len(available) < 2:
        logger.error("Impossible de trouver 2 créneaux libres")
        return f"""Bonjour {sender_name},

Je n'ai pas réussi à trouver deux créneaux disponibles dans les prochains jours.
Pourriez-vous me proposer vos disponibilités ?

Cordialement,
Bilimbilim Mombo"""

    slot1, slot2 = available[0], available[1]

    creneau_1_label = format_date_fr(slot1)
    creneau_2_label = format_date_fr(slot2)

    # --- Génération du proposal_id ---
    proposal_id = str(uuid.uuid4())[:8]

    # --- Sauvegarde RAG ---
    proposal_data = {
        "proposal_id": proposal_id,
        "sender_email": sender_email,
        "sender_name": sender_name,
        "creneau_1_date": slot1.strftime("%Y-%m-%d"),
        "creneau_1_heure": slot1.strftime("%H:%M"),
        "creneau_1_label": creneau_1_label,
        "creneau_2_date": slot2.strftime("%Y-%m-%d"),
        "creneau_2_heure": slot2.strftime("%H:%M"),
        "creneau_2_label": creneau_2_label,
        "sujet": "Rendez-vous",
        "duree_minutes": 30,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }

    if chroma_db:
        save_proposal_to_rag(proposal_data)
    else:
        logger.warning("ChromaDB indisponible, proposition non sauvegardée")

    # --- Construction de l'email ---
    email_response = f"""Bonjour {sender_name},

Merci pour votre message.

Je vous propose les deux créneaux suivants :

- Créneau 1 : {creneau_1_label}
- Créneau 2 : {creneau_2_label}

Merci de me confirmer le créneau qui vous convient.

Cordialement,
Bilimbilim Mombo

[Réf : {proposal_id}]"""

    logger.info(f"Email de proposition généré (réf {proposal_id})")
    return email_response


# ======================================================
# CONFIRMATION DE RENDEZ-VOUS
# 

def confirm_meeting_from_email(state: AgentState, detection: dict) -> str:
    """
    1. Recherche dans ChromaDB (RAG) la proposition en attente
    2. Détermine quel créneau a été choisi
    3. Crée l'événement dans Google Calendar
    4. Met à jour le statut dans ChromaDB
    5. Retourne l'email de confirmation
    """
    sender_email = state.email.sender_email
    sender_name = state.email.sender_name
    email_content = state.email.content

    logger.info(f"[confirm_meeting] Confirmation pour {sender_name} <{sender_email}>")

    if not chroma_db:
        return f"""Bonjour {sender_name},

La base de données est indisponible. Impossible de confirmer le rendez-vous.

Cordialement,
Bilimbilim Mombo"""

    # --- Recherche RAG ---
    proposal_data = search_pending_proposal(sender_email, email_content)

    if not proposal_data:
        return f"""Bonjour {sender_name},

Je ne retrouve pas de proposition de rendez-vous en attente pour votre adresse.
Souhaitez-vous que je vous propose de nouveaux créneaux ?

Cordialement,
Bilimbilim Mombo"""

    # --- Détermination du créneau choisi ---
    creneau_num = detection.get("creneau_choisie")

    if creneau_num is None:
        # Utiliser le LLM pour extraire le créneau depuis l'email
        try:
            extraction_chain = extraction_prompt | llm
            raw_result = extraction_chain.invoke({"text": email_content})
            clean = raw_result.strip().replace("```json", "").replace("```", "")
            extracted = json.loads(clean)
            creneau_num = int(extracted.get("creneau_choisi", 1))
            logger.info(f"LLM a détecté le créneau : {creneau_num}")
        except Exception as e:
            logger.warning(f"Extraction LLM échouée ({e}), créneau 1 par défaut")
            creneau_num = 1

    # Sélection des données du créneau
    if creneau_num == 2:
        date_str = proposal_data["creneau_2_date"]
        heure_str = proposal_data["creneau_2_heure"]
        label = proposal_data["creneau_2_label"]
    else:
        date_str = proposal_data["creneau_1_date"]
        heure_str = proposal_data["creneau_1_heure"]
        label = proposal_data["creneau_1_label"]

    duree = proposal_data.get("duree_minutes", 30)
    sujet = proposal_data.get("sujet", "Rendez-vous")

    logger.info(f"Créneau retenu : {label} (créneau {creneau_num})")

    # --- Création de l'événement Google Calendar ---
    start_datetime = f"{date_str}T{heure_str}:00"
    end_datetime = (
        datetime.fromisoformat(start_datetime) + timedelta(minutes=duree)
    ).isoformat()

    try:
        service = get_calendar_service()
        event = {
            "summary": sujet,
            "description": f"Rendez-vous confirmé avec {sender_name}",
            "start": {"dateTime": start_datetime, "timeZone": "Europe/Paris"},
            "end": {"dateTime": end_datetime, "timeZone": "Europe/Paris"},
            "attendees": [{"email": sender_email}]
        }
        created_event = service.events().insert(
            calendarId="primary",
            body=event,
            sendUpdates="all"
        ).execute()
        calendar_event_id = created_event["id"]
        logger.info(f"Événement créé dans Google Calendar : {calendar_event_id}")
    except Exception as e:
        logger.error(f"Erreur création événement Calendar : {e}")
        return f"""Bonjour {sender_name},

Une erreur est survenue lors de la création de l'événement dans le calendrier.
Veuillez réessayer ou me contacter directement.

Cordialement,
Bilimbilim Mombo"""

    # --- Mise à jour RAG : pending → confirmed ---
    update_proposal_status(proposal_data, "confirmed", calendar_event_id)

    # --- Email de confirmation ---
    return f"""Bonjour {sender_name},

Votre rendez-vous est confirmé.

Date : {label}
Durée : {duree} minutes
Objet : {sujet}

Une invitation a été envoyée à votre adresse email.

Cordialement,
Bilimbilim Mombo"""


# ======================================================
# POINT D'ENTRÉE PRINCIPAL
# ======================================================

def execute_calendar_agent(state: AgentState) -> str:
    logger.info("=" * 50)
    logger.info("AGENT CALENDAR — Activation")

    detection = detect_email_type(state.email.content)
    email_type = detection["type"]

    logger.info(f"Type détecté : {email_type}")

    if email_type == "confirmation_rdv":
        logger.info("→ Traitement : confirmation de rendez-vous")
        return confirm_meeting_from_email(state, detection)

    logger.info("→ Traitement : nouvelle demande de rendez-vous")
    return propose_meeting(state)