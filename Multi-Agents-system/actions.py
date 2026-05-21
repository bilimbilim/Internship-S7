import logging
import base64
from datetime import datetime
from pathlib import Path
from email.message import EmailMessage
import chromadb
from gmail_auth import get_gmail_service
from state import AgentState
from agent_email import execute_email_agent
from calendar_agent import execute_calendar_agent
logger = logging.getLogger("agent")

# MÉMOIRE VECTORIELLE
client = chromadb.PersistentClient(path="./chroma_email_db")
collection = client.get_or_create_collection(name="email_memory")
logger.info(" ChromaDB persistant initialisé → ./chroma_email_db")


def route_to_agent(state: AgentState) -> str:
    """
    Route vers le bon agent en fonction de l'intention
    
    Args:
        state: AgentState avec decision.intent
        
    Returns:
        str: Texte de la réponse générée par l'agent
    """
    intent = state.decision.intent
    
    logger.info(f" ROUTAGE - Intent détecté: {intent}")
    
    # Routage vers les agents
    if intent == "calendar":
        return execute_calendar_agent(state)
    
    elif intent == "email":
        return execute_email_agent(state)
    
    elif intent == "document":
        # TODO: Implémenter agent Document
        logger.info(" Agent Document pas encore implémenté, utilisation Agent Email")
        return execute_email_agent(state)
    
    else:
        # Par défaut, utiliser l'agent Email
        logger.info(f" Intent '{intent}' → Agent Email par défaut")
        return execute_email_agent(state)


def send_email_gmail(state: AgentState, reply_text: str):
    """
    Envoie l'email via Gmail
    
    Args:
        state: AgentState
        reply_text: Texte de la réponse à envoyer
    """
    logger.info(" Envoi de l'email via Gmail")
    
    service = get_gmail_service()
    
    # Construction du message
    message = EmailMessage()
    message["To"] = state.email.sender_email
    message["From"] = "me"
    message["Subject"] = f"Re: {state.email.subject}"
    message.set_content(reply_text)
    
    # Encodage
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {"raw": raw}
    
    # Conserver le thread si disponible
    if state.gmail_thread_id:
        body["threadId"] = state.gmail_thread_id
    
    # Envoi
    service.users().messages().send(
        userId="me",
        body=body
    ).execute()
    
    # Marquer comme lu
    if state.gmail_message_id:
        service.users().messages().modify(
            userId="me",
            id=state.gmail_message_id,
            body={"removeLabelIds": ["UNREAD"]}
        ).execute()
    
    logger.info(" Email envoyé et marqué comme lu")


def save_local_trace(state: AgentState, reply_text: str):
    """
    Sauvegarde une trace locale de l'email
    
    Args:
        state: AgentState
        reply_text: Texte de la réponse envoyée
    """
    outbox = Path("emails/outbox")
    outbox.mkdir(parents=True, exist_ok=True)
    
    filename = outbox / f"sent_{datetime.now():%Y%m%d_%H%M%S}.txt"
    
    filename.write_text(
        f"""TO : {state.email.sender_name} <{state.email.sender_email}>
SUBJECT : Re: {state.email.subject}
INTENT : {state.decision.intent}
PRIORITY : {state.decision.priority}
ATTACHMENTS : {len(state.attachments)}
TIMESTAMP : {datetime.now():%Y-%m-%d %H:%M:%S}

--- RÉPONSE GÉNÉRÉE ---
{reply_text}

--- EMAIL ORIGINAL ---
{state.email.content}
""",
        encoding="utf-8"
    )
    
    logger.info(f" Trace locale : {filename}")


def store_in_memory(state: AgentState, reply_text: str = ""):
    """
    Stocke l'échange complet (email reçu + réponse envoyée) dans la mémoire vectorielle.
    Le document indexé combine les deux pour permettre une recherche sémantique cohérente.
    
    Args:
        state: AgentState
        reply_text: Réponse générée et envoyée
    """
    try:
        # Document indexé = email reçu + réponse envoyée
        # Permet de retrouver l'échange complet lors d'une recherche sémantique
        document = (
            f"[EMAIL REÇU]\n"
            f"De : {state.email.sender_name} <{state.email.sender_email}>\n"
            f"Objet : {state.email.subject}\n"
            f"{state.email.content}\n\n"
            f"[RÉPONSE ENVOYÉE]\n"
            f"{reply_text}"
        )

        collection.add(
            documents=[document],
            metadatas=[{
                "sender_name": state.email.sender_name,
                "sender_email": state.email.sender_email,
                "subject": state.email.subject,
                "intent": state.decision.intent,
                "priority": state.decision.priority,
                "has_reply": bool(reply_text),
                "timestamp": datetime.now().isoformat()
            }],
            ids=[f"email_{datetime.now():%Y%m%d_%H%M%S}"]
        )
        logger.info(" Échange (email + réponse) sauvegardé en mémoire")
    except Exception as e:
        logger.warning(f"  Erreur sauvegarde mémoire : {e}")


def execute_action(state: AgentState) -> AgentState:
    """
    ACTION - Exécute l'action finale (envoi de l'email)
    
    Workflow :
    1. Route vers le bon agent (Calendar ou Email)
    2. Envoie l'email via Gmail
    3. Sauvegarde une trace locale
    4. Stocke en mémoire vectorielle
    
    Args:
        state: AgentState
        
    Returns:
        AgentState: State mis à jour
    """
    logger.info(" ACTION - Exécution")
    
    # 1. Routage vers le bon agent
    reply_text = route_to_agent(state)
    
    # 2. Envoi de l'email
    send_email_gmail(state, reply_text)
    
    # 3. Trace locale
    save_local_trace(state, reply_text)
    
    # 4. Mémoire vectorielle (email reçu + réponse envoyée)
    store_in_memory(state, reply_text)
    
    # Mise à jour du résultat
    state.result = f" Réponse envoyée via Agent {state.decision.intent.upper()}"
    
    logger.info(f" ACTION TERMINÉE - {state.result}")
    
    return state