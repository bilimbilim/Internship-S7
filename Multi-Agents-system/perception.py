import base64
from typing import List, Optional
from state import AgentState, EmailModel
from gmail_auth import get_gmail_service
from agent_email import get_thread_history   # ← AJOUT
from pathlib import Path
from email.utils import parseaddr
import logging

logger = logging.getLogger("agent")


def extract_sender_info(sender_raw: str) -> tuple[str, str]:
    """
    Extrait nom et email depuis le format 'Nom <email@domain.com>'
    Args:
        sender_raw: Chaîne brute du champ From
    Returns:
        tuple: (nom, email)
    """
    # Fallback si sender_raw est vide ou None
    if not sender_raw:
        logger.warning("  Champ 'From' vide ou absent")
        return "Expéditeur inconnu", "inconnu@inconnu.com"

    name, email = parseaddr(sender_raw)

    # Si parseaddr n'a pas réussi à extraire un email valide
    if not email or "@" not in email:
        logger.warning(f"  Email invalide extrait depuis : '{sender_raw}'")
        email = "inconnu@inconnu.com"

    # Si pas de nom, utiliser la partie locale de l'email
    if not name:
        name = email.split("@")[0]

    return name.strip(), email.strip()


def extract_body(payload: dict) -> str:
    """
    Extrait le corps texte d'un email Gmail, qu'il soit simple ou multipart.

    Stratégie :
        1. Email multipart → chercher 'text/plain' dans les parts
        2. Email simple    → lire directement payload['body']['data']
        3. Fallback        → chercher récursivement dans les sous-parts

    Args:
        payload: payload du message Gmail

    Returns:
        str: Corps du message (vide si non trouvé)
    """

    def decode_data(data: str) -> str:
        """Décode le base64 Gmail en texte."""
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"  Erreur décodage base64 : {e}")
            return ""

    def search_parts(parts: list) -> Optional[str]:
        """Cherche récursivement 'text/plain' dans les parts imbriquées."""
        for part in parts:
            mime = part.get("mimeType", "")
            body = part.get("body", {})

            # Part texte simple avec données
            if mime == "text/plain" and "data" in body:
                return decode_data(body["data"])

            # Part imbriquée → descendre récursivement
            if "parts" in part:
                result = search_parts(part["parts"])
                if result:
                    return result

        return None

    # --- CAS 1 : Email multipart ---
    if "parts" in payload:
        body = search_parts(payload["parts"])
        if body:
            return body
        logger.warning("  Aucune part 'text/plain' trouvée dans l'email multipart")

    # --- CAS 2 : Email simple (pas de parts) ---
    simple_body = payload.get("body", {})
    if "data" in simple_body:
        logger.info("  Email simple (pas de parts), lecture directe du body")
        return decode_data(simple_body["data"])

    # --- CAS 3 : Aucun body trouvé ---
    logger.warning("  Impossible d'extraire le corps de l'email")
    return ""


def get_header(headers: list, name: str, default: str = "") -> str:
    """
    Extrait un header Gmail par nom avec une valeur par défaut.

    Args:
        headers: Liste des headers du message Gmail
        name: Nom du header à chercher (ex: 'From', 'Subject')
        default: Valeur retournée si le header est absent

    Returns:
        str: Valeur du header ou default
    """
    value = next(
        (h["value"] for h in headers if h["name"].lower() == name.lower()),
        default
    )
    if not value and default == "":
        logger.warning(f"  Header '{name}' absent ou vide")
    return value


def download_attachments(service, msg) -> List[str]:
    """
    Télécharge les pièces jointes d'un email Gmail

    Args:
        service: Service Gmail API
        msg: Message Gmail

    Returns:
        List[str]: Liste des chemins des fichiers téléchargés
    """
    save_dir = Path("emails/attachments")
    save_dir.mkdir(parents=True, exist_ok=True)
    attachments = []

    def extract_parts(parts):
        for part in parts:
            filename = part.get("filename")

            if filename:
                body = part["body"]

                if "attachmentId" in body:
                    try:
                        attachment = service.users().messages().attachments().get(
                            userId="me",
                            messageId=msg["id"],
                            id=body["attachmentId"]
                        ).execute()

                        data = base64.urlsafe_b64decode(attachment["data"])
                        path = save_dir / filename

                        with open(path, "wb") as f:
                            f.write(data)

                        attachments.append(str(path))
                        logger.info(f"📎 Pièce jointe téléchargée : {filename}")

                    except Exception as e:
                        # Ne pas planter si une pièce jointe échoue
                        logger.warning(f"  Erreur téléchargement pièce jointe '{filename}' : {e}")

            if "parts" in part:
                extract_parts(part["parts"])

    if "parts" in msg.get("payload", {}):
        extract_parts(msg["payload"]["parts"])

    return attachments


def fetch_all_unread_ids() -> List[str]:
    """
    Récupère les IDs de tous les emails non lus dans la boîte de réception.

    Returns:
        List[str]: Liste des IDs de messages non lus (du plus récent au plus ancien)
    """
    try:
        service = get_gmail_service()
    except Exception as e:
        logger.error(f"  Impossible de se connecter à Gmail : {e}")
        return []

    all_ids = []
    page_token = None

    while True:
        try:
            kwargs = {
                "userId": "me",
                "maxResults": 100,
                "q": "is:inbox is:unread"
            }
            if page_token:
                kwargs["pageToken"] = page_token

            results = service.users().messages().list(**kwargs).execute()
        except Exception as e:
            logger.error(f"  Erreur liste des messages : {e}")
            break

        messages = results.get("messages", [])
        all_ids.extend(m["id"] for m in messages)

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    logger.info(f" {len(all_ids)} email(s) non lu(s) trouvé(s)")
    return all_ids


def read_email_gmail(state: AgentState) -> AgentState:
    """
    PERCEPTION - Lit un email Gmail dont l'ID est fourni dans state.gmail_message_id.
    Si aucun ID n'est fourni, lit le premier email non lu.

    Args:
        state: AgentState (doit contenir gmail_message_id si appelé en boucle)

    Returns:
        AgentState: State mis à jour avec les données de l'email
    """
    logger.info(" PERCEPTION - Lecture email Gmail")

    try:
        service = get_gmail_service()
    except Exception as e:
        logger.error(f"  Impossible de se connecter à Gmail : {e}")
        state.result = f"Erreur connexion Gmail : {e}"
        return state

    # Utilise l'ID déjà fourni dans le state, sinon prend le premier non lu
    msg_id = state.gmail_message_id

    if not msg_id:
        try:
            results = service.users().messages().list(
                userId="me",
                maxResults=1,
                q="is:inbox is:unread"
            ).execute()
        except Exception as e:
            logger.error(f"  Erreur liste des messages : {e}")
            state.result = f"Erreur récupération emails : {e}"
            return state

        messages = results.get("messages", [])
        if not messages:
            logger.info(" Aucun email non lu")
            state.result = "Aucun email non lu"
            return state

        msg_id = messages[0]["id"]

    try:
        msg = service.users().messages().get(
            userId="me",
            id=msg_id,
            format="full"
        ).execute()
    except Exception as e:
        logger.error(f"  Erreur récupération message {msg_id} : {e}")
        state.result = f"Erreur lecture email : {e}"
        return state

    # Pièces jointes
    attachments = download_attachments(service, msg)
    if attachments:
        logger.info(f"📎 {len(attachments)} pièce(s) jointe(s) téléchargée(s)")

    # --- HEADERS avec fallbacks ---
    headers = msg.get("payload", {}).get("headers", [])

    if not headers:
        logger.warning("  Aucun header trouvé dans l'email")

    sender_raw   = get_header(headers, "From",    default="")
    subject      = get_header(headers, "Subject", default="(Sans objet)")
    date_header  = get_header(headers, "Date",    default="")

    sender_name, sender_email = extract_sender_info(sender_raw)

    # --- BODY avec fallbacks ---
    body = extract_body(msg.get("payload", {}))

    if not body:
        logger.warning("  Corps de l'email vide, utilisation d'un contenu par défaut")
        body = "(Contenu non disponible)"

    # =========================================================
    # Récupération de l'historique de la conversation
    # =========================================================
    thread_id = msg.get("threadId", "")
    conversation_history = get_thread_history(thread_id)

    if conversation_history:
        logger.info(f" Historique chargé : {len(conversation_history)} messages dans ce thread")
    else:
        logger.info(" Pas d'historique : premier message de cette conversation")
    # =========================================================

    logger.info(f" Email reçu de : {sender_name} <{sender_email}> | Sujet : {subject} | Date : {date_header}")

    state.email = EmailModel(
        sender_name=sender_name,
        sender_email=sender_email,
        subject=subject,
        content=body,
        conversation_history=conversation_history
    )
    state.gmail_message_id = msg["id"]
    state.gmail_thread_id  = thread_id
    state.attachments      = attachments

    return state