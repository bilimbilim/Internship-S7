import logging
import base64
from datetime import datetime
from pathlib import Path
from email.message import EmailMessage
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from gmail_auth import get_gmail_service
from state import AgentState, EmailModel, DecisionModel

logger = logging.getLogger("agent")

# Constante pour la limite d'historique
MAX_HISTORY_MESSAGES = 200

# Réutilisation de la collection ChromaDB créée dans actions.py
# (un seul PersistentClient par path autorisé dans le même process)
def _get_memory_collection():
    """
    Récupère la collection email_memory depuis actions.py pour éviter
    un double PersistentClient sur le même path SQLite.
    Import différé pour éviter les imports circulaires.
    """
    from actions import collection
    return collection

MEMORY_SCORE_THRESHOLD = 1.2   # Distance L2 : plus bas = plus similaire
MEMORY_MAX_RESULTS = 3         # Nombre max d'échanges passés à injecter


def get_past_exchanges(sender_email: str, current_content: str) -> str:
    """
    Recherche dans chroma_email_db les échanges passés pertinents,
    en filtrant par expéditeur et par similarité sémantique avec l'email courant.

    Args:
        sender_email: Adresse email de l'expéditeur
        current_content: Contenu de l'email courant (utilisé pour la recherche sémantique)

    Returns:
        str: Bloc de texte formaté avec les échanges passés, ou chaîne vide si aucun résultat
    """
    try:
        collection = _get_memory_collection()

        # ChromaDB lève une exception si n_results > nombre de docs dans la collection
        # On récupère d'abord le count filtré par expéditeur pour adapter n_results
        try:
            count_result = collection.get(where={"sender_email": sender_email})
            available = len(count_result.get("ids", []))
        except Exception:
            available = 0

        if available == 0:
            return ""

        n = min(MEMORY_MAX_RESULTS, available)

        results = collection.query(
            query_texts=[current_content],
            n_results=n,
            where={"sender_email": sender_email},
            include=["documents", "metadatas", "distances"]
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return ""

        relevant = [
            (doc, meta, dist)
            for doc, meta, dist in zip(docs, metas, distances)
            if dist <= MEMORY_SCORE_THRESHOLD
        ]

        if not relevant:
            return ""

        logger.info(f" {len(relevant)} échange(s) passé(s) trouvé(s) pour {sender_email}")

        lines = ["=== ÉCHANGES PASSÉS AVEC CET EXPÉDITEUR ===\n"]
        for i, (doc, meta, dist) in enumerate(relevant, 1):
            date = meta.get("timestamp", "date inconnue")[:10]
            subject = meta.get("subject", "sans objet")
            lines.append(f"[Échange {i} — {date} | Objet : {subject}]")
            lines.append(doc.strip())
            lines.append("-" * 60 + "\n")
        lines.append("=== FIN DES ÉCHANGES PASSÉS ===\n")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"  Erreur lecture mémoire : {e}")
        return ""


def get_thread_history(thread_id: str, max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    """
    Récupère l'historique d'un thread Gmail (conversation)
    
    Args:
        thread_id: ID du thread Gmail
        max_messages: Nombre maximum de messages à récupérer (défaut: 100)
        
    Returns:
        Liste de dict avec {sender, sender_name, content, date, is_from_me}
    """
    if not thread_id:
        logger.warning(" Aucun thread_id fourni, historique vide")
        return []
    
    logger.info(f" Récupération historique thread {thread_id[:10]}... (max {max_messages} messages)")
    
    try:
        service = get_gmail_service()
        thread = service.users().threads().get(
            userId='me',
            id=thread_id,
            format='full'
        ).execute()
        
        history = []
        messages = thread.get('messages', [])
        
        # Limiter au nombre max de messages (prendre les plus récents)
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
            logger.info(f" Thread tronqué à {max_messages} messages (total: {len(thread.get('messages', []))})")
        
        for msg in messages:
            try:
                headers = {h['name'].lower(): h['value'] for h in msg['payload']['headers']}
                
                # Extraire le corps du message
                body = ""
                payload = msg['payload']
                
                if 'parts' in payload:
                    for part in payload['parts']:
                        if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                            body = base64.urlsafe_b64decode(
                                part['body']['data']
                            ).decode('utf-8', errors='ignore')
                            break
                elif 'body' in payload and 'data' in payload['body']:
                    body = base64.urlsafe_b64decode(
                        payload['body']['data']
                    ).decode('utf-8', errors='ignore')
                
                # Extraire le nom de l'expéditeur
                sender_full = headers.get('from', '')
                sender_name = sender_full.split('<')[0].strip().strip('"') if '<' in sender_full else sender_full
                
                # Détecter si c'est moi qui ai envoyé
                is_from_me = 'me' in msg.get('labelIds', [])
                
                history.append({
                    'sender': sender_full,
                    'sender_name': sender_name,
                    'content': body[:1000],  # Limiter à 1000 caractères par message
                    'date': headers.get('date', ''),
                    'is_from_me': is_from_me
                })
                
            except Exception as e:
                logger.warning(f" Erreur traitement message {msg.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f" {len(history)} messages récupérés dans l'historique")
        return history
        
    except Exception as e:
        logger.error(f" Erreur récupération historique thread : {e}")
        return []


# LLM pour génération de réponses
llm = OllamaLLM(
    model="llama3",
    temperature=0.5,
    top_p=0.80
)

# PROMPT GÉNÉRATION RÉPONSE EMAIL AVEC MÉMOIRE
reply_prompt = PromptTemplate(
    input_variables=["sender", "subject", "content", "intent", "history", "past_exchanges"],
    template="""Tu es un assistant de rédaction d'emails professionnel expert avec MÉMOIRE CONVERSATIONNELLE.

# MISSION
Rédiger une réponse concise et professionnelle en tenant compte de l'HISTORIQUE du thread Gmail
et des ÉCHANGES PASSÉS avec cet expéditeur stockés en mémoire.

# ÉCHANGES PASSÉS AVEC CET EXPÉDITEUR (mémoire long-terme)
{past_exchanges}

# HISTORIQUE DU THREAD ACTUEL (mémoire court-terme)
{history}

# RÈGLES STRICTES
1. CONTEXTE & MÉMOIRE
   - Consulte d'abord les échanges passés pour vérifier si ce sujet a déjà été traité
   - Si un sujet a déjà été abordé dans un échange passé, assure la cohérence avec ce qui a été dit
   - Ne redemande pas d'informations déjà fournies dans les échanges passés
   - Prends en compte les messages du thread actuel
   - N'invente AUCUNE information non mentionnée dans les échanges passés ou l'email actuel

   IMPORTANT :

N'utilise les échanges passés QUE s'ils sont directement liés au sujet de l'email actuel.

Si les échanges passés ne sont PAS pertinents :
- Ignore-les complètement
- Ne fais AUCUNE référence à eux

Exemples :
- Email actuel sur un événement → ignorer un ancien échange sur du phishing
- Sujet différent → ignorer la mémoire

Ne JAMAIS forcer une référence à un échange passé.

2. CONTINUITÉ
   - Si un échange passé est pertinent : "Comme mentionné lors de notre échange du [date]..."
   - Si c'est une suite du thread actuel : "Suite à notre échange..."
   - Si premier contact : traite comme un nouveau sujet

3. CONTENU
   - Réponds UNIQUEMENT à ce qui est demandé explicitement
   - Utilise les échanges passés pour contextualiser et assurer la cohérence
   - N'ajoute AUCUN détail non sollicité

4. TON & STYLE
   - Professionnel mais naturel
   - Concis : maximum 3-4 phrases courtes
   - Adapte selon l'historique (plus direct si plusieurs échanges)
   - Reste courtois sans être obséquieux

5. STRUCTURE OBLIGATOIRE
   Bonjour {sender},
   
   [Référence au contexte si nécessaire]
   [1-2 phrases de réponse directe]
   
   Cordialement,
   Bilimbilim Mombo

# INTERDICTIONS
-  Phrases creuses type "J'espère que vous allez bien"
-  Formules trop longues ou redondantes
-  Informations non demandées
-  Promesses impossibles à tenir
-  Inventer des informations non présentes dans la mémoire ou l'email

# FORMAT DE SORTIE
Retourne UNIQUEMENT le texte de l'email, sans balises, sans commentaires.

---

Intention détectée : {intent}

Email actuel :
Objet : {subject}
De : {sender}

{content}

Rédige la réponse maintenant :"""
)


def generate_email_reply(email: EmailModel, decision: DecisionModel) -> str:
    """
    Génère une réponse email via LLM local avec mémoire conversationnelle
    
    Args:
        email: EmailModel avec sender_name, subject, content, conversation_history
        decision: DecisionModel avec intent
    
    Returns:
        str: Réponse générée
    """
    logger.info(f" AGENT EMAIL - Génération réponse pour intent={decision.intent}")
    
    try:
        # 1. Recherche des échanges passés avec cet expéditeur (mémoire long-terme)
        past_exchanges = get_past_exchanges(email.sender_email, email.content)
        if past_exchanges:
            logger.info(" Échanges passés trouvés et injectés dans le prompt")
        else:
            past_exchanges = "Aucun échange passé trouvé avec cet expéditeur."
            logger.info(" Pas d'échanges passés pour cet expéditeur")

        # 2. Construire l'historique du thread actuel (mémoire court-terme)
        history_text = ""
        if email.conversation_history and len(email.conversation_history) > 0:            # Exclure le dernier message (c'est l'email actuel)
            past_messages = email.conversation_history[:-1] if len(email.conversation_history) > 1 else []
            
            if past_messages:
                history_text = "=== MESSAGES PRÉCÉDENTS DE CETTE CONVERSATION ===\n"
                history_text += f"(Historique de {len(past_messages)} message(s))\n\n"
                
                # Afficher les 5 derniers messages en détail, résumer les autres
                if len(past_messages) > 5:
                    old_count = len(past_messages) - 5
                    history_text += f"[... {old_count} messages plus anciens ...]\n\n"
                    messages_to_show = past_messages[-5:]
                else:
                    messages_to_show = past_messages
                
                for i, msg in enumerate(messages_to_show, 1):
                    sender_label = "MOI" if msg.get('is_from_me', False) else msg.get('sender_name', 'Expéditeur')
                    date = msg.get('date', 'Date inconnue')
                    content = msg.get('content', '').strip()
                    
                    history_text += f"[Message {i}] {sender_label} ({date}):\n"
                    history_text += f"{content}\n"
                    history_text += "-" * 60 + "\n\n"
                
                history_text += "=== FIN DE L'HISTORIQUE ===\n"
                logger.info(f" Historique formaté : {len(past_messages)} messages précédents")
            else:
                history_text = "Aucun message précédent - Premier échange de cette conversation."
                logger.info(" Pas d'historique (premier message)")
        else:
            history_text = "Aucun message précédent - Premier échange de cette conversation."
            logger.info(" Pas d'historique fourni")
        
        formatted_prompt = reply_prompt.format(
            sender=email.sender_name,
            subject=email.subject,
            content=email.content,
            intent=decision.intent,
            history=history_text,
            past_exchanges=past_exchanges
        )
        
        response = llm.invoke(formatted_prompt)
        cleaned_response = response.strip()
        
        logger.info(f" Réponse générée ({len(cleaned_response)} caractères)")
        return cleaned_response
        
    except Exception as e:
        logger.error(f" Erreur génération LLM : {e}")
        
        # Fallback simple
        return f"""Bonjour {email.sender_name},

J'ai bien reçu votre message. Je reviens vers vous rapidement.

Cordialement,
Bilimbilim Mombo"""


def execute_email_agent(state: AgentState) -> str:
    """
    Agent Email - Génère et retourne la réponse email
    
    Args:
        state: AgentState
        
    Returns:
        str: Texte de la réponse générée
    """
    logger.info(" AGENT EMAIL - Activation")
    
    reply_text = generate_email_reply(state.email, state.decision)
    
    logger.info(" AGENT EMAIL - Réponse prête")
    return reply_text