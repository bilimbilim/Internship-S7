import json
import logging
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from state import AgentState, DecisionModel

logger = logging.getLogger("agent")

# LLM LOCAL pour l'analyse
llm = OllamaLLM(
    model="llama3",
    temperature=0
)

# PROMPT D'ANALYSE
analysis_prompt = PromptTemplate(
    input_variables=["email"],
    template="""Tu es un assistant IA chargé d'analyser un email professionnel.

Ton objectif est d'identifier l'intention principale de l'email et de retourner un JSON valide.

FORMAT STRICT :
{{
  "intent": "calendar | email | document | general | unknown",
  "priority": "low | normal | high | urgent",
  "action": "reply | archive | review"
}}

Tu dois retourner UNIQUEMENT le JSON, sans aucun texte supplémentaire.

----------------------
RÈGLES D'INTENTION :
----------------------

1. calendar :
UNIQUEMENT si :
- l'utilisateur propose une date ou une heure
- l'utilisateur demande une disponibilité
- l'utilisateur souhaite planifier, replanifier ou confirmer un rendez-vous

Exemples :
- "Es-tu disponible demain ?"
- "Je propose mardi à 14h"
- "Peut-on décaler la réunion ?"

IMPORTANT :
La présence d'une date dans l'email NE signifie PAS que c'est une demande de rendez-vous.
Ne PAS classer comme "calendar" si :
- la réunion est déjà planifiée
- l'utilisateur demande des informations
- il n'y a aucune demande de date ou de disponibilité

Exemples (DOIVENT être classés en email) :
- "Comment va se passer la réunion ?"
- "Quel est l'ordre du jour ?"
- "Peux-tu me donner plus de détails sur la réunion ?"

----------------------

2. email :
- demande d'information
- question directe
- discussion ou échange
- toute demande sans notion de planification

Exemples :
- "Pouvez-vous m'envoyer le rapport ?"
- "Quelle est la procédure ?"
- "Comment va se passer la réunion ?"

----------------------

3. document :
- email contenant ou demandant des documents
- pièces jointes à analyser
- demande d'envoi de fichiers

----------------------

4. general :
- message informatif
- notification
- message sans action claire

----------------------

5. unknown :
- message ambigu
- contenu insuffisant

----------------------
RÈGLES DE PRIORITÉ :
----------------------

- urgent : urgence explicite, délai immédiat
- high : important avec contrainte de temps
- normal : standard
- low : informatif sans urgence

----------------------
RÈGLES D'ACTION :
----------------------

- calendar → reply
- email → reply
- document → review
- general → reply
- unknown → review

----------------------
EMAIL À ANALYSER :
----------------------

{email}
Retourne UNIQUEMENT le JSON, sans texte supplémentaire.
"""
)


def analyze_email(state: AgentState) -> AgentState:
    """
    Analyse l'email et détermine l'intention pour router vers le bon agent.
    """
    logger.info("ANALYSE - Détection de l'intention")

    email = state.email
    if email is None:
        logger.warning("Aucun email à analyser")
        return state

    # Construction du texte à analyser
    email_text = (
        f"De: {email.sender_name} <{email.sender_email}>\n"
        f"Objet: {email.subject}\n\n"
        f"{email.content}"
    )

    # FIX : initialiser response avant le try pour éviter UnboundLocalError
    response = ""

    try:
        # Appel au LLM
        response = llm.invoke(analysis_prompt.format(email=email_text))

        # Nettoyage de la réponse
        response = response.strip().replace("```json", "").replace("```", "")
        logger.info(f"[ANALYSE] RAW RESPONSE:\n{response}")

        data = extract_json(response)

        # Validation des valeurs
        valid_intents = {"calendar", "email", "document", "general", "unknown"}
        valid_priorities = {"low", "normal", "high", "urgent"}
        valid_actions = {"reply", "archive", "review"}

        intent = data.get("intent", "general")
        priority = data.get("priority", "normal")
        action = data.get("action", "reply")

        # Correction si valeurs invalides
        if intent not in valid_intents:
            logger.warning(f"Intent invalide '{intent}', utilisation de 'general'")
            intent = "general"
        if priority not in valid_priorities:
            logger.warning(f"Priority invalide '{priority}', utilisation de 'normal'")
            priority = "normal"
        if action not in valid_actions:
            logger.warning(f"Action invalide '{action}', utilisation de 'reply'")
            action = "reply"

        # Mise à jour du state
        state.decision = DecisionModel(
            intent=intent,
            priority=priority,
            action=action
        )

        logger.info(f"Intention détectée: {intent} | Priorité: {priority} | Action: {action}")

    except Exception as e:
        logger.error(f"[ANALYSE ERROR] {e}")
        logger.error(f"[ANALYSE RESPONSE] {response}")
        state.decision = DecisionModel(
            intent="general",
            priority="normal",
            action="reply"
        )

    return state


def extract_json(text: str) -> dict:
    """
    Extrait le premier objet JSON trouvé dans le texte.
    Retourne un dict de fallback si aucun JSON valide n'est trouvé.
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.error(f"[JSON ERROR] {e}")
        logger.error(f"[BAD RESPONSE] {text}")

    # Fallback safe
    return {
        "intent": "general",
        "priority": "normal",
        "action": "reply"
    }