#!/usr/bin/env python3
"""
SYS MULTI-AGENT - Architecture modulaire

Architecture :
    PERCEPTION → ANALYSE → AGENTS → ACTION
    
    1. PERCEPTION : Lit l'email depuis Gmail
    2. ANALYSE   : LLM détermine l'intention (calendar/email/document)
    3. AGENTS    : Route vers Agent Calendar ou Agent Email
    4. ACTION    : Envoie la réponse et trace
"""

import logging
import warnings
from langgraph.graph import StateGraph, END
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)


from state import AgentState
from perception import read_email_gmail, fetch_all_unread_ids
from analyse import analyze_email
from actions import execute_action

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent")

# Suppression warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_agent_graph():
    """
    Construit le graph LangGraph de l'agent
    
    Architecture :
        START → PERCEPTION → ANALYSE → ACTION → END
                                
                        (route automatique)
                                
                      Calendar ou Email Agent
    
    Returns:
        CompiledGraph: Graph compilé
    """
    logger.info("Construction du graph agent")
    
    # Création du graph
    graph = StateGraph(AgentState)
    
    # Ajout des noeuds
    graph.add_node("PERCEPTION", read_email_gmail)
    graph.add_node("ANALYSE", analyze_email)
    graph.add_node("ACTION", execute_action)
    
    # Définition du flow
    graph.set_entry_point("PERCEPTION")
    graph.add_edge("PERCEPTION", "ANALYSE")
    graph.add_edge("ANALYSE", "ACTION")
    graph.add_edge("ACTION", END)
    
    # Compilation
    compiled_graph = graph.compile()
    
    logger.info("Graph agent construit")
    return compiled_graph

agent = build_agent_graph()



def main():
    """
    Point d'entrée principal de l'agent.
    Traite TOUS les emails non lus dans la boîte de réception.
    """
    logger.info("=" * 60)
    logger.info("SYS MULTI-AGENT")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Architecture : PERCEPTION → ANALYSE → AGENTS → ACTION")
    logger.info("")

    # Récupération de tous les IDs non lus
    unread_ids = fetch_all_unread_ids()

    unread_ids = unread_ids[:1]  

    if not unread_ids:
        logger.info("Aucun email non lu. Agent terminé.")
        return

    logger.info(f"Traitement de {len(unread_ids)} email(s) non lu(s)...")
    logger.info("")

    success = 0
    errors = 0

    for idx, msg_id in enumerate(unread_ids, start=1):
        logger.info(f"--- Email {idx}/{len(unread_ids)} (ID: {msg_id}) ---")

        # On pré-renseigne l'ID dans le state pour que PERCEPTION lise ce message précis
        initial_state = AgentState(gmail_message_id=msg_id)

        try:
            result = agent.invoke(initial_state)
            logger.info(f" Traité : {result.get('result', 'OK')}")
            success += 1
        except Exception as e:
            logger.error(f" Erreur sur l'email {msg_id} : {e}")
            errors += 1

        logger.info("")

    logger.info("=" * 60)
    logger.info(f"AGENT TERMINÉ — {success} traité(s), {errors} erreur(s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()