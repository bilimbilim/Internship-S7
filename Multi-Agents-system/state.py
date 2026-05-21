from pydantic import BaseModel
from typing import Optional, Literal


class EmailModel(BaseModel):
    """Modèle représentant un email"""
    sender_name: str
    sender_email: str
    subject: str
    content: str
    thread_id: Optional[str] = None         
    conversation_history: list[dict] = []

class DecisionModel(BaseModel):
    """Modèle représentant la décision de l'analyse"""
    intent: Literal[
        "calendar",    # Demande de rendez-vous → Agent Calendar
        "email",       # Question/demande classique → Agent Email
        "document",    # Document à analyser → Agent Document
        "general",     # Information générale → Agent Email
        "unknown"      # Inconnu → Agent Email
    ]
    priority: Literal["low", "normal", "high", "urgent"]
    action: Literal["reply", "archive", "review"]


class AgentState(BaseModel):
    """État global de l'agent"""
    # PERCEPTION
    email: Optional[EmailModel] = None
    attachments: list[str] = []
    gmail_message_id: Optional[str] = None
    gmail_thread_id: Optional[str] = None
    
    # ANALYSE
    decision: Optional[DecisionModel] = None
    
    # ACTION
    result: Optional[str] = None
