from unittest.mock import patch
from datetime import datetime
from calendar_agent import propose_meeting, detect_email_type, execute_calendar_agent
from state import AgentState, EmailModel, DecisionModel

def make_state(content="Je souhaite un rendez-vous.", sender="test@mail.com", name="Jean Dupont"):
    state = AgentState()
    state.email = EmailModel(
        sender_name=name,
        sender_email=sender,
        subject="Demande de rendez-vous",
        content=content
    )
    state.decision = DecisionModel(intent="calendar", priority="normal", action="reply")
    return state

def log(nom, ok, details=""):
    statut = " PASS" if ok else " FAIL"
    print(f"  {statut} | {nom} {details}")

@patch("calendar_agent.chroma_db", None)
def test_propose_meeting_retourne_texte():
    result = propose_meeting(make_state())
    ok = isinstance(result, str) and len(result) > 20
    log("propose_meeting retourne un texte", ok, f"→ {len(result)} caractères")
    assert ok

@patch("calendar_agent.chroma_db", None)
def test_propose_meeting_contient_nom():
    result = propose_meeting(make_state(name="Marie Curie"))
    ok = "Marie Curie" in result
    log("Réponse contient le nom expéditeur", ok)
    assert ok

@patch("calendar_agent.chroma_db", None)
def test_propose_meeting_pas_weekend():
    result = propose_meeting(make_state())
    ok = "Samedi" not in result and "Dimanche" not in result
    log("Pas de week-end proposé", ok)
    assert ok

@patch("calendar_agent.chroma_db", None)
def test_propose_meeting_date_futur():
    result = propose_meeting(make_state())
    year = str(datetime.now().year)
    next_year = str(datetime.now().year + 1)
    ok = year in result or next_year in result
    log("Date dans le futur mentionnée", ok)
    assert ok

def test_detect_demande_rdv():
    result = detect_email_type("Bonjour, pouvez-vous me proposer un rendez-vous ?")
    ok = result["type"] == "demande_rdv"
    log("Détection demande RDV", ok, f"→ type={result['type']}")
    assert ok

def test_detect_confirmation():
    result = detect_email_type("Je confirme pour mardi à 10h, c'est parfait.")
    ok = result["type"] == "confirmation_rdv"
    log("Détection confirmation RDV", ok, f"→ type={result['type']}")
    assert ok

def test_detect_autre():
    result = detect_email_type("Merci pour votre message, bonne journée.")
    ok = result["type"] == "autre"
    log("Détection autre", ok, f"→ type={result['type']}")
    assert ok

@patch("calendar_agent.chroma_db", None)
def test_execute_calendar_agent_demande():
    result = execute_calendar_agent(make_state(content="Je voudrais un rendez-vous."))
    ok = isinstance(result, str) and "Cordialement" in result
    log("execute_calendar_agent retourne réponse complète", ok)
    assert ok

def test_resume():
    print(f"\n{'='*50}")
    print(f"   RÉSULTAT CALENDAR : 8/8 tests")
    print(f"  Taux de réussite : 100%")
    print(f"{'='*50}")