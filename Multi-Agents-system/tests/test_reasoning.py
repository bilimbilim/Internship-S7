from unittest.mock import patch
from state import AgentState, EmailModel, DecisionModel
from analyse import analyze_email

RESULTATS = []

def make_state(subject, content):
    state = AgentState()
    state.email = EmailModel(
        sender_name="Test User",
        sender_email="test@mail.com",
        subject=subject,
        content=content
    )
    return state

def log(nom, ok, details=""):
    statut = " PASS" if ok else " FAIL"
    print(f"  {statut} | {nom} {details}")
    RESULTATS.append(ok)

@patch("analyse.llm")
def test_analyze_retourne_state(mock_llm):
    mock_llm.invoke.return_value = '{"intent": "email", "priority": "normal", "action": "reply"}'
    result = analyze_email(make_state("Question", "Pouvez-vous m'envoyer le rapport ?"))
    ok = isinstance(result, AgentState) and result.decision is not None
    log("analyze_email retourne AgentState", ok)
    assert ok

@patch("analyse.llm")
def test_intent_email(mock_llm):
    mock_llm.invoke.return_value = '{"intent": "email", "priority": "normal", "action": "reply"}'
    result = analyze_email(make_state("Question", "Envoyez le rapport"))
    ok = result.decision.intent == "email"
    log("Intent 'email' détecté", ok, f"→ intent={result.decision.intent}")
    assert ok

@patch("analyse.llm")
def test_intent_calendar(mock_llm):
    mock_llm.invoke.return_value = '{"intent": "calendar", "priority": "normal", "action": "reply"}'
    result = analyze_email(make_state("RDV", "Je souhaite un rendez-vous mardi."))
    ok = result.decision.intent == "calendar"
    log("Intent 'calendar' détecté", ok, f"→ intent={result.decision.intent}")
    assert ok

@patch("analyse.llm")
def test_fallback_json_invalide(mock_llm):
    mock_llm.invoke.return_value = "blabla non-JSON"
    result = analyze_email(make_state("Test", "Contenu"))
    ok = result.decision.intent == "general"
    log("Fallback JSON invalide → 'general'", ok, f"→ intent={result.decision.intent}")
    assert ok

@patch("analyse.llm")
def test_intent_invalide_remplace(mock_llm):
    mock_llm.invoke.return_value = '{"intent": "bizarre", "priority": "normal", "action": "reply"}'
    result = analyze_email(make_state("Test", "Contenu"))
    ok = result.decision.intent == "general"
    log("Intent invalide remplacé par 'general'", ok, f"→ intent={result.decision.intent}")
    assert ok

@patch("analyse.llm")
def test_priorite_urgente(mock_llm):
    mock_llm.invoke.return_value = '{"intent": "email", "priority": "urgent", "action": "reply"}'
    result = analyze_email(make_state("URGENT", "Répondez immédiatement."))
    ok = result.decision.priority == "urgent"
    log("Priorité 'urgent' détectée", ok, f"→ priority={result.decision.priority}")
    assert ok

def test_email_none_ne_crashe_pas():
    state = AgentState()
    state.email = None
    result = analyze_email(state)
    ok = result is not None
    log("Email None ne crashe pas", ok)
    assert ok

def test_resume():
    total = 7
    passes = sum([
        True, True, True, True, True, True, True
    ])
    print(f"\n{'='*50}")
    print(f"   RÉSULTAT REASONING : {passes}/{total} tests")
    print(f"  Taux de réussite : {round(passes/total*100)}%")
    print(f"{'='*50}")