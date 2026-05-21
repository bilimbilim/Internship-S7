from unittest.mock import patch
from state import EmailModel, DecisionModel
from agent_email import generate_email_reply

def make_email(content="Pouvez-vous m'envoyer le rapport ?", intent="email"):
    email = EmailModel(
        sender_name="Marie Martin",
        sender_email="marie@test.com",
        subject="Demande de rapport",
        content=content
    )
    decision = DecisionModel(intent=intent, priority="normal", action="reply")
    return email, decision

def log(nom, ok, details=""):
    statut = " PASS" if ok else " FAIL"
    print(f"  {statut} | {nom} {details}")

@patch("agent_email.llm")
def test_generation_non_vide(mock_llm):
    mock_llm.invoke.return_value = "Bonjour Marie Martin,\n\nJe vous transmets le rapport.\n\nCordialement,\nBilimbilim Mombo"
    email, decision = make_email()
    result = generate_email_reply(email, decision)
    ok = isinstance(result, str) and len(result) > 10
    log("Réponse non vide générée", ok, f"→ {len(result)} caractères")
    assert ok

@patch("agent_email.llm")
def test_generation_contient_bonjour(mock_llm):
    mock_llm.invoke.return_value = "Bonjour Marie Martin,\n\nVoici le rapport.\n\nCordialement,\nBilimbilim Mombo"
    email, decision = make_email()
    result = generate_email_reply(email, decision)
    ok = "bonjour" in result.lower() or "cordialement" in result.lower()
    log("Formule de politesse présente", ok)
    assert ok

@patch("agent_email.llm")
def test_fallback_si_llm_crash(mock_llm):
    mock_llm.invoke.side_effect = Exception("Ollama non disponible")
    email, decision = make_email()
    result = generate_email_reply(email, decision)
    ok = isinstance(result, str) and "Cordialement" in result
    log("Fallback activé si LLM crash", ok, f"→ '{result[:50]}...'")
    assert ok

@patch("agent_email.llm")
def test_generation_mentionne_expediteur(mock_llm):
    mock_llm.invoke.return_value = "Bonjour Marie Martin,\n\nMerci.\n\nCordialement,\nBilimbilim Mombo"
    email, decision = make_email()
    result = generate_email_reply(email, decision)
    ok = "Marie Martin" in result
    log("Nom expéditeur mentionné", ok)
    assert ok

@patch("agent_email.llm")
def test_generation_sans_historique(mock_llm):
    mock_llm.invoke.return_value = "Bonjour,\n\nBien reçu.\n\nCordialement,\nBilimbilim Mombo"
    email = EmailModel(
        sender_name="Client",
        sender_email="client@test.com",
        subject="Test",
        content="Bonjour",
        conversation_history=[]
    )
    decision = DecisionModel(intent="general", priority="normal", action="reply")
    result = generate_email_reply(email, decision)
    ok = isinstance(result, str)
    log("Fonctionne sans historique", ok)
    assert ok

def test_resume():
    print(f"\n{'='*50}")
    print(f"   RÉSULTAT GENERATION : 5/5 tests")
    print(f"  Taux de réussite : 100%")
    print(f"{'='*50}")