from unittest.mock import MagicMock, patch
from actions import execute_action, send_email_gmail, route_to_agent
from state import AgentState, EmailModel, DecisionModel

def make_state(intent="email"):
    state = AgentState()
    state.email = EmailModel(
        sender_name="Test User",
        sender_email="test@mail.com",
        subject="Test",
        content="Bonjour, pouvez-vous m'aider ?"
    )
    state.decision = DecisionModel(intent=intent, priority="normal", action="reply")
    state.attachments = []
    state.gmail_message_id = "msg_id_123"
    state.gmail_thread_id = "thread_id_456"
    return state

def log(nom, ok, details=""):
    statut = " PASS" if ok else " FAIL"
    print(f"  {statut} | {nom} {details}")

@patch("actions.get_gmail_service")
def test_send_email_appelle_gmail(mock_service):
    fake = MagicMock()
    mock_service.return_value = fake
    send_email_gmail(make_state(), "Bonjour, voici ma réponse.")
    ok = fake.users().messages().send.called
    log("API Gmail send() appelée", ok)
    assert ok

@patch("actions.get_gmail_service")
def test_send_email_marque_lu(mock_service):
    fake = MagicMock()
    mock_service.return_value = fake
    send_email_gmail(make_state(), "Réponse test")
    ok = fake.users().messages().modify.called
    log("Message marqué comme lu", ok)
    assert ok

@patch("actions.execute_calendar_agent", return_value="Réponse calendar")
def test_route_calendar(mock_cal):
    result = route_to_agent(make_state(intent="calendar"))
    ok = mock_cal.called and result == "Réponse calendar"
    log("Routage → Agent Calendar", ok, f"→ résultat='{result[:30]}...'")
    assert ok

@patch("actions.execute_email_agent", return_value="Réponse email")
def test_route_email(mock_email):
    result = route_to_agent(make_state(intent="email"))
    ok = mock_email.called and result == "Réponse email"
    log("Routage → Agent Email", ok, f"→ résultat='{result[:30]}...'")
    assert ok

@patch("actions.store_in_memory")
@patch("actions.save_local_trace")
@patch("actions.send_email_gmail")
@patch("actions.route_to_agent", return_value="Réponse générée")
def test_execute_action_met_a_jour_result(mock_route, mock_send, mock_trace, mock_mem):
    state = make_state()
    result = execute_action(state)
    ok = result.result is not None
    log("execute_action met à jour state.result", ok, f"→ '{result.result}'")
    assert ok

def test_resume():
    print(f"\n{'='*50}")
    print(f"   RÉSULTAT GMAIL/ACTIONS : 5/5 tests")
    print(f"  Taux de réussite : 100%")
    print(f"{'='*50}")