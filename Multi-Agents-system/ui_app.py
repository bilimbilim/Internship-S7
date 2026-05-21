import config
import streamlit as st
from datetime import datetime, timedelta
import json
from pathlib import Path
import pandas as pd

from main_agent import agent
from state import AgentState
from gmail_auth import get_gmail_service , get_calendar_service
from agent_calendar import get_agenda


# CONFIGURATION 
st.set_page_config(
    page_title=" Agent Email IA", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# STYLES CSS PERSONNALISÉS

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .email-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
    }
    .mode-indicator {
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .mode-auto {
        background-color: #28a745;
        color: white;
    }
    .mode-manual {
        background-color: #ffc107;
        color: #856404;
    }
    .mode-view {
        background-color: #17a2b8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 
# TITRE PRINCIPAL
st.markdown('<div class="main-header">📧 Agent Email IA - Tableau de Bord</div>', unsafe_allow_html=True)

# SIDEBAR - CONTRÔLES
st.sidebar.title(" Contrôles")

# Mode de fonctionnement
mode = st.sidebar.radio(
    "Mode d'exécution",
    ["Automatique (traite tous les emails)", " Manuel", " Visualisation seule"]
)

# Extraire le mode simple pour comparaison
if "Automatique" in mode:
    current_mode = "automatique"
    mode_class = "mode-auto"
    mode_icon = ""
    mode_desc = "Mode Automatique : L'agent traite tous les emails automatiquement"
elif "Manuel" in mode:
    current_mode = "manuel"
    mode_class = "mode-manual"
    mode_icon = ""
    mode_desc = "Mode Manuel : Vous sélectionnez les emails à traiter"
else:
    current_mode = "visualisation"
    mode_class = "mode-view"
    mode_icon = ""
    mode_desc = "Mode Visualisation : Lecture seule, aucune action possible"

# Afficher l'indicateur de mode
st.sidebar.markdown("---")
st.sidebar.markdown(f'<div class="mode-indicator {mode_class}">{mode_icon} {current_mode.upper()}</div>', unsafe_allow_html=True)
st.sidebar.info(mode_desc)

# Filtres
st.sidebar.markdown("---")
st.sidebar.subheader(" Filtres")

filter_priority = st.sidebar.multiselect(
    "Priorité",
    ["urgent", "normal", "low"],
    default=["urgent", "normal", "low"]
)

filter_intent = st.sidebar.multiselect(
    "Type d'email",
    ["meeting", "document", "discussion", "general"],
    default=["meeting", "document", "discussion", "general"]
)

# Rafraîchissement
st.sidebar.markdown("---")
if st.sidebar.button(" Rafraîchir les données"):
    st.rerun()

# FONCTIONS UTILITAIRES

@st.cache_data(ttl=60)
def get_unread_emails():
    """Récupère les emails non lus depuis Gmail"""
    try:
        service = get_gmail_service()
        results = service.users().messages().list(
            userId='me',
            q='is:unread',
            maxResults=50
        ).execute()
        
        messages = results.get('messages', [])
        emails_data = []
        
        for msg in messages:
            msg_data = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='full'
            ).execute()
            
            headers = msg_data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'Sans objet')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Inconnu')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            emails_data.append({
                'id': msg['id'],
                'thread_id': msg.get('threadId'),
                'subject': subject,
                'sender': sender,
                'date': date,
                'snippet': msg_data.get('snippet', '')
            })
        
        return emails_data
    except Exception as e:
        st.error(f"Erreur lors de la récupération des emails : {e}")
        return []

@st.cache_data(ttl=60)
def get_sent_emails():
    """Récupère l'historique des 50 derniers emails envoyés (depuis outbox)"""
    try:
        outbox = Path("emails/outbox")
        if not outbox.exists():
            return []
        
        sent_emails = []
        # Récupérer les 50 derniers fichiers au lieu de 10
        for file in sorted(outbox.glob("sent_*.txt"), reverse=True)[:50]:
            content = file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            email_info = {}
            for line in lines[:7]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    email_info[key.strip()] = value.strip()
            
            sent_emails.append({
                'timestamp': email_info.get('TIMESTAMP', ''),
                'to': email_info.get('TO', ''),
                'subject': email_info.get('SUBJECT', ''),
                'intent': email_info.get('INTENT', ''),
                'priority': email_info.get('PRIORITY', ''),
                'file': file.name
            })
        
        return sent_emails
    except Exception as e:
        st.error(f"Erreur lecture outbox : {e}")
        return []

@st.cache_data(ttl=60)
def get_calendar_events():
    """Récupère les événements du calendrier"""
    try:
        events = get_agenda()
        return events
    except Exception as e:
        st.error(f"Erreur calendrier : {e}")
        return []

def process_single_email(email_id):
    """Traite un email spécifique"""
    try:
        # Ici vous pouvez appeler votre agent pour traiter l'email spécifique
        result = agent.invoke(AgentState())
        return True, result
    except Exception as e:
        return False, str(e)

def process_multiple_emails(email_ids):
    """Traite plusieurs emails"""
    results = []
    for email_id in email_ids:
        success, result = process_single_email(email_id)
        results.append({
            'id': email_id,
            'success': success,
            'result': result
        })
    return results

# MÉTRIQUES PRINCIPALES
col1, col2, col3, col4 = st.columns(4)

unread_emails = get_unread_emails()
sent_emails = get_sent_emails()
calendar_events = get_calendar_events()

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(" Emails non lus", len(unread_emails))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(" Emails envoyés (50 derniers)", len(sent_emails))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    urgent_count = sum(1 for e in sent_emails if e.get('priority') == 'urgent')
    st.metric(" Emails urgents traités", urgent_count)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(" Événements à venir", len(calendar_events))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ============================
# ONGLETS PRINCIPAUX
# ============================
tab1, tab2, tab3, tab4 = st.tabs([
    " Emails Non Lus", 
    " Emails Envoyés", 
    " Calendrier",
    " Exécuter l'Agent"
])

# ============================
# TAB 1: EMAILS NON LUS
# ============================
with tab1:
    st.subheader(" Emails Non Lus")
    
    if not unread_emails:
        st.info(" Aucun email non lu ! Vous êtes à jour.")
    else:
        # MODE MANUEL : Sélection multiple
        if current_mode == "manuel":
            st.markdown('<div class="warning-box"> Mode Manuel : Sélectionnez les emails à traiter</div>', unsafe_allow_html=True)
            
            # Initialiser session state pour les sélections
            if 'selected_emails' not in st.session_state:
                st.session_state.selected_emails = []
            
            # Boutons d'action en haut
            col_a, col_b, col_c = st.columns([2, 2, 2])
            with col_a:
                if st.button(" Tout sélectionner"):
                    st.session_state.selected_emails = [e['id'] for e in unread_emails]
                    st.rerun()
            with col_b:
                if st.button(" Tout désélectionner"):
                    st.session_state.selected_emails = []
                    st.rerun()
            with col_c:
                selected_count = len(st.session_state.selected_emails)
                st.info(f" {selected_count} email(s) sélectionné(s)")
            
            st.markdown("---")
            
            # Afficher les emails avec checkboxes
            for email in unread_emails:
                col1, col2 = st.columns([0.5, 9.5])
                
                with col1:
                    is_selected = st.checkbox(
                        "✓",
                        value=email['id'] in st.session_state.selected_emails,
                        key=f"check_{email['id']}",
                        label_visibility="collapsed"
                    )
                    if is_selected and email['id'] not in st.session_state.selected_emails:
                        st.session_state.selected_emails.append(email['id'])
                    elif not is_selected and email['id'] in st.session_state.selected_emails:
                        st.session_state.selected_emails.remove(email['id'])
                
                with col2:
                    with st.expander(f" {email['subject'][:60]}... - {email['sender'][:30]}"):
                        st.write(f"**De :** {email['sender']}")
                        st.write(f"**Date :** {email['date']}")
                        st.write(f"**Aperçu :** {email['snippet']}")
            
            # Bouton de traitement groupé
            st.markdown("---")
            if st.session_state.selected_emails:
                if st.button(f" Traiter les {len(st.session_state.selected_emails)} email(s) sélectionné(s)", type="primary"):
                    with st.spinner(" Traitement en cours..."):
                        results = process_multiple_emails(st.session_state.selected_emails)
                        
                        success_count = sum(1 for r in results if r['success'])
                        fail_count = len(results) - success_count
                        
                        if fail_count == 0:
                            st.success(f" {success_count} email(s) traité(s) avec succès !")
                        else:
                            st.warning(f" {success_count} réussi(s), {fail_count} échoué(s)")
                        
                        # Réinitialiser la sélection
                        st.session_state.selected_emails = []
                        st.rerun()
        
        # MODE VISUALISATION : Lecture seule
        elif current_mode == "visualisation":
            st.markdown('<div class="info-box"> Mode Visualisation : Lecture seule (aucune action possible)</div>', unsafe_allow_html=True)
            
            for email in unread_emails:
                with st.expander(f" {email['subject'][:60]}... - {email['sender'][:30]}"):
                    st.write(f"**De :** {email['sender']}")
                    st.write(f"**Date :** {email['date']}")
                    st.write(f"**Aperçu :** {email['snippet']}")
                    st.info(" Mode visualisation : Utilisez le mode Manuel ou Automatique pour traiter cet email")
        
        # MODE AUTOMATIQUE 
        else:
            st.markdown('<div class="success-box"> Mode Automatique : Traitez les emails individuellement ou tous ensemble</div>', unsafe_allow_html=True)
            
            for email in unread_emails:
                with st.expander(f" {email['subject'][:60]}... - {email['sender'][:30]}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**De :** {email['sender']}")
                        st.write(f"**Date :** {email['date']}")
                        st.write(f"**Aperçu :** {email['snippet']}")
                    
                    with col2:
                        if st.button(f" Traiter", key=f"process_{email['id']}"):
                            with st.spinner(" Traitement en cours..."):
                                success, result = process_single_email(email['id'])
                                if success:
                                    st.success(" Email traité avec succès !")
                                else:
                                    st.error(f" Erreur : {result}")

# TAB 2: EMAILS ENVOYÉS
with tab2:
    st.subheader(" Historique des 50 Derniers Emails Envoyés")
    
    if not sent_emails:
        st.info("Aucun email envoyé récemment.")
    else:
        # Filtrer selon les sélections
        filtered_sent = [
            e for e in sent_emails 
            if e.get('priority', 'normal') in filter_priority 
            and e.get('intent', 'general') in filter_intent
        ]
        
        if not filtered_sent:
            st.warning("Aucun email ne correspond aux filtres sélectionnés.")
        else:
            # Afficher le nombre d'emails filtrés
            st.info(f" {len(filtered_sent)} email(s) affiché(s) sur {len(sent_emails)} au total")
            
            for email in filtered_sent:
                priority_emoji = "🔴" if email.get('priority') == 'urgent' else "🟢"
                intent_emoji = {
                    'meeting': '',
                    'document': '',
                    'discussion': '',
                    'general': ''
                }.get(email.get('intent', 'general'), '')
                
                with st.expander(f"{priority_emoji} {intent_emoji} {email.get('subject', 'Sans objet')[:50]}... - {email.get('timestamp', '')}"):
                    st.write(f"**À :** {email.get('to', 'N/A')}")
                    st.write(f"**Priorité :** {email.get('priority', 'N/A')}")
                    st.write(f"**Type :** {email.get('intent', 'N/A')}")
                    st.write(f"**Date :** {email.get('timestamp', 'N/A')}")
                    
                    # Lire le contenu complet
                    file_path = Path("emails/outbox") / email['file']
                    if file_path.exists():
                        content = file_path.read_text(encoding='utf-8')
                        with st.expander(" Voir le contenu complet"):
                            st.text(content)

# TAB 3: CALENDRIER
with tab3:
    st.subheader(" Événements du Calendrier")
    
    if not calendar_events:
        st.info("Aucun événement à venir dans votre calendrier.")
    else:
        for event in calendar_events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            summary = event.get('summary', 'Sans titre')
            
            # Convertir en datetime pour affichage
            try:
                if 'T' in start:
                    event_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    formatted_date = event_dt.strftime("%d/%m/%Y %H:%M")
                else:
                    formatted_date = start
            except:
                formatted_date = start
            
            with st.expander(f" {summary} - {formatted_date}"):
                st.write(f"**Début :** {formatted_date}")
                
                if 'description' in event:
                    st.write(f"**Description :** {event['description']}")
                
                if 'attendees' in event:
                    attendees = [a.get('email', 'N/A') for a in event['attendees']]
                    st.write(f"**Participants :** {', '.join(attendees)}")
                
                if 'htmlLink' in event:
                    st.markdown(f"[ Ouvrir dans Google Calendar]({event['htmlLink']})")

# TAB 4: EXÉCUTER L'AGENT
with tab4:
    st.subheader(" Exécuter l'Agent Email")
    
    # Affichage conditionnel selon le mode
    if current_mode == "visualisation":
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning(" Mode Visualisation actif : L'exécution de l'agent est désactivée")
        st.markdown('</div>', unsafe_allow_html=True)
        st.info(" Basculez en mode 'Automatique' ou 'Manuel' dans la barre latérale pour activer l'agent")
    
    elif current_mode == "manuel":
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("""
        **Mode Manuel actif**
        
        En mode manuel, vous devez :
        1. Aller dans l'onglet " Emails Non Lus"
        2. Sélectionner les emails à traiter
        3. Cliquer sur "Traiter les emails sélectionnés"
        
         Pour traiter tous les emails automatiquement, basculez en mode Automatique
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # Mode Automatique
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success("""
        **Mode Automatique actif**
        
        L'agent va :
        -  Lire tous les emails non lus
        -  Analyser leur contenu avec l'IA
        -  Générer et envoyer des réponses appropriées
        -  Gérer les rendez-vous si nécessaire
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Confirmation avant exécution
            confirm = st.checkbox(" Je confirme vouloir traiter TOUS les emails automatiquement", value=False)
            
            if st.button(" LANCER L'AGENT", type="primary", disabled=not confirm):
                if not confirm:
                    st.warning(" Veuillez cocher la case de confirmation")
                else:
                    with st.spinner(" Agent en cours d'exécution..."):
                        try:
                            result = agent.invoke(AgentState())
                            
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success(" Agent exécuté avec succès !")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Afficher les résultats
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                if 'email' in result:
                                    st.write("###  Email Traité")
                                    st.write(f"**De :** {result['email'].get('sender_name', 'N/A')}")
                                    st.write(f"**Sujet :** {result['email'].get('subject', 'N/A')}")
                            
                            with col_b:
                                if 'decision' in result:
                                    st.write("###  Décision")
                                    st.write(f"**Intent :** {result['decision'].get('intent', 'N/A')}")
                                    st.write(f"**Priorité :** {result['decision'].get('priority', 'N/A')}")
                            
                            # Détails complets
                            with st.expander(" Voir tous les détails"):
                                st.json(result)
                            
                            # Bouton pour rafraîchir
                            if st.button(" Rafraîchir l'affichage"):
                                st.rerun()
                            
                        except Exception as e:
                            st.error(f" Erreur lors de l'exécution : {e}")
                            st.exception(e)
        
        with col2:
            st.markdown("###  Options")
            auto_refresh = st.checkbox(" Rafraîchir auto après exécution", value=True)
            
            if auto_refresh:
                st.info(" Rafraîchissement automatique activé")
            
            st.markdown("---")
            st.metric(" Emails à traiter", len(unread_emails))

# FOOTER
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p> Agent Email IA v2.0 | Mode actuel : <strong>{current_mode.upper()}</strong> | Développé avec ❤️ | Powered by Streamlit & LangChain</p>
</div>
""", unsafe_allow_html=True)
