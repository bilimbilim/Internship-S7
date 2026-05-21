from fastapi import FastAPI, Request, UploadFile, File
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import subprocess
import base64
import threading
import time
from threading import Lock
from email.message import EmailMessage
from datetime import datetime
from datetime import datetime, timedelta
from gmail_auth import get_gmail_service, get_calendar_service
from agent_rag import rag_agent
from rag_space.index_documents import (
    index_all_documents, index_single_file,
    load_tracking, save_tracking, get_file_hash,
    DOCUMENTS_DIR, PERSIST_DIR, LocalEmbeddings
)
from langchain_chroma import Chroma
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# INITIALISATION

app = FastAPI()


# ── WATCHDOG — surveillance automatique des documents ────────

class DocumentHandler(FileSystemEventHandler):
    def __init__(self):
        self.embeddings = LocalEmbeddings()
        self.vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings
        )

    def _process(self, file_path: str):
        import os
        if not any(file_path.endswith(ext) for ext in (".pdf",".txt",".csv",".xlsx",".docx",".json")):
            return
        filename = os.path.basename(file_path)
        tracking = load_tracking()
        current_hash = get_file_hash(file_path)
        if filename in tracking and tracking[filename]["hash"] == current_hash:
            return
        print(f"[WATCH] Nouveau fichier détecté : {filename}")
        if index_single_file(file_path, self.vectordb):
            tracking[filename] = {"hash": current_hash, "path": file_path}
            save_tracking(tracking)
            print(f"[WATCH] ✓ {filename} indexé avec succès")

    def on_created(self, event):
        if not event.is_directory:
            time.sleep(1)
            self._process(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            time.sleep(1)
            self._process(event.src_path)


def start_watcher():
    import os
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    print("[WATCH] Indexation initiale des documents existants...")
    index_all_documents()
    handler = DocumentHandler()
    observer = Observer()
    observer.schedule(handler, path=DOCUMENTS_DIR, recursive=False)
    observer.start()
    print(f"[WATCH] Surveillance active sur : {DOCUMENTS_DIR}")
    try:
        while True:
            time.sleep(2)
    except Exception:
        observer.stop()
    observer.join()


@app.on_event("startup")
def startup_event():
    watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    watcher_thread.start()
    scheduler_thread = threading.Thread(target=trigger_scheduler, daemon=True)
    scheduler_thread.start()

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODÈLES ──────────────────────────────────────────────────

class RagQuestion(BaseModel):
    question: str


class ReplyPayload(BaseModel):
    message_id: str
    to: str
    subject: str
    body: str

# ── INTERFACE ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ── AGENT ────────────────────────────────────────────────────

@app.post("/api/agent/execute")
def execute_agent():
    try:
        subprocess.run(["python", "main_agent.py"], check=True)
        return {"success": True, "result": {"message": "Agent exécuté avec succès"}}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ── RAG CHAT ─────────────────────────────────────────────

@app.post("/api/rag/chat")
def rag_chat(payload: RagQuestion):
    try:
        answer = rag_agent(payload.question)
        return {"success": True, "answer": answer}
    except Exception as e:
        return {"success": False, "answer": f"Erreur : {str(e)}"}

# ── RAG UPLOAD ───────────────────────────────────────────────

@app.post("/api/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        dest = os.path.join(DOCUMENTS_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        # Indexation immédiate
        from langchain_chroma import Chroma
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=LocalEmbeddings())
        success = index_single_file(dest, vectordb)
        if success:
            tracking = load_tracking()
            tracking[file.filename] = {"hash": get_file_hash(dest), "path": dest}
            save_tracking(tracking)
        return {"success": success, "filename": file.filename}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/rag/documents")
def rag_documents():
    try:
        if not os.path.exists(DOCUMENTS_DIR):
            return {"files": []}
        files = [f for f in os.listdir(DOCUMENTS_DIR)
                 if f.lower().endswith((".pdf",".txt",".csv",".xlsx",".docx",".json"))]
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}

# ── EMAILS NON LUS ───────────────────────────────────────────

@app.get("/api/emails/unread")
def get_unread_emails():
    service = get_gmail_service()

    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX", "UNREAD"],
        maxResults=50
    ).execute()

    messages = results.get("messages", [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="metadata",
            metadataHeaders=["Subject", "From", "Date"]
        ).execute()

        headers = msg_data["payload"]["headers"]

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
        sender  = next((h["value"] for h in headers if h["name"] == "From"),    "")
        date    = next((h["value"] for h in headers if h["name"] == "Date"),    "")

        emails.append({
            "id":      msg["id"],   # ← nécessaire pour répondre
            "subject": subject,
            "sender":  sender,
            "date":    date,
            "snippet": msg_data.get("snippet", "")
        })

    return {"emails": emails}

# ── EMAILS ENVOYÉS ───────────────────────────────────────────

@app.get("/api/emails/sent")
def get_sent_emails():
    service = get_gmail_service()

    results = service.users().messages().list(
        userId="me",
        labelIds=["SENT"],
        maxResults=50
    ).execute()

    messages = results.get("messages", [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="metadata",
            metadataHeaders=["Subject", "To", "Date"]
        ).execute()

        headers = msg_data["payload"]["headers"]

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
        to      = next((h["value"] for h in headers if h["name"] == "To"),      "")
        date    = next((h["value"] for h in headers if h["name"] == "Date"),    "")

        emails.append({
            "id":        msg["id"],
            "subject":   subject,
            "to":        to,
            "timestamp": date,
            "snippet":   msg_data.get("snippet", ""),
            "intent":    "email",
            "priority":  "normal"
        })

    return {"emails": emails}

# ── RÉPONDRE À UN EMAIL ──────────────────────────────────────

@app.post("/api/emails/reply")
def send_reply(payload: ReplyPayload):
    try:
        service = get_gmail_service()

        msg = EmailMessage()
        msg["To"]      = payload.to
        msg["Subject"] = payload.subject
        msg.set_content(payload.body)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        service.users().messages().send(
            userId="me",
            body={"raw": raw}
        ).execute()

        return {"success": True}

    except Exception as e:
        return {"success": False, "error": str(e)}

# ── CALENDRIER ───────────────────────────────────────────────

@app.get("/api/calendar/events")
def get_calendar_events(year: int = None, month: int = None):
    import calendar

    service = get_calendar_service()
    now = datetime.utcnow()

    # Mois affiché
    target_year  = year or now.year
    target_month = month or now.month

    # Bornes strictes du mois
    start_date = datetime(target_year, target_month, 1)
    last_day = calendar.monthrange(target_year, target_month)[1]
    end_date = datetime(target_year, target_month, last_day, 23, 59, 59)

    time_min = start_date.isoformat() + "Z"
    time_max = end_date.isoformat() + "Z"

    all_events = []
    page_token = None

    while True:
        kwargs = dict(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=250,
            singleEvents=True,
            orderBy="startTime"
        )

        if page_token:
            kwargs["pageToken"] = page_token

        result = service.events().list(**kwargs).execute()
        all_events.extend(result.get("items", []))

        page_token = result.get("nextPageToken")
        if not page_token:
            break

    #  FILTRE FINAL SÉCURITÉ (important)
    formatted_events = []
    for event in all_events:
        start = event["start"].get("dateTime", event["start"].get("date"))

        try:
            dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except Exception:
            continue

        if dt.year == target_year and dt.month == target_month:
            formatted_events.append({
                "title": event.get("summary", "Sans titre"),
                "start": start,
                "description": event.get("description", ""),
                "attendees": [a["email"] for a in event.get("attendees", [])]
            })

    return {"events": formatted_events}


# ── STATISTIQUES CALENDRIER ──────────────────────────────────

# @app.get("/api/calendar/stats")
# def get_calendar_stats(year: int = None, month: int = None):
#     """
#     Retourne les stats pour un mois donné (défaut = mois courant).

#     Params:
#         year  : année  (ex: 2025)
#         month : mois   (ex: 3 pour mars)
#     """
#     from collections import Counter
#     import calendar

#     service = get_calendar_service()
#     now     = datetime.utcnow()

#     # Mois cible (défaut = mois courant)
#     target_year  = year  or now.year
#     target_month = month or now.month

#     # Bornes du mois cible uniquement
#     first_day = datetime(target_year, target_month, 1)
#     last_day  = datetime(
#         target_year,
#         target_month,
#         calendar.monthrange(target_year, target_month)[1],
#         23, 59, 59
#     )

#     # ── 1 seul appel API ──────────────────────────────────────
#     result = service.events().list(
#         calendarId   = "primary",
#         timeMin      = first_day.isoformat() + "Z",
#         timeMax      = last_day.isoformat()  + "Z",
#         maxResults   = 250,   # un mois ne dépasse jamais 250 events
#         singleEvents = True,
#         orderBy      = "startTime"
#     ).execute()

#     month_events = result.get("items", [])

#     # ── Analyse ───────────────────────────────────────────────
#     weekly   = Counter()
#     daily    = Counter()
#     types    = Counter()
#     stop_words = {"de","du","le","la","les","un","une","des","et","en",
#                   "à","au","avec","pour","sur","par","the","of","a"}
#     day_names  = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]

#     for event in month_events:
#         start_raw = event["start"].get("dateTime", event["start"].get("date", ""))
#         if not start_raw:
#             continue
#         try:
#             dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
#         except Exception:
#             continue

#         # Semaine dans le mois (S01, S02, S03, S04, S05)
#         week_num = ((dt.day - 1) // 7) + 1
#         weekly[f"S{week_num:02d}"] += 1

#         # Jour de la semaine
#         daily[day_names[dt.weekday()]] += 1

#         # Type d'événement (premier mot significatif du titre)
#         title   = event.get("summary", "Sans titre").strip()
#         words   = [w.lower() for w in title.split()
#                    if w.lower() not in stop_words and len(w) > 2]
#         keyword = words[0].capitalize() if words else "Autre"
#         types[keyword] += 1

#     # ── Navigation mois précédent / suivant ───────────────────
#     prev_month = target_month - 1 if target_month > 1 else 12
#     prev_year  = target_year      if target_month > 1 else target_year - 1
#     next_month = target_month + 1 if target_month < 12 else 1
#     next_year  = target_year      if target_month < 12 else target_year + 1

#     def counter_sorted(c, keys):
#         return [{"label": k, "count": c.get(k, 0)} for k in keys]

#     return {
#         "period": {
#             "year":       target_year,
#             "month":      target_month,
#             "label":      first_day.strftime("%B %Y"),
#             "prev":       {"year": prev_year,  "month": prev_month},
#             "next":       {"year": next_year,  "month": next_month},
#             "is_current": (target_year == now.year and target_month == now.month)
#         },
#         "total_month": len(month_events),
#         "weekly":      [{"label": k, "count": weekly[k]} for k in sorted(weekly)],
#         "daily":       counter_sorted(daily, day_names),
#         "types":       [{"label": k, "count": v} for k, v in types.most_common(10)],
#     }

@app.get("/api/calendar/stats")
def get_calendar_stats(year: int = None, month: int = None, week: int = None, view: str = "month"):
    """
    Retourne les statistiques du calendrier avec catégorisation intelligente.

    Params:
        year  : année  (ex: 2025)
        month : mois   (ex: 3 pour mars)  — utilisé si view="month"
        week  : numéro de semaine ISO     — utilisé si view="week"
        view  : "week" | "month" | "year"
    """
    import calendar
    from collections import Counter, defaultdict

    service = get_calendar_service()
    now     = datetime.utcnow()

    target_year  = year  or now.year
    target_month = month or now.month

    # ─────────────────────────────────────────────────────────
    # CATÉGORIES — mots-clés par famille
    # Chaque événement sera coloré selon sa catégorie
    # ─────────────────────────────────────────────────────────
    CATEGORIES = {
        "Réunion":      {
            "color": "#2563eb",   # bleu
            "keywords": ["réunion", "meeting", "standup", "stand-up", "sprint",
                         "planning", "review", "retrospective", "sync", "call",
                         "conférence", "conference", "comité", "comite", "briefing",
                         "kick-off", "kickoff", "workshop", "atelier"]
        },
        "Cours / Étudiant": {
            "color": "#7c3aed",   # violet
            "keywords": ["cours", "class", "étudiant", "etudiant", "student",
                         "lecture", "séminaire", "seminaire", "td", "tp", "exam",
                         "examen", "soutenance", "tutorat", "tuto", "formation",
                         "apprentissage", "enseignement", "amphi", "correction",
                         "devoir", "projet étudiant", "jury"]
        },
        "Transport":    {
            "color": "#0891b2",   # cyan
            "keywords": ["train", "tgv", "ter", "avion", "flight", "vol",
                         "bus", "metro", "taxi", "uber", "trajet", "voyage",
                         "départ", "depart", "arrivée", "arrivee", "aéroport",
                         "airport", "gare", "navette", "transfert", "eurostar",
                         "intercités", "intercites", "rer"]
        },
        "Rendez-vous":  {
            "color": "#db2777",   # rose
            "keywords": ["rdv", "rendez-vous", "rendez vous", "appointment",
                         "consultation", "médecin", "medecin", "docteur", "doctor",
                         "dentiste", "kiné", "kine", "pharmacie", "clinique",
                         "hôpital", "hopital", "notaire", "avocat", "expert",
                         "entretien", "interview", "visite"]
        },
        "Déjeuner / Repas": {
            "color": "#ea580c",   # orange
            "keywords": ["déjeuner", "dejeuner", "lunch", "dîner", "diner",
                         "dinner", "repas", "restaurant", "café", "cafe",
                         "petit-déjeuner", "breakfast", "brunch", "dégustation",
                         "pot", "apéro", "apero", "soirée", "soiree"]
        },
        "Sport":        {
            "color": "#16a34a",   # vert
            "keywords": ["sport", "gym", "fitness", "running", "course",
                         "natation", "piscine", "tennis", "football", "foot",
                         "basket", "vélo", "velo", "cyclisme", "yoga",
                         "pilates", "crossfit", "musculation", "escalade",
                         "randonnée", "rando", "marche", "match"]
        },
        "Perso / Famille": {
            "color": "#f59e0b",   # jaune
            "keywords": ["famille", "family", "anniversaire", "birthday",
                         "mariage", "wedding", "vacances", "holiday", "congé",
                         "conge", "weekend", "week-end", "fête", "fete",
                         "noël", "noel", "christmas", "pâques", "paques",
                         "enfant", "kids", "école", "ecole", "parent"]
        },
        "Administratif": {
            "color": "#6b7280",   # gris
            "keywords": ["administratif", "admin", "facture", "invoice",
                         "déclaration", "declaration", "impôt", "impot",
                         "banque", "bank", "assurance", "contrat", "contract",
                         "signature", "document", "dossier", "formulaire",
                         "préfecture", "mairie", "urssaf", "comptabilité"]
        },
    }

    def categorize(title: str) -> str:
        """Retourne la catégorie d'un événement selon son titre."""
        title_lower = title.lower()
        for cat, cfg in CATEGORIES.items():
            for kw in cfg["keywords"]:
                if kw in title_lower:
                    return cat
        return "Autre"

    # ─────────────────────────────────────────────────────────
    # BORNES selon la vue
    # ─────────────────────────────────────────────────────────
    if view == "year":
        first_day = datetime(target_year, 1, 1)
        last_day  = datetime(target_year, 12, 31, 23, 59, 59)
    elif view == "week":
        # Semaine ISO — lundi au dimanche
        iso_week  = week or now.isocalendar()[1]
        first_day = datetime.fromisocalendar(target_year, iso_week, 1)  # lundi
        last_day  = datetime.fromisocalendar(target_year, iso_week, 7).replace(hour=23, minute=59, second=59)
    else:  # month (défaut)
        first_day = datetime(target_year, target_month, 1)
        last_day  = datetime(
            target_year, target_month,
            calendar.monthrange(target_year, target_month)[1],
            23, 59, 59
        )

    # ─────────────────────────────────────────────────────────
    # APPEL API Google Calendar (paginé si > 250 events/an)
    # ─────────────────────────────────────────────────────────
    all_events = []
    page_token = None
    while True:
        kwargs = dict(
            calendarId   = "primary",
            timeMin      = first_day.isoformat() + "Z",
            timeMax      = last_day.isoformat()  + "Z",
            maxResults   = 250,
            singleEvents = True,
            orderBy      = "startTime",
        )
        if page_token:
            kwargs["pageToken"] = page_token
        result     = service.events().list(**kwargs).execute()
        all_events.extend(result.get("items", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            break

    total = len(all_events)

    # ─────────────────────────────────────────────────────────
    # ANALYSE
    # ─────────────────────────────────────────────────────────
    day_names  = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    cat_counter   = Counter()
    weekly_counter = Counter()   # "S01" … "S05"  ou  semaine ISO
    daily_counter  = Counter()   # Lun … Dim
    monthly_counter = Counter()  # Jan … Déc  (utilisé en vue année)
    month_names = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Août","Sep","Oct","Nov","Déc"]

    # Pour la durée moyenne par catégorie
    cat_durations = defaultdict(list)   # catégorie → liste de durées en minutes

    # ← AJOUT : détail par catégorie pour chaque période de la timeline
    timeline_cats = defaultdict(lambda: defaultdict(int))

    for event in all_events:
        start_raw = event["start"].get("dateTime", event["start"].get("date", ""))
        end_raw   = event["end"].get("dateTime",   event["end"].get("date",   ""))
        if not start_raw:
            continue
        try:
            dt_start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
            dt_end   = datetime.fromisoformat(end_raw.replace("Z", "+00:00")) if end_raw else dt_start
        except Exception:
            continue

        title    = event.get("summary", "Sans titre").strip()
        category = categorize(title)
        cat_counter[category] += 1

        # Durée en minutes
        duration = (dt_end - dt_start).total_seconds() / 60
        if 0 < duration < 1440:   # ignore all-day et valeurs aberrantes
            cat_durations[category].append(duration)

        # Semaine dans le mois (vue mois) ou semaine ISO (vue année)
        if view == "year":
            period_label = month_names[dt_start.month - 1]
            monthly_counter[period_label] += 1
        elif view == "week":
            period_label = day_names[dt_start.weekday()]
            weekly_counter[period_label] += 1
        else:
            week_num = ((dt_start.day - 1) // 7) + 1
            period_label = f"S{week_num:02d}"
            weekly_counter[period_label] += 1

        # ← AJOUT : compter la catégorie pour cette période
        timeline_cats[period_label][category] += 1

        # Jour de la semaine (toujours)
        daily_counter[day_names[dt_start.weekday()]] += 1

    # ─────────────────────────────────────────────────────────
    # CONSTRUCTION DES RÉSULTATS
    # ─────────────────────────────────────────────────────────

    # Catégories avec pourcentage + couleur
    categories_result = []
    for cat, count in cat_counter.most_common():
        pct = round(count / total * 100, 1) if total else 0
        avg_dur = round(sum(cat_durations[cat]) / len(cat_durations[cat])) if cat_durations[cat] else None
        categories_result.append({
            "label":       cat,
            "count":       count,
            "percentage":  pct,
            "color":       CATEGORIES.get(cat, {}).get("color", "#94a3b8"),
            "avg_duration_min": avg_dur,   # durée moyenne en minutes
        })

    # Série temporelle — avec détail par catégorie pour le diagramme empilé
    if view == "year":
        timeline = [
            {"label": m, "count": monthly_counter.get(m, 0), **timeline_cats.get(m, {})}
            for m in month_names
        ]
    elif view == "week":
        timeline = [
            {"label": d, "count": weekly_counter.get(d, 0), **timeline_cats.get(d, {})}
            for d in day_names
        ]
    else:
        sorted_weeks = sorted(weekly_counter.keys())
        timeline = [
            {"label": w, "count": weekly_counter[w], **timeline_cats.get(w, {})}
            for w in sorted_weeks
        ]

    # Répartition par jour de semaine (toutes vues)
    daily_result = [{"label": d, "count": daily_counter.get(d, 0)} for d in day_names]

    # ─────────────────────────────────────────────────────────
    # NAVIGATION
    # ─────────────────────────────────────────────────────────
    if view == "year":
        prev_nav = {"year": target_year - 1, "view": "year"}
        next_nav = {"year": target_year + 1, "view": "year"}
        period_label = str(target_year)
        is_current   = target_year == now.year
    elif view == "week":
        iso_week = week or now.isocalendar()[1]
        prev_w   = iso_week - 1 if iso_week > 1 else 52
        prev_y   = target_year if iso_week > 1 else target_year - 1
        next_w   = iso_week + 1 if iso_week < 52 else 1
        next_y   = target_year if iso_week < 52 else target_year + 1
        prev_nav = {"year": prev_y, "week": prev_w, "view": "week"}
        next_nav = {"year": next_y, "week": next_w, "view": "week"}
        period_label = f"Semaine {iso_week} — {target_year}"
        is_current   = (target_year == now.year and iso_week == now.isocalendar()[1])
    else:
        prev_month = target_month - 1 if target_month > 1 else 12
        prev_year  = target_year      if target_month > 1 else target_year - 1
        next_month = target_month + 1 if target_month < 12 else 1
        next_year  = target_year      if target_month < 12 else target_year + 1
        prev_nav = {"year": prev_year,  "month": prev_month, "view": "month"}
        next_nav = {"year": next_year,  "month": next_month, "view": "month"}
        period_label = first_day.strftime("%B %Y")
        is_current   = (target_year == now.year and target_month == now.month)

    # Jour le plus chargé
    busiest = max(daily_result, key=lambda x: x["count"], default={"label": "—", "count": 0})

    return {
        "period": {
            "year":       target_year,
            "month":      target_month if view == "month" else None,
            "view":       view,
            "label":      period_label,
            "prev":       prev_nav,
            "next":       next_nav,
            "is_current": is_current,
        },
        "total":        total,
        "total_month":  total,        
        "categories":   categories_result,
        "types":        categories_result,   
        "timeline":     timeline,
        "weekly":       timeline,      
        "daily":        daily_result,
        "busiest_day":  busiest,
    }
    
    
    #  TRIGGER CONFIG

TRIGGER_CONFIG = {
    "active": False,
    "interval_minutes": 15,
    "last_run": None
}

trigger_lock = Lock()

# ── TRIGGER API ─────────────────────────────────────


class TriggerConfig(BaseModel):
    active: bool
    interval_minutes: int


@app.get("/api/triggers/config")
def get_trigger_config():
    return TRIGGER_CONFIG


@app.post("/api/triggers/config")
def update_trigger(config: TriggerConfig):
    with trigger_lock:
        TRIGGER_CONFIG["active"] = config.active
        TRIGGER_CONFIG["interval_minutes"] = config.interval_minutes
    return {"success": True}

# ── TRIGGER SCHEDULER ─────────────────────────────────

def trigger_scheduler():
    while True:
        with trigger_lock:
            active = TRIGGER_CONFIG["active"]
            interval = TRIGGER_CONFIG["interval_minutes"]
            last_run = TRIGGER_CONFIG["last_run"]

        if active:
            now = datetime.utcnow()

            if last_run is None or (now - last_run) >= timedelta(minutes=interval):
                print(f"[TRIGGER] Exécution automatique ({interval} min)")
                try:
                    subprocess.Popen(["python", "main_agent.py"])
                    with trigger_lock:
                        TRIGGER_CONFIG["last_run"] = now.isoformat()
                except Exception as e:
                    print("[TRIGGER ERROR]", e)

        time.sleep(30)  