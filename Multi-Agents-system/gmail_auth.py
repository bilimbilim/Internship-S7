import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# # ===============================
# # SCOPES GMAIL
# # ===============================
# SCOPES = [
#     "https://www.googleapis.com/auth/gmail.readonly",
#     "https://www.googleapis.com/auth/gmail.send",
#     "https://www.googleapis.com/auth/gmail.modify",
#     "https://www.googleapis.com/auth/calendar"

# ]

# def get_gmail_service():
#     """
#     Initialise la connexion OAuth Gmail et retourne un service Gmail prêt à l'emploi.
#     """
#     creds = None

#     if os.path.exists("token.json"):
#         creds = Credentials.from_authorized_user_file("token.json", SCOPES)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 "credentials.json", SCOPES
#             )
#             creds = flow.run_local_server(
#                 port=0,
#                 prompt="consent",
#                 authorization_prompt_message="Autorise l'accès Gmail dans ton navigateur"
#             )

#         with open("token.json", "w") as token:
#             token.write(creds.to_json())

#     service = build("gmail", "v1", credentials=creds)
#     return service


# # TEST LOCAL (NE S’EXÉCUTE PAS À L’IMPORT)
# if __name__ == "__main__":
#     service = get_gmail_service()
#     results = service.users().messages().list(
#         userId="me",
#         maxResults=1,
#         q="is:inbox"
#     ).execute()
#     print(results)





SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar"
]


def get_google_credentials():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds


def get_gmail_service():
    creds = get_google_credentials()
    return build("gmail", "v1", credentials=creds)


def get_calendar_service():
    creds = get_google_credentials()
    return build("calendar", "v3", credentials=creds)
