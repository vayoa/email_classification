# gmail_2000_to_csv.py
"""
Fetch the 2,000 most recent Gmail messages (no time limit) and save sender, subject, body (plain text) to CSV.

Setup:
  1) Enable Gmail API in Google Cloud; create OAuth Client ID (Desktop App).
  2) Download as credentials.json next to this script.
  3) pip install -U google-api-python-client google-auth-httplib2 google-auth-oauthlib html2text pandas

Run:
  python gmail_2000_to_csv.py
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ---------------- Config ---------------- #
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"

LIMIT = 2000  # number of most recent messages to export
LIST_PAGE_SIZE = 500  # Gmail allows up to 500 per list page
BATCH_SIZE = 100  # up to ~100 sub-requests per batch is safe

OUTPUT_CSV = "gmail_2000_recent.csv"

# Ask for a minimal subset of fields to reduce payload size.
FIELDS = (
    "id,threadId,snippet,internalDate,"
    "payload/headers(name,value),"
    "payload/mimeType,"
    "payload/body/data,"
    "payload/parts(mimeType,filename,body/data,headers)"
)


# ---------------- Auth ---------------- #
def get_gmail_service() -> Any:
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError(
            "credentials.json not found. Put your OAuth client (Desktop App) here."
        )
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ---------------- Helpers ---------------- #
def b64url_decode(b64: str | None) -> str:
    if not b64:
        return ""
    return base64.urlsafe_b64decode(b64).decode("utf-8", errors="replace")


def headers_to_dict(headers_list: List[Dict[str, str]] | None) -> Dict[str, str]:
    headers_list = headers_list or []
    return {h.get("name", "").lower(): h.get("value", "") for h in headers_list}


def html_to_text_fast(html: str) -> str:
    # Fast, decent-quality HTML -> text
    # html2text returns Markdown-like text; good for modeling.
    import html2text

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(html or "").strip()


def extract_body_from_payload(payload: Dict[str, Any]) -> str:
    """
    Prefer text/plain parts; if absent, fall back to HTML->text.
    Skip attachments (parts with a filename).
    """
    if not payload:
        return ""

    plains: List[str] = []
    htmls: List[str] = []

    stack = [payload]
    while stack:
        p = stack.pop()
        mime = (p.get("mimeType") or "").lower()
        body = p.get("body", {}) or {}
        filename = (p.get("filename") or "").strip()

        # Skip attachments by filename presence
        if filename:
            continue

        data = b64url_decode(body.get("data"))
        if data:
            if mime == "text/plain":
                plains.append(data.strip())
            elif mime == "text/html":
                htmls.append(data)

        # Recurse into children
        for child in p.get("parts", []) or []:
            stack.append(child)

    if plains:
        # Join multiple plain parts with a divider
        return "\n\n---\n\n".join([t for t in plains if t])

    if htmls:
        # Convert each HTML part to text, then join
        texts = [html_to_text_fast(h) for h in htmls if h.strip()]
        texts = [t for t in texts if t]
        return "\n\n---\n\n".join(texts)

    return ""


# ---------------- Core fetch ---------------- #
def list_recent_message_ids(service, limit: int = LIMIT) -> List[str]:
    """
    Return up to `limit` most recent message IDs.
    (messages.list returns newest-first by default)
    """
    ids: List[str] = []
    req = service.users().messages().list(userId="me", maxResults=LIST_PAGE_SIZE)
    while req and len(ids) < limit:
        resp = req.execute()
        ids.extend(m["id"] for m in resp.get("messages", []) if "id" in m)
        req = service.users().messages().list_next(req, resp)
    return ids[:limit]


def fetch_sender_subject_body_batched(
    service, ids: List[str], batch_size: int = BATCH_SIZE
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Use Gmail batch HTTP with format='full' to fetch many messages efficiently.
    Returns (rows, errors). Each row: {id, from, subject, body}
    """
    results: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []

    def cb(request_id, response, exception):
        if exception:
            errors.append({"id": request_id, "error": str(exception)})
            return
        try:
            payload = response.get("payload", {}) or {}
            h = headers_to_dict(payload.get("headers"))
            sender = h.get("from", "")
            subject = h.get("subject", "")

            body_text = extract_body_from_payload(payload)
            results[response["id"]] = {
                "id": response.get("id", ""),
                "from": sender,
                "subject": subject,
                "body": body_text,
            }
        except Exception as e:
            errors.append({"id": request_id, "error": f"parse_error: {e}"})

    # Use the service-provided batch helper to get the correct batch endpoint
    for i in range(0, len(ids), batch_size):
        batch = service.new_batch_http_request(callback=cb)
        chunk = ids[i : i + batch_size]
        for mid in chunk:
            req = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=mid,
                    format="full",
                    fields=FIELDS,
                )
            )
            batch.add(req, request_id=mid)
        batch.execute()

    # Keep original order
    rows = [
        results.get(mid, {"id": mid, "from": "", "subject": "", "body": ""})
        for mid in ids
    ]
    return rows, errors


# ---------------- CSV ---------------- #
def write_csv_with_pandas(rows: List[Dict[str, Any]], path: str = OUTPUT_CSV) -> None:
    df = pd.DataFrame(rows, columns=["id", "from", "subject", "body"])
    # Normalize newlines for CSV readability
    df["body"] = df["body"].fillna("").str.replace("\r", "", regex=False)
    df["text"] = "Subject: " + df["subject"] + "\n\n" + df["body"]
    df.to_csv(path, index=False, encoding="utf-8")


# ---------------- Main ---------------- #
def main():
    service = get_gmail_service()
    print(f"Listing up to {LIMIT} most recent message IDs...")
    ids = list_recent_message_ids(service, limit=LIMIT)
    print(
        f"Found {len(ids)} message IDs. Fetching details in batches of {BATCH_SIZE}..."
    )

    rows, errs = fetch_sender_subject_body_batched(service, ids, batch_size=BATCH_SIZE)
    if errs:
        print(f"Completed with {len(errs)} errors (continuing).")

    write_csv_with_pandas(rows, OUTPUT_CSV)
    print(f"Done. Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
