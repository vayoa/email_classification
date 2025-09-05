from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Gmail API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Inference (ModernBERT multitask)
import torch
from transformers import AutoTokenizer

# Reuse model & utils from your inference script
from infer_email_multitask import (  # noqa: E402
    MultiTaskEmailModel,
    predict,
)

# Embeddings + Vector store
from sentence_transformers import SentenceTransformer
import chromadb


# ---------------- Config defaults ---------------- #
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"

LIST_PAGE_SIZE = 500  # Gmail allows up to 500 per list page
FETCH_BATCH_SIZE = 100  # Fetch & process in 100s as requested (keeps model alive)
MAX_INFER_BATCH = 64  # Tokenization/inference micro-batch (GPU/CPU friendly)

FIELDS = (
    "id,threadId,snippet,internalDate,"
    "payload/headers(name,value),"
    "payload/mimeType,"
    "payload/body/data,"
    "payload/parts(mimeType,filename,body/data,headers)"
)


# ---------------- Data classes ---------------- #
@dataclass
class ModelBundle:
    model: MultiTaskEmailModel
    tok: AutoTokenizer
    device: str
    cfg: Dict[str, Any]
    id2label_cat: Dict[int, str]


# ---------------- Gmail Auth ---------------- #
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
        return "\n\n---\n\n".join([t for t in plains if t])

    if htmls:
        texts = [html_to_text_fast(h) for h in htmls if (h or "").strip()]
        texts = [t for t in texts if t]
        return "\n\n---\n\n".join(texts)

    return ""


# ---------------- Gmail fetch ---------------- #
def list_recent_message_ids(service, limit: int) -> List[str]:
    """
    Return up to `limit` most recent message IDs (newest-first by default).
    """
    ids: List[str] = []
    req = service.users().messages().list(userId="me", maxResults=LIST_PAGE_SIZE)
    while req and len(ids) < limit:
        resp = req.execute()
        ids.extend(m["id"] for m in resp.get("messages", []) if "id" in m)
        req = service.users().messages().list_next(req, resp)
    return ids[:limit]


def fetch_sender_subject_body_batched(
    service, ids: List[str], batch_size: int = FETCH_BATCH_SIZE
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

            # NEW: capture Gmail's internalDate (ms since epoch, UTC)
            internal_ms = response.get("internalDate")
            if internal_ms is not None:
                internal_ms = int(internal_ms)
                dt = datetime.fromtimestamp(internal_ms / 1000.0, tz=timezone.utc)
                iso = dt.isoformat()
            else:
                iso = None

            results[response["id"]] = {
                "id": response.get("id", ""),
                "from": sender or "",
                "subject": subject or "",
                "body": body_text or "",
                "date_epoch_ms": internal_ms or "",
                "date_iso": iso or "",
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

    # Keep original order and ensure defaults
    rows = [
        results.get(mid, {"id": mid, "from": "", "subject": "", "body": ""})
        for mid in ids
    ]
    return rows, errors


def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# ---------------- Inference & Embeddings ---------------- #
def init_multitask_model(model_dir: str) -> ModelBundle:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = os.path.join(model_dir, "multitask_config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])

    num_cat = cfg["num_category"]
    model = MultiTaskEmailModel(cfg["base_model"], num_cat=num_cat)
    state = torch.load(
        os.path.join(model_dir, "pytorch_model_multitask.bin"), map_location="cpu"
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    id2label_cat = {int(k): v for k, v in cfg["id2label_category"].items()}

    return ModelBundle(
        model=model, tok=tok, device=device, cfg=cfg, id2label_cat=id2label_cat
    )


def classify_rows(rows: List[Dict[str, Any]], mb: ModelBundle) -> List[Dict[str, Any]]:
    """
    Run multitask classification on a batch of rows using the already-loaded model.
    Returns list of per-row prediction dicts, in the same order as rows.
    """
    import math

    df = pd.DataFrame(rows, columns=["id", "from", "subject", "body"])
    texts = (
        df["subject"].astype(str).fillna("")
        + "\n\n"
        + df["body"].astype(str).fillna("")
    ).tolist()

    ycat_ids, ycat_probs, yimp_ids, yimp_p1 = predict(
        mb.model,
        mb.tok,
        texts,
        mb.device,
        max_length=mb.cfg.get("max_length", 512),
        batch_size=MAX_INFER_BATCH,
    )
    cat_labels = [mb.id2label_cat[int(i)] for i in ycat_ids]
    imp_labels = ["important" if p >= 0.5 else "not_important" for p in yimp_p1]

    preds = []
    for i, rid in enumerate(df["id"].tolist()):
        # Probability of the predicted class only
        try:
            idx = int(ycat_ids[i])
            cat_p = float(ycat_probs[i][idx])
        except Exception:
            # Fallback: take max prob if indexing is out of bounds
            cat_p = float(max(ycat_probs[i])) if len(ycat_probs[i]) else 0.0

        preds.append(
            {
                "id": rid,
                "pred_category": cat_labels[i],
                "pred_category_p": cat_p,
                "pred_important": imp_labels[i],
                "pred_important_p": float(yimp_p1[i]),
            }
        )
    return preds


def init_vector_store(persist_dir: str, collection_name: str):
    # Persistent client (non-ephemeral)
    client = chromadb.PersistentClient(path=persist_dir)
    # Create/get collection; we will pass precomputed embeddings
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


def embed_and_upsert(
    rows: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
    collection,
    sbert: SentenceTransformer,
):
    """
    Embed 'text' = Subject + Body with MiniLM, and upsert to Chroma with metadata
    that includes the chosen classification + its probability, and importance + its probability.
    """
    if not rows:
        return

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    pred_by_id = {p["id"]: p for p in preds}

    for r in rows:
        rid = r["id"]
        subj = (r.get("subject") or "").strip()
        body = (r.get("body") or "").strip()
        text = f"Subject: {subj}\n\n{body}"
        p = pred_by_id.get(rid, {})

        meta = {
            "from": (r.get("from") or "").strip(),
            "subject": subj,
            "pred_category": p.get("pred_category"),
            "pred_category_p": float(p.get("pred_category_p") or 0.0),
            "pred_important": p.get("pred_important"),
            "pred_important_p": float(p.get("pred_important_p") or 0.0),
            "date_iso": r.get("date_iso") or None,
            "date_epoch_ms": (
                int(r["date_epoch_ms"]) if r.get("date_epoch_ms") is not None else None
            ),
        }

        ids.append(rid)
        texts.append(text)
        metadatas.append(meta)

    embs = sbert.encode(
        texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embs,
    )


# ---------------- Pipeline ---------------- #
def process_batch(
    rows: List[Dict[str, Any]],
    mb: ModelBundle,
    collection,
    sbert: SentenceTransformer,
) -> int:
    """
    Drop empty rows, classify the rest, embed, and upsert.
    Returns number of indexed rows.
    """
    # 1) Clear rows with no sender, subject, and body (all empty) — ~30% historically.
    cleaned = []
    for r in rows:
        sender = (r.get("from") or "").strip()
        subj = (r.get("subject") or "").strip()
        body = (r.get("body") or "").strip()
        if sender or subj or body:
            cleaned.append(
                {
                    "id": r["id"],
                    "from": sender,
                    "subject": subj,
                    "body": body,
                    "date_epoch_ms": r["date_epoch_ms"],
                    "date_iso": r["date_iso"],
                }
            )
    if not cleaned:
        return 0

    # 2) Multitask inference on this batch with the live model
    preds = classify_rows(cleaned, mb)

    # 3) MiniLM embeddings + Chroma upsert (store classifications in metadata)
    embed_and_upsert(cleaned, preds, collection, sbert)

    return len(cleaned)


def run_indexer(
    limit: int,
    model_dir: str,
    persist_dir: str,
    collection_name: str,
) -> None:
    # Init Gmail service
    service = get_gmail_service()

    # List message IDs
    print(f"Listing up to {limit} most recent message IDs...")
    ids = list_recent_message_ids(service, limit=limit)
    print(f"Found {len(ids)} message IDs.")

    # Init ModernBERT multitask model once (kept alive across all batches)
    print("Loading multitask classifier...")
    mb = init_multitask_model(model_dir)

    # Init MiniLM embedder + Chroma persistent store
    print("Initializing MiniLM embedder and ChromaDB (persistent)...")
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    client, collection = init_vector_store(persist_dir, collection_name)

    # Pipeline: fetch -> submit classify+index to a worker while continuing to fetch
    total_indexed = 0
    total_errors = 0
    futures = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for chunk_ids in chunked(ids, FETCH_BATCH_SIZE):
            # Fetch this chunk's rows (network)
            rows, errs = fetch_sender_subject_body_batched(
                service, chunk_ids, batch_size=FETCH_BATCH_SIZE
            )
            total_errors += len(errs)

            # Submit CPU/GPU work (classification + embedding + upsert) to worker
            futures.append(ex.submit(process_batch, rows, mb, collection, sbert))

        # Drain
        for fut in as_completed(futures):
            total_indexed += fut.result() or 0

    # Persist (older Chroma versions)
    try:
        client.persist()  # type: ignore[attr-defined]
    except Exception:
        pass

    print(
        f"Done. Indexed {total_indexed} messages into Chroma (collection='{collection_name}', dir='{persist_dir}')."
    )
    if total_errors:
        print(f"Completed with {total_errors} fetch errors (skipped those).")


# ---------------- CLI ---------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Gmail -> (Classifier + ChromaDB) indexer")
    ap.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="How many most-recent messages to process",
    )
    ap.add_argument(
        "--model-dir",
        type=str,
        default="outputs/modernbert_multitask",
        help="Path to trained multitask model directory (with multitask_config.json & weights)",
    )
    ap.add_argument(
        "--persist-dir",
        type=str,
        default="./db",
        help="Directory for persistent ChromaDB",
    )
    ap.add_argument(
        "--collection-name",
        type=str,
        default="emails",
        help="Chroma collection name",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_indexer(
        limit=args.limit,
        model_dir=args.model_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )
