from sentence_transformers import SentenceTransformer
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
from google import genai
from google.genai import types
from enum import Enum
from pydantic import BaseModel
from dotenv import load_dotenv
import hashlib
import json
import inspect
from rich import print
from indexer import (
    init_multitask_model,
    html_to_text_fast,
    classify_rows,
    embed_and_upsert,
    get_gmail_service,
    run_indexer,
)
from datetime import datetime

load_dotenv()


class Category(str, Enum):
    spam = "spam"
    updates = "updates"
    forum = "forum"
    social_media = "social_media"
    verify_code = "verify_code"
    promotions = "promotions"


class Email(BaseModel):
    id: str
    date: datetime
    sender: str
    subject: str | None = None
    text: str | None = None
    predicted_class: Category
    predicted_important: bool


@st.cache_resource(show_spinner=False)
def get_chrome_collection() -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path="./db")
    return chroma_client.get_collection(
        "emails",
        # TODO: Fix
        # SentenceTransformerEmbeddingFunction(
        #     "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
        # ),
    )


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",
    )


@st.cache_resource(show_spinner=False)
def get_classifier():
    return init_multitask_model("outputs/modernbert_multitask")


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    return genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )


@st.cache_resource(show_spinner=False)
def index_pasted_email(
    body: str,
    subject: str = "",
    sender: str = "",
) -> dict:
    """
    Classify+embed+upsert one email. Returns the prediction dict.
    - If subject is empty, we try to use the first line (<=120 chars) as subject.
    - Uses a deterministic id (sha256 of sender/subject/body), so re-running updates the same record.
    """

    def _make_id(sender: str, subject: str, body: str) -> str:
        h = hashlib.sha256()
        h.update((sender or "").encode("utf-8"))
        h.update(b"\x00")
        h.update((subject or "").encode("utf-8"))
        h.update(b"\x00")
        h.update((body or "").encode("utf-8"))
        return h.hexdigest()

    body = body.strip()
    if not subject and "\n" in body:
        first, rest = body.split("\n", 1)
        if len(first.strip()) <= 120:
            subject = first.strip()
            body = rest.strip()
    body = html_to_text_fast(body)

    rid = _make_id(sender.strip(), subject.strip(), body.strip())
    rows = [
        {
            "id": rid,
            "from": sender.strip(),
            "subject": subject.strip(),
            "body": body.strip(),
        }
    ]

    preds = classify_rows(rows, get_classifier())
    embed_and_upsert(rows, preds, get_chrome_collection(), get_embedder())
    return preds[0]


def retrieve(
    query: str = None,
    pred_category: Category = None,
    pred_important: bool = None,
    before: str = None,
    after: str = None,
    k: int = 5,
) -> list[Email]:

    where = {"$and": []}
    if pred_category:
        where["$and"].append({"pred_category": {"$eq": pred_category}})
    if pred_important is not None:
        where["$and"].append(
            {
                "pred_important": {
                    "$eq": "important" if pred_important else "not_important"
                }
            }
        )
    if before:
        before = datetime.strptime(before, "%Y-%m-%d")
        where["$and"].append(
            {"date_epoch_ms": {"$lte": int(before.timestamp() * 1000)}}
        )
    if after:
        after = datetime.strptime(after, "%Y-%m-%d")
        where["$and"].append({"date_epoch_ms": {"$gte": int(after.timestamp() * 1000)}})

    if len(where["$and"]) == 1:
        where = where["$and"][0]

    collection = get_chrome_collection()

    if query is None:
        res = collection.get(
            limit=k,
            where=where or None,
            include=["documents", "metadatas", "uris"],
        )
        ids = res["ids"]
        documents = res["documents"]
        metadatas = res["metadatas"]
    else:
        res = collection.query(
            # query_texts=[query],
            query_embeddings=[
                get_embedder().encode(
                    query,
                    batch_size=128,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            ],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances", "uris"],
        )
        ids = res["ids"][0]
        documents = res["documents"][0]
        metadatas = res["metadatas"][0]

    emails = []
    for id, meta, text in zip(ids, metadatas, documents):
        emails.append(
            Email(
                id=id,
                date=meta.get("date_iso") or meta.get("date"),
                sender=meta.get("from") or "",
                subject=meta.get("subject"),
                text=text,
                predicted_class=meta.get("pred_category"),
                predicted_important=(
                    (meta.get("pred_important") or "") == "important"
                    if isinstance(meta.get("pred_important"), str)
                    else bool(meta.get("pred_important"))
                ),
            )
        )
    return emails


def search_by_text(
    query: str,
    category_filter: str | None = None,
    important_filter: bool | None = None,
    sent_before: str | None = None,
    sent_after: str | None = None,
    limit: int | None = 5,
) -> list[Email]:
    """Semantic email search with optional category/importance filters.

    Args:
        query: User's natural-language query, e.g., 'invoices from ACME last quarter'.
        category_filter: Optional predicted category to restrict the candidate set.
        important_filter: Optional predicted flag to restrict the candidate set to only important to read emails.
        sent_before: Optional date filter, "%Y-%m-%d" formatted.
        sent_after: Optional date filter, "%Y-%m-%d" formatted.
        limit: Number of results to return. Default is 5, max are 20.

    Returns:
        A list of emails that match the query.
    """
    return retrieve(
        query=query,
        pred_category=category_filter,
        pred_important=important_filter,
        before=sent_before,
        after=sent_after,
        k=min(limit or 5, 20),
    )


def search_by_category(
    category_filter: str,
    important_filter: bool = None,
    query: str = None,
    sent_before: str | None = None,
    sent_after: str | None = None,
    limit: int | None = 5,
) -> list[Email]:
    """Category-first retrieval with optional importance filter or semantic refinement.

    Args:
        category_filter: Predicted category to filter on.
        important_filter: Optional predicted flag to restrict the candidate set to only important to read emails.
        query: Optional natural-language query to refine within the class bucket, e.g., 'from ACME in Q2'. If omitted, results are ranked by recency.
        sent_before: Optional date filter, "%Y-%m-%d" formatted.
        sent_after: Optional date filter, "%Y-%m-%d" formatted.
        limit: Number of results to return. Default is 5, max are 20.

    Returns:
        A list of emails that match the query.
    """
    return retrieve(
        query=query,
        pred_category=category_filter,
        pred_important=important_filter,
        before=sent_before,
        after=sent_after,
        k=min(limit or 5, 20),
    )


def _as_email(obj) -> Email | None:
    try:
        if isinstance(obj, Email):
            return obj
        if hasattr(obj, "model_dump"):
            data = obj.model_dump()
        elif isinstance(obj, dict):
            data = obj
        else:
            return None
        data = {
            "id": data.get("id") or "",
            "date": data.get("date") or data.get("date_iso") or data.get("date_epoch_ms"),
            "sender": data.get("sender") or data.get("from") or "",
            "subject": data.get("subject"),
            "text": data.get("text") or data.get("body"),
            "predicted_class": data.get("predicted_class") or data.get("pred_category"),
            "predicted_important": data.get("predicted_important")
            if data.get("predicted_important") is not None
            else data.get("pred_important"),
        }
        # derive subject from first line of text if missing
        if not data.get("subject") and isinstance(data.get("text"), str):
            first = data["text"].strip().splitlines()[:1]
            if first:
                fl = first[0].strip()
                if 0 < len(fl) <= 120:
                    data["subject"] = fl
        return Email(**data)
    except Exception:
        return None


 


_HAS_DIALOG = hasattr(st, "dialog")

if _HAS_DIALOG:

    @st.dialog("Source Email")
    def _show_source_dialog(item: Email):
        st.markdown(f"From: {item.sender}")
        st.markdown(
            f"Date: {item.date.strftime('%Y-%m-%d') if isinstance(item.date, datetime) else str(item.date)}"
        )
        cat = (
            item.predicted_class.value
            if isinstance(item.predicted_class, Category)
            else str(item.predicted_class)
        )
        meta_bits = [b for b in [cat.replace("_", " ").title(), ("Important" if item.predicted_important else "Not important")] if b]
        if meta_bits:
            st.caption(" • ".join(meta_bits))
        if item.subject:
            st.subheader(item.subject)
        if item.text:
            st.write(item.text)


def _render_inline_source_detail(item: Email):
    with st.container(border=True):
        st.markdown(f"From: {item.sender}")
        st.markdown(
            f"Date: {item.date.strftime('%Y-%m-%d') if isinstance(item.date, datetime) else str(item.date)}"
        )
        cat = (
            item.predicted_class.value
            if isinstance(item.predicted_class, Category)
            else str(item.predicted_class)
        )
        meta_bits = [b for b in [cat.replace("_", " ").title(), ("Important" if item.predicted_important else "Not important")] if b]
        if meta_bits:
            st.caption(" • ".join(meta_bits))
        if item.subject:
            st.subheader(item.subject)
        if item.text:
            st.write(item.text)


def _render_sources_panel_v2(items: list[Email]):
    if not items:
        return
    with st.expander(f"Sources ({len(items)})", expanded=False):
        cols_per_row = 4
        max_items = min(len(items), 12)

        def _cat_key(cat_val) -> str:
            if isinstance(cat_val, Category):
                s = cat_val.value
            else:
                s = str(cat_val or "")
                if "." in s:
                    s = s.split(".")[-1]
            return s.replace("_", " ").strip().lower()

        def _cat_icon(cat_val) -> str:
            c = _cat_key(cat_val)
            if not c:
                return "✉️"
            return {
                "promotions": "📢",
                "social media": "💬",
                "social_media": "💬",
                "updates": "🔔",
                "forum": "🧵",
                "verify code": "🔑",
                "verify_code": "🔑",
                "spam": "🚫",
            }.get(c, "✉️")

        def _imp_icon(b: bool | None) -> str:
            return "⭐" if b else "☆"

        def _trunc(s: str, n: int = 24) -> str:
            if not s:
                return "(no subject)"
            return s if len(s) <= n else s[: n - 1] + "…"

        clicked = None
        for i, it in enumerate(items[:max_items]):
            if i % cols_per_row == 0:
                row = st.columns(cols_per_row, gap="small")
            col = row[i % cols_per_row]
            with col:
                date_str = it.date.strftime("%Y-%m-%d") if isinstance(it.date, datetime) else str(it.date)
                label = f"{_cat_icon(it.predicted_class)} {_imp_icon(it.predicted_important)}  {date_str} · {_trunc(it.subject or '')}"
                key = f"srcpill_{hash((id(items), it.id, it.sender, date_str, it.subject))}_{i}"
                if st.button(label, key=key, use_container_width=True):
                    clicked = it

        if clicked is not None:
            if _HAS_DIALOG:
                _show_source_dialog(clicked)
            else:
                st.session_state["_inline_source_detail"] = clicked

        if (not _HAS_DIALOG) and st.session_state.get("_inline_source_detail"):
            _render_inline_source_detail(st.session_state.get("_inline_source_detail"))


def _render_sources_panel(items: list[Email]):
    if not items:
        return
    # Collapsed by default to save space
    with st.expander(f"Sources ({len(items)})", expanded=False):
        # Render pills horizontally in a grid
        cols_per_row = 4
        max_items = min(len(items), 12)

        def _cat_icon(cat: Category) -> str:
            c = cat.value.lower()
            return {
                "promotions": "📢",
                "social_media": "💬",
                "updates": "🔔",
                "forum": "🧵",
                "verify_code": "🔑",
                "spam": "🚫",
            }[c]

        def _imp_icon(b: bool | None) -> str:
            return "⭐" if b else "☆"

        def _trunc(s: str, n: int = 24) -> str:
            if not s:
                return "(no subject)"
            return s if len(s) <= n else s[: n - 1] + "…"

        clicked = None
        for i, it in enumerate(items[:max_items]):
            if i % cols_per_row == 0:
                row = st.columns(cols_per_row, gap="small")
            col = row[i % cols_per_row]
            with col:
                label = f"{_cat_icon(it.get('category'))} {_imp_icon(it.get('important'))}  {it.get('date','')} · {_trunc(it.get('subject',''))}"
                key = f"srcpill_{hash((id(items), it.get('id'), it.get('from'), it.get('date'), it.get('subject')))}_{i}"
                if st.button(label, key=key, use_container_width=True):
                    clicked = it

        # On click show a dialog if available, otherwise inline details
        if clicked is not None:
            if _HAS_DIALOG:
                _show_source_dialog(clicked)
            else:
                st.session_state["_inline_source_detail"] = clicked

        if (not _HAS_DIALOG) and st.session_state.get("_inline_source_detail"):
            _render_inline_source_detail(st.session_state.get("_inline_source_detail"))


def _call_tool_from_function_call(part_fc) -> list:
    """Replays a tool call from a function_call part to recover sources.

    Falls back gracefully if schema differs. Only calls known tools.
    """
    try:
        name = getattr(part_fc, "name", None) or getattr(part_fc, "function_name", None)
        args = (
            getattr(part_fc, "args", None) or getattr(part_fc, "arguments", None) or {}
        )
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        tools = {
            "search_by_text": search_by_text,
            "search_by_category": search_by_category,
        }
        func = tools.get(name)
        if not func:
            return []
        sig = inspect.signature(func)
        call_kwargs = {k: v for k, v in args.items() if k in sig.parameters}
        return func(**call_kwargs)
    except Exception:
        return []


def _fingerprint_function_call(part_fc) -> str:
    try:
        name = (
            getattr(part_fc, "name", None)
            or getattr(part_fc, "function_name", None)
            or ""
        )
        args = (
            getattr(part_fc, "args", None) or getattr(part_fc, "arguments", None) or {}
        )
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if not isinstance(args, dict):
            try:
                args = dict(args)
            except Exception:
                args = {"value": str(args)}
        return name + "::" + json.dumps(args, sort_keys=True, default=str)
    except Exception:
        return ""


def _gather_existing_tool_fps(messages) -> set[str]:
    fps = set()
    try:
        for msg in messages:
            for part in getattr(msg, "parts", []) or []:
                fc = getattr(part, "function_call", None)
                if fc:
                    fp = _fingerprint_function_call(fc)
                    if fp:
                        fps.add(fp)
    except Exception:
        pass
    return fps


with st.sidebar:

    model = st.selectbox(
        label="model",
        index=1,
        options=[
            "models/gemini-2.0-flash",
            "models/gemini-2.5-flash",
            "models/gemini-2.5-pro",
        ],
    )

    classify_tab, auth_tab = st.tabs(["Classify", "Fetch"])

    with classify_tab:
        st.title("Classify")

        sender = st.text_input(
            "Paste your email sender here.", value="example@gmail.com"
        )
        subject = st.text_input("Paste your email subject here.")
        body = st.text_area("Paste your email body here.", height="stretch")

        index = st.button("Index", type="primary", width="stretch")
        if index:
            pred = index_pasted_email(
                sender=sender,
                body=body,
                subject=subject,
            )
            col1, col2 = st.columns(2)
            col1.metric(
                "Category",
                pred["pred_category"].replace("_", " ").title(),
                pred["pred_category_p"],
                border=True,
            )
            col2.metric(
                "Importance",
                pred["pred_important"].replace("_", " ").title(),
                pred["pred_important_p"],
                border=True,
            )


    def _read_token_status(path: str = "token.json") -> tuple[str, str] | None:
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r") as f:
                data = json.load(f)
            exp = data.get("expiry")
            if not exp:
                return ("unknown", "No expiry in token.json")
            # Handle trailing Z
            try:
                if exp.endswith("Z"):
                    exp_dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                else:
                    exp_dt = datetime.fromisoformat(exp)
            except Exception:
                return ("unknown", f"Unparseable expiry: {exp}")
            # Compare as naive UTC to avoid aware/naive mismatches
            now = datetime.utcnow()
            exp_dt_naive = exp_dt.replace(tzinfo=None)
            # Treat token as expired if within 1 minute of expiry
            delta = (exp_dt_naive - now).total_seconds()
            if delta <= 60:
                return ("expired", exp_dt.isoformat())
            return ("valid", exp_dt.isoformat())
        except Exception as e:
            return ("unknown", str(e))

    with auth_tab:
        st.title("Fetch")
        st.caption(
            "Authenticate with Gmail and index recent emails into the vector DB."
        )

        creds_exists = os.path.exists("credentials.json")
        token_state = _read_token_status()

        if not creds_exists:
            st.error(
                "Missing credentials.json. Add your Google OAuth Desktop Client file next to app.py."
            )
        else:
            if token_state is None:
                st.warning("No token.json found. You will be asked to sign in.")
            else:
                status, detail = token_state
                if status == "valid":
                    st.success(f"Gmail token valid. Expires at {detail} UTC.")
                elif status == "expired":
                    st.warning(f"Token expired or near expiry (at {detail} UTC). Will refresh.")
                else:
                    st.info(f"Token status unknown: {detail}")

        limit = st.number_input(
            "How many recent emails to fetch",
            min_value=100,
            max_value=10000,
            value=2000,
            step=100,
            help="Newest first; batches of 100 for classification and indexing.",
        )

        fetch = st.button(
            "Fetch & Index from Gmail",
            type="primary",
            use_container_width=True,
            help="Handles auth (sign-in or refresh) and runs the indexer.",
        )

        if fetch:
            if not creds_exists:
                st.error("credentials.json not found. Cannot authenticate with Gmail.")
            else:
                try:
                    with st.spinner("Authenticating with Gmail..."):
                        # Triggers refresh or OAuth flow as needed and writes token.json
                        _ = get_gmail_service()
                    st.success("Authenticated with Gmail.")
                except Exception as e:
                    st.error(f"Auth failed: {e}")
                else:
                    try:
                        with st.spinner("Indexing emails (this may take a while)..."):
                            run_indexer(
                                limit=int(limit),
                                model_dir="outputs/modernbert_multitask",
                                persist_dir="./db",
                                collection_name="emails",
                            )
                        st.success("Done indexing. You can now search in the chat.")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

CONFIG = types.GenerateContentConfig(
    system_instruction=f"""You are an ai assistant that has access to the user's personal gmail emails in a vectorized db. Today is {datetime.now().strftime('%Y-%m-%d')}.
The emails in that db have also been classified by a custom model that predicts email categories (out of {Category._member_names_}) and whether that email is a priority email (important). This was done to help you with retrieval.
Try to be as useful as you can to the user. Answering it's questions while making sure to attribute your findings.
When you're asked a question, you'll first try to retrieve the relevant emails, read through them and then answer carefully. Make sure to quote yourself to your sources and try to always mention the email category + importance if you're listing them. Do not give answers that are too short.""",
    thinking_config=(
        None if "2.0" in model else types.ThinkingConfig(thinking_budget=-1)
    ),
    tools=[
        search_by_text,
        search_by_category,
    ],
)


# Streamed response emulator
def response_generator(messages=None):
    added_toolchain = False
    collected_sources = []
    # Reset per-turn sources and establish baseline of prior tool calls
    st.session_state._last_sources = []
    baseline_fps = st.session_state.get("_baseline_tool_fps", set()) or set()
    seen_fps = set()
    for chunk in get_gemini_client().models.generate_content_stream(
        model=model,
        config=CONFIG,
        contents=messages or st.session_state.messages,
    ):
        if not added_toolchain and (
            history := chunk.automatic_function_calling_history
        ):
            toolchain = []
            for content in history:
                for part in content.parts:
                    if part.function_call or part.function_response:
                        toolchain.append(content)
                        # Collect sources by replaying only new function calls
                        if part.function_call:
                            fp = _fingerprint_function_call(part.function_call)
                            if fp and (fp not in baseline_fps) and (fp not in seen_fps):
                                seen_fps.add(fp)
                                results = _call_tool_from_function_call(part.function_call) or []
                                for r in results:
                                    em = _as_email(r)
                                    if em is not None:
                                        collected_sources.append(em)
                        break
            if "messages" in st.session_state:
                st.session_state.messages += toolchain
                added_toolchain = True
            # Persist sources for the upcoming assistant message (dedup by id)
            norm = []
            seen = set()
            for em in collected_sources:
                try:
                    key = em.id if isinstance(em, Email) else None
                except Exception:
                    key = None
                if not key:
                    # Build a robust tuple key
                    try:
                        sender = em.sender if isinstance(em, Email) else None
                        subject = em.subject if isinstance(em, Email) else None
                        datev = em.date if isinstance(em, Email) else None
                        key = (sender, subject, datev)
                    except Exception:
                        key = id(em)
                if key in seen:
                    continue
                seen.add(key)
                norm.append(em if isinstance(em, Email) else em)
            st.session_state._last_sources = norm
        if chunk.text:
            yield chunk.text


st.title("Gmail Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_by_turn" not in st.session_state:
    st.session_state.sources_by_turn = []

# Display chat messages from history on app rerun
model_turn_index = 0
for message in st.session_state.messages:
    if any(p.text is not None for p in message.parts):
        with st.chat_message("assistant" if message.role == "model" else "user"):
            # Render sources above assistant/model answers when available
            if message.role == "model":
                if model_turn_index < len(st.session_state.sources_by_turn):
                    _render_sources_panel_v2(
                        st.session_state.sources_by_turn[model_turn_index]
                    )
                model_turn_index += 1
            st.markdown("".join(p.text for p in message.parts if p.text is not None))

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Establish baseline fingerprints of any prior tool calls
        st.session_state._baseline_tool_fps = _gather_existing_tool_fps(
            st.session_state.messages
        )
        # Ensure no carry-over sources before stream begins
        st.session_state._last_sources = []
        # Placeholder for sources to appear above the live streamed text
        sources_ph = st.empty()

        def _stream_with_live_sources():
            rendered = False
            for chunk in response_generator():
                # As soon as sources are collected during streaming, render them once
                if (not rendered) and st.session_state.get("_last_sources"):
                    with sources_ph.container():
                        _render_sources_panel_v2(st.session_state.get("_last_sources", []))
                    rendered = True
                yield chunk

        response = st.write_stream(_stream_with_live_sources())
    # Add assistant response to chat history
    st.session_state.messages.append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=response)],
        )
    )
    # Attach sources collected during streaming to this turn
    st.session_state.sources_by_turn.append(st.session_state.pop("_last_sources", []))
    st.session_state.pop("_baseline_tool_fps", None)
