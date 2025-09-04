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
from indexer import (
    init_multitask_model,
    html_to_text_fast,
    classify_rows,
    embed_and_upsert,
)

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
    sender: str
    text: str | None
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
    k: int = 5,
) -> list[Email]:
    where = {}
    if pred_category and pred_important is not None:
        where = {
            "$and": [
                {"pred_category": pred_category},
                {"pred_important": pred_important},
            ]
        }
    else:
        if pred_category:
            where["pred_category"] = pred_category
        if pred_important is not None:
            where["pred_important"] = pred_important

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
                sender=meta["from"],
                text=text,
                predicted_class=meta["pred_category"],
                predicted_important=meta["pred_important"] == "important",
            )
        )
    return emails


def search_by_text(
    query: str,
    category_filter: str | None = None,
    important_filter: bool | None = None,
    limit: int = 5,
) -> list[Email]:
    """Semantic email search with optional category/importance filters.

    Args:
        query: User's natural-language query, e.g., 'invoices from ACME last quarter'.
        category_filter: Optional predicted category to restrict the candidate set.
        important_filter: Optional predicted flag to restrict the candidate set to only important to read emails.
        limit: Number of results to return. Default is 5, max are 20.

    Returns:
        A list of emails that match the query.
    """
    return retrieve(
        query=query,
        pred_category=category_filter,
        pred_important=important_filter,
        k=min(limit, 20),
    )


def search_by_category(
    category_filter: str,
    important_filter: bool = None,
    query: str = None,
    limit: int = 5,
) -> list[Email]:
    """Category-first retrieval with optional importance filter or semantic refinement.

    Args:
        category_filter: Predicted category to filter on.
        important_filter: Optional predicted flag to restrict the candidate set to only important to read emails.
        query: Optional natural-language query to refine within the class bucket, e.g., 'from ACME in Q2'. If omitted, results are ranked by recency.
        limit: Number of results to return. Default is 5, max are 20.

    Returns:
        A list of emails that match the query.
    """
    return retrieve(
        query=query,
        pred_category=category_filter,
        pred_important=important_filter,
        k=min(limit, 20),
    )


MODEL = "gemini-2.5-flash"
CONFIG = types.GenerateContentConfig(
    system_instruction=f"""You are an ai assistant that has access to the user's personal gmail emails in a vectorized db.
The emails in that db have also been classified by a custom model that predicts email categories (out of {Category._member_names_}) and whether that email is a priority email (important). This was done to help you with retrieval.
Try to be as useful as you can to the user. Answering it's questions while making sure to attribute your findings.
When you're asked a question, you'll first try to retrieve the relevant emails, read through them and then answer carefully.""",
    thinking_config=types.ThinkingConfig(thinking_budget=0),
    tools=[
        search_by_text,
        search_by_category,
    ],
)


# Streamed response emulator
def response_generator(messages=None):
    added_toolchain = False
    for chunk in get_gemini_client().models.generate_content_stream(
        model=MODEL,
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
                        break
            if "messages" in st.session_state:
                st.session_state.messages += toolchain
                added_toolchain = True
        if chunk.text:
            yield chunk.text


st.title("Gmail Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if any(p.text is not None for p in message.parts):
        with st.chat_message(message.role):
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
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=response)],
        )
    )


with st.sidebar:
    classify_tab, auth_tab = st.tabs(["Classify", "Authenticate"])

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
