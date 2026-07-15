"""
Excel RAG with Azure OpenAI + Azure AI Search
----------------------------------------------
Builds a production RAG pipeline for Excel/CSV files using:
  - Azure OpenAI  : embeddings (text-embedding-3-large) + chat (gpt-4o)
  - Azure AI Search : vector store (replaces local Chroma)

Usage
-----
1. Fill in your credentials in the CONFIG section below.
2. First run  : uncomment `vector_store = index_documents(docs)` — indexes data
3. Later runs : use `vector_store = load_index()` — skips re-indexing
4. Run        : python excel_rag_ai_search.py
"""

import os
import yaml
import pandas as pd
from datetime import datetime

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import AzureSearch
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# Load prompts from config
# ─────────────────────────────────────────────

_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "config", "prompts.yaml")
with open(_PROMPTS_PATH) as f:
    _PROMPTS = yaml.safe_load(f)["prompts"]["excel_rag"]

SYSTEM_PROMPT        = _PROMPTS["system"].strip()
USER_PROMPT_TEMPLATE = _PROMPTS["user"].strip()

# ─────────────────────────────────────────────
# CONFIG — credentials loaded from environment / .env
# ─────────────────────────────────────────────
# Copy .env.example to .env and fill in your Azure credentials.
# Never hardcode secrets in this file.

from dotenv import load_dotenv

load_dotenv()

_REQUIRED_ENV = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_KEY",
]
_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing)}. "
        "Copy .env.example to .env and fill in your Azure credentials."
    )

AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"   # your embedding deployment name
AZURE_CHAT_DEPLOYMENT      = "gpt-4o"                   # your chat deployment name
AZURE_API_VERSION          = "2024-08-01-preview"

AZURE_SEARCH_INDEX = "food-reviews-index"   # index name — auto-created on first run

# Data file
DATA_FILE = "data/food_reviews_1k.csv"      # supports .csv and .xlsx


# ─────────────────────────────────────────────
# STEP 1 — Load Excel / CSV file
# ─────────────────────────────────────────────

def prepare_data(file_path: str, max_rows: int = None) -> list[Document]:
    """Load a CSV or Excel file and convert each row into a LangChain Document."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    if max_rows:
        df = df.head(max_rows)

    docs = [
        Document(
            page_content="\n".join(f"{col}: {val}" for col, val in row.items()),
            metadata={"row_index": int(i)}
        )
        for i, row in df.iterrows()
    ]
    print(f"[load]  {len(docs)} rows loaded from '{file_path}'")
    return docs


# ─────────────────────────────────────────────
# STEP 2 — Shared embeddings builder
# ─────────────────────────────────────────────

def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=AZURE_API_VERSION,
        chunk_size=16,
    )


# ─────────────────────────────────────────────
# STEP 3 — Index documents into Azure AI Search
#          Run ONCE when data changes
# ─────────────────────────────────────────────

def index_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> AzureSearch:
    """Chunk docs → embed → push to Azure AI Search index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"[index] {len(chunks)} chunks created — uploading to Azure AI Search...")

    embeddings = get_embeddings()

    vector_store = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_search_key=os.environ["AZURE_SEARCH_KEY"],
        index_name=AZURE_SEARCH_INDEX,
        embedding_function=embeddings.embed_query,
    )
    vector_store.add_documents(chunks)
    print(f"[index] Done — {len(chunks)} chunks indexed into '{AZURE_SEARCH_INDEX}'")
    return vector_store


# ─────────────────────────────────────────────
# STEP 4 — Load existing index (skip re-indexing)
#          Use on all subsequent runs
# ─────────────────────────────────────────────

def load_index() -> AzureSearch:
    """Connect to an already-populated Azure AI Search index."""
    embeddings = get_embeddings()
    vector_store = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_search_key=os.environ["AZURE_SEARCH_KEY"],
        index_name=AZURE_SEARCH_INDEX,
        embedding_function=embeddings.embed_query,
    )
    print(f"[load]  Connected to existing index '{AZURE_SEARCH_INDEX}'")
    return vector_store


# ─────────────────────────────────────────────
# STEP 5 — RAG: retrieve + generate answer
# ─────────────────────────────────────────────

MAX_HISTORY_TURNS = 10   # keep last N turns (each turn = 1 user + 1 assistant message)


def generate_answer(
    vector_store: AzureSearch,
    query: str,
    chat_history: list,
    k: int = 5,
) -> str:
    """Retrieve top-k chunks and generate an answer, passing chat history for memory."""
    retrieved = vector_store.similarity_search_with_relevance_scores(query, k=k)
    context = "\n\n".join(doc.page_content for doc, _ in retrieved)

    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }

    # Current user message — template filled with retrieved context + query
    user_message = {
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(context=context, query=query),
    }

    # Build full message list: system + trimmed history + current question
    messages = [system_message] + chat_history + [user_message]

    model = AzureChatOpenAI(
        azure_deployment=AZURE_CHAT_DEPLOYMENT,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=AZURE_API_VERSION,
    )
    response = model.invoke(messages)

    usage = response.response_metadata.get("token_usage", {})
    print(
        f"[tokens] prompt={usage.get('prompt_tokens')} | "
        f"completion={usage.get('completion_tokens')} | "
        f"total={usage.get('total_tokens')}"
    )
    return response.content


# ─────────────────────────────────────────────
# STEP 6 — Interactive chat loop with memory
# ─────────────────────────────────────────────

def chat_loop(vector_store: AzureSearch) -> None:
    """Keep asking the user for questions until they type 'exit' or 'quit'.
    Maintains chat history as a context window across turns.
    """
    print("\n=== RAG Chat Ready ===")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    chat_history = []   # stores previous turns as {role, content} dicts

    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        query = input(f"[{ts}] You: ").strip()

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        answer = generate_answer(vector_store, query, chat_history)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] Answer:\n{answer}")
        print("-" * 60)

        # Save this turn to history with timestamps
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({"role": "user",      "content": f"[{ts}] {query}"})
        chat_history.append({"role": "assistant", "content": f"[{ts}] {answer}"})

        # Trim to last MAX_HISTORY_TURNS turns to avoid token overflow
        chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # --- Load data ---
    # docs = prepare_data(DATA_FILE, 20)

    # --- First run: index documents ---
    # vector_store = index_documents(docs)

    # --- Later runs: load existing index (comment out index_documents above) ---
    vector_store = load_index()
    print(vector_store)

    # --- Interactive chat loop ---
    chat_loop(vector_store)
