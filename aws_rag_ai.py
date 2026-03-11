"""
AWS RAG with Amazon Bedrock + Amazon OpenSearch Serverless
----------------------------------------------------------
Unified RAG pipeline supporting both PDF and Excel/CSV files using:
  - PyMuPDF (fitz)                      : PDF text extraction with page metadata
  - pandas                              : Excel/CSV row loading
  - Amazon Bedrock                      : embeddings (amazon.titan-embed-text-v2:0)
                                          + chat (anthropic.claude-3-5-sonnet)
  - Amazon OpenSearch Serverless (AOSS) : vector store

Usage
-----
1. Set DATA_MODE = "pdf" or "excel" in CONFIG below.
2. Fill in your AWS credentials (or rely on env vars / IAM role / ~/.aws/credentials).
3. First run  : uncomment index_documents() in __main__ — chunks, embeds, indexes data.
4. Later runs : use load_index() — connects to existing index, skips re-indexing.
5. Run        : python aws_rag_ai.py
"""

import os
import yaml
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime

import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# Load prompts from config
# ─────────────────────────────────────────────

_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "config", "prompts.yaml")
with open(_PROMPTS_PATH) as f:
    _PROMPTS = yaml.safe_load(f)["prompts"]["aws_rag"]

USER_PROMPT_TEMPLATE = _PROMPTS["user"].strip()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_MODE = "pdf"   # "pdf" or "excel"

# AWS credentials — leave empty strings to use env vars / IAM role / ~/.aws/credentials
AWS_REGION            = "us-east-1"
AWS_ACCESS_KEY_ID     = ""   # or set env var AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = ""   # or set env var AWS_SECRET_ACCESS_KEY

# Amazon Bedrock model IDs
BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
BEDROCK_CHAT_MODEL      = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# Amazon OpenSearch Serverless
OPENSEARCH_ENDPOINT = "https://<your-collection-id>.us-east-1.aoss.amazonaws.com"
OPENSEARCH_INDEX    = "rag-index"   # auto-created on first run

# Data files
PDF_FILES  = [
    "data/Misconduct_Under_Presidio_India_HR_Policy.pdf",
]
EXCEL_FILE = "data/food_reviews_1k.csv"   # supports .csv and .xlsx


# ─────────────────────────────────────────────
# STEP 1a — Load PDF files
# ─────────────────────────────────────────────

def prepare_pdf(file_path: str) -> list[Document]:
    """Extract text from each PDF page using PyMuPDF."""
    docs = []
    pdf = fitz.open(file_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text().strip()

        if not text:   # skip blank / image-only pages
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": page_num + 1,
                    "total_pages": len(pdf),
                    "type": "pdf",
                },
            )
        )

    pdf.close()
    print(f"[load]  {len(docs)} pages extracted from '{file_path}'")
    return docs


def prepare_multiple_pdfs(file_paths: list[str]) -> list[Document]:
    """Load multiple PDF files into a single document list."""
    all_docs = []
    for fp in file_paths:
        all_docs.extend(prepare_pdf(fp))
    print(f"[load]  {len(all_docs)} total pages across {len(file_paths)} PDF(s)")
    return all_docs


# ─────────────────────────────────────────────
# STEP 1b — Load Excel / CSV file
# ─────────────────────────────────────────────

def prepare_excel(file_path: str, max_rows: int = None) -> list[Document]:
    """Load a CSV or Excel file and convert each row into a Document."""
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    if max_rows:
        df = df.head(max_rows)

    docs = [
        Document(
            page_content="\n".join(f"{col}: {val}" for col, val in row.items()),
            metadata={
                "row_index": int(i),
                "source": os.path.basename(file_path),
                "type": "excel",
            },
        )
        for i, row in df.iterrows()
    ]
    print(f"[load]  {len(docs)} rows loaded from '{file_path}'")
    return docs


# ─────────────────────────────────────────────
# STEP 2 — AWS auth + embeddings
# ─────────────────────────────────────────────

def _get_aws_auth() -> AWS4Auth:
    """Build SigV4 auth for Amazon OpenSearch Serverless (service = aoss)."""
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID or None,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
        region_name=AWS_REGION,
    )
    creds = session.get_credentials().get_frozen_credentials()
    return AWS4Auth(
        creds.access_key,
        creds.secret_key,
        AWS_REGION,
        "aoss",
        session_token=creds.token,
    )


def get_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(
        model_id=BEDROCK_EMBEDDING_MODEL,
        region_name=AWS_REGION,
    )


# ─────────────────────────────────────────────
# STEP 3 — Index documents into OpenSearch Serverless
#          Run ONCE when data changes
# ─────────────────────────────────────────────

def index_documents(
    docs: list[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> OpenSearchVectorSearch:
    """Chunk docs → embed → push to Amazon OpenSearch Serverless."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"[index] {len(chunks)} chunks created — uploading to OpenSearch Serverless...")

    vector_store = OpenSearchVectorSearch.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        opensearch_url=OPENSEARCH_ENDPOINT,
        index_name=OPENSEARCH_INDEX,
        http_auth=_get_aws_auth(),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        is_aoss=True,
        bulk_size=500,
    )
    print(f"[index] Done — {len(chunks)} chunks indexed into '{OPENSEARCH_INDEX}'")
    return vector_store


# ─────────────────────────────────────────────
# STEP 4 — Load existing index (skip re-indexing)
#          Use on all subsequent runs
# ─────────────────────────────────────────────

def load_index() -> OpenSearchVectorSearch:
    """Connect to an already-populated OpenSearch Serverless index."""
    vector_store = OpenSearchVectorSearch(
        index_name=OPENSEARCH_INDEX,
        embedding_function=get_embeddings(),
        opensearch_url=OPENSEARCH_ENDPOINT,
        http_auth=_get_aws_auth(),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        is_aoss=True,
    )
    print(f"[load]  Connected to existing index '{OPENSEARCH_INDEX}'")
    return vector_store


# ─────────────────────────────────────────────
# STEP 5 — RAG: retrieve + generate answer
# ─────────────────────────────────────────────

MAX_HISTORY_TURNS = 10


def generate_answer(
    vector_store: OpenSearchVectorSearch,
    query: str,
    chat_history: list,
    mode: str,
    k: int = 5,
) -> str:
    """Retrieve top-k chunks and generate an answer with page citations and chat memory."""
    retrieved = vector_store.similarity_search_with_relevance_scores(query, k=k)

    if mode == "pdf":
        context = "\n\n".join(
            f"[{doc.metadata.get('source')} — Page {doc.metadata.get('page')}]:\n{doc.page_content}"
            for doc, _ in retrieved
        )
    else:
        context = "\n\n".join(doc.page_content for doc, _ in retrieved)

    system_message = {
        "role": "system",
        "content": _PROMPTS[f"{mode}_system"].strip(),
    }

    user_message = {
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(context=context, query=query),
    }

    messages = [system_message] + "\n" + chat_history + "\n"  + [user_message]

    model = ChatBedrock(
        model_id=BEDROCK_CHAT_MODEL,
        region_name=AWS_REGION,
        model_kwargs={"max_tokens": 1024},
    )
    response = model.invoke(messages)

    usage = response.response_metadata.get("usage", {})
    print(
        f"[tokens] input={usage.get('input_tokens')} | "
        f"output={usage.get('output_tokens')}"
    )
    return response.content


# ─────────────────────────────────────────────
# STEP 6 — Interactive chat loop with memory
# ─────────────────────────────────────────────

def chat_loop(vector_store: OpenSearchVectorSearch, mode: str) -> None:
    """Keep asking the user for questions until they type 'exit' or 'quit'."""
    print(f"\n=== AWS RAG Chat Ready ({mode.upper()} mode) ===")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    chat_history = []

    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        query = input(f"[{ts}] You: ").strip()

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        answer = generate_answer(vector_store, query, chat_history, mode)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] Answer:\n{answer}")
        print("-" * 60)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({"role": "user",      "content": f"[{ts}] {query}"})
        chat_history.append({"role": "assistant", "content": f"[{ts}] {answer}"})

        chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if DATA_MODE == "pdf":
        # --- First run: load + index ---
        docs = prepare_multiple_pdfs(PDF_FILES)
        vector_store = index_documents(docs, chunk_size=1500, chunk_overlap=150)

        # --- Later runs: load existing index ---
        vector_store = load_index()

    elif DATA_MODE == "excel":
        # --- First run: load + index ---
        docs = prepare_excel(EXCEL_FILE)
        vector_store = index_documents(docs, chunk_size=500, chunk_overlap=50)

        # --- Later runs: load existing index ---
        vector_store = load_index()

    chat_loop(vector_store, DATA_MODE)
