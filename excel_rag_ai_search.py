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
import pandas as pd

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import AzureSearch
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# CONFIG — fill in your Azure credentials here
# ─────────────────────────────────────────────

# Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"]  = "your-azure-openai-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"

AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"   # your embedding deployment name
AZURE_CHAT_DEPLOYMENT      = "gpt-4o"                   # your chat deployment name
AZURE_API_VERSION          = "2024-08-01-preview"

# Azure AI Search
os.environ["AZURE_SEARCH_ENDPOINT"] = "https://your-search-service.search.windows.net"
os.environ["AZURE_SEARCH_KEY"]      = "your-azure-search-admin-key"

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

def generate_answer(vector_store: AzureSearch, query: str, k: int = 5) -> str:
    """Retrieve top-k chunks from Azure AI Search and generate an answer."""
    retrieved = vector_store.similarity_search_with_relevance_scores(query, k=k)
    context = "\n\n".join(doc.page_content for doc, _score in retrieved)

    prompt = f"""Answer the question below using only the provided context.
If the answer is not in the context, say "I don't know."

Question: {query}

Context:
{context}
"""
    model = AzureChatOpenAI(
        azure_deployment=AZURE_CHAT_DEPLOYMENT,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=AZURE_API_VERSION,
    )
    response = model.invoke(prompt)

    usage = response.response_metadata.get("token_usage", {})
    print(
        f"[tokens] prompt={usage.get('prompt_tokens')} | "
        f"completion={usage.get('completion_tokens')} | "
        f"total={usage.get('total_tokens')}"
    )
    return response.content


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # --- Load data ---
    docs = prepare_data(DATA_FILE)

    # --- First run: index documents ---
    vector_store = index_documents(docs)

    # --- Later runs: load existing index (comment out index_documents above) ---
    # vector_store = load_index()

    # --- Ask questions ---
    queries = [
        "What are the best reviews?",
        "Give me reviews with low scores and negative feedback.",
        "List reviews where people suggest improvements.",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        print("A:", generate_answer(vector_store, q))
        print("-" * 60)
