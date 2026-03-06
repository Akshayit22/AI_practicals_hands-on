"""
PDF RAG with Azure OpenAI + Azure AI Search
--------------------------------------------
Builds a RAG pipeline for PDF files (including large books) using:
  - PyMuPDF (fitz)  : PDF text extraction with page metadata
  - Azure OpenAI    : embeddings (text-embedding-3-large) + chat (gpt-4o)
  - Azure AI Search : vector store

Usage
-----
1. Fill in your credentials in the CONFIG section below.
2. First run  : uses index_documents() — extracts, chunks, indexes PDF
3. Later runs : uses load_index()      — skips re-indexing
4. Run        : python pdf_rag_ai_search.py
"""

import os
import fitz  # PyMuPDF
from datetime import datetime

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import AzureSearch
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# CONFIG — fill in your Azure credentials here
# ─────────────────────────────────────────────

# Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"]  = "***REMOVED-AZURE-OPENAI-KEY***"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://akstel395.openai.azure.com/"

AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"   # your embedding deployment name
AZURE_CHAT_DEPLOYMENT      = "gpt-4o"                   # your chat deployment name
AZURE_API_VERSION          = "2024-08-01-preview"

# Azure AI Search
os.environ["AZURE_SEARCH_ENDPOINT"] = "https://akshay395.search.windows.net"
os.environ["AZURE_SEARCH_KEY"]      = "***REMOVED-AZURE-SEARCH-KEY***"

AZURE_SEARCH_INDEX = "pdf-rag-index-policy-doc"   # auto-created on first run

# PDF files — add as many as needed, all go into the same index
PDF_FILES = [
    # "data/Famous old receipts.pdf",
    "data/Misconduct_Under_Presidio_India_HR_Policy.pdf",
]

# ─────────────────────────────────────────────
# STEP 1 — Load PDF and extract text per page
# ─────────────────────────────────────────────

def prepare_pdf(file_path: str) -> list[Document]:
    """
    Extract text from each PDF page using PyMuPDF.
    Each page becomes one Document with page number stored in metadata.
    """
    docs = []
    pdf = fitz.open(file_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text().strip()

        if not text:          # skip blank pages (images-only pages)
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": page_num + 1,          # 1-based page number
                    "total_pages": len(pdf),
                }
            )
        )

    pdf.close()
    print(f"[load]  {len(docs)} pages extracted from '{file_path}'")
    return docs


def prepare_multiple_pdfs(file_paths: list[str]) -> list[Document]:
    """Load multiple PDF files and combine into a single list of Documents."""
    all_docs = []
    for file_path in file_paths:
        all_docs.extend(prepare_pdf(file_path))
    print(f"[load]  {len(all_docs)} total pages across {len(file_paths)} PDF(s)")
    return all_docs


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
    chunk_size: int = 1500,      # larger than CSV — book sentences need more context
    chunk_overlap: int = 150,    # higher overlap — avoid cutting mid-sentence
) -> AzureSearch:
    """Chunk pages → embed → push to Azure AI Search index."""
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
# STEP 5 — RAG: retrieve + generate answer with page citations
# ─────────────────────────────────────────────

MAX_HISTORY_TURNS = 10


def generate_answer(
    vector_store: AzureSearch,
    query: str,
    chat_history: list,
    k: int = 8,
) -> str:
    """
    Retrieve top-k chunks and generate an answer with page citations and chat memory.
    """
    retrieved = vector_store.similarity_search_with_relevance_scores(query, k=k)

    # Print retrieved sources
    # print("\n[retrieved sources]")
    # for doc, score in retrieved:
    #     print(f"  {doc.metadata.get('source')} | Page {doc.metadata.get('page')} | score={score:.2f} | {doc.page_content[:80].strip()}...")

    # Build context with file + page labels
    context = "\n\n".join(
        f"[{doc.metadata.get('source')} — Page {doc.metadata.get('page')}]:\n{doc.page_content}"
        for doc, _ in retrieved
    )

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant analyzing PDF documents.\n"
            "Use the retrieved context to answer questions directly and clearly.\n"
            "Cite the source file and page number(s) in your answer (e.g. 'According to airbnb-deck.pdf, page 5...').\n"
            "If nothing relevant exists in the context, say 'I don't know.'"
        ),
    }

    user_message = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}",
    }

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
    """Keep asking the user for questions until they type 'exit' or 'quit'."""
    print("\n=== PDF RAG Chat Ready ===")
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

        answer = generate_answer(vector_store, query, chat_history)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] Answer:\n{answer}")
        print("-" * 60)

        # Save turn to history with timestamps
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({"role": "user",      "content": f"[{ts}] {query}"})
        chat_history.append({"role": "assistant", "content": f"[{ts}] {answer}"})

        # Trim to last MAX_HISTORY_TURNS
        chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # --- Load multiple PDFs ---
    # docs = prepare_multiple_pdfs(PDF_FILES)

    # --- First run: index documents ---
    # vector_store = index_documents(docs)

    # --- Later runs: load existing index ---
    vector_store = load_index()

    # --- Interactive chat loop ---
    chat_loop(vector_store)
#claude --resume 3e1e0a45-1bae-4e8e-98b4-6fa662a0fa00