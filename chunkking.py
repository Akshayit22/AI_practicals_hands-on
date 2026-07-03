

# Semantic Chunking
# SemanticChunker splits text at semantically meaningful boundaries using embeddings.
# Sentences that are topically similar are grouped into the same chunk.

# !pip install langchain-experimental -q

import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_aws import BedrockEmbeddings

# --- Bedrock API Key (no IAM credentials needed) ---
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "bedrock-api-key-YmVkcm9jay5hbWF6b25hd3MuY29tLz9BY3Rpb249Q2FsbFdpdGhCZWFyZXJUb2tlbiZYLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFTSUE0WTJZSVZSNFpNRUdUNUlUJTJGMjAyNjAzMTMlMkZ1cy1lYXN0LTElMkZiZWRyb2NrJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjAzMTNUMDkwMjQyWiZYLUFtei1FeHBpcmVzPTQzMjAwJlgtQW16LVNlY3VyaXR5LVRva2VuPUlRb0piM0pwWjJsdVgyVmpFTUglMkYlMkYlMkYlMkYlMkYlMkYlMkYlMkYlMkYlMkZ3RWFDWFZ6TFdWaGMzUXRNU0pHTUVRQ0lHUWRmdlhUQzBndDJIVjkxeEluY3VCNlF1Sjc4NmtOazBjdmtxMEpJJTJCeXJBaUFpRWhMOHJKQiUyRmpVNkpUcjdVNnpmb2ZzV2l1VEduc2xhYURyM1BpRmNabXlxOUJBaUslMkYlMkYlMkYlMkYlMkYlMkYlMkYlMkYlMkYlMkY4QkVBSWFERGczTnprMk9UQTFPRGt6TnlJTXBzQ2xsamZiS2h6MlVvNkNLcEVFN3RYNEdRaW0xaDlzekclMkJYdG9zaGYwV0Q1c1BNVXdxdlFxRSUyQmclMkZTUyUyQmI1bUg0MlBJYWJ3b25QWDFUVzJMSmd1JTJGJTJCNVhpamVsJTJGdTZWbUw2JTJGM0pIZm1pelhDUXhWZmh0Z3d3dSUyQnRhT1JYUEg3T3FndW5yWWg5eFFmdWx3a3M3NlpkOUdxVWdlSTJmSVVYd01NYkdkUTB6RjJpaDNPblA2RUZmNFJBMXJXbjRJc3NhSXRQTEg1VUVtQ2s4amgwN0tlQ1doT0NtVjZTSnlPcVdibFp0RkNUSiUyQmJIS3RvSWNlcjVON0s3dmRBVzQxeldGajhaQ09mYWhFVlp0NVd3WTJOYmRXZmdkRmJYNWZMWm1SY1h3bUpMQmhFN0VOcSUyRkQyTWc1Mk9adEZkYnQzT3FPVWpPTkJYNWplS1FaJTJCJTJGelpQcU8wZkdWTElySWs3dzhjWGdidHFWNWJjVlF1a3IyczVJRnVXRSUyQnElMkZHJTJGZFZzMFglMkI1TEdhRTQlMkJvRVJwNkd3U244TGdneEtYa0lBQ3N0ME5MTTZ5N1UzUWhOOWdKNHR6ZSUyRkFjTVVGaXFUTG9UaTYzT0hUcVVSd2wxQ3hFamNNMExzcSUyRkglMkZ6MjZyMHNtYkZoOFRKUGdBaW9NJTJCTVRYZHpmYTYyQ2FEWU5JQlBqUm4zOFU2aXM4MUxqM09rdjZneXg3cEVLQzE3dG1acUtUa0dOOE9vUnJlZ214ciUyQjBFQ2olMkZVOVdUbElxbmJoJTJGMTlwM0k5ZiUyRkFjbUZNdXZlJTJGdXFuNnZwMDFqeFZ0Y3VwOUhCQk83Zkg4QjA0Q0xWOUNtOEY1dGM1TnlqMTBXM1NMS0ZrZWhzdSUyQlkwNENlUFhzU0c5Q21FZDFqdjdseTlvVUU0bnJvSDZPWUlYRXJKZWQ1d05xMUVzUVdRb3h0Wml2YSUyRjlBZkZjb3Z4MWhNNVoyaDBSY1o0VFREQm84JTJGTkJqckVBamtqT3VMejNSMjl1cWVsb3QzeGI2QWVKZHNuTEJTUmZReThKbGlRRzJvS3JvNW1KdWVLVkdLQU5QRGdIdW1ObG9kSHIlMkZiZWFyRHUwZ0xzQVg0aXZMTSUyRkppRDVjcDk1dldIbkNSJTJGdjU2JTJCJTJGSGhnRjFxTyUyRkJoalFmNUpPclVCZXpyYlJ6bVI5bXglMkZ6TmQlMkZObjh5TVgzd0RzbkNrVHF2ZlAyUjJJc0w3Y0FzY0tmcHklMkJNZHZkaCUyRlJ6UWs1NjZuVFdGaGRyYkhhcGlNYTZncWtGc0wlMkY4cDRiTnprVVBKck9NaFZqMmN4U3pIOElRWlFRYSUyRmNhMHclMkZGTlBzS0UyTVM0YTVCVVNkSXVLVnQyaG1vb0J6RVBXZnBCdSUyQmElMkJYUWVBT0I4Ykc1T1JFSDVUZGJVUnpvYmtxT1ZsSjNUOExxU2hGZkxMRXVRdHk0SEh6UWcydEdBbVkwakdpcWYzU1A3aGUwN1dpcHVIVGVyY0QyenIxMnM1M1MyMnJLVG5FdGFGVXZ1cGJ5eVNqdyUyQnlVRDBrV2Q4YzhBbzJvcFpuODNqbkk5JTJGcVAlMkI5WVRseDdjWExiQSUzRCUzRCZYLUFtei1TaWduYXR1cmU9Y2Q2ZTI2MGQxYjRmYzdkNWQzZWE3NTczYmE4YmYwYzkzYTNlMjVmMjg2NzE2M2M0M2MwNjU4MzE4YmY4ODVjYiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmVmVyc2lvbj0x"

# --- Sample paragraph covering 3 distinct topics ---
sample_text = """
The solar system consists of the Sun and the objects that orbit it, including eight planets,
their moons, and various smaller bodies such as asteroids and comets. Jupiter is the largest
planet, while Mercury is the smallest. Earth is the only known planet to support life.Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed. Supervised learning uses labeled data to train models,
while unsupervised learning finds hidden patterns in unlabeled data. Deep learning uses neural
networks with many layers to solve complex problems.The French Revolution began in 1789 and fundamentally transformed France's political landscape.
It led to the rise of Napoleon Bonaparte and spread democratic ideals across Europe. The storming of the Bastille on July 14, 1789 became a powerful symbol of the uprising against monarchy.
"""

AWS_REGION      = "us-east-1"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

def semantic_chunking():
    # --- Initialize SemanticChunker with AWS Bedrock embeddings ---
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL,
        region_name=AWS_REGION,
    )

    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=0.3,
    )

    # --- Split and print chunks ---
    chunks = semantic_splitter.create_documents([sample_text])

    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content.strip())
        print()
        


# ─────────────────────────────────────────────
# TOKEN-BASED CHUNKING
# Splits by fixed token count using a real transformer model tokenizer.
# Each chunk contains exactly chunk_size tokens (model-accurate, not char-estimate).
# Uses HuggingFace AutoTokenizer — same tokenizer the embedding model uses.
# ─────────────────────────────────────────────

from transformers import AutoTokenizer

def token_based_chunking(chunk_size=55, chunk_overlap=0):
    # Load transformer model tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Convert text → token IDs
    token_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    # Split token IDs into fixed-size windows with overlap
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += chunk_size - chunk_overlap

    print(f"=== TOKEN-BASED (Transformer: all-MiniLM-L6-v2) | Total chunks: {len(chunks)} ===\n")
    for i, chunk in enumerate(chunks):
        token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
        print(f"--- Chunk {i+1} ({token_count} tokens) ---")
        print(chunk.strip())
        print()


# ─────────────────────────────────────────────
# CONTENT-AWARE CHUNKING
# Splits on document structure: headers, paragraphs, newlines.
# Tries larger separators first, falls back to smaller ones.
# Best for structured docs (markdown, reports, HTML).
# ─────────────────────────────────────────────

from langchain_text_splitters import RecursiveCharacterTextSplitter

structured_text = """
# Solar System
The solar system consists of the Sun and the objects that orbit it, including eight planets,
their moons, and various smaller bodies. Jupiter is the largest planet.

## Machine Learning
Machine learning is a branch of artificial intelligence. Supervised learning uses labeled data.
Unsupervised learning finds hidden patterns. Deep learning uses neural networks.

## French Revolution
The French Revolution began in 1789. It led to the rise of Napoleon Bonaparte.
The storming of the Bastille became a symbol of the uprising against monarchy.
"""

def content_aware_chunking():
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n# ", "\n\n", "\n", " "],  # tries these in order
        chunk_size=200,
        chunk_overlap=100,
    )
    chunks = splitter.create_documents([structured_text])

    print(f"=== CONTENT-AWARE | Total chunks: {len(chunks)} ===\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content.strip())
        print()


# ─────────────────────────────────────────────
# AGENTIC CHUNKING
# Uses an LLM to decide chunk boundaries intelligently.
# The LLM reads the text and groups sentences into
# propositions (self-contained facts), then clusters them.
# Most accurate but slowest — uses LLM API calls.
# ─────────────────────────────────────────────

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

BEDROCK_CHAT_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

def agentic_chunking():
    llm = ChatBedrock(
        model_id=BEDROCK_CHAT_MODEL,
        region_name=AWS_REGION,
        model_kwargs={"max_tokens": 1024},
    )

    prompt = f"""Split the following text into logical, self-contained chunks.
Each chunk should cover one distinct topic or idea.
Return ONLY the chunks separated by the delimiter: ---CHUNK---
Text:
{sample_text}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw_chunks = response.content.strip().split("---CHUNK---")
    chunks = [c.strip() for c in raw_chunks if c.strip()]

    print(f"=== AGENTIC (LLM-based) | Total chunks: {len(chunks)} ===\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # print("\n" + "="*60)
    # semantic_chunking()

    # print("\n" + "="*60)
    # token_based_chunking()

    # print("\n" + "="*60)
    # content_aware_chunking()

    print("\n" + "="*60)
    agentic_chunking()