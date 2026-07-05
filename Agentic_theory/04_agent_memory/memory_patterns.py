"""
Agent Memory — buffer, windowed, summary, and long-term (persisted) memory
---------------------------------------------------------------------------
Read Agentic_AI/04_agent_memory/README.md first.

PARTS 1-3 (short-term memory) run every time you execute the script and
show three strategies for managing a growing conversation history.

PART 4 (long-term memory) is deliberately designed to be run TWICE:
  - 1st run : no saved memory found -> saves a few facts to a FAISS index
              on disk, then exits, telling you to run it again.
  - 2nd run : finds the saved index -> loads it and retrieves the facts
              relevant to a brand-new question, in a brand-new process.
This is the only way to honestly demonstrate "memory survives past this
process exiting" -- anything shown within a single run would just be the
in-memory list from Parts 1-3 again.

Requires: OPENAI_API_KEY in your environment.
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────
# 1. Buffer memory — keep everything (simplest, doesn't scale)
# ─────────────────────────────────────────────

def demo_buffer_memory():
    history = []

    def chat(user_input: str) -> str:
        history.append({"role": "user", "content": user_input})
        resp = client.chat.completions.create(model=MODEL, messages=history)
        reply = resp.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        return reply

    chat("My name is Priya and I work on the payments team.")
    reply = chat("What team do I work on?")
    print(f"  Buffer memory recalled: {reply}")
    print(f"  Buffer size after 2 turns: {len(history)} messages")
    print("  -> Keeps growing forever. Fine for short chats, expensive/")
    print("     eventually over the context limit for long ones.")


# ─────────────────────────────────────────────
# 2. Windowed memory — keep only the last k turns
# ─────────────────────────────────────────────

def demo_windowed_memory(window_size: int = 2):
    history = []

    def chat(user_input: str) -> str:
        history.append({"role": "user", "content": user_input})
        # Only send the model the last `window_size` *pairs* of messages.
        windowed = history[-(window_size * 2):]
        resp = client.chat.completions.create(model=MODEL, messages=windowed)
        reply = resp.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        return reply

    chat("My name is Priya and I work on the payments team.")
    chat("I like my coffee black.")
    chat("My favorite programming language is Go.")
    # By now, with window_size=2, the model can only see the last 2 turns --
    # the very first message (the name) has fallen out of the window.
    reply = chat("What is my name?")
    print(f"  Windowed memory (window={window_size} turns) recalled: {reply}")
    print("  -> Cheap and simple, but forgets anything older than the window.")


# ─────────────────────────────────────────────
# 3. Summary memory — compress old turns instead of dropping them
# ─────────────────────────────────────────────

def summarize(messages: list[dict]) -> str:
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"Summarize the key facts from this conversation in one short sentence:\n\n{transcript}",
        }],
    )
    return resp.choices[0].message.content


def demo_summary_memory():
    raw_history = [
        {"role": "user", "content": "My name is Priya and I work on the payments team."},
        {"role": "assistant", "content": "Nice to meet you, Priya!"},
        {"role": "user", "content": "I like my coffee black."},
        {"role": "assistant", "content": "Noted."},
    ]
    # Instead of dropping the old turns (windowed) or keeping them raw
    # (buffer), compress them into one summary message.
    summary = summarize(raw_history)
    print(f"  Compressed summary: {summary}")

    # From here on, only the summary + new turns get sent -- much cheaper
    # than the raw transcript, but (unlike windowed memory) the old facts
    # aren't silently lost, just compressed.
    messages = [
        {"role": "system", "content": f"Conversation summary so far: {summary}"},
        {"role": "user", "content": "What team do I work on again?"},
    ]
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    print(f"  Recalled from summary: {resp.choices[0].message.content}")


# ─────────────────────────────────────────────
# 4. Long-term memory — persisted across sessions with a vector store
#    (this is RAG, applied to "facts about the user" instead of documents)
# ─────────────────────────────────────────────

MEMORY_INDEX_DIR = os.path.join(os.path.dirname(__file__), "long_term_memory_index")


def demo_long_term_memory():
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    embeddings = OpenAIEmbeddings()

    if not os.path.exists(MEMORY_INDEX_DIR):
        # "Session 1" -- nothing saved yet. Store a few facts and stop.
        facts = [
            "The user's name is Priya.",
            "Priya works on the payments team.",
            "Priya prefers concise, bullet-point answers.",
            "Priya's timezone is IST (India Standard Time).",
        ]
        docs = [Document(page_content=f) for f in facts]
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(MEMORY_INDEX_DIR)
        print("  No prior memory found -- this is session 1.")
        print(f"  Saved {len(facts)} facts to disk at: {MEMORY_INDEX_DIR}")
        print("  Run this script again to see them recalled in a new process.")
        return

    # "Session 2" (or later) -- load the persisted index from disk. Note
    # this is a totally fresh Python process; nothing in memory carried
    # over except what's sitting on disk.
    db = FAISS.load_local(
        MEMORY_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    query = "What do I usually ask you to keep in mind when answering?"
    results = db.similarity_search(query, k=2)

    print("  Prior memory found on disk -- this is a later session.")
    print(f"  Query: {query!r}")
    print("  Recalled facts (via similarity search, not exact match):")
    for r in results:
        print(f"    - {r.page_content}")


if __name__ == "__main__":
    print("=" * 70)
    print("1. BUFFER MEMORY")
    print("=" * 70)
    demo_buffer_memory()

    print()
    print("=" * 70)
    print("2. WINDOWED MEMORY")
    print("=" * 70)
    demo_windowed_memory(window_size=2)

    print()
    print("=" * 70)
    print("3. SUMMARY MEMORY")
    print("=" * 70)
    demo_summary_memory()

    print()
    print("=" * 70)
    print("4. LONG-TERM MEMORY (persisted across process restarts)")
    print("=" * 70)
    demo_long_term_memory()
