# 4. Agent Memory — Short-Term vs. Long-Term

## The problem

Every agent loop you've built so far (modules 1–3) starts from an empty `messages` list every
time you call the function. Run the script twice — the second run knows nothing about the first.
That's fine for a demo, useless for a real assistant. "Memory" is just the general name for
*carrying state across turns (or across sessions)* so the agent doesn't relearn everything from
zero each time.

There are two genuinely different kinds of memory, solving two different problems:

## Short-term memory: the current conversation

This is just... the `messages` list you've already been building. The "memory" is nothing more
than not throwing away previous turns. The only real engineering problem here is **context window
limits** (you covered this in Week 1, `t307`) — a long conversation eventually won't fit in the
model's context window, and even if it does, stuffing in every single past message wastes tokens
and money on irrelevant history.

Three common strategies, in increasing sophistication:

1. **Buffer memory** — keep everything. Simplest, works fine for short conversations, breaks down
   as the conversation grows (cost + context limit).
2. **Windowed memory** — keep only the last *k* turns. Cheap and simple, but the agent forgets
   anything older than the window — bad if the user references something from 20 messages ago.
3. **Summary memory** — periodically ask the LLM to compress older turns into a short summary,
   and keep that summary + the last few raw turns. Balances cost against not forgetting entirely.

## Long-term memory: across sessions

Short-term memory dies when the process exits. Long-term memory means **persisting facts to
disk/a database** so a brand-new session can recall something from a week ago — "the user's
timezone is EST", "the user prefers concise answers." The most common implementation you already
have the tools for: embed facts (same embedding step from your RAG work) and store them in a
vector store (FAISS, same as your `aws_rag_ai.py`). Retrieving relevant memories for a new query is
then... a similarity search. **Long-term memory for agents is RAG, applied to a store of facts
about the user/conversation instead of a store of documents.** You already know how to build this.

## Run it

```bash
cd Agentic_AI/04_agent_memory
python memory_patterns.py
```

The script demonstrates, in order:
1. Buffer memory hitting the "too much context" problem directly.
2. Windowed memory forgetting something outside the window.
3. Summary memory retaining the gist of a long conversation cheaply.
4. Long-term memory: facts saved in one "session" (one Python process) and correctly recalled
   in a second, independent run of the script (loaded from a FAISS index on disk).

## Next

[`05_langchain_agent`](../05_langchain_agent/) — now that you've built the loop, tools, and
memory by hand, see how LangChain packages all three so you don't hand-roll them in every project.
