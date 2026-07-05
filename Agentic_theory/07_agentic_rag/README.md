# 7. Agentic RAG

## What changes vs. the RAG you already built

Your existing RAG scripts (`pdf_rag_ai_search.py`, `excel_rag_ai_search.py`, `aws_rag_ai.py`) all
follow the same fixed pipeline, once per request:

```
query → embed → similarity_search(k) → stuff chunks into prompt → LLM → answer
```

This always retrieves **exactly once**, with the **user's original query, unmodified**, no
matter what the question actually needs. That's a workflow (module 1's terminology) — good, simple,
predictable, and it's the right choice for most single-hop questions ("what does the PDF say about X").

**Agentic RAG** turns retrieval into a *tool* the LLM can call zero, one, or many times, with
whatever query it decides to write — the same "retriever as a tool" idea you already saw briefly
in module 5. The concrete things this unlocks that fixed RAG can't do:

| Fixed RAG (what you have) | Agentic RAG |
|---|---|
| Always retrieves, even for questions that don't need it | Skips retrieval for general questions |
| Retrieves once with the user's literal query | Can rewrite the query before searching (e.g. expand an acronym) |
| Can't combine two different sources in one answer | Can call the retriever tool twice, once per source, and combine results |
| No self-correction | Can notice "these chunks don't answer the question" and retrieve again with a different query |

The cost: more LLM calls, non-deterministic retrieval behavior, and harder debugging (this is the
same workflow-vs-agent tradeoff from module 1 — you're trading predictability for flexibility).
**Use agentic RAG when questions genuinely span multiple sources or need reformulation. Use your
existing fixed RAG scripts when a question always maps to "search this one index."**

## What this demo shows

Two small document sets that a single question needs *both* of:
- **Product FAQ index** — what a "Pro plan" includes.
- **Shipping policy index** — how returns/refunds work.

Question: *"If I'm on the Pro plan and want to return a defective unit, what's covered and how
long do I have?"* — this needs a fact from *each* index. A fixed single-retrieval RAG pipeline
would only search one of them (whichever index you hardcoded), and give an incomplete answer. The
agent, given both indexes as separate tools, calls **both**, because it reasons that the question
needs both.

## Run it

```bash
cd Agentic_AI/07_agentic_rag
python agentic_rag_demo.py
```

Watch the trace: on a simple factual question, the agent calls only one retriever tool (or none).
On the combined question, it calls both — a decision your fixed pipeline could never make.

## Next

[`08_multi_agent_langgraph`](../08_multi_agent_langgraph/) — instead of one agent with many tools,
split responsibilities across multiple specialized agents, coordinated by a supervisor.
