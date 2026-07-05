# Agentic AI — Learning Path

You already know: **tokens, embeddings, chunking, RAG, prompting**. This folder picks up where
that left off and covers **agents** — the part of the curriculum where an LLM stops just
answering questions and starts *doing things*: calling tools, looping, remembering, and
coordinating with other agents.

Written for a full-stack dev who is new to AI. Every concept has:
1. A **README.md** — the "why", explained with analogies to normal backend engineering.
2. A **runnable `.py` file** — heavily commented, so you learn by reading + running, not just reading.

No notebooks here on purpose — agents are about control flow (loops, branches, function calls),
which reads more naturally as plain Python than as notebook cells.

---

## Mental model (read this first)

A **RAG pipeline** (what you already built) is a fixed pipe:

```
question → embed → vector search → stuff context into prompt → LLM → answer
```

Every request goes through the exact same steps, in the exact same order. That's a **workflow**.

An **agent** replaces that fixed pipe with a loop where the LLM itself decides what to do next:

```
question → LLM thinks → "I should call tool X" → run tool X → feed result back to LLM
         → LLM thinks again → "I have enough info now" → final answer
```

The LLM is no longer just generating text — it's generating **decisions**. Your RAG retriever
becomes just one tool among many that the LLM can *choose* to call, or not, or call twice, or
call with a different query the second time. That shift — from "code decides the steps" to
"LLM decides the steps" — is the entire idea behind agents. Everything else (tool calling,
memory, multi-agent, MCP) is machinery in service of that one loop.

---

## Learning path

Go in order — each module builds on the previous one's vocabulary.

| # | Folder | Concept | Curriculum week |
|---|--------|---------|------------------|
| 1 | [`01_agent_fundamentals`](01_agent_fundamentals/) | Workflow vs. agent, the agent loop | Week 3 |
| 2 | [`02_tool_calling`](02_tool_calling/) | Function/tool calling — how an LLM "calls code" | Week 3 |
| 3 | [`03_react_agent_from_scratch`](03_react_agent_from_scratch/) | ReAct pattern, built with zero frameworks | Week 3 |
| 4 | [`04_agent_memory`](04_agent_memory/) | Short-term vs. long-term memory | Week 3 |
| 5 | [`05_langchain_agent`](05_langchain_agent/) | Building the same agent with LangChain (production way) | Week 4 |
| 6 | [`06_mcp_protocol`](06_mcp_protocol/) | MCP — the "USB-C port" for tools | Week 4 |
| 7 | [`07_agentic_rag`](07_agentic_rag/) | Agentic RAG — retrieval as a decision, not a step | Week 4 |
| 8 | [`08_multi_agent_langgraph`](08_multi_agent_langgraph/) | Multi-agent systems with LangGraph (supervisor pattern) | Week 5 |
| 9 | [`09_observability_evals`](09_observability_evals/) | Tracing, evals, guardrails (pointers only) | Week 6 |

**Suggested order of attack:** 1 → 2 → 3 → 4 (understand agents with no framework at all) → 5
(see how a framework saves you the boilerplate) → 7 (apply it to your existing RAG code) → 6
(understand MCP conceptually) → 8 (scale to multiple agents) → 9 (skim, it's mostly links).

---

## Setup

```bash
# from repo root
pip install -r requirements.txt
```

Each module's code defaults to **OpenAI** (`gpt-4o-mini`) because it's the simplest to run with
one env var, but every file notes how to swap in `ChatBedrock` / Claude, since your other scripts
(`aws_rag_ai.py`, `pdf_rag_ai_search.py`) already use `langchain_aws`. Set whichever key you use:

```bash
export OPENAI_API_KEY="sk-..."
# or, if you prefer Bedrock/Claude like your existing RAG scripts:
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

Each `.py` file is standalone — `cd` into its folder and `python <file>.py`.

---

## Vocabulary cheat-sheet (fill in as you go)

| Term | One-line definition |
|---|---|
| **Agent** | An LLM running in a loop, choosing which tools to call until it decides it's done. |
| **Tool / Function calling** | A JSON schema you give the LLM describing a function; the LLM replies "call this function with these args" instead of plain text. |
| **ReAct** | "Reason + Act" — the LLM alternates between writing a thought and taking an action, in the same loop. |
| **Agent memory** | State carried across turns (chat history) or across sessions (a database of facts/summaries). |
| **AgentExecutor** | LangChain's built-in implementation of the agent loop, so you don't hand-write it. |
| **MCP** | A standard protocol so any LLM app can talk to any tool server, instead of writing custom integration code per tool. |
| **Agentic RAG** | RAG where the LLM decides *if/when/how many times* to retrieve, instead of always retrieving once up front. |
| **Multi-agent system** | Several agents, each with a narrow job, coordinated by a router/supervisor agent. |
| **LangGraph** | A graph-based orchestration layer on top of LangChain for building agents/multi-agent systems as explicit state machines. |
