# 5. Building the Same Agent With LangChain

## Why bother, if you already built one by hand?

Modules 1–4 built an agent loop, tool calling, and memory from raw API calls. That was on purpose
— you now know exactly what's happening under the hood. But hand-rolling the loop every time you
start a project is repetitive and easy to get subtly wrong (forgetting the `max_iterations` guard,
mismatching `tool_call_id`s, etc). LangChain packages the pattern you already understand into a
few reusable pieces:

| Your hand-rolled version (modules 1-4) | LangChain equivalent |
|---|---|
| The `while` loop calling the LLM, executing tools, appending results | `AgentExecutor` |
| `@tool`-decorated functions + JSON schema (already used in module 2) | Same `@tool` decorator |
| `PYTHON_FUNCTIONS = {"name": fn}` dispatch dict | Handled internally by `AgentExecutor` |
| `messages` list you appended to manually | `chat_history` passed into the agent |
| Your FAISS-backed RAG script (`aws_rag_ai.py`) | Wrapped as a `Tool` the agent can *choose* to call |

Nothing conceptually new happens here — you're looking at the same ReAct-style loop from module 3,
just assembled by a library instead of by hand.

## The pieces

- **`create_tool_calling_agent(llm, tools, prompt)`** — builds the "decide what to do next" step.
  Takes your LLM + your list of tools + a prompt template, returns a `Runnable` that, given
  `chat_history` and tool results so far, decides the next action (or the final answer).
- **`AgentExecutor(agent=..., tools=..., verbose=True)`** — this *is* the loop. Call
  `.invoke({"input": ...})` and it repeatedly runs the agent, executes whatever tool it picks,
  feeds the result back, until the agent returns a final answer. `verbose=True` prints the
  Thought/Action/Observation trace for free — no need to hand-print it like in module 3.
- **Tools list** — a mix of a plain calculator tool and a **retriever tool** built from a small
  FAISS store, exactly like your `aws_rag_ai.py` / `pdf_rag_ai_search.py`, except now the agent
  *decides* whether to search it at all, instead of your code always searching it first. That
  distinction is exactly what module 7 (Agentic RAG) explores in depth — this module is your
  first taste of it.

## Run it

```bash
cd Agentic_AI/05_langchain_agent
python langchain_agent_executor.py
```

Two things to watch in the `verbose=True` trace:
1. On a general question, the agent skips the retriever tool entirely — it decided it didn't
   need it. Nobody coded that decision.
2. On a policy question, it calls the retriever tool with a query *it wrote itself* (not
   necessarily the user's exact words), then answers using the retrieved chunk.

## Next

[`06_mcp_protocol`](../06_mcp_protocol/) — every tool so far has been a Python function living in
the same process as the agent. MCP standardizes how an agent talks to tools that live *elsewhere*.
