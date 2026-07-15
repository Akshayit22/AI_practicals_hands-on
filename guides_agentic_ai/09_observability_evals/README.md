# 9. Observability, Evaluation & Guardrails (Week 6 — pointers)

This module is intentionally lighter than 1–8: it's Week 6 territory, and it's more about
*knowing these tools exist and roughly how they fit* than building from scratch. Once you have
agents running (modules 1–8), these are the three problems that show up immediately in practice.

## 1. Observability — "what is my agent actually doing?"

Every module so far printed its Thought/Action/Observation trace to your terminal with
`verbose=True` or manual `print()`. That does not scale to production — you can't tail
`stdout` across a fleet of running agents, and you have no history once the process exits.

**LangFuse** is a tracing/observability backend built specifically for LLM apps. It captures every
step your LangChain agent takes — prompts sent, tokens used, tools called, latency per step — and
gives you a searchable UI instead of scrollback. Integration is usually just adding a callback
handler:

```python
# pip install langfuse
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()  # reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from env

# any LangChain Runnable (including AgentExecutor from module 5, or a LangGraph app
# from module 8) accepts callbacks the same way:
executor.invoke({"input": "..."}, config={"callbacks": [langfuse_handler]})
```

See `langfuse_tracing_demo.py` in this folder for a runnable version (needs a free LangFuse
account + API keys — https://langfuse.com).

## 2. Evaluation — "is my agent actually any good?"

Once tracing tells you *what* happened, evaluation tells you whether it was *correct*. LangChain's
**AgentEvals** benchmarks an agent on things like:
- **Correctness** — did it reach the right final answer?
- **Trajectory match** — did it call the *right sequence* of tools (not just get lucky)?
- **Latency** — how long did the full loop take?
- **Hallucination rate** — did it state something not supported by tool results?

This needs a labeled test set (question → expected answer / expected tool sequence) specific to
your use case, so there's no generic runnable demo here — but the pattern is: run your agent
against a fixed set of questions, compare against expected answers/trajectories, track the score
over time as you change prompts/tools. Treat it like a test suite for a non-deterministic system —
same idea as regression tests, just with a fuzzier pass/fail.

## 3. Guardrails — "how do I stop my agent doing something it shouldn't?"

Tool calling (module 2) means the model can request *anything* your schema allows — including
things you don't want, if the model is confused or manipulated (prompt injection, see your Week 2
material on prompt hacking). **NeMo Guardrails** (NVIDIA) lets you define input/output contracts —
e.g. "reject any user message that looks like a jailbreak attempt," "never let the agent output a
raw account number," "if the agent's tool call touches `delete_*`, require human confirmation
first." Conceptually this is a validation layer wrapped around your agent loop, same instinct as
input validation at an API boundary — never trust the model's output any more than you'd trust
unvalidated client input.

## 4. AWS Bedrock Agents — a managed alternative

Everything in modules 1–8 was self-hosted (you run the loop, the memory, the tools). **AWS Bedrock
Agents** is AWS's managed version of the same idea: you configure an agent (system prompt +
"action groups," which are Bedrock's name for tool schemas) through the AWS console/API, and AWS
runs the agent loop for you, with built-in integration to a Bedrock Knowledge Base (their managed
RAG) instead of your own FAISS/OpenSearch code. Worth knowing this exists as the "managed service"
end of the spectrum, the same way you'd choose between self-hosting Postgres vs. RDS.

## Next steps from here

- Get a free LangFuse account and run `langfuse_tracing_demo.py` against the module 5 agent.
- Revisit your curriculum's Week 6 assignments (`i301-6`, `i302-6`) — they ask you to wire
  LangFuse + Guardrails + AgentEval into an agent you already built. Modules 5 and 8 are good
  candidates to instrument.
