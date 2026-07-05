"""
LangFuse tracing wired into a LangChain agent.
--------------------------------------------------------------------
Read Agentic_AI/09_observability_evals/README.md first.

This reuses the exact same agent-building code from module 5
(calculator + HR retriever tools) -- the only new thing here is the
`CallbackHandler`, which is how you plug tracing into ANY LangChain
Runnable (an AgentExecutor, a LangGraph app, a plain chain) without
changing its logic at all.

Requires:
    pip install langfuse
    OPENAI_API_KEY            (same as every other module)
    LANGFUSE_PUBLIC_KEY       (free account at https://langfuse.com)
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST             (optional, defaults to LangFuse Cloud)

If the LangFuse env vars aren't set, this script still runs the agent
normally -- it just won't have anywhere to send traces. Uncomment the
handler wiring below once you have keys.
"""

import os
import sys

# Reuse module 5's agent-building code directly instead of duplicating it.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "05_langchain_agent"),
)
from langchain_agent_executor import build_agent_executor  # noqa: E402


def main():
    executor = build_agent_executor()

    callbacks = []
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        from langfuse.langchain import CallbackHandler
        callbacks = [CallbackHandler()]
        print("LangFuse tracing enabled -- check your LangFuse dashboard after this runs.")
    else:
        print("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set -- running without tracing.")
        print("Get free keys at https://langfuse.com to see this agent's full trace.")

    result = executor.invoke(
        {"input": "How many days of remote work am I allowed per week?", "chat_history": []},
        config={"callbacks": callbacks},
    )
    print("\nFinal answer:", result["output"])


if __name__ == "__main__":
    main()
