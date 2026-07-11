"""
LinkedIn post generator — writer <-> reviewer loop with web search.

Requires: langgraph>=1.0  langchain>=1.0  langchain-openai  langchain-tavily
          pydantic  python-dotenv  rich
"""

from __future__ import annotations

import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from rich import print
from typing_extensions import TypedDict

load_dotenv()

MAX_ATTEMPTS = 3        # writer drafts, reviewer rejects -> at most this many rounds
MAX_SEARCH_ROUNDS = 2   # hard stop on the writer <-> tools ping-pong


# --------------------------------------------------------------------------
# config / models
# --------------------------------------------------------------------------
def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


API_KEY = _require("AZURE_AI_API_KEY")
BASE_URL = _require("AZURE_AI_BASE_URL")
MODEL = os.getenv("AZURE_AI_MODEL", "DeepSeek-V3.2")


def make_llm(temperature: float, max_tokens: int = 2000) -> ChatOpenAI:
    # NOTE: if you're on Azure OpenAI proper, use AzureChatOpenAI.
    # If you're on Azure AI Foundry, langchain-azure-ai's AzureAIChatCompletionsModel
    # is the first-class integration. ChatOpenAI + base_url works, but you're
    # relying on the endpoint being OpenAI-compatible.
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,   # was 1000 — too tight, it can truncate mid-post
        timeout=60,
        max_retries=3,           # transient 429/5xx retries, was missing
        default_headers={"api-key": API_KEY},
    )


search_tool = TavilySearch(max_results=3)
tools = [search_tool]
tool_node = ToolNode(tools)

writer_llm = make_llm(temperature=0.7)
writer_llm_with_tools = writer_llm.bind_tools(tools)
reviewer_llm = make_llm(temperature=0.2)


class Review(BaseModel):
    """Structured verdict — replaces the brittle 'APPROVED' in text.upper() check."""

    approved: bool = Field(description="True only if the post meets EVERY criterion.")
    feedback: str = Field(description="Short, concrete, actionable feedback.")


# If your endpoint rejects json_schema mode, fall back to:
#   reviewer_llm.with_structured_output(Review, method="function_calling")
reviewer = reviewer_llm.with_structured_output(Review)


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
class State(TypedDict):
    topic: str
    messages: Annotated[list[AnyMessage], add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int
    max_attempts: int  # was hardcoded as `>= 2` inside the router


WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Write engaging, professional "
    "LinkedIn posts about the given topic. If the topic needs up-to-date facts, "
    "statistics, or current trends, use the web search tool BEFORE writing. "
    "Rules: strong hook in the first line; exactly one clear takeaway; short, "
    "skimmable paragraphs; 150-200 words; end with a question or CTA; no hashtags. "
    "When you have finished researching, reply with ONLY the post text — no "
    "preamble, no 'Here is your post:', no commentary."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. Judge whether the post is "
    "publish-ready against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Skimmable — short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone, not corporate-robotic\n"
    "7. No hashtags\n"
    "8. Actually on-topic for the stated topic\n\n"
    "Be strict but fair. Approve only if it genuinely meets all criteria."
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _text(msg: AnyMessage) -> str:
    """content can be a str or a list of content blocks — handle both."""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict)
    )


def _search_rounds(messages: list[AnyMessage]) -> int:
    return sum(1 for m in messages if isinstance(m, ToolMessage))


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------
def writer_node(state: State) -> dict:
    """Writes (or rewrites) the post. May call Tavily first."""
    messages = state["messages"]

    # Resuming after a search: hand the tool results back to the model.
    if messages and isinstance(messages[-1], ToolMessage):
        # FIX: cap the tools loop. Past the cap, strip the tools so the model
        # is forced to actually write instead of searching forever.
        llm = (
            writer_llm
            if _search_rounds(messages) >= MAX_SEARCH_ROUNDS
            else writer_llm_with_tools
        )
        response = llm.invoke([SystemMessage(WRITER_SYSTEM_PROMPT), *messages])
        return {"messages": [response]}  # don't bump `attempt`

    attempt = state["attempt"] + 1

    if attempt == 1:
        instruction = (
            f"Write a LinkedIn post about: {state['topic']}\n"
            f"If you need current facts or statistics, search the web first."
        )
    else:
        instruction = (
            f"Your draft was REJECTED by the reviewer.\n\n"
            f"Reviewer feedback:\n{state['review_feedback']}\n\n"
            f"Rewrite the post so that every point above is fixed. Reuse the "
            f"research already in this conversation — only search again if you "
            f"need a fact you don't have. Do not repeat the same mistakes."
        )

    new_messages = [HumanMessage(instruction)]

    # FIX: pass the FULL thread, not just [system, human]. This is what keeps the
    # Tavily results from attempt 1 alive on attempts 2 and 3 — otherwise the
    # rewrite has no facts and starts inventing statistics.
    response = writer_llm_with_tools.invoke(
        [SystemMessage(WRITER_SYSTEM_PROMPT), *messages, *new_messages]
    )

    return {"messages": [*new_messages, response], "attempt": attempt}


def extract_draft_node(state: State) -> dict:
    """Pull the post text out of the last real AI message."""
    # FIX: was `state['messages'][-1].content` — could be "" or a block list.
    draft = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = _text(msg).strip()
            if text:
                draft = text
                break

    print(f"\n[bold]--- Draft (attempt {state['attempt']}) ---[/bold]\n{draft}\n")
    return {"draft": draft}


def reviewer_node(state: State) -> dict:
    """Approve or reject with feedback."""
    # FIX: give the reviewer the topic too, so it can check relevance.
    review: Review = reviewer.invoke(
        [
            SystemMessage(REVIEWER_SYSTEM_PROMPT),
            HumanMessage(
                f"Topic: {state['topic']}\n\nDraft:\n{state['draft']}"
            ),
        ]
    )

    verdict = "[green]APPROVED[/green]" if review.approved else "[red]REJECTED[/red]"
    print(f"Verdict: {verdict}\nFeedback: {review.feedback}\n")

    return {"is_approved": review.approved, "review_feedback": review.feedback}


# --------------------------------------------------------------------------
# routers
# --------------------------------------------------------------------------
def route_after_writer(state: State) -> Literal["tools", "extract_draft"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "extract_draft"


def route_after_reviewer(state: State) -> Literal["writer", "__end__"]:
    if state["is_approved"]:
        return END
    if state["attempt"] >= state["max_attempts"]:
        print("[yellow]Max attempts reached — returning the last draft.[/yellow]")
        return END
    return "writer"


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------
builder = StateGraph(State)

builder.add_node("writer", writer_node)
builder.add_node("tools", tool_node)
builder.add_node("extract_draft", extract_draft_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "writer")
builder.add_conditional_edges(
    "writer",
    route_after_writer,
    {"tools": "tools", "extract_draft": "extract_draft"},  # explicit map -> better diagrams
)
builder.add_edge("tools", "writer")
builder.add_edge("extract_draft", "reviewer")
builder.add_conditional_edges(
    "reviewer",
    route_after_reviewer,
    {"writer": "writer", END: END},
)

# A checkpointer gives you durable state, resumability, and is the prerequisite
# for human-in-the-loop `interrupt()` if you ever want a real human reviewer.
# Swap InMemorySaver for SqliteSaver / PostgresSaver in production.
app = builder.compile(checkpointer=InMemorySaver())


# --------------------------------------------------------------------------
# CLI — guarded so `langgraph dev` / imports / tests don't trigger input()
# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 55)
    print("LinkedIn Post Generator")
    print("=" * 55)

    topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()
    if not topic:
        print("\nNo topic given. Exiting.")
        return

    initial_state: State = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
        "max_attempts": MAX_ATTEMPTS,
    }
    config = {"configurable": {"thread_id": "linkedin-session-1"}}

    print("\nGenerating...\n")
    # stream_mode="updates" gives you live progress instead of prints buried in nodes
    for chunk in app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name in chunk:
            print(f"[dim]· {node_name}[/dim]")

    final = app.get_state(config).values

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final["draft"])
    print("=" * 55)
    print(f"Attempts: {final['attempt']} | Approved: {final['is_approved']}")


if __name__ == "__main__":
    main()