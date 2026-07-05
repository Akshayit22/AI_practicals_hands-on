"""
Multi-agent supervisor pattern with LangGraph.
--------------------------------------------------------------------
Read Agentic_AI/08_multi_agent_langgraph/README.md first.

Three "agents" as graph nodes:
  - supervisor    : classifies the query as IT or Finance, decides the route
  - it_agent      : a full ReAct agent (via create_react_agent) with IT tools
  - finance_agent : a full ReAct agent with Finance tools

This directly mirrors your curriculum's Week 5 assignment (i301-5) --
adapt the tools/docs below for your actual IT/Finance FAQ content.

Requires: OPENAI_API_KEY in your environment.
Requires: pip install langgraph (see requirements.txt)

Swap-in note: replace ChatOpenAI with ChatBedrock (langchain_aws) to run
the whole graph against Claude on Bedrock -- no other code changes needed.
"""

import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import create_react_agent

MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────
# Mock "ReadFile" tools -- stand-ins for real internal docs
# ─────────────────────────────────────────────

IT_DOCS = (
    "VPN setup: install Cisco AnyConnect, connect to vpn.company.com using your SSO login. "
    "Approved software: Slack, VS Code, Docker Desktop, Zoom. "
    "New laptop requests: submit a ticket in the IT portal; approval takes 2 business days."
)

FINANCE_DOCS = (
    "Reimbursement: submit expense reports via Concur within 30 days; attach receipts over $25. "
    "Budget reports: available on the Finance SharePoint, updated monthly. "
    "Payroll: processed on the last business day of each month."
)


@tool
def read_it_docs(query: str) -> str:
    """Read internal IT documentation covering VPN setup, approved software, and hardware requests."""
    return IT_DOCS


@tool
def read_finance_docs(query: str) -> str:
    """Read internal finance documentation covering reimbursement, budget reports, and payroll."""
    return FINANCE_DOCS


@tool
def web_search(query: str) -> str:
    """Search the public web for general information not covered by internal docs. Stubbed here."""
    return f"[stubbed web result for: {query}] -- swap for a real search tool (Tavily/DuckDuckGo) in production."


# ─────────────────────────────────────────────
# Shared graph state
# ─────────────────────────────────────────────

class AgentState(MessagesState):
    next: str  # written by the supervisor, read by the conditional edge


# ─────────────────────────────────────────────
# Supervisor node: classify the query, decide the route
# ─────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    user_query = state["messages"][-1].content
    llm = ChatOpenAI(model=MODEL, temperature=0)
    decision = llm.invoke([
        {
            "role": "system",
            "content": (
                "Classify the user's query as exactly one word: 'IT' or 'Finance'. "
                "IT covers VPN, software, hardware, technical issues. "
                "Finance covers reimbursement, budget, payroll, invoices."
            ),
        },
        {"role": "user", "content": user_query},
    ])
    category = decision.content.strip().lower()
    route = "it_agent" if "it" in category else "finance_agent"
    print(f"  [supervisor] classified as '{category}' -> routing to {route}")
    return {"next": route}


# ─────────────────────────────────────────────
# Specialist agents: built with LangGraph's own prebuilt ReAct agent,
# same pattern you built by hand in module 3 and with AgentExecutor in
# module 5 -- just reused here as a node inside a bigger graph.
# ─────────────────────────────────────────────

def build_it_node():
    agent = create_react_agent(ChatOpenAI(model=MODEL, temperature=0), tools=[read_it_docs, web_search])

    def _run(state: AgentState) -> dict:
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][-1:]}  # append only the final response

    return _run


def build_finance_node():
    agent = create_react_agent(ChatOpenAI(model=MODEL, temperature=0), tools=[read_finance_docs, web_search])

    def _run(state: AgentState) -> dict:
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][-1:]}

    return _run


# ─────────────────────────────────────────────
# Assemble the graph
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("it_agent", build_it_node())
    graph.add_node("finance_agent", build_finance_node())

    graph.set_entry_point("supervisor")

    # This is the routing table: after "supervisor" runs, look at state["next"]
    # and go to whichever node it names. Nobody hardcoded which query goes
    # where -- the supervisor's LLM call decided it, per request.
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"it_agent": "it_agent", "finance_agent": "finance_agent"},
    )

    graph.add_edge("it_agent", END)
    graph.add_edge("finance_agent", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    queries = [
        "How do I set up VPN on my new laptop?",
        "How do I file a reimbursement for a client dinner?",
    ]

    for q in queries:
        print("=" * 70)
        print("Query:", q)
        print("=" * 70)
        result = app.invoke({"messages": [HumanMessage(content=q)], "next": ""})
        print("\nFinal answer:", result["messages"][-1].content)
        print()
