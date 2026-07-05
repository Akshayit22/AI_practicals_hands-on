"""
The same ReAct-style agent from modules 1-3, this time built with
LangChain's create_tool_calling_agent + AgentExecutor instead of a
hand-rolled while loop.
--------------------------------------------------------------------
Read Agentic_AI/05_langchain_agent/README.md first.

One of the three tools is a RAG retriever (FAISS-backed, same pattern as
your aws_rag_ai.py / pdf_rag_ai_search.py) wrapped as a tool the agent can
choose to call -- or not. That's your first taste of "agentic RAG"
(module 7 covers it properly).

Requires: OPENAI_API_KEY in your environment.

Swap-in note: replace ChatOpenAI/OpenAIEmbeddings below with
ChatBedrock/BedrockEmbeddings from langchain_aws (same as aws_rag_ai.py)
to run this whole file against Claude on Bedrock instead -- nothing else
in this file needs to change.
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────
# Tool 1: a plain calculator (same idea as module 3, LangChain style)
# ─────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '45 * 3'."""
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────
# Tool 2: a RAG retriever, exactly like your existing RAG scripts,
# except now it's a *tool* the agent decides whether to invoke.
# ─────────────────────────────────────────────

HR_POLICY_DOCS = [
    "Refund policy: customers can request a full refund within 14 days of the original charge.",
    "PTO policy: full-time employees accrue 1.5 days of paid time off per month.",
    "Remote work policy: employees may work remotely up to 3 days per week with manager approval.",
    "Expense policy: reimbursement requests must be submitted within 30 days with a receipt attached.",
]


def build_hr_retriever_tool():
    embeddings = OpenAIEmbeddings()
    docs = [Document(page_content=d) for d in HR_POLICY_DOCS]
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    @tool
    def search_hr_policy(query: str) -> str:
        """Search internal HR policy documents (refunds, PTO, remote work, expenses)."""
        results = retriever.invoke(query)
        return "\n".join(r.page_content for r in results)

    return search_hr_policy


# ─────────────────────────────────────────────
# Assemble the agent
# ─────────────────────────────────────────────

def build_agent_executor() -> AgentExecutor:
    llm = ChatOpenAI(model=MODEL, temperature=0)
    tools = [calculator, build_hr_retriever_tool()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful internal assistant. Use tools only when needed."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),  # where the agent's tool-call reasoning lives
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    # This IS the loop from module 3 -- verbose=True prints the
    # Thought/Action/Observation trace for you, no manual print() needed.
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)


if __name__ == "__main__":
    executor = build_agent_executor()
    chat_history = []  # plain list of BaseMessage objects -- short-term memory (module 4)

    questions = [
        "What is 128 * 7?",                                  # no retriever needed
        "How many days of remote work am I allowed per week?",  # needs the HR retriever tool
    ]

    for q in questions:
        print("=" * 70)
        print("Question:", q)
        print("=" * 70)
        result = executor.invoke({"input": q, "chat_history": chat_history})
        print("\nFinal answer:", result["output"])

        # Update short-term memory so the next question has context, same
        # idea as the buffer memory demo in module 4.
        from langchain_core.messages import HumanMessage, AIMessage
        chat_history.append(HumanMessage(content=q))
        chat_history.append(AIMessage(content=result["output"]))
        print()
