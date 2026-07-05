"""
Agentic RAG — retrieval as a tool the agent decides to call (0, 1, or
many times), instead of a fixed step that always runs first.
--------------------------------------------------------------------
Read Agentic_AI/07_agentic_rag/README.md first.

Two separate FAISS indexes, exposed as two separate tools:
  - search_product_faq      -> what the Pro plan includes
  - search_shipping_policy  -> return/refund rules

A question that only needs one index gets one tool call. A question
that needs facts from BOTH gets two tool calls, with queries the agent
writes itself -- not the user's literal words.

Requires: OPENAI_API_KEY in your environment.

Swap-in note: replace ChatOpenAI/OpenAIEmbeddings with
ChatBedrock/BedrockEmbeddings (langchain_aws) to run this against Claude
on Bedrock, same as your aws_rag_ai.py.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MODEL = "gpt-4o-mini"

PRODUCT_FAQ_DOCS = [
    "The Pro plan includes unlimited projects, priority email support, and up to 10 team seats.",
    "The Free plan is limited to 1 project and community support only.",
    "The Pro plan does not include phone support -- that requires the Enterprise plan.",
]

SHIPPING_POLICY_DOCS = [
    "Defective units can be returned within 30 days of delivery for a full refund or replacement.",
    "Non-defective returns are accepted within 14 days, minus a 10% restocking fee.",
    "Digital/software products are non-refundable once a license key has been issued.",
]


def build_retriever_tool(name: str, description: str, docs: list[str]):
    """Same FAISS-backed retriever pattern as your RAG scripts, wrapped as a tool."""
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents([Document(page_content=d) for d in docs], embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    def _search(query: str) -> str:
        results = retriever.invoke(query)
        return "\n".join(r.page_content for r in results)

    # Build the @tool dynamically so name/description can vary per index.
    _search.__name__ = name
    _search.__doc__ = description
    return tool(_search)


def build_agentic_rag_executor() -> AgentExecutor:
    tools = [
        build_retriever_tool(
            "search_product_faq",
            "Search the product FAQ for plan features and pricing details.",
            PRODUCT_FAQ_DOCS,
        ),
        build_retriever_tool(
            "search_shipping_policy",
            "Search the shipping/returns policy for refund and return rules.",
            SHIPPING_POLICY_DOCS,
        ),
    ]

    llm = ChatOpenAI(model=MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You answer questions using the available search tools. Only call a "
            "tool if the question actually requires information from that source. "
            "If a question needs facts from more than one source, call each "
            "relevant tool separately before answering."
        )),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)


if __name__ == "__main__":
    executor = build_agentic_rag_executor()

    print("=" * 70)
    print("Q1: single-source question (only needs the product FAQ)")
    print("=" * 70)
    r1 = executor.invoke({"input": "Does the Pro plan include phone support?"})
    print("\nAnswer:", r1["output"])

    print()
    print("=" * 70)
    print("Q2: cross-source question (needs BOTH product FAQ and shipping policy)")
    print("=" * 70)
    r2 = executor.invoke({
        "input": (
            "I'm on the Pro plan and got a defective unit -- what does my plan "
            "cover, and how long do I have to return it?"
        )
    })
    print("\nAnswer:", r2["output"])

    print()
    print("Notice Q2 required calling BOTH retriever tools -- a fixed, single-")
    print("retrieval RAG pipeline searching only one index could not have answered it.")
