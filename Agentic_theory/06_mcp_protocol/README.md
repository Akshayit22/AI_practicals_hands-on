# 6. MCP — Model Context Protocol

## The problem MCP solves

Every tool in modules 1–5 was a Python function living in the *same process* as your agent. Fine
for a demo. But think about a real company: you might want your agent to reach a Slack API, a
Postgres database, a Jira instance, an internal HR system, a file system — and you might want to
use that *same* set of tools from three different apps (a LangChain agent, a Claude Desktop
session, a custom internal chatbot). Without a standard, you write **N × M** custom integrations
(N tools × M client apps), each with its own auth, its own schema conventions, its own bugs.

**MCP (Model Context Protocol, introduced by Anthropic in late 2024)** standardizes the wire
format between "things that provide tools/data" (**MCP servers**) and "things that use an LLM to
call tools" (**MCP clients**). Write one MCP server for your Jira integration, and *any* MCP
client — Claude Desktop, a LangChain agent, your own app — can use it without custom glue code.

The analogy your curriculum's reading (`t304-4`) makes: it's like **USB-C**. Before USB-C, every
device had its own proprietary port and cable. USB-C is one physical + protocol standard that
everything now plugs into. MCP does the same thing for "LLM app talks to tool/data source."
Another good analogy if you've done editor tooling: it's the same idea as **LSP** (Language
Server Protocol) — one protocol so any editor can talk to any language's tooling, instead of each
editor writing custom support for each language.

## The three things an MCP server can expose

| Concept | What it is | Analogy |
|---|---|---|
| **Tool** | A callable function with a schema (exactly like module 2's tool calling) | A POST endpoint |
| **Resource** | Read-only data the client can fetch (a file, a DB row, a doc) | A GET endpoint |
| **Prompt** | A reusable prompt template the server provides | A saved query / stored procedure |

This module focuses on **tools** since that's what connects directly to what you already know
from module 2 — MCP tools use the *same* JSON-schema-based function-calling idea, just exposed
over a protocol instead of living in your process.

## Client ↔ Server, concretely

```
┌─────────────┐   MCP protocol   ┌──────────────┐
│  MCP Client │ ───────────────► │  MCP Server  │
│ (your agent)│ ◄─────────────── │ (tool host)  │
└─────────────┘   (stdio / HTTP) └──────────────┘
```

- The **server** declares its tools (name, description, schema) — same shape as module 2's
  `TOOLS_SCHEMA`, just returned by the server instead of hand-written by you.
- The **client** connects, asks "what tools do you have?" (`list_tools`), gets the schemas, hands
  them to the LLM exactly like `bind_tools()` did in module 2, and when the LLM picks one, the
  client sends the call *over the protocol* to the server, which executes it and returns the result.
- Transport is usually **stdio** (server runs as a local subprocess — what this demo uses) or
  **HTTP/SSE** (server runs remotely, e.g. as a hosted service).

## Run it

```bash
cd Agentic_AI/06_mcp_protocol
pip install mcp   # if not already installed via requirements.txt
python mcp_client.py
```

`mcp_client.py` launches `mcp_server.py` as a subprocess (stdio transport), asks it what tools it
has, and calls one. You never import `mcp_server.py` directly — the whole point is that client and
server are separate processes talking a shared protocol, exactly like a browser and a web server.

## Where this plugs into what you already know

`langchain-mcp-adapters` (a small extra package) converts an MCP server's tools directly into
LangChain `@tool`-shaped objects, so you can drop them straight into the `AgentExecutor` from
module 5:

```python
# pip install langchain-mcp-adapters
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "my_tools": {"command": "python", "args": ["mcp_server.py"], "transport": "stdio"}
})
tools = await client.get_tools()          # ready to pass straight into create_tool_calling_agent
```

Not run in this demo (keeps the dependency list small) — but this is exactly the reference your
curriculum links for building a custom MCP tool (`i301-4`).

## Next

[`07_agentic_rag`](../07_agentic_rag/) — back to your existing RAG scripts, but retrieval becomes
something the agent decides to do, rather than something that always runs first.
