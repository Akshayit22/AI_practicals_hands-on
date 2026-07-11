"""
A minimal MCP client — connects to mcp_server.py over stdio, lists its
tools, and calls one. No LLM involved yet; this file is purely about
seeing the client<->server protocol work, isolated from agent logic.
--------------------------------------------------------------------
Read Agentic_AI/06_mcp_protocol/README.md first.

Run:
    python mcp_client.py

Under the hood: this launches `python mcp_server.py` as a subprocess,
speaks MCP over its stdin/stdout, and shuts it down when done. In a real
deployment, the server might instead run as a long-lived remote process
reached over HTTP/SSE -- the client code barely changes either way.
"""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()  # MCP handshake

            # Ask the server what tools it exposes -- this is the same
            # "here's my schema" idea as module 2's TOOLS_SCHEMA, except
            # the server tells the client instead of you writing it by hand.
            tools_response = await session.list_tools()
            print("Tools exposed by the server:")
            for t in tools_response.tools:
                print(f"  - {t.name}: {t.description}")
            print()

            # Call a tool over the protocol. The server process actually
            # runs the function; we just sent it a request and got a result.
            result = await session.call_tool(
                "get_order_status", arguments={"order_id": "ORD-1002"}
            )
            print("get_order_status(order_id='ORD-1001') ->")
            print(" ", result.content[0].text)
            print()

            result = await session.call_tool(
                "convert_currency",
                arguments={"amount": 125, "from_currency": "usd", "to_currency": "inr"},
            )
            print("convert_currency(100, 'usd', 'inr') ->")
            print(" ", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
