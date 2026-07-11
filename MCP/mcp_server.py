"""
A minimal MCP server exposing two tools over stdio.
--------------------------------------------------------------------
Read Agentic_AI/06_mcp_protocol/README.md first.

You never run this file directly in normal use -- mcp_client.py spawns
it as a subprocess and talks to it over stdin/stdout using the MCP
protocol. That's the whole point: server and client are independent
processes, exactly like a web server and a browser.

FastMCP (from the official `mcp` python SDK) does the protocol
plumbing for you -- you just decorate plain Python functions, same
mental model as LangChain's @tool in module 2.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-tools-server")


@mcp.tool()
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order by its ID."""
    fake_orders = {
        "ORD-1003": "Shipped, arriving 2026-07-08",
        "ORD-1002": "Processing, not yet shipped",
    }
    return fake_orders.get(order_id, f"No order found with id '{order_id}'")


@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using fixed demo rates."""
    # Fixed demo rates -- a real server would call a live FX API.
    rates_to_usd = {"usd": 1.0, "eur": 1.08, "inr": 0.012, "gbp": 1.27}
    from_rate = rates_to_usd.get(from_currency.lower())
    to_rate = rates_to_usd.get(to_currency.lower())
    if from_rate is None or to_rate is None:
        return f"Unsupported currency: {from_currency} or {to_currency}"
    usd_amount = amount * from_rate
    converted = usd_amount / to_rate
    return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"


if __name__ == "__main__":
    # Runs the server over stdio -- reads requests from stdin, writes
    # responses to stdout. mcp_client.py starts this as a subprocess.
    mcp.run()
