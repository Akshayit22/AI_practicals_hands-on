"""
ReAct (Reason + Act) loop, built from scratch on top of the real
tool-calling API you learned in module 2.
--------------------------------------------------------------------
Read Agentic_AI/03_react_agent_from_scratch/README.md first.

Pattern: Thought -> Action -> Observation, repeated until the model
stops requesting tools and just answers.

  Thought      = the free-text reasoning the model writes before/instead
                 of calling a tool (response.content)
  Action       = the tool_call the model requests
  Observation  = the string we feed back after running the tool

Requires: OPENAI_API_KEY in your environment.
"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────
# Tools available to the agent
# ─────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely (no eval of arbitrary code)."""
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
        return f"Error evaluating '{expression}': {e}"


def lookup_product_price(product: str) -> str:
    """Look up the unit price of a product from a tiny mock catalog."""
    catalog = {"widget": 12.50, "gadget": 45.00, "gizmo": 8.75}
    price = catalog.get(product.lower())
    return f"${price:.2f}" if price else f"No price found for '{product}'"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '12.5 * 40'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_product_price",
            "description": "Look up the unit price of a product by name.",
            "parameters": {
                "type": "object",
                "properties": {"product": {"type": "string"}},
                "required": ["product"],
            },
        },
    },
]

PYTHON_FUNCTIONS = {
    "calculator": calculator,
    "lookup_product_price": lookup_product_price,
}


# ─────────────────────────────────────────────
# The ReAct loop itself
# ─────────────────────────────────────────────

def run_react_loop(question: str, max_iterations: int = 6) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful assistant. Before calling a tool, briefly "
                "explain your reasoning in one sentence. Only call tools that "
                "are actually necessary. When you have the final answer, reply "
                "with plain text and no tool call."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, max_iterations + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message

        # "Thought" -- whatever reasoning text the model wrote this turn.
        if msg.content:
            print(f"  Thought ({iteration}): {msg.content}")

        # No tool calls means the model considers itself done.
        if not msg.tool_calls:
            return msg.content

        messages.append(msg)

        # "Action" + "Observation" for each tool the model requested.
        for call in msg.tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)
            print(f"  Action  ({iteration}): {fn_name}({fn_args})")

            result = PYTHON_FUNCTIONS[fn_name](**fn_args)
            print(f"  Observation ({iteration}): {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return "(gave up: exceeded max_iterations without a final answer)"


if __name__ == "__main__":
    # This question genuinely needs two different tools, in sequence:
    # 1. look up the unit price  2. do the multiplication for a bulk order.
    question = "If I order 40 units of 'widget', what's the total cost?"

    print("=" * 70)
    print(f"Question: {question}")
    print("=" * 70)
    answer = run_react_loop(question)
    print()
    print("Final answer:", answer)
