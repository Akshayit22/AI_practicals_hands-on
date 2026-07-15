"""
Tool Calling (Function Calling) — the real, provider-native version
--------------------------------------------------------------------
Read Agentic_AI/02_tool_calling/README.md first.

PART A — raw OpenAI tool-calling API
    You hand-write a JSON schema describing a function. The model replies
    with a structured `tool_calls` field instead of loose text. You parse
    it, run the *real* Python function yourself, and send the result back.

PART B — the same thing with LangChain's @tool decorator
    Instead of hand-writing JSON schemas, you decorate a normal typed
    Python function. LangChain introspects the signature + docstring and
    builds the schema for you. This is what every later module uses.

Requires: OPENAI_API_KEY in your environment.

Swap-in note: `ChatBedrock` (langchain_aws) supports `.bind_tools()` with
the exact same @tool-decorated functions used in Part B -- see the
commented block at the bottom for the one-line swap.
"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────
# PART A — raw OpenAI tool-calling API
# ─────────────────────────────────────────────

# Step 1: the real Python function that actually does something.
def get_weather(city: str) -> str:
    # Stubbed for the demo -- in real life this calls a weather API.
    fake_data = {"austin": "91°F, sunny", "seattle": "62°F, rainy"}
    return fake_data.get(city.lower(), f"No data for {city}")


# Step 2: the schema describing that function, in the shape the API expects.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Austin'"}
                },
                "required": ["city"],
            },
        },
    }
]

# Step 3: a lookup so we can dispatch by name once the model asks for one.
PYTHON_FUNCTIONS = {"get_weather": get_weather}


def run_raw_tool_calling_demo():
    messages = [{"role": "user", "content": "What's the weather in Austin right now?"}]

    # First call: give the model the question + the tools it's allowed to use.
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
    )
    msg = resp.choices[0].message

    if not msg.tool_calls:
        print("  Model answered directly, no tool needed:", msg.content)
        return

    # The model wants to call one or more tools. It NEVER runs them itself --
    # we do, right here, in our own process, with our own permissions.
    messages.append(msg)  # keep the assistant's tool-call message in history

    for call in msg.tool_calls:
        fn_name = call.function.name
        fn_args = json.loads(call.function.arguments)  # guaranteed valid JSON
        print(f"  Model requested: {fn_name}({fn_args})")

        result = PYTHON_FUNCTIONS[fn_name](**fn_args)

        # Tool results go back as a "tool" role message, tagged with the
        # call id so the model knows which request this answers.
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": str(result),
        })

    # Second call: model now has the tool result and writes the final answer.
    final = client.chat.completions.create(model=MODEL, messages=messages)
    print("  Final answer:", final.choices[0].message.content)


# ─────────────────────────────────────────────
# PART B — the same tool, defined the LangChain way
# ─────────────────────────────────────────────

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_weather_lc(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: City name, e.g. 'Austin'
    """
    # Same stub as Part A -- the point here is the *definition* style, not the logic.
    fake_data = {"austin": "91°F, sunny", "seattle": "62°F, rainy"}
    return fake_data.get(city.lower(), f"No data for {city}")


def run_langchain_tool_calling_demo():
    llm = ChatOpenAI(model=MODEL)
    llm_with_tools = llm.bind_tools([get_weather_lc])  # <- schema auto-generated

    # To use ChatBedrock/Claude instead, this is the only line that changes:
    #   from langchain_aws import ChatBedrock
    #   llm = ChatBedrock(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    #   llm_with_tools = llm.bind_tools([get_weather_lc])
    # Everything below is identical regardless of provider -- that's the point
    # of using LangChain's abstraction instead of each provider's raw SDK.

    messages = [{"role": "user", "content": "What's the weather in Seattle?"}]
    ai_msg = llm_with_tools.invoke(messages)

    if not ai_msg.tool_calls:
        print("  Model answered directly:", ai_msg.content)
        return

    messages.append(ai_msg)
    for call in ai_msg.tool_calls:
        print(f"  Model requested: {call['name']}({call['args']})")
        result = get_weather_lc.invoke(call["args"])
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": str(result),
        })

    final = llm_with_tools.invoke(messages)
    print("  Final answer:", final.content)


if __name__ == "__main__":
    print("=" * 70)
    print("PART A: raw OpenAI tool-calling API")
    print("=" * 70)
    run_raw_tool_calling_demo()

    print()
    print("=" * 70)
    print("PART B: LangChain @tool + bind_tools")
    print("=" * 70)
    run_langchain_tool_calling_demo()

    print()
    print("Both parts do the same RPC-like dance:")
    print("  LLM proposes a call -> your code executes it -> result goes back")
    print("  -> LLM writes the final answer. The model never ran any code itself.")
