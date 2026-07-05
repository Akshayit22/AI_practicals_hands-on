"""
Workflow vs. Agent — the same problem solved two ways
--------------------------------------------------------------
Read Agentic_AI/01_agent_fundamentals/README.md first.

Scenario: a customer support ticket comes in. We need to figure out what
kind of request it is and produce a reply.

  PART A — WORKFLOW
    Fixed steps, written by us: classify -> pick a branch -> respond.
    The LLM is used *inside* one step, but never decides the control flow.

  PART B — AGENT (hand-rolled, no framework, no official tool-calling API)
    The LLM sees the ticket + a list of "actions" it's allowed to take and,
    turn by turn, decides which action to take next. We just execute
    whatever it asks for and feed the result back. This is the rawest
    possible version of an agent loop -- intentionally not using OpenAI's
    real function-calling API yet (that's module 2) so nothing hides the
    control-flow idea.

Requires: OPENAI_API_KEY in your environment.
    export OPENAI_API_KEY="sk-..."

Swap-in note: every `client.chat.completions.create(...)` call below can be
replaced with `ChatBedrock(model_id="us.anthropic.claude-3-5-sonnet-...")`
from `langchain_aws`, same as your aws_rag_ai.py, if you'd rather use Claude
via Bedrock. The loop logic doesn't change -- only the model call does.
"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────
# PART A — WORKFLOW: fixed steps, decided by us at code-writing time
# ─────────────────────────────────────────────

def classify_ticket(ticket: str) -> str:
    """The only place the LLM is used in the workflow version."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Classify this support ticket into exactly one word: "
                "'billing', 'technical', or 'general'.\n\n"
                f"Ticket: {ticket}\n\nRespond with only the one word."
            ),
        }],
    )
    return resp.choices[0].message.content.strip().lower()


def billing_handler(ticket: str) -> str:
    return f"[billing team] Routed for refund/invoice review: '{ticket}'"


def tech_handler(ticket: str) -> str:
    return f"[technical team] Routed to on-call engineer: '{ticket}'"


def general_handler(ticket: str) -> str:
    return f"[general support] Routed to first-line support: '{ticket}'"


def handle_ticket_as_workflow(ticket: str) -> str:
    """
    Every branch here was written by a human. No matter what the ticket
    says, execution can only ever go through one of these three paths.
    """
    category = classify_ticket(ticket)
    print(f"  [workflow] classifier decided: {category}")

    if "billing" in category:
        return billing_handler(ticket)
    elif "technical" in category:
        return tech_handler(ticket)
    else:
        return general_handler(ticket)


# ─────────────────────────────────────────────
# PART B — AGENT: the LLM decides the next step, turn by turn
# ─────────────────────────────────────────────

# The "tools" our agent is allowed to use. In a real system these would
# hit a database/API. Here they're stubs so you can focus on the loop.
def lookup_account(customer_id: str) -> str:
    return f"Account {customer_id}: Plan=Pro, last charge=$49 on 2026-06-01, status=active"


def check_refund_policy(topic: str) -> str:
    return "Refund policy: full refund within 14 days of charge, no questions asked."


AVAILABLE_ACTIONS = {
    "lookup_account": lookup_account,
    "check_refund_policy": check_refund_policy,
}

AGENT_SYSTEM_PROMPT = f"""
You are a support triage agent. You solve tickets by taking actions, one at a time.

Available actions:
- lookup_account(customer_id: str) -> account details
- check_refund_policy(topic: str) -> policy text
- final_answer(content: str) -> ends the loop and returns your reply to the customer

Each turn, respond with ONLY a JSON object, no other text, in this exact shape:
{{"action": "<action name>", "args": {{...}}, "thought": "<why you're doing this>"}}

Call actions until you have enough information, then call final_answer with your reply.
"""


def ask_llm_what_to_do_next(messages: list[dict]) -> dict:
    """The 'decision' step of the agent loop. Returns a parsed action dict."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},  # forces valid JSON back
    )
    raw = resp.choices[0].message.content
    return json.loads(raw)


def handle_ticket_as_agent(ticket: str, max_turns: int = 5) -> str:
    """
    Nobody wrote 'if category == billing' here. The LLM looks at the
    growing conversation and decides, turn by turn, what to do next.
    The number of turns is not known ahead of time -- it emerges.
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Ticket: {ticket}"},
    ]

    for turn in range(max_turns):
        decision = ask_llm_what_to_do_next(messages)
        action = decision.get("action")
        args = decision.get("args", {})
        thought = decision.get("thought", "")
        print(f"  [agent turn {turn+1}] thought: {thought}")
        print(f"  [agent turn {turn+1}] action:  {action}({args})")

        if action == "final_answer":
            return args.get("content", "(no content)")

        if action not in AVAILABLE_ACTIONS:
            # The LLM hallucinated an action we don't support -- tell it so
            # and let it try again. This is exactly why real tool-calling
            # APIs (module 2) validate the schema for you.
            result = f"Error: unknown action '{action}'"
        else:
            result = AVAILABLE_ACTIONS[action](**args)

        messages.append({"role": "assistant", "content": json.dumps(decision)})
        messages.append({"role": "user", "content": f"Result: {result}"})

    return "(agent did not finish within max_turns)"


# ─────────────────────────────────────────────
# Run both on the same ticket to compare control flow
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ticket = (
        "I was charged twice for my subscription this month, customer id CUST-4471. "
        "Can I get a refund for the duplicate charge?"
    )

    print("=" * 70)
    print("PART A: WORKFLOW (fixed steps)")
    print("=" * 70)
    print(handle_ticket_as_workflow(ticket))

    print()
    print("=" * 70)
    print("PART B: AGENT (LLM decides the steps)")
    print("=" * 70)
    print(handle_ticket_as_agent(ticket))

    print()
    print("Notice: the workflow took exactly 1 LLM call (classification).")
    print("The agent took multiple calls and *chose* to look up the account")
    print("and the refund policy before answering -- nobody coded that path.")
