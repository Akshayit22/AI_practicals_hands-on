# 1. Workflow vs. Agent — and the Agent Loop

## The question this answers

You already write workflows every day: `validate → process → save → respond`. That's just a
function calling other functions in a fixed order. So what's actually *new* about an "AI agent"?
Isn't it just... a function that calls an LLM?

The honest answer: an agent is a workflow where **one of the steps is "ask the LLM what the next
step should be"**, and you loop on that until the LLM says "I'm done." That's it. No magic.

## Workflow: you write the `if/else`

```python
def handle_ticket(ticket: str) -> str:
    category = classify(ticket)          # step 1 — always runs
    if category == "billing":
        return billing_handler(ticket)   # step 2a
    elif category == "technical":
        return tech_handler(ticket)      # step 2b
    else:
        return general_handler(ticket)   # step 2c
```

Every possible path through this function was decided by *you*, at code-writing time. The LLM
(inside `classify`) only fills in one blank. This is predictable, testable, cheap, and fast — and
for 80% of backend problems, it's the right answer. Don't reach for an agent if a workflow works.

## Agent: the LLM writes the `if/else`, one step at a time, at runtime

```python
def handle_ticket_as_agent(ticket: str) -> str:
    messages = [{"role": "user", "content": ticket}]
    while True:
        decision = ask_llm_what_to_do_next(messages)   # LLM picks the next action
        if decision["action"] == "final_answer":
            return decision["content"]
        result = run_action(decision["action"], decision["args"])  # you execute it
        messages.append({"role": "assistant", "content": str(decision)})
        messages.append({"role": "user", "content": f"Result: {result}"})
```

Nobody wrote `if category == "billing"` anywhere. The LLM looks at the conversation so far and
decides, turn by turn: "I should look up the account", then "I should check refund policy", then
"I have enough to answer." **The number of steps and their order are not known in advance** — they
emerge from the conversation. That's the defining property of an agent.

## Why this matters (the actual tradeoff)

| | Workflow | Agent |
|---|---|---|
| Steps known ahead of time? | Yes | No — decided at runtime |
| Predictable / testable | Easy | Hard (non-deterministic) |
| Handles novel/unexpected input | Poorly — breaks outside the paths you coded | Well — LLM improvises |
| Cost per request | 1 LLM call (usually) | N LLM calls (1 per loop turn) |
| Debuggability | Stack trace | Need tracing/logs (see module 9) |
| When to use | Requirements are known and stable | Task shape varies request-to-request |

Rule of thumb from Anthropic's own engineering guidance (linked in your curriculum,
`t301-3`): **start with the simplest workflow that solves the problem. Only introduce an agent
loop when the task genuinely requires dynamic, multi-step decisions that you can't enumerate in
advance.** Agents are slower and more expensive per request than workflows — that's the price of
flexibility, not a free upgrade.

## Run it

```bash
cd Agentic_AI/01_agent_fundamentals
python workflow_vs_agent.py
```

The script runs the *same* support-ticket scenario twice — once as a rigid workflow, once as an
agent loop with two toy tools — so you can see the control flow difference directly in the printed
trace. There's no "real" tool-calling API used yet (that's module 2) — the agent here just asks
the LLM to reply with a small JSON object describing its decision, which we parse ourselves. This
is deliberately the most primitive possible version of an agent, so the loop itself is the star.

## Next

[`02_tool_calling`](../02_tool_calling/) — replace our hand-rolled JSON decision format with the
real function-calling API that OpenAI/Anthropic/Bedrock models support natively.
