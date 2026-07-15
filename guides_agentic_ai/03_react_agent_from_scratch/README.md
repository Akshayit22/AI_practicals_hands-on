# 3. ReAct — Reason + Act, Built With Zero Frameworks

## What ReAct actually is

ReAct (from the 2022 paper *"ReAct: Synergizing Reasoning and Acting in Language Models"*, the
paper your curriculum's `t303-3` AI Unplugged session is based on) is a prompting pattern where the
model alternates between two things, out loud, in the same response stream:

```
Thought: I need to know the account balance before I can approve this refund.
Action: lookup_account(customer_id="CUST-4471")
Observation: Account CUST-4471: balance=$49, status=active
Thought: The charge is within the 14-day refund window, so I can approve it.
Action: final_answer("Refund approved for $49.")
```

That's the whole pattern: **Thought → Action → Observation, repeated until a final answer.**
The "Thought" step is the key idea the paper introduced — forcing the model to explain its
reasoning *before* acting measurably improves tool-use accuracy, compared to letting it jump
straight to actions. It's the LLM equivalent of writing a comment above a tricky line of code
before you write the line.

Historically (2022), this was implemented with **plain text prompting** — no native tool-calling
API existed yet, so the "Action:" line was just text that a regex parser extracted. Modern APIs
(what you learned in module 2) give you structured `tool_calls` instead of parsing regex, but the
underlying *pattern* — reason, then act, then observe, repeat — is identical. This module builds
it by hand, on top of the real tool-calling API, so you see exactly how the loop is assembled
before you let a framework (module 5) do it for you.

## Why build it "from scratch" if frameworks exist?

Because when a LangChain agent misbehaves in production — loops forever, calls the wrong tool,
ignores a result — you need to know what's happening *inside* `AgentExecutor.invoke()`. It is
this loop. Nothing more. If you've never written it yourself, debugging it feels like magic.
After this module, it won't.

## The loop, precisely

```
messages = [user's question]
repeat up to max_iterations:
    response = llm(messages, tools=available_tools)   # "Thought" happens inside response.content
    if response has no tool_calls:
        return response.content                        # model decided it's done
    for each tool_call in response.tool_calls:
        result = execute(tool_call)                     # "Observation"
        messages.append(tool result)
    messages.append(response)                            # keep reasoning in history
give up after max_iterations (guard against infinite loops)
```

That `max_iterations` guard is not optional in real systems — an agent with no cap can loop
forever and burn your API budget if a tool keeps returning something the model doesn't like.
Always cap it.

## Run it

```bash
cd Agentic_AI/03_react_agent_from_scratch
python react_loop.py
```

Watch the printed trace — you'll see the model's reasoning text (`Thought`), the tool it picks
(`Action`), and the result you feed back (`Observation`), across multiple turns, for a question
that genuinely requires 2+ tool calls to answer.

## Next

[`04_agent_memory`](../04_agent_memory/) — right now, every run starts from a blank slate. Real
agents need to remember earlier turns (and sometimes earlier *sessions*).
