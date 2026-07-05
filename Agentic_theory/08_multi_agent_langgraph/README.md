# 8. Multi-Agent Systems With LangGraph — Supervisor Pattern

## Why one agent isn't always enough

Module 5's agent had two tools (calculator, HR retriever) and one system prompt describing both.
That's fine for 2 tools. Now imagine 15 tools spanning IT, Finance, HR, and Sales — one prompt
trying to describe all of them gets long, the model gets worse at picking the *right* tool among
many similar ones, and every team touching this "one big agent" steps on each other's prompt
changes. The fix mirrors something you already do in backend systems: **split by responsibility**.
Instead of one agent that knows about everything, build several narrow agents, each with its own
small toolset and prompt, and add a **router** in front that sends each request to the right one.

This is exactly microservices vs. a monolith, applied to agents:

| Monolith agent (module 5) | Multi-agent system (this module) |
|---|---|
| One prompt describing every tool | Each sub-agent's prompt only describes its own tools |
| Adding a tool risks breaking unrelated behavior | Sub-agents are isolated — IT team can change the IT agent without touching Finance |
| One model call picks from *all* tools | Supervisor picks an agent; that agent picks from its own smaller toolset |

## The supervisor pattern

```
                     ┌───────────────┐
   user query  ───►  │  Supervisor   │  (classifies: "this is an IT question")
                     └───────┬───────┘
                             │ routes to exactly one specialist
              ┌──────────────┼──────────────┐
              ▼                             ▼
      ┌───────────────┐             ┌───────────────┐
      │   IT Agent     │             │ Finance Agent │
      │ tools:         │             │ tools:        │
      │  - read_it_docs│             │  - read_fin.. │
      │  - web_search  │             │  - web_search │
      └───────────────┘             └───────────────┘
```

This is the exact structure your curriculum's Week 5 assignment (`i301-5`) asks you to build:
a Supervisor Agent that classifies queries as IT or Finance and routes accordingly, where each
specialist has its own tools. This module's code is a direct, working reference implementation of
that pattern — read it, then adapt it for the assignment.

## Why LangGraph instead of plain LangChain for this

`AgentExecutor` (module 5) runs *one* agent's loop. Coordinating *several* agents — deciding which
one runs, in what order, whether to loop back to the supervisor after a sub-agent finishes — needs
something to model that flow explicitly. **LangGraph represents your agent system as a graph**:
nodes are agents (or plain functions), edges are "what runs next," and a shared **state** object
flows through every node. This is a much better fit for multi-agent systems than nesting
`AgentExecutor`s inside each other, because the routing logic (who runs next) is explicit and
inspectable instead of buried inside prompts.

Key building blocks used in the code:
- **`StateGraph(AgentState)`** — the graph, parameterized by a shared state schema (here, just the
  running message list + a `next` field the supervisor writes to).
- **`create_react_agent(llm, tools)`** — LangGraph's own prebuilt helper that builds a full
  ReAct-style agent (the same pattern as module 3/5) as a ready-to-use graph node. This is what
  each specialist (IT, Finance) is built from — you don't need to hand-write `AgentExecutor` again.
- **`add_conditional_edges(from_node, routing_fn, mapping)`** — this is the routing table: after
  the supervisor node runs, `routing_fn` reads the state and decides which node runs next.

## Run it

```bash
cd Agentic_AI/08_multi_agent_langgraph
pip install langgraph   # see requirements.txt
python supervisor_multi_agent.py
```

Two queries are run — one IT, one Finance — and you'll see the supervisor's routing decision
printed before each specialist agent takes over.

## Next

[`09_observability_evals`](../09_observability_evals/) — once you have agents (single or multi),
the next problem is: how do you know what they're doing in production, and whether they're any good?
