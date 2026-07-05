# 2. Tool Calling (a.k.a. Function Calling)

## The problem it solves

In module 1's agent, we made the LLM invent its own JSON format for describing actions, and we
parsed it by hand with `json.loads()`. That works, but it's fragile: the model can typo a field
name, forget to close a brace, or invent an action that doesn't exist, and your parser breaks.

**Tool calling** is the model provider (OpenAI, Anthropic, AWS Bedrock) baking this pattern
directly into the model's training and API, so it's reliable instead of "hope the JSON parses."

## How it actually works — think of it as an RPC contract

You already know this shape from building APIs. Tool calling is just:

1. **You publish a schema** — like an OpenAPI spec, but for one function:
   ```json
   {
     "name": "get_weather",
     "description": "Get current weather for a city",
     "parameters": {
       "type": "object",
       "properties": { "city": { "type": "string" } },
       "required": ["city"]
     }
   }
   ```
2. **You send this schema alongside the user's message** to the LLM API.
3. **The model replies with either:**
   - normal text (it didn't need a tool), **or**
   - a structured `tool_calls` field: `{"name": "get_weather", "arguments": "{\"city\": \"Austin\"}"}`
4. **Your code — not the model — actually executes the function.** The LLM never runs code. It
   only ever outputs "please call this function with these arguments." You parse `arguments`
   (guaranteed valid JSON matching your schema), run the real Python function, and send the
   return value back to the model as a new message.
5. The model sees the result and either calls another tool or writes the final answer.

This is the exact same request/response idea as a webhook: you don't execute the caller's code,
you just get a validated payload telling you what they want.

## The critical mental model

**The LLM never executes anything.** It only *proposes* a function call. Your application code is
what actually runs `get_weather(city="Austin")`. This means:
- You can validate/sanitize arguments before running anything (never trust LLM output blindly —
  same as never trusting client input in a normal API).
- You control side effects. The model can "want" to delete a database row; whether that actually
  happens is 100% up to your code.
- Tool calling doesn't require an agent loop — you can use it for a single call-and-respond too.
  The *loop* (call tool → feed result back → maybe call another tool) is what turns it into an agent.

## Two ways to define tools in this file

- **Raw OpenAI API** — you write the JSON schema by hand. Good for understanding the wire format.
- **LangChain `@tool`** — you decorate a normal Python function with a type-hinted signature and
  docstring, and LangChain generates the schema for you from that. This is what you'll use in
  every real project (module 5 onward) — nobody hand-writes JSON schemas in production.

## Run it

```bash
cd Agentic_AI/02_tool_calling
python tool_calling_basics.py
```

## Next

[`03_react_agent_from_scratch`](../03_react_agent_from_scratch/) — put tool calling inside a real
loop and implement the ReAct pattern by hand.
