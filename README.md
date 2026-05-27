# LLM Agent

A ReAct-style LLM agent served over a FastAPI REST API. The agent reasons step by step, selects from a set of tools, observes their output, and iterates until it has enough information to produce a final answer. Full run traces and multi-turn conversation history are logged to Postgres.

This is the top-level orchestrator in a four-project ML stack (P2 feature store, P3 drift monitor, P4 RAG pipeline, P5 agent).

## Stack

| Component | Library |
|---|---|
| API | FastAPI + uvicorn (port 8083) |
| LLM | mlx-lm + Mistral-7B-Instruct-v0.3-4bit (Apple Silicon) |
| Agent loop | Custom ReAct parser |
| Persistence | PostgreSQL + SQLAlchemy |
| Streaming | Server-Sent Events (SSE) via StreamingResponse |

## Tools

| Tool | Description | Target |
|---|---|---|
| `calculate` | Safe AST-based arithmetic evaluator | Local |
| `search_documents` | Hybrid BM25 + vector search against the RAG pipeline | P4 :8082 |
| `lookup_entity` | Feature store lookup by entity ID | P2 :8080 |
| `drift_monitor` | Drift event counts and scheduler status | P3 :8081 |
| `summarise` | Extractive summarisation for long tool outputs | Local |

## Setup

```bash
# Create database
createdb llm_agent

# Install dependencies
pip install -r requirements.txt

# Set upstream service URLs in .env (defaults shown)
# P2_API_URL=http://localhost:8080   (feature store)
# P3_API_URL=http://localhost:8081   (drift monitor)
# P4_API_URL=http://localhost:8082   (RAG pipeline)
```

## Running

```bash
# Start API server (downloads Mistral-7B on first run, ~4GB)
make serve

# Run end-to-end demo
make demo

# Run tests
make test
```

## API Endpoints

### Agent

| Method | Path | Description |
|---|---|---|
| POST | `/agent/run` | Run the agent on a query, returns trace + final answer |
| POST | `/agent/run/stream` | Same but streams events as SSE (text/event-stream) |
| POST | `/agent/chat` | Stateful multi-turn chat; pass session_id to continue a conversation |
| GET | `/agent/sessions/{id}/history` | Full conversation history for a session |
| GET | `/tools` | List registered tools with descriptions |

### Observability

| Method | Path | Description |
|---|---|---|
| GET | `/agent/runs` | Recent runs (run_id, query, answer, steps) |
| GET | `/agent/runs/{run_id}/trace` | Full step-by-step trace for a run |
| GET | `/agent/stats` | Aggregate success rate, avg steps, tool usage counts |
| GET | `/health` | Server status |

Interactive docs at `http://localhost:8083/docs`.

## Agent Loop

The agent follows the Thought/Action/Observation cycle:

```
Thought: <reasoning about what to do next>
Action: <tool_name>
Action Input: {"key": "value"}
```

The tool result is appended as `Observation:` and the model continues. When the model emits `Final Answer:` the loop exits. If `max_steps` is reached, the model is prompted once more to produce its best answer from accumulated context.

## Conversation Memory

`POST /agent/chat` accepts an optional `session_id`. Each turn (user query + agent answer) is persisted as `ConversationTurn` rows. On the next request with the same session_id, prior turns are injected into the prompt so the agent can answer follow-up questions with full context.

```bash
# Start a session
curl -s -X POST http://localhost:8083/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the drift status?"}' | jq .session_id

# Follow-up with memory
curl -s -X POST http://localhost:8083/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Should we retrain now?", "session_id": "<id above>"}'
```

## SSE Streaming

`POST /agent/run/stream` emits newline-delimited `data:` events:

```
data: {"event": "start", "run_id": "abc123", "query": "..."}
data: {"event": "thought", "step": 0, "content": "I should check drift status"}
data: {"event": "tool_call", "step": 0, "tool": "drift_monitor", "input": {"metric": "all"}}
data: {"event": "observation", "step": 0, "content": "Drift summary: 3 total events..."}
data: {"event": "final_answer", "run_id": "abc123", "content": "...", "success": true}
data: [DONE]
```

## Project Structure

```
app/
  agent.py      model loading, ReAct loop, response parser, SSE generator
  tools.py      tool registry + implementations (calculate, search, drift, etc.)
  main.py       FastAPI app, run/chat/stream endpoints, observability
  models.py     SQLAlchemy models (AgentRun, ToolCall, Conversation, ConversationTurn)
  database.py   engine + session
docs/
  architecture.md   full system diagram and integration map
notebooks/
  demo.py       demo queries exercising each tool
tests/
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system diagram showing how P2/P3/P4/P5 connect.
