# LLM Agent

A ReAct-style LLM agent served over a FastAPI REST API. The agent reasons step by step, selects from a set of tools, observes their output, and iterates until it has enough information to produce a final answer. Full run traces and multi-turn conversation history are persisted to Postgres.

This is the top-level orchestrator in a four-project ML stack:

| Project | Service | Port |
|---|---|---|
| ml-platform (P2) | Feature store + model registry | 8080 |
| ml-drift-monitor (P3) | Drift detection + scheduled retraining | 8081 |
| rag-pipeline (P4) | Hybrid BM25 + vector RAG with reranking | 8082 |
| llm-agent (P5, this) | ReAct agent with memory and SSE streaming | 8083 |

## Stack

| Component | Library |
|---|---|
| API | FastAPI + uvicorn (port 8083) |
| LLM | mlx-lm + Mistral-7B-Instruct-v0.3-4bit (Apple Silicon) |
| Agent loop | Custom ReAct parser |
| Persistence | PostgreSQL + SQLAlchemy |
| Streaming | Server-Sent Events via StreamingResponse |

## Tools

| Tool | Description | Calls |
|---|---|---|
| `calculate` | Safe AST-based arithmetic evaluator | Local |
| `search_documents` | Hybrid BM25 + dense search with cross-encoder reranking | rag-pipeline :8082 |
| `lookup_entity` | Feature store lookup by entity ID | ml-platform :8080 |
| `drift_monitor` | Drift event counts and scheduler health | ml-drift-monitor :8081 |
| `summarise` | Extractive summarisation for long tool outputs | Local |

## Setup

```bash
# Create the database
createdb llm_agent

# Install dependencies
pip install -r requirements.txt

# Set environment variables (defaults shown), e.g. in a .env file

DATABASE_URL=postgresql://localhost/llm_agent
GEN_MODEL=mlx-community/Mistral-7B-Instruct-v0.3-4bit
P2_API_URL=http://localhost:8080   # ml-platform
P3_API_URL=http://localhost:8081   # ml-drift-monitor
P4_API_URL=http://localhost:8082   # rag-pipeline
MAX_STEPS=6
```

## Running

```bash
# Start API server (downloads Mistral-7B on first run, ~4GB)
make serve

# Run end-to-end demo (exercises all 5 tools)
make demo

# Run tests (no model download required for tool unit tests)
make test
```

## Running all 4 projects together

```bash
# Terminal 1: Feature store + model registry
cd ../ml-platform && make serve       # :8080

# Terminal 2: Drift monitor
cd ../ml-drift-monitor && make serve   # :8081

# Terminal 3: RAG pipeline
cd ../rag-pipeline && make serve       # :8082

# Terminal 4: Agent (orchestrates the above)
cd ../llm-agent && make serve       # :8083
```

## API Endpoints

### Agent

| Method | Path | Description |
|---|---|---|
| POST | `/agent/run` | Run the agent on a query; returns full trace and final answer |
| POST | `/agent/run/stream` | Same but streams steps as SSE (text/event-stream) |
| POST | `/agent/chat` | Stateful multi-turn chat; pass session_id to continue |
| GET | `/agent/sessions/{id}/history` | Full turn history for a session |
| GET | `/tools` | List registered tools with descriptions and parameters |

### Observability

| Method | Path | Description |
|---|---|---|
| GET | `/agent/runs` | Recent runs (run_id, query, answer, steps taken) |
| GET | `/agent/runs/{run_id}/trace` | Full step-by-step ReAct trace |
| GET | `/agent/stats` | Success rate, average steps, per-tool usage counts |
| GET | `/health` | Liveness check |

Interactive docs at `http://localhost:8083/docs`.

## Usage examples

### One-shot query

```bash
curl -s -X POST http://localhost:8083/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the current drift status and should we retrain?"}' \
  | jq '{answer: .final_answer, steps: .steps_taken}'
```

### Multi-turn conversation

```bash
# Turn 1: ask about drift
SESSION=$(curl -s -X POST http://localhost:8083/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How many drift events have occurred today?"}' \
  | jq -r .session_id)

# Turn 2: follow-up (agent remembers turn 1)
curl -s -X POST http://localhost:8083/agent/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Given that context, what would you recommend?\", \"session_id\": \"$SESSION\"}" \
  | jq .final_answer
```

### List available tools

```bash
curl -s http://localhost:8083/tools | jq
```

### SSE streaming

```bash
curl -N -X POST http://localhost:8083/agent/run/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Search for information about model retraining thresholds.", "max_steps": 3}'

# Output:
# data: {"event": "start", "run_id": "a1b2c3d4", ...}
# data: {"event": "thought", "step": 0, "content": "I should search the knowledge base..."}
# data: {"event": "tool_call", "step": 0, "tool": "search_documents", ...}
# data: {"event": "observation", "step": 0, "content": "[1] (score=0.87) ..."}
# data: {"event": "final_answer", "run_id": "a1b2c3d4", "content": "...", "success": true}
# data: [DONE]
```

## Agent Loop

The agent follows the Thought/Action/Observation cycle until it reaches a final answer:

```
Thought: I need to check the drift monitor before recommending retraining.
Action: drift_monitor
Action Input: {"metric": "all"}

Observation: Drift summary: 7 total events, model version=v3, last drift at=2026-05-26 18:30...
             Scheduler: running=True, interval=15min, checks_run=12, auto_retrains=1...

Thought: There have been 7 drift events and 1 auto-retrain already. The model was recently
         retrained, so further manual retraining is likely unnecessary right now.
Final Answer: The drift monitor shows 7 total events with 1 automatic retrain already triggered...
```

## Conversation Memory

Each turn in `/agent/chat` is stored as a `ConversationTurn` row tied to a `session_id`. On the next request in the same session, all prior turns are injected into the prompt prefix so the agent maintains full context across requests without any client-side state.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system diagram showing how ml-platform (P2), ml-drift-monitor (P3), rag-pipeline (P4), and llm-agent (P5) interconnect.

## Project Structure

```
app/
  agent.py      model loading, ReAct loop, SSE generator, memory-aware runner
  tools.py      tool registry (calculate, search_documents, lookup_entity, drift_monitor, summarise)
  main.py       FastAPI app: run, chat, stream endpoints + observability
  models.py     SQLAlchemy: AgentRun, ToolCall, Conversation, ConversationTurn
  database.py   engine + session factory
docs/
  architecture.md   ASCII system diagrams and integration map
infra/
  main.tf       Terraform: VPC, ALB, ECS Fargate, RDS Postgres, ECR
  variables.tf
  outputs.tf
notebooks/
  demo.py       end-to-end demo queries
tests/
  test_agent.py tool unit tests + API integration tests
```

## Deployment

Terraform in `infra/` provisions ECS Fargate (4096 cpu / 16384 MB for the 4-bit quantized Mistral model) behind an ALB on port 80, with RDS Postgres in a private subnet. Set `p2_api_url` (ml-platform), `p3_api_url` (ml-drift-monitor), and `p4_api_url` (rag-pipeline) variables to point at the ALB DNS names from the other three project deployments.

```bash
cd infra
terraform init
terraform apply -var="db_password=<secret>" \
  -var="p2_api_url=http://<p2-alb-dns>" \
  -var="p3_api_url=http://<p3-alb-dns>" \
  -var="p4_api_url=http://<p4-alb-dns>"
```
