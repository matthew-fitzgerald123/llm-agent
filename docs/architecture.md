# Architecture: LLM Agent (Project 5)

The agent is the top-level orchestrator in a four-project ML stack. It takes natural-language questions, reasons over them using a ReAct loop, and calls specialized downstream services to fetch real data before producing a final answer.

## System diagram

```
                        User / Client
                             |
                      POST /agent/chat
                      POST /agent/run
                      POST /agent/run/stream (SSE)
                             |
                    +--------v---------+
                    |   LLM Agent      |
                    |   FastAPI :8083  |
                    |                  |
                    |  ReAct loop      |
                    |  Mistral-7B-4bit |
                    |  (Apple Silicon) |
                    |                  |
                    |  Session memory  |
                    |  (Postgres)      |
                    +--+--+--+--+------+
                       |  |  |  |
          +------------+  |  |  +------------------+
          |               |  |                     |
          v               v  v                     v
  +-------+------+  +-----+--+------+  +-----------+------+
  | calculate    |  | drift_monitor |  | search_documents |
  | (local AST)  |  | P3 :8081      |  | P4 RAG :8082     |
  +--------------+  |               |  |                  |
                    | /drift/summary|  | BM25 + dense     |
                    | /scheduler/   |  | cross-encoder    |
                    |  status       |  | reranking        |
                    +-------+-------+  +--------+---------+
                            |                   |
                    +-------v-------+   +--------v---------+
                    | ML Drift      |   | RAG Pipeline     |
                    | Monitor (P3)  |   | (P4)             |
                    |               |   |                  |
                    | ADWIN drift   |   | ChromaDB (EFS)   |
                    | APScheduler   |   | BM25 index       |
                    | auto-retrain  |   | citations        |
                    +-------+-------+   +------------------+
                            |
                    +-------v-------+
                    | ML Platform   |
                    | (P2) :8080    |
                    |               |
                    | Feature store |  <-- also used by lookup_entity tool
                    | Model registry|
                    | MLflow        |
                    +---------------+
```

## Data flow for a multi-turn conversation

```
Turn 1: "What is the drift status?"
  Agent --> drift_monitor("all")
        --> GET P3 /drift/summary + /scheduler/status
        <-- drift event counts, last retrain time
  Agent --> Final Answer: "3 drift events detected in the last hour..."

Turn 2: "Should we retrain now?"
  Agent receives: [prior turn 1 context injected into prompt]
  Agent --> search_documents("retraining thresholds")
        --> POST P4 /query
        <-- relevant docs from knowledge base
  Agent --> Final Answer: "Based on the 3 drift events and your threshold of 5..."
```

## Agent step flow (ReAct)

```
Prompt (system + tools + [memory] + question)
         |
         v
    Mistral-7B generates:
    Thought: ...
    Action: <tool>
    Action Input: {...}
         |
         v
    Tool dispatcher --> tool function --> external service
         |
         v
    Observation: <tool output>
         |
         v
    Append to history, repeat
         |
         v (when model emits Final Answer:)
    Store AgentRun + ToolCalls + ConversationTurns in Postgres
    Return response to caller
```

## Project integrations

| Tool | Target Project | Endpoint |
|---|---|---|
| `drift_monitor` | P3 (ml-drift-monitor :8081) | GET /drift/summary, GET /scheduler/status |
| `search_documents` | P4 (rag-pipeline :8082) | POST /query |
| `lookup_entity` | P2 (ml-platform :8080) | GET /features/{set}/{entity_id} |
| `calculate` | Local | AST evaluator, no network call |
| `summarise` | Local | Extractive sentence sampling |

## Infrastructure

Each project deploys independently on AWS ECS Fargate behind an ALB:

```
Internet --> ALB :80 --> ECS Fargate task --> FastAPI app
                              |
                         RDS Postgres (private subnet)
                         ECR (image registry)
```

Project 4 additionally mounts an EFS volume for ChromaDB persistence across task restarts.

## Key design decisions

- **ReAct over chain-of-thought**: the structured Thought/Action/Observation loop makes it easy to log, debug, and replay agent reasoning.
- **SSE streaming**: `/agent/run/stream` emits events as each step completes, so UIs can show a live trace without waiting for the full run.
- **Session memory in Postgres**: storing conversation turns in the DB (not in-memory) means memory survives server restarts and can be audited.
- **Lazy model loading**: Mistral-7B loads on first request (not at startup) so the server starts instantly and tests that don't need the model skip the 4GB download.
- **Tool isolation**: each tool handles its own error path and returns a string, so the agent loop never crashes on a failed tool call.
