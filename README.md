# LLM Agent

A ReAct-style LLM agent served over a FastAPI REST API. The agent reasons step by step, selects from a set of tools, observes their output, and iterates until it has enough information to produce a final answer. Full run traces are logged to Postgres.

## Stack

| Component | Library |
|---|---|
| API | FastAPI + uvicorn (port 8083) |
| LLM | mlx-lm + Mistral-7B-Instruct-v0.3-4bit (Apple Silicon) |
| Agent loop | Custom ReAct parser |
| Persistence | PostgreSQL + SQLAlchemy |

## Tools

| Tool | Description |
|---|---|
| `calculate` | Safe AST-based arithmetic evaluator |
| `search_documents` | Vector search against the RAG pipeline (project_04, port 8082) |
| `lookup_entity` | Feature store lookup by entity ID (project_02, port 8080) |
| `summarise` | Extractive summarisation for long tool outputs |

## Setup

```bash
# Create database
createdb llm_agent

# Install dependencies
pip install -r requirements.txt

# Optional: set upstream service URLs in .env
# P2_API_URL=http://localhost:8080   (feature store)
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

The agent follows the Thought/Action/Observation cycle. Each step, the model emits:

```
Thought: <reasoning>
Action: <tool_name>
Action Input: {"key": "value"}
```

The tool result is appended as `Observation:` and the model continues. When the model emits `Final Answer:` the loop exits. If `max_steps` is reached without a final answer, the model is prompted once more to produce its best answer from accumulated context.

## Project Structure

```
app/
  agent.py      model loading, ReAct loop, response parser
  tools.py      tool registry + implementations
  main.py       FastAPI app, run + trace persistence
  models.py     SQLAlchemy models (AgentRun, ToolCall)
  database.py   engine + session
notebooks/
  demo.py       demo queries exercising each tool
tests/
```
