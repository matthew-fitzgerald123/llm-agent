from __future__ import annotations
import httpx, ast, operator, os
from dotenv import load_dotenv

load_dotenv()

P2_API_URL = os.getenv("P2_API_URL", "http://localhost:8080")
P3_API_URL = os.getenv("P3_API_URL", "http://localhost:8081")
P4_API_URL = os.getenv("P4_API_URL", "http://localhost:8082")

# ── Tool registry ─────────────────────────────────────────

TOOLS: dict[str, dict] = {}

def tool(name: str, description: str, params: dict):
    """Decorator that registers a function as a tool."""
    def decorator(fn):
        TOOLS[name] = {
            "name":        name,
            "description": description,
            "params":      params,
            "fn":          fn,
        }
        return fn
    return decorator

# ── Tool implementations ──────────────────────────────────

@tool(
    name="calculate",
    description="Evaluates a safe arithmetic expression. Use for any math: sums, ratios, percentages, averages. Input must be a valid Python arithmetic expression using only numbers and operators (+, -, *, /, **, %).",
    params={"expression": "string: arithmetic expression, e.g. (120 + 95) / 2"},
)
def calculate(expression: str) -> str:
    # The model sometimes wraps the value in its own quotes ("'2 + 2'") or a
    # trailing equals sign; unwrap before validating so a formatting quirk does
    # not surface as a tool error the agent then has to reason around.
    expression = expression.strip().strip("'\"").strip().rstrip("=").strip()
    allowed = set("0123456789+-*/.() %**")
    if not all(c in allowed for c in expression.replace(" ", "")):
        return "Error: expression contains disallowed characters"
    try:
        result = _safe_eval(expression)
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"

def _safe_eval(expr: str):
    """AST-based safe arithmetic evaluator."""
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        else:
            raise ValueError(f"Unsupported expression: {type(node)}")
    return _eval(ast.parse(expr, mode="eval").body)


@tool(
    name="search_documents",
    description="Searches the document knowledge base for relevant information. Use when you need facts, definitions, or context about a topic. Returns the most relevant passages.",
    params={"query": "string: natural language search query", "top_k": "integer: number of results, default 3"},
)
def search_documents(query: str, top_k: int = 3) -> str:
    try:
        r = httpx.post(
            f"{P4_API_URL}/query",
            json={"query": query, "top_k": top_k},
            timeout=30.0,
        )
        if r.status_code != 200:
            return f"Search unavailable (status {r.status_code})"
        data = r.json()
        chunks = data.get("chunks", [])
        if not chunks:
            return "No relevant documents found."
        return "\n\n".join([
            f"[{i+1}] (score={c['score']}) {c['text']}"
            for i, c in enumerate(chunks)
        ])
    except httpx.ConnectError:
        return "Search service unavailable: P4 RAG pipeline not running"


@tool(
    name="lookup_entity",
    description="Looks up stored features for a known entity from the feature store. Use when you need structured data about a specific entity like a user, company, or asset.",
    params={
        "entity_id": "string: the entity identifier",
        "feature_set": "string: the feature set name e.g. 'credit_signals'",
    },
)
def lookup_entity(entity_id: str, feature_set: str) -> str:
    try:
        r = httpx.get(
            f"{P2_API_URL}/features/{feature_set}/{entity_id}",
            timeout=10.0,
        )
        if r.status_code == 404:
            return f"No features found for entity '{entity_id}' in '{feature_set}'"
        if r.status_code != 200:
            return f"Feature store unavailable (status {r.status_code})"
        return str(r.json())
    except httpx.ConnectError:
        return "Feature store unavailable: P2 ML platform not running"


@tool(
    name="summarise",
    description="Summarises a long piece of text into key points. Use when context is too long to reason over directly, or to compress tool outputs before forming a final answer.",
    params={
        "text": "string: the text to summarise",
        "max_sentences": "integer: target summary length in sentences, default 3",
    },
)
def summarise(text: str, max_sentences: int = 3) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if len(sentences) <= max_sentences:
        return text
    if max_sentences == 1:
        return sentences[0] + "."
    step = max(1, len(sentences) // max_sentences)
    selected = sentences[::step][:max_sentences]
    return ". ".join(selected) + "."


@tool(
    name="drift_monitor",
    description="Checks the drift monitoring system (P3) for model drift events and scheduler status. Use when you need to know if a model is drifting, how many drift events have occurred, when retraining last ran, or the health of the drift detection pipeline. Specify metric='summary' for drift event counts, 'scheduler' for scheduler state, or 'all' for both.",
    params={"metric": "string: one of 'summary', 'scheduler', or 'all' (default 'all')"},
)
def drift_monitor(metric: str = "all") -> str:
    parts = []
    try:
        if metric in ("summary", "all"):
            r = httpx.get(f"{P3_API_URL}/drift/summary", timeout=10.0)
            if r.status_code == 200:
                d = r.json()
                parts.append(
                    f"Drift summary: {d.get('total_drift_events', 0)} total events, "
                    f"model version={d.get('model_version', 'unknown')}, "
                    f"last drift at={d.get('last_drift_at') or 'never'}, "
                    f"feature drift counts={d.get('feature_drift_counts', {})}"
                )
            else:
                parts.append(f"Drift summary unavailable (status {r.status_code})")

        if metric in ("scheduler", "all"):
            r = httpx.get(f"{P3_API_URL}/scheduler/status", timeout=10.0)
            if r.status_code == 200:
                s = r.json()
                parts.append(
                    f"Scheduler: running={s.get('running')}, "
                    f"interval={s.get('interval_minutes')}min, "
                    f"checks_run={s.get('checks_run')}, "
                    f"auto_retrains={s.get('auto_retrains')}, "
                    f"last_check={s.get('last_check') or 'never'}, "
                    f"next_check={s.get('next_check') or 'pending'}"
                )
            else:
                parts.append(f"Scheduler status unavailable (status {r.status_code})")

        if not parts:
            return "Unknown metric. Use 'summary', 'scheduler', or 'all'."
        return "\n".join(parts)

    except httpx.ConnectError:
        return "Drift monitor unavailable: P3 drift monitor not running"


def call_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call by name. Returns string output."""
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available tools: {list(TOOLS.keys())}"
    try:
        fn = TOOLS[name]["fn"]
        return fn(**inputs)
    except TypeError as e:
        return f"Error calling {name}: {e}"
    except Exception as e:
        return f"Error calling {name}: {e}"


def tools_prompt() -> str:
    """Formats tool definitions for injection into the system prompt."""
    lines = ["Available tools:\n"]
    for t in TOOLS.values():
        params = ", ".join([f"{k}: {v}" for k, v in t["params"].items()])
        lines.append(f"- {t['name']}({params})")
        lines.append(f"  {t['description']}\n")
    return "\n".join(lines)
