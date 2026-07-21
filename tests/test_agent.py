from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, ".")

from app.main import app
from app.tools import call_tool, TOOLS

client = TestClient(app)

# Tool unit tests - no model needed

def test_calculate_basic():
    result = call_tool("calculate", {"expression": "2 + 2"})
    assert result == "4"

def test_calculate_complex():
    result = call_tool("calculate", {"expression": "(100 + 200) / 3"})
    assert float(result) == pytest.approx(100.0, rel=1e-3)

def test_calculate_strips_model_added_quotes():
    """The model sometimes emits {"expression": "'100 + 250'"}; the quotes are its
    formatting, not part of the expression, and must not fail validation."""
    assert call_tool("calculate", {"expression": "'100 + 250'"}) == "350"
    assert call_tool("calculate", {"expression": '"7 * 8"'}) == "56"


def test_calculate_strips_trailing_equals():
    assert call_tool("calculate", {"expression": "12 * 12 ="}) == "144"


def test_calculate_rejects_unsafe():
    result = call_tool("calculate", {"expression": "__import__('os')"})
    assert "Error" in result

def test_calculate_ratio():
    result = call_tool("calculate", {"expression": "120 / 80"})
    assert float(result) == pytest.approx(1.5, rel=1e-3)

def test_summarise_short_text():
    text = "Machine learning is powerful. It learns from data. Models improve over time."
    result = call_tool("summarise", {"text": text, "max_sentences": 2})
    assert len(result) > 0

def test_summarise_long_text():
    text = ". ".join([f"Sentence {i}" for i in range(20)])
    result = call_tool("summarise", {"text": text, "max_sentences": 3})
    assert result.count(".") <= 5  # compressed

def test_unknown_tool():
    result = call_tool("nonexistent_tool", {})
    assert "Error" in result

def test_tools_registered():
    assert "calculate" in TOOLS
    assert "search_documents" in TOOLS
    assert "lookup_entity" in TOOLS
    assert "summarise" in TOOLS
    assert "drift_monitor" in TOOLS


def test_drift_monitor_unavailable():
    result = call_tool("drift_monitor", {"metric": "all"})
    assert "unavailable" in result.lower() or "drift" in result.lower()


def test_drift_monitor_summary_only():
    result = call_tool("drift_monitor", {"metric": "summary"})
    assert isinstance(result, str) and len(result) > 0


def test_drift_monitor_scheduler_only():
    result = call_tool("drift_monitor", {"metric": "scheduler"})
    assert isinstance(result, str) and len(result) > 0


def test_drift_monitor_unknown_metric():
    result = call_tool("drift_monitor", {"metric": "bogus"})
    assert "Unknown metric" in result or "unavailable" in result.lower()

# ── API tests ─────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_list_tools():
    r = client.get("/tools")
    assert r.status_code == 200
    names = [t["name"] for t in r.json()]
    assert "calculate" in names
    assert "summarise" in names

def test_agent_run_math_query():
    r = client.post("/agent/run", json={
        "query": "What is (450 + 320) divided by 7? Round to 2 decimal places.",
        "max_steps": 4,
    })
    assert r.status_code == 200
    data = r.json()
    assert "final_answer" in data
    assert "run_id" in data
    assert data["steps_taken"] >= 1

def test_agent_trace_logged():
    r = client.post("/agent/run", json={
        "query": "Calculate 15% of 840.",
        "max_steps": 3,
    })
    run_id = r.json()["run_id"]
    r = client.get(f"/agent/runs/{run_id}/trace")
    assert r.status_code == 200
    assert r.json()["run_id"] == run_id

def test_agent_stats():
    r = client.get("/agent/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_runs" in data
    assert "success_rate" in data

def test_agent_runs_list():
    r = client.get("/agent/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_agent_stream_returns_sse():
    import json as _json
    with client.stream("POST", "/agent/run/stream", json={
        "query": "What is 3 * 7?",
        "max_steps": 4,
    }) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        lines = []
        for line in r.iter_lines():
            if line.startswith("data: "):
                lines.append(line[6:])
        assert "[DONE]" in lines
        events = [_json.loads(l) for l in lines if l != "[DONE]"]
        event_types = [e["event"] for e in events]
        assert "start" in event_types
        assert "final_answer" in event_types


def test_agent_chat_new_session():
    r = client.post("/agent/chat", json={"query": "What is 12 times 12?"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert "final_answer" in data
    assert data["turns_in_context"] == 0


def test_agent_chat_remembers_prior_turn():
    r1 = client.post("/agent/chat", json={"query": "Calculate 100 + 200."})
    assert r1.status_code == 200
    session_id = r1.json()["session_id"]

    r2 = client.post("/agent/chat", json={"query": "What did I ask you about?", "session_id": session_id})
    assert r2.status_code == 200
    assert r2.json()["turns_in_context"] == 2  # user + agent from turn 1


def test_session_history_endpoint():
    r = client.post("/agent/chat", json={"query": "Calculate 7 * 8."})
    session_id = r.json()["session_id"]
    r2 = client.get(f"/agent/sessions/{session_id}/history")
    assert r2.status_code == 200
    data = r2.json()
    assert data["session_id"] == session_id
    assert len(data["turns"]) == 2
    assert data["turns"][0]["role"] == "user"
    assert data["turns"][1]["role"] == "agent"


def test_session_history_not_found():
    r = client.get("/agent/sessions/nonexistent-session-xyz/history")
    assert r.status_code == 404


def test_agent_stream_final_answer_has_run_id():
    import json as _json
    with client.stream("POST", "/agent/run/stream", json={
        "query": "Calculate 50 + 50.",
        "max_steps": 4,
    }) as r:
        lines = [l[6:] for l in r.iter_lines() if l.startswith("data: ") and l[6:] != "[DONE]"]
        events = [_json.loads(l) for l in lines]
        final = next((e for e in events if e["event"] == "final_answer"), None)
        assert final is not None
        assert "run_id" in final
        assert "content" in final


# Parser: imagined-trajectory handling

def test_parse_response_action_wins_over_imagined_final_answer():
    """When the model emits Action, a fabricated Observation, and a Final Answer
    in one generation, only the Action (which came first) must be honored."""
    from app.agent import parse_response
    text = (
        "Thought: I need the drift count.\n"
        "Action: drift_monitor\n"
        'Action Input: {"metric": "summary"}\n'
        "Observation: 4 drift events\n"
        "Final Answer: There have been 4 drift events."
    )
    step = parse_response(text)
    assert step.is_final is False
    assert step.tool_name == "drift_monitor"
    assert step.tool_input == {"metric": "summary"}

def test_parse_response_genuine_final_answer_still_works():
    from app.agent import parse_response
    text = "Thought: I have enough information.\nFinal Answer: The total is 8."
    step = parse_response(text)
    assert step.is_final is True
    assert step.final_answer == "The total is 8."

def test_truncate_at_observation():
    from app.agent import truncate_at_observation
    text = "Thought: x\nAction: calculate\nAction Input: {}\nObservation: fake\nmore"
    assert "fake" not in truncate_at_observation(text)
    assert "Action: calculate" in truncate_at_observation(text)


# Loop control: termination after a good observation

def test_is_repeat_call_detects_identical_call():
    from app.agent import AgentStep, is_repeat_call
    prior = AgentStep(tool_name="calculate", tool_input={"expression": "2+2"})
    repeat = AgentStep(tool_name="calculate", tool_input={"expression": "2+2"})
    assert is_repeat_call(repeat, [prior]) is True


def test_is_repeat_call_allows_different_input():
    from app.agent import AgentStep, is_repeat_call
    prior = AgentStep(tool_name="calculate", tool_input={"expression": "2+2"})
    fresh = AgentStep(tool_name="calculate", tool_input={"expression": "3+3"})
    assert is_repeat_call(fresh, [prior]) is False


def test_is_repeat_call_allows_different_tool():
    from app.agent import AgentStep, is_repeat_call
    prior = AgentStep(tool_name="calculate", tool_input={"expression": "2+2"})
    other = AgentStep(tool_name="summarise", tool_input={"expression": "2+2"})
    assert is_repeat_call(other, [prior]) is False


def test_is_repeat_call_ignores_steps_without_a_tool():
    from app.agent import AgentStep, is_repeat_call
    thought_only = AgentStep(thought="just thinking")
    assert is_repeat_call(thought_only, [thought_only]) is False


def test_is_repeat_call_empty_history():
    from app.agent import AgentStep, is_repeat_call
    step = AgentStep(tool_name="calculate", tool_input={"expression": "2+2"})
    assert is_repeat_call(step, []) is False


def test_continue_prompt_directs_toward_final_answer():
    """The post-observation nudge must tell the model it may answer now."""
    from app.agent import CONTINUE_PROMPT
    assert "Final Answer" in CONTINUE_PROMPT
    assert "Observation" in CONTINUE_PROMPT


def test_continue_prompt_demands_the_observed_value():
    """Must tell the model to reuse the tool's exact number.

    An earlier wording said 'never restate the Observation', which pushed the
    model into inventing a different figure than the tool returned."""
    from app.agent import CONTINUE_PROMPT
    assert "exactly as given" in CONTINUE_PROMPT
    assert "never recompute" in CONTINUE_PROMPT.lower()
    assert "never restate" not in CONTINUE_PROMPT.lower()


def test_force_final_prompt_stops_tool_use():
    from app.agent import FORCE_FINAL_PROMPT
    assert "Final Answer" in FORCE_FINAL_PROMPT
    assert "Stop calling tools" in FORCE_FINAL_PROMPT
