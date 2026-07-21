from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import sys
sys.path.insert(0, ".")

from app.main import app, _persist_stream_run
from app.ui import STATIC_DIR

client = TestClient(app)

# UI route

def test_root_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "LLM Agent" in r.text

def test_root_excluded_from_openapi():
    assert "/" not in app.openapi()["paths"]

def test_index_targets_real_endpoints():
    html = (STATIC_DIR / "index.html").read_text()
    assert "/agent/run/stream" in html
    assert "/services/health" in html
    assert "/agent/runs" in html
    assert "/agent/stats" in html

def test_index_handles_all_stream_event_types():
    """The UI must handle every event type run_agent_stream emits."""
    html = (STATIC_DIR / "index.html").read_text()
    for event in ("start", "thought", "tool_call", "observation", "final_answer"):
        assert event in html, f"stream event {event!r} not handled"
    assert "[DONE]" in html

def test_index_follows_system_theme_with_no_manual_toggle():
    html = (STATIC_DIR / "index.html").read_text()
    assert "prefers-color-scheme" in html
    assert "theme-toggle" not in html
    assert "localStorage" not in html

# /services/health

def test_services_health_shape():
    r = client.get("/services/health")
    assert r.status_code == 200
    services = r.json()["services"]
    names = {s["name"] for s in services}
    assert names == {"agent", "feature-store", "drift-monitor", "rag"}
    agent = next(s for s in services if s["name"] == "agent")
    assert agent["ok"] is True

# streamed-run persistence

def _stream_payloads():
    return [
        json.dumps({"event": "start", "run_id": "abc12345", "query": "What is 2+2?"}),
        json.dumps({"event": "thought", "step": 0, "content": "I should calculate."}),
        json.dumps({"event": "tool_call", "step": 0, "tool": "calculate", "input": {"expression": "2+2"}}),
        json.dumps({"event": "observation", "step": 0, "content": "4"}),
        json.dumps({"event": "final_answer", "run_id": "abc12345", "content": "4", "steps_taken": 2, "success": True}),
    ]

def test_persist_stream_run_writes_run_and_tool_calls():
    db = MagicMock()
    with patch("app.main.SessionLocal", return_value=db):
        _persist_stream_run(_stream_payloads())
    assert db.commit.called
    added = [call.args[0] for call in db.add.call_args_list]
    kinds = [type(a).__name__ for a in added]
    assert "AgentRun" in kinds
    assert "ToolCall" in kinds
    run = next(a for a in added if type(a).__name__ == "AgentRun")
    assert run.run_id == "abc12345"
    assert run.success is True
    tc = next(a for a in added if type(a).__name__ == "ToolCall")
    assert tc.tool_name == "calculate"
    assert tc.tool_output == "4"

def test_persist_stream_run_ignores_incomplete_stream():
    """A stream without a final_answer (client disconnect) must not persist."""
    db = MagicMock()
    with patch("app.main.SessionLocal", return_value=db):
        _persist_stream_run(_stream_payloads()[:2])
    assert not db.commit.called

def test_persist_stream_run_swallows_db_errors():
    db = MagicMock()
    db.commit.side_effect = RuntimeError("db down")
    with patch("app.main.SessionLocal", return_value=db):
        _persist_stream_run(_stream_payloads())  # must not raise
    assert db.rollback.called
