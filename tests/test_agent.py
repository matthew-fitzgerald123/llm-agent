from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, ".")

from app.main import app
from app.tools import call_tool, TOOLS

client = TestClient(app)

# ── Tool unit tests — no model needed ─────────────────────

def test_calculate_basic():
    result = call_tool("calculate", {"expression": "2 + 2"})
    assert result == "4"

def test_calculate_complex():
    result = call_tool("calculate", {"expression": "(100 + 200) / 3"})
    assert float(result) == pytest.approx(100.0, rel=1e-3)

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
