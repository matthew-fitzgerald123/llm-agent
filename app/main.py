from __future__ import annotations
import asyncio
import json
import logging
import uuid
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Any, Optional
from dotenv import load_dotenv

import httpx

from app.database import get_db, engine, SessionLocal
from app.models import Base, AgentRun, ToolCall, Conversation, ConversationTurn
from app.agent import run_agent, run_agent_with_memory, run_agent_stream, get_model
from app.tools import TOOLS, P2_API_URL, P3_API_URL, P4_API_URL
from app.ui import mount_ui

logger = logging.getLogger(__name__)

load_dotenv()
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield

app = FastAPI(title="LLM Agent", version="1.0.0", lifespan=lifespan)
mount_ui(app)

# ── Agent ─────────────────────────────────────────────────

class AgentReq(BaseModel):
    query: str
    max_steps: int = 6

@app.post("/agent/run", tags=["agent"])
def agent_run(req: AgentReq, db: Session = Depends(get_db)):
    result = run_agent(req.query, max_steps=req.max_steps)

    run = AgentRun(
        run_id=result.run_id,
        query=result.query,
        final_answer=result.final_answer,
        steps_taken=len(result.steps),
        success=result.success,
    )
    db.add(run)

    for i, step in enumerate(result.steps):
        if step.tool_name:
            tc = ToolCall(
                run_id=result.run_id,
                step=i,
                tool_name=step.tool_name,
                tool_input=step.tool_input,
                tool_output=step.observation,
            )
            db.add(tc)

    db.commit()

    return {
        "run_id":       result.run_id,
        "query":        result.query,
        "final_answer": result.final_answer,
        "steps_taken":  len(result.steps),
        "success":      result.success,
        "trace": [
            {
                "step":        i,
                "thought":     s.thought,
                "tool":        s.tool_name,
                "tool_input":  s.tool_input,
                "observation": s.observation,
                "is_final":    s.is_final,
            }
            for i, s in enumerate(result.steps)
        ],
    }

class ChatReq(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_steps: int = 6

@app.post("/agent/chat", tags=["agent"])
def agent_chat(req: ChatReq, db: Session = Depends(get_db)):
    session_id = req.session_id or str(uuid.uuid4())[:12]

    conv = db.query(Conversation).filter_by(session_id=session_id).first()
    if not conv:
        conv = Conversation(session_id=session_id)
        db.add(conv)
        db.flush()

    prior = (
        db.query(ConversationTurn)
        .filter_by(session_id=session_id)
        .order_by(ConversationTurn.id)
        .all()
    )
    turns = [{"role": t.role, "content": t.content} for t in prior]

    result = run_agent_with_memory(req.query, turns, max_steps=req.max_steps)

    run = AgentRun(
        run_id=result.run_id,
        query=result.query,
        final_answer=result.final_answer,
        steps_taken=len(result.steps),
        success=result.success,
    )
    db.add(run)
    db.add(ConversationTurn(session_id=session_id, role="user", content=req.query, run_id=result.run_id))
    db.add(ConversationTurn(session_id=session_id, role="agent", content=result.final_answer, run_id=result.run_id))
    db.commit()

    return {
        "session_id":   session_id,
        "run_id":       result.run_id,
        "query":        result.query,
        "final_answer": result.final_answer,
        "steps_taken":  len(result.steps),
        "success":      result.success,
        "turns_in_context": len(turns),
    }


@app.get("/agent/sessions", tags=["agent"])
def list_sessions(limit: int = 20, db: Session = Depends(get_db)):
    sessions = (
        db.query(Conversation)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "session_id": s.session_id,
            "created_at": str(s.created_at),
            "turns": db.query(ConversationTurn).filter_by(session_id=s.session_id).count(),
        }
        for s in sessions
    ]


@app.get("/agent/sessions/{session_id}/history", tags=["agent"])
def session_history(session_id: str, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter_by(session_id=session_id).first()
    if not conv:
        raise HTTPException(404, "Session not found")
    turns = (
        db.query(ConversationTurn)
        .filter_by(session_id=session_id)
        .order_by(ConversationTurn.id)
        .all()
    )
    return {
        "session_id": session_id,
        "created_at": str(conv.created_at),
        "turns": [
            {"role": t.role, "content": t.content, "run_id": t.run_id, "created_at": str(t.created_at)}
            for t in turns
        ],
    }


def _persist_stream_run(payloads: list[str]) -> None:
    """Persist a completed streamed run so it shows up in /agent/runs like batch runs.

    Best-effort: a database failure must never break an already-delivered stream."""
    try:
        events = [json.loads(p) for p in payloads]
        start = next((e for e in events if e.get("event") == "start"), None)
        final = next((e for e in events if e.get("event") == "final_answer"), None)
        if not start or not final:
            return

        observations = {
            e["step"]: e.get("content", "")
            for e in events if e.get("event") == "observation"
        }
        db = SessionLocal()
        try:
            db.add(AgentRun(
                run_id=start["run_id"],
                query=start["query"],
                final_answer=final.get("content", ""),
                steps_taken=final.get("steps_taken", 0),
                success=final.get("success", False),
            ))
            for e in events:
                if e.get("event") == "tool_call":
                    db.add(ToolCall(
                        run_id=start["run_id"],
                        step=e["step"],
                        tool_name=e["tool"],
                        tool_input=e.get("input", {}),
                        tool_output=observations.get(e["step"], ""),
                    ))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    except Exception:
        logger.warning("Failed to persist streamed run", exc_info=True)


@app.post("/agent/run/stream", tags=["agent"])
async def agent_run_stream(req: AgentReq):
    async def event_gen():
        payloads: list[str] = []
        async for payload in run_agent_stream(req.query, max_steps=req.max_steps):
            payloads.append(payload)
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"
        _persist_stream_run(payloads)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

# ── Observability ─────────────────────────────────────────

@app.get("/agent/runs", tags=["observability"])
def list_runs(limit: int = 20, db: Session = Depends(get_db)):
    runs = (
        db.query(AgentRun)
        .order_by(AgentRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "run_id":      r.run_id,
            "query":       r.query,
            "answer":      r.final_answer,
            "steps_taken": r.steps_taken,
            "success":     r.success,
            "created_at":  str(r.created_at),
        }
        for r in runs
    ]

@app.get("/agent/runs/{run_id}/trace", tags=["observability"])
def get_trace(run_id: str, db: Session = Depends(get_db)):
    run = db.query(AgentRun).filter_by(run_id=run_id).first()
    if not run:
        raise HTTPException(404, "Run not found")
    tool_calls = (
        db.query(ToolCall)
        .filter_by(run_id=run_id)
        .order_by(ToolCall.step)
        .all()
    )
    return {
        "run_id":  run.run_id,
        "query":   run.query,
        "answer":  run.final_answer,
        "success": run.success,
        "tool_calls": [
            {
                "step":   tc.step,
                "tool":   tc.tool_name,
                "input":  tc.tool_input,
                "output": tc.tool_output,
            }
            for tc in tool_calls
        ],
    }

@app.get("/agent/stats", tags=["observability"])
def stats(db: Session = Depends(get_db)):
    runs = db.query(AgentRun).all()
    if not runs:
        return {"message": "No runs yet"}
    success_rate = round(sum(1 for r in runs if r.success) / len(runs), 4)
    avg_steps = round(sum(r.steps_taken for r in runs) / len(runs), 2)
    tool_calls = db.query(ToolCall).all()
    tool_counts: dict[str, int] = {}
    for tc in tool_calls:
        tool_counts[tc.tool_name] = tool_counts.get(tc.tool_name, 0) + 1
    return {
        "total_runs":   len(runs),
        "success_rate": success_rate,
        "avg_steps":    avg_steps,
        "tool_usage":   tool_counts,
    }

@app.get("/tools", tags=["agent"])
def list_tools():
    return [
        {"name": t["name"], "description": t["description"], "params": t["params"]}
        for t in TOOLS.values()
    ]

@app.get("/services/health", tags=["observability"])
async def services_health():
    """Reachability of this service and the three siblings the tools call out to."""
    targets = [
        ("feature-store", P2_API_URL),
        ("drift-monitor", P3_API_URL),
        ("rag", P4_API_URL),
    ]

    async def ping(client: httpx.AsyncClient, url: str) -> bool:
        try:
            r = await client.get(f"{url}/health")
            return r.status_code < 500
        except Exception:
            return False

    async with httpx.AsyncClient(timeout=1.5) as client:
        results = await asyncio.gather(*(ping(client, url) for _, url in targets))

    services = [{"name": "agent", "url": "self", "ok": True}]
    services += [
        {"name": name, "url": url, "ok": ok}
        for (name, url), ok in zip(targets, results)
    ]
    return {"services": services}


@app.get("/health")
def health():
    return {"status": "ok"}
