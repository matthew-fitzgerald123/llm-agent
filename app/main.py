from __future__ import annotations
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Any
from dotenv import load_dotenv

from app.database import get_db, engine
from app.models import Base, AgentRun, ToolCall
from app.agent import run_agent, get_model
from app.tools import TOOLS

load_dotenv()
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield

app = FastAPI(title="LLM Agent", version="1.0.0", lifespan=lifespan)

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

@app.get("/health")
def health():
    return {"status": "ok"}
