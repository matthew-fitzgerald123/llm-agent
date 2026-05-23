from __future__ import annotations
from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Text, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class AgentRun(Base):
    __tablename__ = "agent_runs"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(String, unique=True, nullable=False, index=True)
    query       = Column(Text, nullable=False)
    final_answer = Column(Text, nullable=True)
    steps_taken = Column(Integer, default=0)
    success     = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow, index=True)

class ToolCall(Base):
    __tablename__ = "tool_calls"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(String, nullable=False, index=True)
    step        = Column(Integer, nullable=False)
    tool_name   = Column(String, nullable=False)
    tool_input  = Column(JSON, nullable=False)
    tool_output = Column(Text, nullable=True)
    error       = Column(Text, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
