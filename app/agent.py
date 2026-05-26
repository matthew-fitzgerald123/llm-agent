from __future__ import annotations
import re, json, uuid, asyncio
from dataclasses import dataclass, field
from mlx_lm import load, generate
from dotenv import load_dotenv
import os

from app.tools import call_tool, tools_prompt, TOOLS

load_dotenv()

# ── Model ─────────────────────────────────────────────────

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        model_id = os.getenv("GEN_MODEL", "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        print(f"Loading agent model: {model_id}")
        _model, _tokenizer = load(model_id)
        print("Agent model ready.")
    return _model, _tokenizer

# ── Agent step types ──────────────────────────────────────

@dataclass
class Thought:
    content: str

@dataclass
class ToolUse:
    tool_name: str
    tool_input: dict

@dataclass
class Observation:
    content: str

@dataclass
class FinalAnswer:
    content: str

@dataclass
class AgentStep:
    thought: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    observation: str = ""
    is_final: bool = False
    final_answer: str = ""

# ── System prompt ─────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise AI assistant that reasons step by step and uses tools to answer questions accurately.

{tools}

You MUST follow this exact format for every response:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"key": "value"}}

Or if you have enough information to answer:

Thought: [your reasoning]
Final Answer: [your complete answer]

Rules:
- Always start with Thought:
- Only call one tool per response
- Action Input must be valid JSON
- Never make up facts — use tools to get real information
- If a tool returns an error, try a different approach
- Give Final Answer only when you have enough information
"""

# ── Parser ────────────────────────────────────────────────

def parse_response(text: str) -> AgentStep:
    step = AgentStep()

    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
    if thought_match:
        step.thought = thought_match.group(1).strip()

    final_match = re.search(r"Final Answer:\s*(.+?)$", text, re.DOTALL)
    if final_match:
        step.is_final = True
        step.final_answer = final_match.group(1).strip()
        return step

    action_match = re.search(r"Action:\s*(\w+)", text)
    if action_match:
        step.tool_name = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)
    if input_match:
        try:
            step.tool_input = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            raw = input_match.group(1)
            pairs = re.findall(r'"(\w+)":\s*"([^"]*)"', raw)
            step.tool_input = {k: v for k, v in pairs}

    return step

# ── Agent runner ──────────────────────────────────────────

@dataclass
class RunResult:
    run_id: str
    query: str
    final_answer: str
    steps: list[AgentStep]
    success: bool

def run_agent(query: str, max_steps: int = None) -> RunResult:
    if max_steps is None:
        max_steps = int(os.getenv("MAX_STEPS", 6))

    model, tokenizer = get_model()
    run_id = str(uuid.uuid4())[:8]

    system = SYSTEM_PROMPT.format(tools=tools_prompt())
    history = f"[INST] {system}\n\nQuestion: {query} [/INST]"

    steps = []
    final_answer = ""
    success = False

    for step_num in range(max_steps):
        response = generate(
            model,
            tokenizer,
            prompt=history,
            max_tokens=512,
            verbose=False,
        )

        step = parse_response(response)
        steps.append(step)

        if step.is_final:
            final_answer = step.final_answer
            success = True
            break

        if step.tool_name:
            observation = call_tool(step.tool_name, step.tool_input)
            step.observation = observation
            history += f"\n{response}\nObservation: {observation}\n[INST] Continue. [/INST]"
        else:
            history += f"\n{response}\n[INST] Please use the format: Thought/Action/Action Input or Thought/Final Answer [/INST]"

    if not success:
        prompt = history + "\n[INST] You have reached the step limit. Give your best Final Answer now based on what you have found. [/INST]"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        final_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL)
        if final_match:
            final_answer = final_match.group(1).strip()
        else:
            final_answer = response.strip()

    return RunResult(
        run_id=run_id,
        query=query,
        final_answer=final_answer,
        steps=steps,
        success=success,
    )


def run_agent_with_memory(query: str, prior_turns: list[dict], max_steps: int = None) -> RunResult:
    """Run the agent with prior conversation turns injected into context."""
    if max_steps is None:
        max_steps = int(os.getenv("MAX_STEPS", 6))

    model, tokenizer = get_model()
    run_id = str(uuid.uuid4())[:8]

    system = SYSTEM_PROMPT.format(tools=tools_prompt())

    if prior_turns:
        lines = ["Previous conversation:"]
        for t in prior_turns:
            prefix = "User" if t["role"] == "user" else "Agent"
            lines.append(f"{prefix}: {t['content']}")
        context = "\n".join(lines) + "\n\n"
    else:
        context = ""

    history = f"[INST] {system}\n\n{context}Question: {query} [/INST]"

    steps = []
    final_answer = ""
    success = False

    for step_num in range(max_steps):
        response = generate(model, tokenizer, prompt=history, max_tokens=512, verbose=False)
        step = parse_response(response)
        steps.append(step)

        if step.is_final:
            final_answer = step.final_answer
            success = True
            break

        if step.tool_name:
            observation = call_tool(step.tool_name, step.tool_input)
            step.observation = observation
            history += f"\n{response}\nObservation: {observation}\n[INST] Continue. [/INST]"
        else:
            history += f"\n{response}\n[INST] Please use the format: Thought/Action/Action Input or Thought/Final Answer [/INST]"

    if not success:
        prompt = history + "\n[INST] You have reached the step limit. Give your best Final Answer now based on what you have found. [/INST]"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        final_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL)
        final_answer = final_match.group(1).strip() if final_match else response.strip()

    return RunResult(run_id=run_id, query=query, final_answer=final_answer, steps=steps, success=success)


async def run_agent_stream(query: str, max_steps: int = None):
    """Async generator that yields JSON event strings for SSE."""
    if max_steps is None:
        max_steps = int(os.getenv("MAX_STEPS", 6))

    model, tokenizer = get_model()
    run_id = str(uuid.uuid4())[:8]

    yield json.dumps({"event": "start", "run_id": run_id, "query": query})

    system = SYSTEM_PROMPT.format(tools=tools_prompt())
    history = f"[INST] {system}\n\nQuestion: {query} [/INST]"

    steps = []

    for step_num in range(max_steps):
        response = await asyncio.to_thread(
            generate, model, tokenizer, prompt=history, max_tokens=512, verbose=False
        )

        step = parse_response(response)
        steps.append(step)

        if step.thought:
            yield json.dumps({"event": "thought", "step": step_num, "content": step.thought})

        if step.is_final:
            yield json.dumps({
                "event": "final_answer",
                "run_id": run_id,
                "content": step.final_answer,
                "steps_taken": len(steps),
                "success": True,
            })
            return

        if step.tool_name:
            yield json.dumps({"event": "tool_call", "step": step_num, "tool": step.tool_name, "input": step.tool_input})
            observation = await asyncio.to_thread(call_tool, step.tool_name, step.tool_input)
            step.observation = observation
            yield json.dumps({"event": "observation", "step": step_num, "content": observation})
            history += f"\n{response}\nObservation: {observation}\n[INST] Continue. [/INST]"
        else:
            history += f"\n{response}\n[INST] Please use the format: Thought/Action/Action Input or Thought/Final Answer [/INST]"

    prompt = history + "\n[INST] You have reached the step limit. Give your best Final Answer now based on what you have found. [/INST]"
    response = await asyncio.to_thread(generate, model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
    final_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL)
    final_answer = final_match.group(1).strip() if final_match else response.strip()

    yield json.dumps({
        "event": "final_answer",
        "run_id": run_id,
        "content": final_answer,
        "steps_taken": len(steps),
        "success": False,
    })
