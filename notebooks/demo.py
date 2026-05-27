"""
LLM Agent demo: shows ReAct loop with tool use.
Run: make serve then make demo
"""
from __future__ import annotations
import requests, json

BASE = "http://localhost:8083"

def run(query: str, max_steps: int = 6):
    return requests.post(f"{BASE}/agent/run", json={
        "query": query,
        "max_steps": max_steps,
    }).json()

def get(path: str):
    return requests.get(f"{BASE}{path}").json()

print("\n=== LLM Agent Demo ===\n")

# 1. List available tools
tools = get("/tools")
print("1. Available tools:")
for t in tools:
    print(f"   {t['name']}: {t['description'][:60]}...")

# 2. Math reasoning query
print("\n2. Math query...")
r = run("What is the average of 847, 932, 1204, and 678? Then what is 15% of that average?")
print(f"   Query:  {r['query']}")
print(f"   Answer: {r['final_answer']}")
print(f"   Steps:  {r['steps_taken']}  Success: {r['success']}")
print("   Trace:")
for step in r["trace"]:
    if step["thought"]:
        print(f"     [{step['step']}] Thought: {step['thought'][:80]}...")
    if step["tool"]:
        print(f"     [{step['step']}] Tool: {step['tool']}({step['tool_input']})")
        print(f"     [{step['step']}] Obs:  {str(step['observation'])[:80]}...")

# 3. Knowledge retrieval query (requires P4 running)
print("\n3. Knowledge query (requires P4 RAG pipeline on :8082)...")
r = run("What is overfitting and how do you prevent it?", max_steps=4)
print(f"   Query:  {r['query']}")
print(f"   Answer: {r['final_answer'][:300]}...")
print(f"   Steps:  {r['steps_taken']}")

# 4. Feature lookup query (requires P2 running)
print("\n4. Feature lookup query (requires P2 ML Platform on :8080)...")
r = run("Look up the credit signals for user_000 and summarise what you find.", max_steps=4)
print(f"   Query:  {r['query']}")
print(f"   Answer: {r['final_answer'][:300]}...")
print(f"   Steps:  {r['steps_taken']}")

# 5. Stats
print("\n5. Agent stats:")
stats = get("/agent/stats")
print(f"   {json.dumps(stats, indent=4)}")

print(f"\nAPI docs → http://localhost:8083/docs")
print("\nDone.")
