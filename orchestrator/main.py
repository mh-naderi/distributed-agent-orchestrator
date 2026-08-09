"""
Entry point for running the orchestrator locally (outside Kubernetes),
for the Week 1 Day 7 milestone: get an end-to-end run working before
adding infra complexity.
"""

from orchestrator.graph import build_graph
from orchestrator.state import AgentState


def run(task: str) -> str:
    graph = build_graph()
    initial_state: AgentState = {"messages": [{"role": "user", "content": task}], "iterations": 0}
    final_state = graph.invoke(initial_state)
    return final_state["messages"][-1]


if __name__ == "__main__":
    result = run("Research the current state of MCP adoption and summarize it.")
    print(result)
