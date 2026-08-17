"""
Entry point for running the orchestrator locally (outside Kubernetes),
for the Week 1 Day 7 milestone: get an end-to-end run working before
adding infra complexity.

Prerequisites for a run:
  - Ollama serving a tool-calling model (see orchestrator/config.py)
  - the three agent servers running, at the URLs in config.AGENT_URLS
"""

import asyncio
import logging

from orchestrator.graph import SYSTEM_PROMPT, build_graph
from orchestrator.llm import get_provider
from orchestrator.mcp_client import MCPToolRegistry


async def arun(task: str) -> str:
    """
    Run one task through the graph.

    Tool discovery happens once, up front, before the graph is built: the
    model can't be asked to choose a tool until we know what tools exist. This
    is also the first thing to check when debugging - if discovery returns
    nothing, the agents aren't reachable and the loop will never call anything.
    """
    registry = MCPToolRegistry()
    await registry.discover()

    if not registry.tools:
        raise RuntimeError(
            "No MCP tools discovered - are the agent servers running? "
            "See orchestrator/config.py for the URLs being tried."
        )

    graph = build_graph(registry, get_provider())

    initial_state = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        "iterations": 0,
    }

    final_state = await graph.ainvoke(initial_state)
    return final_state["messages"][-1].get("content", "")


async def arun_traced(task: str) -> tuple[str, list[str]]:
    """
    Like arun, but also returns the tool names that were called, in order.

    The evaluation harness needs this: "did it reach for the right tools" is a
    separate question from "is the final text any good", and the tool history is
    the only place the first question can be answered. Checking the final text
    for tool names would be guesswork.
    """
    registry = MCPToolRegistry()
    await registry.discover()

    if not registry.tools:
        raise RuntimeError(
            "No MCP tools discovered - are the agent servers running? "
            "See orchestrator/config.py for the URLs being tried."
        )

    graph = build_graph(registry, get_provider())
    final_state = await graph.ainvoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ],
            "iterations": 0,
        }
    )

    tools_called = [
        call["name"]
        for message in final_state["messages"]
        if message["role"] == "assistant"
        for call in message.get("tool_calls", [])
    ]
    return final_state["messages"][-1].get("content", ""), tools_called


def run(task: str) -> str:
    """Synchronous wrapper - eval/run_eval.py drives the system through this."""
    return asyncio.run(arun(task))


def run_traced(task: str) -> tuple[str, list[str]]:
    """Synchronous wrapper around arun_traced."""
    return asyncio.run(arun_traced(task))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run("Research the current state of MCP adoption and summarize it.")
    print(result)
