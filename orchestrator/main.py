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
from dataclasses import dataclass

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


@dataclass
class TraceResult:
    """
    Everything the evaluation harness needs from one run.

    tool_outputs matters as much as the answer: the most important question for
    this system is not "does the answer sound right" but "is it supported by
    what the tools actually returned". Judging the text alone cannot tell the
    difference between a grounded answer and a fluent invention - which this
    project has produced before, with a fabricated statistic.
    """

    answer: str
    tools_called: list[str]
    tool_outputs: list[dict]
    iterations: int


async def arun_traced(task: str) -> TraceResult:
    """
    Like arun, but records what happened along the way.

    "Did it reach for the right tools" and "is the final text any good" are
    separate questions, and the tool history is the only place the first one can
    be answered. Scanning the answer for tool names would be guesswork.
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

    messages = final_state["messages"]
    tools_called = [
        call["name"]
        for message in messages
        if message["role"] == "assistant"
        for call in message.get("tool_calls", [])
    ]
    tool_outputs = [
        {"name": m.get("name", "?"), "output": m.get("content", "")}
        for m in messages
        if m["role"] == "tool"
    ]

    return TraceResult(
        answer=messages[-1].get("content", ""),
        tools_called=tools_called,
        tool_outputs=tool_outputs,
        iterations=final_state["iterations"],
    )


def run(task: str) -> str:
    """Synchronous wrapper - eval/run_eval.py drives the system through this."""
    return asyncio.run(arun(task))


def run_traced(task: str) -> TraceResult:
    """Synchronous wrapper around arun_traced - eval/run_eval.py uses this."""
    return asyncio.run(arun_traced(task))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run("Research the current state of MCP adoption and summarize it.")
    print(result)
