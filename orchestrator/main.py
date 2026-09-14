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
from orchestrator.trace import use_trace

logger = logging.getLogger(__name__)


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

    # The trace has to cover ainvoke, because that is when the tool calls
    # happen and the id has to be set when the client reads it.
    with use_trace() as trace_id:
        logger.info("run start trace=%s task=%r", trace_id, task[:80])
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

    # The id the agents logged this run under. Without it a case that failed is
    # a result with no way back to what the services actually did - and these
    # runs, not the API's, are the ones behind every measurement in the docs.
    trace_id: str | None = None


def tool_history(messages: list[dict]) -> tuple[list[str], list[dict]]:
    """
    What the model asked for, and what each request got back.

    Every output carries the arguments it was produced from. They used to be
    dropped here, while the messages this reads held them all along, and that
    cost a real investigation: a run credited a genuine report to the fictional
    Quazzlemint Foundation after calling search_web, and whether the query it
    sent opened with the subject - the exact shape a coverage-note bug had just
    been found to miss - could not be answered. The output was kept, the query
    that produced it was not, and the model never searched again in the runs
    that followed, so there was nothing to recapture.

    Outputs are paired with requests by tool_call_id, not by position. The model
    can request several tools in one turn, and a positional pairing that drifted
    would attach one call's arguments to another call's output without any
    visible sign - a trace that is confidently wrong, which is worse than one
    that is missing. An output with no matching request says None rather than
    borrowing a neighbour's.
    """
    tools_called = []
    arguments_by_id = {}
    for message in messages:
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls", []):
            tools_called.append(call["name"])
            if call.get("id") is not None:
                arguments_by_id[call["id"]] = call.get("arguments")

    tool_outputs = [
        {
            "name": m.get("name", "?"),
            "arguments": arguments_by_id.get(m.get("tool_call_id")),
            "output": m.get("content", ""),
        }
        for m in messages
        if m["role"] == "tool"
    ]
    return tools_called, tool_outputs


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

    # The eval harness and the experiment runner come through here rather than
    # through the API, so without this the runs behind every measurement in the
    # docs would be the ones that could not be followed across the agents.
    with use_trace() as trace_id:
        logger.info("run start trace=%s task=%r", trace_id, task[:80])
        final_state = await graph.ainvoke(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ],
                "iterations": 0,
            }
        )
        logger.info("run end trace=%s iterations=%s", trace_id, final_state["iterations"])

    messages = final_state["messages"]
    tools_called, tool_outputs = tool_history(messages)

    return TraceResult(
        answer=messages[-1].get("content", ""),
        tools_called=tools_called,
        tool_outputs=tool_outputs,
        iterations=final_state["iterations"],
        trace_id=trace_id,
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
