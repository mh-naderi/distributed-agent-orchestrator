"""
HTTP API around the orchestrator - the first step toward a UI.

WHY A SERVICE AT ALL. orchestrator/main.py is a CLI entry point: it runs the
loop once and prints the answer. That makes the most interesting thing this
project does - the reason -> act -> reason cycle, with tool calls resolving one
at a time - completely invisible. Wrapping the loop in a service that streams
its steps is what makes it watchable, and it is also what would let the
orchestrator run inside the cluster instead of reaching in from outside.

WHY SERVER-SENT EVENTS. The loop produces a sequence of steps over time and the
browser only ever listens - it never pushes anything back mid-run. That is
exactly the shape SSE is for: a one-way stream over ordinary HTTP. WebSockets
would also work but are bidirectional and need more machinery for no benefit
here. Being plain HTTP also means it passes through an Ingress unmodified.

WHY STARLETTE RATHER THAN FASTAPI. Starlette, uvicorn and sse-starlette are
already installed as MCP dependencies. FastAPI is built on Starlette and adds
request-body validation and schema generation, neither of which three endpoints
with a single query parameter need. Using what is already here keeps the
dependency list honest.

Run it:
    .venv/Scripts/python.exe -m uvicorn orchestrator.api:app --port 18080
"""

import json
import logging
from pathlib import Path

from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from orchestrator.config import AGENT_URLS, OLLAMA_MODEL
from orchestrator.graph import SYSTEM_PROMPT, build_graph
from orchestrator.llm import get_provider
from orchestrator.mcp_client import MCPToolRegistry

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"


async def index(request):
    return FileResponse(STATIC / "index.html")


async def health(request):
    """Liveness plus enough configuration to make a failed run diagnosable."""
    return JSONResponse(
        {"status": "ok", "model": OLLAMA_MODEL, "agents": AGENT_URLS}
    )


def _sse(event: str, **payload) -> dict:
    """
    One SSE frame. sse-starlette turns this into `event:` / `data:` lines.

    Note the error event is named run_error, not error: EventSource already
    defines an `error` event for transport-level failures, so a server-sent
    event with that name collides with the browser's own handler and the two
    become indistinguishable on the client.
    """
    return {"event": event, "data": json.dumps(payload)}


async def _run(task: str):
    """
    Drive the graph and translate each step into an event.

    graph.astream() yields one dict per node execution, keyed by node name,
    holding the partial state that node returned. Translating those into
    domain events here means the graph itself stays unaware that anything is
    watching - no streaming concerns leak into the loop.
    """
    registry = MCPToolRegistry()
    try:
        await registry.discover()
    except Exception as exc:
        yield _sse("run_error", message=f"tool discovery failed: {exc}")
        return

    if not registry.tools:
        yield _sse(
            "run_error",
            message="No MCP tools discovered - are the agent servers running?",
        )
        return

    yield _sse("tools", tools=[t["name"] for t in registry.tools])

    state = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        "iterations": 0,
    }

    try:
        async for chunk in build_graph(registry, get_provider()).astream(state):
            for node, update in chunk.items():
                messages = update.get("messages") or []

                if node == "reason":
                    message = messages[-1] if messages else {}
                    if message.get("tool_calls"):
                        for call in message["tool_calls"]:
                            yield _sse(
                                "tool_call",
                                iteration=update.get("iterations"),
                                name=call["name"],
                                arguments=call["arguments"],
                            )
                    else:
                        yield _sse("answer", content=message.get("content", ""))

                elif node == "act":
                    for message in messages:
                        yield _sse(
                            "tool_result",
                            name=message.get("name", "?"),
                            output=message.get("content", ""),
                        )
    except Exception as exc:
        logger.exception("run failed")
        yield _sse("run_error", message=f"{type(exc).__name__}: {exc}")
        return

    yield _sse("done")


async def stream(request):
    task = request.query_params.get("task", "").strip()
    if not task:
        return JSONResponse({"error": "missing ?task="}, status_code=400)
    return EventSourceResponse(_run(task))


app = Starlette(
    routes=[
        Route("/", index),
        Route("/health", health),
        Route("/stream", stream),
    ]
)
