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
import time
from contextlib import asynccontextmanager
from pathlib import Path

from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from prometheus_client import start_http_server
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from orchestrator.config import AGENT_URLS, OLLAMA_MODEL
from orchestrator.graph import SYSTEM_PROMPT, build_graph
from orchestrator.llm import get_provider
from orchestrator.mcp_client import MCPToolRegistry
from orchestrator.metrics import (
    DISCOVERY_FAILURES,
    METRICS_PORT,
    RUN_DURATION,
    RUN_ITERATIONS,
    RUNS,
    TOOLS_DISCOVERED,
)

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
    started = time.perf_counter()
    # Pessimistic default: anything that escapes without setting this - an
    # exception, or the client disconnecting mid-run and closing the
    # generator - is counted as a failure rather than silently not counted.
    outcome = "failed"
    iterations = None

    try:
        registry = MCPToolRegistry()
        try:
            await registry.discover()
        except Exception as exc:
            DISCOVERY_FAILURES.inc()
            outcome = "no_tools"
            yield _sse("run_error", message=f"tool discovery failed: {exc}")
            return

        TOOLS_DISCOVERED.observe(len(registry.tools))

        if not registry.tools:
            DISCOVERY_FAILURES.inc()
            outcome = "no_tools"
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
                    if update.get("iterations") is not None:
                        iterations = update["iterations"]

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
                            outcome = "answered"
                            yield _sse("answer", content=message.get("content", ""))

                    elif node == "truncate":
                        # A separate event, not an answer. The run ended because
                        # the guardrail fired, and calling that an "answer" would
                        # present a stop notice as a result. Previously this path
                        # emitted nothing at all and the page just stopped.
                        outcome = "truncated"
                        message = messages[-1] if messages else {}
                        yield _sse("truncated", content=message.get("content", ""))

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
    finally:
        # A generator's finally runs on close too, so a client that navigates
        # away mid-run is still counted rather than vanishing from the totals.
        RUNS.labels(outcome=outcome).inc()
        RUN_DURATION.observe(time.perf_counter() - started)
        if iterations is not None:
            RUN_ITERATIONS.observe(iterations)


async def stream(request):
    task = request.query_params.get("task", "").strip()
    if not task:
        return JSONResponse({"error": "missing ?task="}, status_code=400)
    return EventSourceResponse(_run(task))


@asynccontextmanager
async def _lifespan(app):
    """
    Start the Prometheus endpoint on its own port, once, at app startup.

    A lifespan hook rather than module-level code, so that importing this
    module - which the tests do - does not bind a port as a side effect.
    (Starlette's older on_startup= list is gone in 1.x; lifespan is the
    replacement, and the tests caught the difference.)

    start_http_server runs a small WSGI server on a daemon thread. That is
    the same mechanism all three agents use, and the thread is what keeps
    scrapes answerable while the event loop is busy inside a long inference
    call.
    """
    start_http_server(METRICS_PORT)
    logger.info("metrics listening on :%d", METRICS_PORT)
    yield


app = Starlette(
    routes=[
        Route("/", index),
        Route("/health", health),
        Route("/stream", stream),
    ],
    lifespan=_lifespan,
)
