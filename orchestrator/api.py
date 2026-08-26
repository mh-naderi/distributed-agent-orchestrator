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

import asyncio
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

from orchestrator.config import (
    AGENT_URLS,
    MAX_CONCURRENT_RUNS,
    MAX_QUEUED_RUNS,
    MCP_DISCOVERY_TTL,
    OLLAMA_MODEL,
)
from orchestrator.graph import SYSTEM_PROMPT, build_graph
from orchestrator.llm import get_provider
from orchestrator.mcp_client import MCPToolRegistry
from orchestrator.metrics import (
    DISCOVERY_FAILURES,
    METRICS_PORT,
    RUN_DURATION,
    RUN_ITERATIONS,
    RUNS,
    RUNS_QUEUED,
    TOOLS_DISCOVERED,
)

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"


async def index(request):
    return FileResponse(STATIC / "index.html")


async def health(request):
    """Liveness plus enough configuration to make a failed run diagnosable."""
    # Deliberately does NOT call the agents: readiness must not cascade, or a
    # wedged agent pulls the UI out of service when the other tools still
    # work. The discovery figures are the cheap substitute - they say what the
    # last discovery FOUND, and how long ago, without touching anything now.
    age = _registry_cache.age
    return JSONResponse(
        {
            "status": "ok",
            "model": OLLAMA_MODEL,
            "agents": AGENT_URLS,
            "discovery": {
                "tools": len(_registry_cache.tools),
                "age_seconds": None if age is None else round(age, 1),
                "ttl_seconds": MCP_DISCOVERY_TTL,
            },
        }
    )


class _RegistryCache:
    """
    One discovery result, shared by every request until it goes stale.

    Discovery is three MCP connect-and-handshake round trips and it used to
    run on every request, before any work started - so every run paid for it
    and the first event was correspondingly late.

    Three rules make caching safe here:

    1. A TTL, so a tool added to an agent becomes visible on its own rather
       than requiring a restart.
    2. An EMPTY result is never stored. Discovery tolerates unreachable
       agents by design - it logs and skips them - so a blip can legitimately
       return nothing, and caching that would convert a moment of downtime
       into a full TTL of it.
    3. A lock, so a burst of concurrent requests performs one discovery
       rather than one each. The re-check inside the lock matters: whoever
       waited for it usually finds the work already done.
    """

    def __init__(self, ttl: float = MCP_DISCOVERY_TTL):
        self._ttl = ttl
        self._lock = asyncio.Lock()
        self._registry: MCPToolRegistry | None = None
        self._fetched_at = 0.0

    def _fresh(self) -> bool:
        return (
            self._registry is not None
            and bool(self._registry.tools)
            and (time.monotonic() - self._fetched_at) < self._ttl
        )

    @property
    def age(self) -> float | None:
        """Seconds since the cached result was fetched, or None if empty."""
        if self._registry is None:
            return None
        return time.monotonic() - self._fetched_at

    @property
    def tools(self) -> list[dict]:
        """Whatever the last successful discovery found; empty if none yet."""
        return self._registry.tools if self._registry is not None else []

    def clear(self) -> None:
        self._registry = None
        self._fetched_at = 0.0

    async def get(self) -> MCPToolRegistry:
        if self._fresh():
            return self._registry

        async with self._lock:
            if self._fresh():
                return self._registry

            registry = MCPToolRegistry()
            await registry.discover()
            TOOLS_DISCOVERED.observe(len(registry.tools))

            if not registry.tools:
                # Deliberately not stored, so the next request retries.
                DISCOVERY_FAILURES.inc()
                return registry

            self._registry = registry
            self._fetched_at = time.monotonic()
            return registry


_registry_cache = _RegistryCache()


class _RunLimiter:
    """
    Caps how many runs execute at once.

    The constraint is physical, not architectural: a run drives inference on a
    single 4GB GPU, and one run with both models resident was measured leaving
    377-582MiB free. The driver reset this project hit twice happened at
    157MiB. A second concurrent run has nowhere to come from, and the failure
    it produces is a machine-wide bugcheck rather than a handled error - which
    is exactly the kind of failure worth spending a semaphore to avoid.

    Queue rather than refuse, because refusing throws away work somebody asked
    for. But queue VISIBLY and with a bound: a silent queue makes the page look
    hung, and an unbounded one just moves the failure to a pile of requests that
    all time out together.

    A class rather than a module-level semaphore so tests can reset it. Shared
    state that cannot be reset produces failures in whichever test happens to
    run second, which is a miserable thing to debug - see the registry cache.
    """

    def __init__(self, limit: int = MAX_CONCURRENT_RUNS, max_waiting: int = MAX_QUEUED_RUNS):
        self._limit = limit
        self._max_waiting = max_waiting
        self._semaphore = asyncio.Semaphore(limit)
        self._waiting = 0

    @property
    def waiting(self) -> int:
        return self._waiting

    def would_wait(self) -> bool:
        """True if every slot is taken, so acquiring will block."""
        return self._semaphore.locked()

    def is_full(self) -> bool:
        """True if the queue is at its cap and further work should be refused."""
        return self._waiting >= self._max_waiting

    async def acquire(self) -> None:
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1

    def release(self) -> None:
        self._semaphore.release()

    def reset(self) -> None:
        self._semaphore = asyncio.Semaphore(self._limit)
        self._waiting = 0


_run_limiter = _RunLimiter()


def _sse(event: str, **payload) -> dict:
    """
    One SSE frame. sse-starlette turns this into `event:` / `data:` lines.

    Note the error event is named run_error, not error: EventSource already
    defines an `error` event for transport-level failures, so a server-sent
    event with that name collides with the browser's own handler and the two
    become indistinguishable on the client.
    """
    return {"event": event, "data": json.dumps(payload)}


async def _run(task: str, escalate: bool = False):
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
        try:
            registry = await _registry_cache.get()
        except Exception as exc:
            DISCOVERY_FAILURES.inc()
            outcome = "no_tools"
            yield _sse("run_error", message=f"tool discovery failed: {exc}")
            return

        if not registry.tools:
            outcome = "no_tools"
            yield _sse(
                "run_error",
                message="No MCP tools discovered - are the agent servers running?",
            )
            return

        yield _sse("tools", tools=[t["name"] for t in registry.tools])

        # Everything above this point is cheap and touches no GPU. The gate
        # goes here, immediately before inference starts.
        if _run_limiter.would_wait():
            if _run_limiter.is_full():
                outcome = "rejected"
                yield _sse(
                    "busy",
                    message=(
                        f"{_run_limiter.waiting} run(s) already waiting - this one was "
                        f"refused rather than added to a queue that cannot drain. "
                        f"Try again shortly."
                    ),
                )
                return
            RUNS_QUEUED.inc()
            # Announced BEFORE waiting. A silent queue is indistinguishable
            # from a hung page.
            yield _sse("queued", ahead=_run_limiter.waiting + 1)

        await _run_limiter.acquire()
        try:
            state = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ],
                "iterations": 0,
            }

            try:
                async for chunk in build_graph(registry, get_provider(escalate)).astream(state):
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
                            # A separate event, not an answer. The run ended
                            # because the guardrail fired, and calling that an
                            # "answer" would present a stop notice as a result.
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
            # Released on every path, including the client disconnecting
            # mid-run - a generator's finally runs when it is closed. Leaking
            # a slot would wedge the service for everyone after.
            _run_limiter.release()
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

    # Escalation is opt-in per request rather than a heuristic. See
    # get_provider for why there is no automatic rule yet.
    escalate = request.query_params.get("escalate", "").lower() in ("1", "true", "yes", "on")
    return EventSourceResponse(_run(task, escalate=escalate))


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
