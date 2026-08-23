"""
Tests for the SSE translation layer in orchestrator/api.py.

What's under test is only the mapping from graph steps to events: does a run
that ends normally produce an answer, and does a run cut short by the
guardrail say so. No LLM, no agent servers, no cluster.

_run() builds its own registry and provider rather than taking them as
arguments the way build_graph() does, so there is no parameter to inject a
fake through. Both are module-level names in orchestrator.api, though, so
monkeypatch can replace them - which is what the fixtures below do.
"""

import json

import pytest
from prometheus_client import REGISTRY

from orchestrator.graph import MAX_ITERATIONS
from orchestrator.llm import LLMResponse, ToolCall
from tests.conftest import FakeRegistry, ScriptedProvider

import orchestrator.api as api


class DiscoverableRegistry(FakeRegistry):
    """FakeRegistry plus the discover() that _run calls before starting."""

    async def discover(self):
        return self.tools


async def collect(task: str) -> list[tuple[str, dict]]:
    """Drive _run to completion and return [(event_name, payload), ...]."""
    return [
        (frame["event"], json.loads(frame["data"]))
        async for frame in api._run(task)
    ]


@pytest.fixture(autouse=True)
def clean_registry_cache():
    """
    Discovery is cached process-wide, so it leaks between tests.

    Without this a test that wires an empty registry still sees whatever a
    previous test discovered. autouse because forgetting it in one test
    produces a failure in a DIFFERENT test, which is a miserable thing to
    debug.
    """
    api._registry_cache.clear()
    yield
    api._registry_cache.clear()


@pytest.fixture
def wire(monkeypatch):
    """Point _run at a fake registry and a scripted provider."""

    def _wire(provider, registry=None):
        registry = registry or DiscoverableRegistry()
        monkeypatch.setattr(api, "MCPToolRegistry", lambda: registry)
        monkeypatch.setattr(api, "get_provider", lambda: provider)
        return registry

    return _wire


async def test_normal_run_streams_the_full_sequence(wire):
    """The documented happy path: tools -> tool_call -> tool_result -> answer -> done."""
    provider = ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "search_web", {"query": "MCP"})],
            ),
            LLMResponse(content="MCP is an open standard."),
        ]
    )
    wire(provider, DiscoverableRegistry(results={"search_web": "MCP is an open standard."}))

    events = await collect("What is MCP?")
    names = [name for name, _ in events]

    assert names == ["tools", "tool_call", "tool_result", "answer", "done"]
    assert dict(events)["answer"]["content"] == "MCP is an open standard."


async def test_truncated_run_reports_itself_instead_of_going_silent(wire):
    """
    The guardrail path must reach the browser.

    Before the truncate node existed this run emitted tool_call events and then
    done, with no answer and nothing explaining why - the page stopped mid-run.
    A stop notice is also deliberately NOT an answer event, so the UI can style
    it differently and the eval harness can tell the two apart.
    """

    class NeverStops:
        async def chat(self, messages, tools):
            return LLMResponse(
                content="", tool_calls=[ToolCall("x", "search_web", {"query": "again"})]
            )

    wire(NeverStops())

    events = await collect("loop forever")
    names = [name for name, _ in events]

    assert "truncated" in names, "guardrail fired but the stream never said so"
    assert "answer" not in names, "a stop notice must not be presented as an answer"
    assert names[-1] == "done"

    payload = dict(events)["truncated"]
    assert payload["content"].strip()
    assert str(MAX_ITERATIONS) in payload["content"]


async def test_discovery_failure_is_reported_and_stops_the_run(wire):
    """No tools means no run - and the page should say why, not hang."""

    class EmptyRegistry(DiscoverableRegistry):
        def __init__(self):
            super().__init__(tools=[])

    wire(ScriptedProvider([]), EmptyRegistry())

    events = await collect("anything")

    assert [name for name, _ in events] == ["run_error"]
    assert "agent servers" in dict(events)["run_error"]["message"]


def runs(outcome: str) -> float:
    """Current value of orchestrator_runs_total for one outcome."""
    value = REGISTRY.get_sample_value(
        "orchestrator_runs_total", {"outcome": outcome}
    )
    return value or 0.0


async def test_a_normal_run_is_counted_as_answered(wire):
    provider = ScriptedProvider([LLMResponse(content="done")])
    wire(provider)

    before = runs("answered")
    await collect("anything")

    assert runs("answered") == before + 1


async def test_a_truncated_run_is_counted_separately(wire):
    """
    The point of the outcome label: a run that hit the guardrail must not be
    indistinguishable from one that answered. Runs piling up under
    "truncated" is the signal that the loop is regularly running out of road.
    """

    class NeverStops:
        async def chat(self, messages, tools):
            return LLMResponse(
                content="", tool_calls=[ToolCall("x", "search_web", {"query": "again"})]
            )

    wire(NeverStops())

    before_truncated = runs("truncated")
    before_answered = runs("answered")
    await collect("loop forever")

    assert runs("truncated") == before_truncated + 1
    assert runs("answered") == before_answered


def test_metrics_are_exported_in_the_exposition_format():
    """
    Exercises generate_latest directly rather than an HTTP route.

    Metrics are served by prometheus_client on its own port at app startup,
    not as a Starlette route - see orchestrator/metrics.py for why. Binding
    that port in a test would be a side effect for no extra coverage: what
    matters is that the series exist and are named as the dashboard expects.
    """
    from prometheus_client import generate_latest

    exposition = generate_latest().decode()

    # Present even at zero, or a panel built on it reads "No data" until the
    # first run rather than showing a legitimate zero.
    assert "orchestrator_runs_total" in exposition
    assert "orchestrator_run_duration_seconds" in exposition
    assert "orchestrator_run_iterations" in exposition


def test_importing_the_app_does_not_bind_the_metrics_port():
    """
    The reason the metrics server lives in a lifespan hook, not at import.

    Importing orchestrator.api must stay free of side effects, or every test
    that touches it competes for a real port - and on this machine ports are
    a scarce, moving target (see docs/RUNBOOK.md).
    """
    import socket

    from orchestrator.metrics import METRICS_PORT

    sock = socket.socket()
    try:
        sock.settimeout(0.3)
        assert sock.connect_ex(("127.0.0.1", METRICS_PORT)) != 0, (
            f"something is listening on {METRICS_PORT} merely from importing the app"
        )
    finally:
        sock.close()


class CountingRegistry(DiscoverableRegistry):
    """Records how many times discovery actually ran."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discoveries = 0

    async def discover(self):
        self.discoveries += 1
        return self.tools


async def test_discovery_is_not_repeated_for_every_run(monkeypatch):
    """The point of the cache: three MCP handshakes once, not once per run."""
    registry = CountingRegistry()
    monkeypatch.setattr(api, "MCPToolRegistry", lambda: registry)
    monkeypatch.setattr(
        api, "get_provider", lambda: ScriptedProvider([LLMResponse(content="ok")])
    )

    for _ in range(3):
        monkeypatch.setattr(
            api, "get_provider", lambda: ScriptedProvider([LLMResponse(content="ok")])
        )
        await collect("anything")

    assert registry.discoveries == 1


async def test_an_empty_discovery_is_never_cached(monkeypatch):
    """
    The rule that keeps a blip from becoming an outage.

    MCPToolRegistry tolerates unreachable agents by logging and skipping them,
    so "no tools" is a legitimate transient result. Caching it would keep the
    system down for a full TTL after the agents came back.
    """
    registry = CountingRegistry(tools=[])
    monkeypatch.setattr(api, "MCPToolRegistry", lambda: registry)
    monkeypatch.setattr(api, "get_provider", lambda: ScriptedProvider([]))

    await collect("first")
    await collect("second")

    # Retried rather than served from a cached failure.
    assert registry.discoveries == 2


async def test_agents_coming_back_are_picked_up_without_a_restart(monkeypatch):
    """A failed discovery must not wedge the process until someone restarts it."""
    registry = CountingRegistry(tools=[])
    monkeypatch.setattr(api, "MCPToolRegistry", lambda: registry)
    monkeypatch.setattr(api, "get_provider", lambda: ScriptedProvider([]))

    events = await collect("while down")
    assert [name for name, _ in events] == ["run_error"]

    # The agents come back.
    registry.tools = DiscoverableRegistry().tools
    monkeypatch.setattr(
        api, "get_provider", lambda: ScriptedProvider([LLMResponse(content="back")])
    )

    events = await collect("after recovery")
    assert [name for name, _ in events] == ["tools", "answer", "done"]


async def test_concurrent_first_requests_discover_once(monkeypatch):
    """The lock: a burst on a cold cache must not stampede into N discoveries."""
    import asyncio

    registry = CountingRegistry()

    async def slow_discover():
        registry.discoveries += 1
        await asyncio.sleep(0.05)  # long enough for the others to pile up
        return registry.tools

    registry.discover = slow_discover
    monkeypatch.setattr(api, "MCPToolRegistry", lambda: registry)
    monkeypatch.setattr(
        api, "get_provider", lambda: ScriptedProvider([LLMResponse(content="ok")])
    )

    await asyncio.gather(*(collect(f"task {i}") for i in range(5)))

    assert registry.discoveries == 1