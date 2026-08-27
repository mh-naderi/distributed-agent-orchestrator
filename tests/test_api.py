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


def asked_by_user(turn: list[dict]) -> list[str]:
    """
    The questions a person asked in one model turn.

    Excludes the nudge, which is a user-role message the loop injects when the
    model narrates a tool call instead of making one - see graph.NUDGE_PROMPT.
    It is user-role for provider portability, not because a person said it.
    """
    from orchestrator.graph import NUDGE_PROMPT

    return [
        m["content"]
        for m in turn
        if m["role"] == "user" and m["content"] != NUDGE_PROMPT
    ]


async def collect_session(task: str, session_id: str) -> list[tuple[str, dict]]:
    """Drive _run with a session so history carries between calls."""
    return [
        (frame["event"], json.loads(frame["data"]))
        async for frame in api._run(task, session_id=session_id)
    ]


@pytest.fixture(autouse=True)
def clean_registry_cache():
    """
    Discovery and the run limiter are process-wide, so both leak between tests.

    Without this a test that wires an empty registry still sees whatever a
    previous test discovered. autouse because forgetting it in one test
    produces a failure in a DIFFERENT test, which is a miserable thing to
    debug.
    """
    api._registry_cache.clear()
    api._run_limiter.reset()
    api._sessions.clear()
    yield
    api._registry_cache.clear()
    api._run_limiter.reset()
    api._sessions.clear()


@pytest.fixture
def wire(monkeypatch):
    """Point _run at a fake registry and a scripted provider."""

    def _wire(provider, registry=None):
        registry = registry or DiscoverableRegistry()
        monkeypatch.setattr(api, "MCPToolRegistry", lambda: registry)
        monkeypatch.setattr(api, "get_provider", lambda *a, **k: provider)
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
        api, "get_provider", lambda *a, **k: ScriptedProvider([LLMResponse(content="ok")])
    )

    for _ in range(3):
        monkeypatch.setattr(
            api, "get_provider", lambda *a, **k: ScriptedProvider([LLMResponse(content="ok")])
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
    monkeypatch.setattr(api, "get_provider", lambda *a, **k: ScriptedProvider([]))

    await collect("first")
    await collect("second")

    # Retried rather than served from a cached failure.
    assert registry.discoveries == 2


async def test_agents_coming_back_are_picked_up_without_a_restart(monkeypatch):
    """A failed discovery must not wedge the process until someone restarts it."""
    registry = CountingRegistry(tools=[])
    monkeypatch.setattr(api, "MCPToolRegistry", lambda: registry)
    monkeypatch.setattr(api, "get_provider", lambda *a, **k: ScriptedProvider([]))

    events = await collect("while down")
    assert [name for name, _ in events] == ["run_error"]

    # The agents come back.
    registry.tools = DiscoverableRegistry().tools
    monkeypatch.setattr(
        api,
        "get_provider",
        lambda *a, **k: ScriptedProvider(
            [LLMResponse(content="back"), LLMResponse(content="back")]
        ),
    )

    events = await collect("after recovery")
    # The scripted answer carries no tool call, so the loop asks once more
    # before accepting it - see graph.NUDGE_PROMPT.
    assert [name for name, _ in events] == ["tools", "nudge", "answer", "done"]


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
        api, "get_provider", lambda *a, **k: ScriptedProvider([LLMResponse(content="ok")])
    )

    await asyncio.gather(*(collect(f"task {i}") for i in range(5)))

    assert registry.discoveries == 1


# ---------------------------------------------------------------------------
# Concurrency: one run touches the GPU at a time
# ---------------------------------------------------------------------------


class BlockingProvider:
    """A model call that parks until released, so overlap is deterministic."""

    def __init__(self):
        import asyncio

        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.concurrent = 0
        self.max_concurrent = 0

    async def chat(self, messages, tools):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.entered.set()
        try:
            await self.release.wait()
            return LLMResponse(content="done")
        finally:
            self.concurrent -= 1


def _wire_blocking(monkeypatch, provider):
    monkeypatch.setattr(api, "MCPToolRegistry", lambda: DiscoverableRegistry())
    monkeypatch.setattr(api, "get_provider", lambda *a, **k: provider)


async def test_a_second_run_waits_instead_of_competing_for_the_gpu(monkeypatch):
    """
    The whole point. Two runs at once would put two models on a 4GB card, and
    the failure measured on this hardware is a driver reset, not a handled
    error.
    """
    import asyncio

    provider = BlockingProvider()
    _wire_blocking(monkeypatch, provider)

    first = asyncio.create_task(collect("first"))
    await provider.entered.wait()

    second = asyncio.create_task(collect("second"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    provider.release.set()
    events_first, events_second = await asyncio.gather(first, second)

    assert provider.max_concurrent == 1, "two runs were inside the model at once"
    assert "queued" in [name for name, _ in events_second]
    assert "queued" not in [name for name, _ in events_first]


async def test_the_queue_is_announced_before_the_wait(monkeypatch):
    """A silent queue is indistinguishable from a hung page."""
    import asyncio

    provider = BlockingProvider()
    _wire_blocking(monkeypatch, provider)

    first = asyncio.create_task(collect("first"))
    await provider.entered.wait()

    stream = api._run("second")
    seen = [(await stream.__anext__())["event"] for _ in range(2)]

    assert seen == ["tools", "queued"]

    provider.release.set()
    await stream.aclose()
    await first


async def test_runs_beyond_the_queue_cap_are_refused(monkeypatch):
    """An unbounded queue just moves the failure to a pile of timeouts."""
    import asyncio

    provider = BlockingProvider()
    _wire_blocking(monkeypatch, provider)
    monkeypatch.setattr(api, "_run_limiter", api._RunLimiter(limit=1, max_waiting=1))

    first = asyncio.create_task(collect("first"))
    await provider.entered.wait()

    queued = asyncio.create_task(collect("queued"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    refused = await collect("refused")

    assert [name for name, _ in refused] == ["tools", "busy"]
    assert "refused" in dict(refused)["busy"]["message"]

    provider.release.set()
    await asyncio.gather(first, queued)


async def test_a_refused_run_is_counted_separately(monkeypatch):
    """rejected must not look like a failure - nothing broke, it was declined."""
    import asyncio

    provider = BlockingProvider()
    _wire_blocking(monkeypatch, provider)
    monkeypatch.setattr(api, "_run_limiter", api._RunLimiter(limit=1, max_waiting=0))

    first = asyncio.create_task(collect("first"))
    await provider.entered.wait()

    before_rejected = runs("rejected")
    before_failed = runs("failed")
    await collect("refused")

    assert runs("rejected") == before_rejected + 1
    assert runs("failed") == before_failed

    provider.release.set()
    await first


async def test_the_slot_is_released_when_a_client_disconnects(monkeypatch):
    """
    A leaked slot wedges the service for everyone after.

    Closing the generator is what a client navigating away does, and the
    release lives in a finally so that path is covered too.
    """
    provider = BlockingProvider()
    _wire_blocking(monkeypatch, provider)

    stream = api._run("abandoned")
    await stream.__anext__()      # tools
    provider.release.set()
    await stream.__anext__()      # answer - the slot is held at this point
    await stream.aclose()         # client goes away

    assert not api._run_limiter.would_wait(), "the slot was never given back"


# ---------------------------------------------------------------------------
# Sessions: a follow-up can see the previous turn
# ---------------------------------------------------------------------------


async def test_without_a_session_each_run_starts_fresh(wire):
    """The previous behaviour, and still right for a one-off question."""
    provider = ScriptedProvider([LLMResponse(content=f"turn {i}") for i in range(6)])
    wire(provider)

    await collect("first question")
    await collect("second question")

    # seen[-1] rather than seen[1]: an answer with no tool call is nudged once,
    # so each run now shows the provider more than one turn.
    assert asked_by_user(provider.seen[-1]) == ["second question"]


async def test_a_session_carries_the_previous_turn_into_the_next(wire):
    """
    The point of the feature: "now summarise that" needs something to refer to.
    """
    provider = ScriptedProvider(
        [LLMResponse(content="MCP is a protocol.")] + [LLMResponse(content="ok")] * 5
    )
    wire(provider)

    await collect_session("what is MCP?", "sess-1")
    await collect_session("summarise that", "sess-1")

    last_turn = provider.seen[-1]
    answers = [m["content"] for m in last_turn if m["role"] == "assistant"]

    assert asked_by_user(last_turn) == ["what is MCP?", "summarise that"]
    assert "MCP is a protocol." in answers


async def test_separate_sessions_do_not_leak_into_each_other(wire):
    provider = ScriptedProvider([LLMResponse(content=f"answer {i}") for i in range(6)])
    wire(provider)

    await collect_session("question A", "sess-a")
    await collect_session("question B", "sess-b")

    turn_for_b = provider.seen[-1]
    assert asked_by_user(turn_for_b) == ["question B"], (
        "session B saw another session's history"
    )


async def test_the_system_prompt_is_not_duplicated_on_a_follow_up(wire):
    """A second copy is wasted context in a window this small."""
    provider = ScriptedProvider([LLMResponse(content="one"), LLMResponse(content="two")])
    wire(provider)

    await collect_session("first", "sess-2")
    await collect_session("second", "sess-2")

    second_turn = provider.seen[1]
    assert sum(1 for m in second_turn if m["role"] == "system") == 1


async def test_stored_history_is_trimmed(wire):
    """
    Enforced on the way IN to the store, not on the way out - otherwise the
    budget is applied one run too late, after the oversized prompt was sent.
    """
    from orchestrator.config import history_budget_chars
    from orchestrator.sessions import estimate_chars

    # Large TOOL output, which is what actually dominates a real history and
    # what trimming is designed to control. A huge assistant answer is a
    # different case: trim will not mangle the model's own text, and says so
    # in a warning rather than pretending it fitted.
    registry = DiscoverableRegistry(results={"search_web": "x" * 20000})
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "search_web", {"query": "q"})]),
            LLMResponse(content="short answer"),
        ]
    )
    wire(provider, registry)

    await collect_session("first", "sess-3")

    stored = api._sessions.get("sess-3").messages
    assert estimate_chars(stored) <= history_budget_chars()


# ---------------------------------------------------------------------------
# The nudge, as the page sees it
# ---------------------------------------------------------------------------


async def test_a_narrated_tool_call_never_reaches_the_page_as_an_answer(wire):
    """
    The reason the answer is held rather than streamed immediately.

    The reason node emits before the router has decided anything, so a narrated
    tool call looks exactly like a final answer at that moment. Emitting it
    would show a wrong answer, then a nudge, then the real one - and the wrong
    one would be the first thing a person read.
    """
    provider = ScriptedProvider(
        [
            LLMResponse(content="I need to search the web. Let's do that first."),
            LLMResponse(content="the real answer"),
        ]
    )
    wire(provider)

    events = await collect("what is a Quazzlemint?")
    names = [name for name, _ in events]
    answers = [payload["content"] for name, payload in events if name == "answer"]

    assert names == ["tools", "nudge", "answer", "done"]
    assert answers == ["the real answer"]
    assert not any("Let's do that first" in a for a in answers)


async def test_the_nudge_is_announced(wire):
    """A silent extra round trip is an unexplained pause on the page."""
    provider = ScriptedProvider(
        [LLMResponse(content="I will search."), LLMResponse(content="done")]
    )
    wire(provider)

    events = await collect("anything")

    assert "nudge" in dict(events)
    assert "without making one" in dict(events)["nudge"]["message"]


async def test_a_run_that_used_a_tool_is_not_nudged(wire):
    """
    The narrow condition. A model that actually called something and then
    answered is finished, and asking again would cost a model call for nothing.
    """
    registry = DiscoverableRegistry(results={"search_web": "found it"})
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "search_web", {"query": "x"})]),
            LLMResponse(content="grounded answer"),
        ]
    )
    wire(provider, registry)

    events = await collect("a real question")

    assert "nudge" not in [name for name, _ in events]
    assert [name for name, _ in events] == ["tools", "tool_call", "tool_result", "answer", "done"]


async def test_the_loop_nudges_at_most_once(wire):
    """Two nudges in a row would be a loop, not a recovery."""
    provider = ScriptedProvider([LLMResponse(content="I will search.")] * 6)
    wire(provider)

    events = await collect("anything")

    assert [name for name, _ in events].count("nudge") == 1
