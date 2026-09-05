"""
Tests for the correlation id that ties four services' logs together.

What matters is that an id exists for the length of a run, that it reaches the
agents by a channel the model never sees, and that its absence is harmless. The
last one carries the most weight: a tool must not stop working because nobody
was watching it.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from orchestrator import trace

AGENT_DIR = Path(__file__).resolve().parents[1] / "agents" / "code_analysis_agent"
sys.path.insert(0, str(AGENT_DIR))

# Imported under its own name rather than loaded from the file, unlike the other
# agent modules in this suite. Loading it fresh would execute it a second time
# and re-register tool_calls_total, which the Prometheus registry refuses -
# collection failed outright with DuplicateTimeseries. Every agent's copy is
# byte-identical and a test enforces that, so whichever loads first serves.
import instrumentation  # noqa: E402


# ---------------------------------------------------------------------------
# the id itself
# ---------------------------------------------------------------------------


def test_there_is_no_id_outside_a_run():
    assert trace.current_trace_id() is None


def test_an_id_lasts_exactly_as_long_as_the_run():
    with trace.use_trace() as trace_id:
        assert trace.current_trace_id() == trace_id

    assert trace.current_trace_id() is None


def test_a_nested_run_does_not_leak_its_id_outward():
    """The tests nest runs; a leaked id would attribute later work to an earlier run."""
    with trace.use_trace() as outer:
        with trace.use_trace() as inner:
            assert inner != outer
            assert trace.current_trace_id() == inner
        assert trace.current_trace_id() == outer


def test_ids_are_short_and_distinct():
    ids = {trace.new_trace_id() for _ in range(200)}

    assert len(ids) == 200, "collisions at this scale would make greps ambiguous"
    assert all(len(i) == trace.TRACE_ID_LENGTH for i in ids)


def test_concurrent_runs_do_not_share_an_id():
    """
    A contextvar rather than a module global, so two runs in flight keep their
    own id instead of both seeing whichever was set last.
    """

    async def run(seen):
        with trace.use_trace() as tid:
            await asyncio.sleep(0)  # yield, so the other task interleaves
            seen.append((tid, trace.current_trace_id()))

    async def both():
        seen = []
        await asyncio.gather(run(seen), run(seen))
        return seen

    seen = asyncio.run(both())

    assert all(expected == actual for expected, actual in seen)
    assert seen[0][0] != seen[1][0]


def test_the_paired_form_restores_the_previous_id():
    with trace.use_trace() as outer:
        inner, token = trace.begin()
        assert trace.current_trace_id() == inner
        trace.reset(token)
        assert trace.current_trace_id() == outer


# ---------------------------------------------------------------------------
# what the agent reads
# ---------------------------------------------------------------------------


class FakeMeta:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeContext:
    def __init__(self, meta=None, raises=None):
        self._meta = meta
        self._raises = raises

    @property
    def request_context(self):
        if self._raises:
            raise self._raises
        return FakeMeta(meta=self._meta)


def test_the_agent_reads_the_id_the_orchestrator_sent():
    context = FakeContext(meta=FakeMeta(traceId="abc12345"))

    assert instrumentation.trace_id_of(context) == "abc12345"


@pytest.mark.parametrize(
    "context",
    [
        FakeContext(meta=None),                       # a caller that sent no meta
        FakeContext(meta=FakeMeta(progressToken=1)),  # meta, but no trace id
        FakeContext(raises=ValueError("no request")), # called outside a request
        FakeContext(raises=AttributeError("none")),
    ],
)
def test_a_missing_id_is_never_an_error(context):
    """
    A tool must not stop working because nobody was watching. Every way the id
    can be absent - a hand-made call, an older orchestrator, no request context
    at all - reads as "-" rather than raising.
    """
    assert instrumentation.trace_id_of(context) == "-"


# ---------------------------------------------------------------------------
# the join between the two halves
# ---------------------------------------------------------------------------


class _Result:
    content = []
    isError = False


class RecordingSession:
    """Captures what the client actually put on the wire."""

    def __init__(self, sink):
        self.sink = sink

    async def call_tool(self, name, arguments, meta=None):
        self.sink.append({"name": name, "arguments": arguments, "meta": meta})
        return _Result()


def _fake_session(sink):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def session(url):
        yield RecordingSession(sink)

    return session


def test_the_client_sends_the_id_as_protocol_metadata(monkeypatch):
    """
    Not as an argument. Arguments appear in the schema the model is shown, so a
    trace id there would be one more field for it to get wrong.
    """
    from orchestrator import mcp_client

    sent = []
    registry = mcp_client.MCPToolRegistry()
    registry._owner = {"retrieve": "http://agent/mcp"}
    monkeypatch.setattr(mcp_client, "_session", _fake_session(sent))

    with trace.use_trace("feedface"):
        asyncio.run(registry.call("retrieve", {"query": "x"}))

    assert sent[0]["meta"] == {"traceId": "feedface"}
    assert "traceId" not in sent[0]["arguments"], "must not reach the tool's arguments"


def test_a_call_outside_a_run_sends_no_metadata(monkeypatch):
    """
    None rather than a placeholder: a call nobody is tracing should look exactly
    as it did before tracing existed.
    """
    from orchestrator import mcp_client

    sent = []
    registry = mcp_client.MCPToolRegistry()
    registry._owner = {"retrieve": "http://agent/mcp"}
    monkeypatch.setattr(mcp_client, "_session", _fake_session(sent))

    asyncio.run(registry.call("retrieve", {"query": "x"}))

    assert sent[0]["meta"] is None
