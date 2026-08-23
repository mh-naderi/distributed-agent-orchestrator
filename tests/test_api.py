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
