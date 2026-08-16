"""
Shared test fixtures and fakes.

The fakes here are the payoff for build_graph() taking its registry and
provider as arguments: the loop can be tested without an LLM and without any
agent servers, so the tests that cover orchestration logic run in
milliseconds and never flake on the network.
"""

import socket
from urllib.parse import urlparse

import pytest

from orchestrator.llm import LLMResponse


class FakeRegistry:
    """Stands in for MCPToolRegistry - records calls, returns canned output."""

    def __init__(self, tools=None, results=None):
        self.tools = tools if tools is not None else [
            {
                "name": "search_web",
                "description": "Search the web.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return self._results.get(name, f"[fake result from {name}]")


class ScriptedProvider:
    """Replays a fixed list of LLM turns and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    async def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        if self.turns:
            return self.turns.pop(0)
        return LLMResponse(content="(out of scripted turns)")


def _reachable(url: str, timeout: float = 0.3) -> bool:
    parsed = urlparse(url)
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((parsed.hostname, parsed.port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


@pytest.fixture
def agents_running():
    """Skip an integration test unless every configured agent is listening."""
    from orchestrator.config import AGENT_URLS

    down = [name for name, url in AGENT_URLS.items() if not _reachable(url)]
    if down:
        pytest.skip(f"agent(s) not running: {', '.join(down)}")
