"""
Integration tests against real MCP servers.

These need the three agents actually running (see the README); they skip
cleanly when the agents aren't up, so `pytest` still passes on a fresh
checkout. Everything here exercises the protocol for real: the initialize
handshake, tool discovery, and live tool execution over Streamable HTTP.

Start the agents first:
    MCP_PORT=18000 python agents/research_agent/server.py
    MCP_PORT=18001 python agents/summarizer_agent/server.py
    MCP_PORT=18002 python agents/code_analysis_agent/server.py
"""

import pytest

from orchestrator.mcp_client import MCPToolRegistry

pytestmark = pytest.mark.integration


@pytest.fixture
async def registry(agents_running):
    reg = MCPToolRegistry()
    await reg.discover()
    return reg


async def test_discovers_every_agents_tools(registry):
    """One registry, three separate services - this is the multi-server payoff."""
    assert {tool["name"] for tool in registry.tools} == {
        "search_web",
        "summarize",
        "analyze_code",
    }


async def test_discovered_tools_carry_usable_schemas(registry):
    """The schema is what gets handed to the model, so it has to be complete."""
    search = next(t for t in registry.tools if t["name"] == "search_web")

    assert search["description"]
    assert search["input_schema"]["type"] == "object"
    assert "query" in search["input_schema"]["properties"]
    assert search["input_schema"]["required"] == ["query"]


@pytest.mark.parametrize(
    "name, arguments, expected_fragment",
    [
        ("search_web", {"query": "MCP adoption"}, "[stub result]"),
        ("summarize", {"text": "some text to summarize"}, "[stub summary]"),
        ("analyze_code", {"code": "def add(a, b): return a + b"}, "[stub analysis]"),
    ],
)
async def test_calls_are_routed_to_the_owning_agent(registry, name, arguments, expected_fragment):
    """Dispatch is a lookup in the ownership map built during discovery."""
    result = await registry.call(name, arguments)
    assert expected_fragment in result


async def test_unknown_tool_returns_an_error_string(registry):
    result = await registry.call("does_not_exist", {})
    assert "no agent exposes a tool named" in result


async def test_invalid_arguments_return_an_error_rather_than_raising(registry):
    """A failed call must stay inside the loop as feedback, not crash it."""
    result = await registry.call("search_web", {"wrong_field": 1})
    assert result.startswith("Error")


async def test_unreachable_agent_is_skipped_not_fatal():
    """Failure isolation: one agent down must not take the orchestrator down."""
    reg = MCPToolRegistry({"ghost": "http://localhost:1/mcp"})
    tools = await reg.discover()
    assert tools == []
