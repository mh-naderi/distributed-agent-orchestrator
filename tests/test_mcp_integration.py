"""
Integration tests against real MCP servers.

These need the three agents actually running (see the README); they skip
cleanly when the agents aren't up, so `pytest` still passes on a fresh
checkout. Everything here exercises the protocol for real: the initialize
handshake, tool discovery, and live tool execution over Streamable HTTP.

Start the agents first:
    MCP_PORT=18000 python agents/research_agent/server.py
    MCP_PORT=18001 python agents/retrieval_agent/server.py
    MCP_PORT=18002 python agents/code_analysis_agent/server.py

The retrieval agent also needs Ollama running with the embedding model pulled.
"""

import re

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
        "index_documents",
        "retrieve",
        "analyze_code",
        "evaluate_expression",
    }


async def test_discovered_tools_carry_usable_schemas(registry):
    """The schema is what gets handed to the model, so it has to be complete."""
    search = next(t for t in registry.tools if t["name"] == "search_web")

    assert search["description"]
    assert search["input_schema"]["type"] == "object"
    assert "query" in search["input_schema"]["properties"]
    assert search["input_schema"]["required"] == ["query"]


async def test_index_then_retrieve_round_trip(registry):
    """
    Exercises both retrieval tools across the network, and incidentally proves
    dispatch works - both tools live on the retrieval agent, search_web doesn't.

    The marker is a FIXED string, and the test accepts it being indexed or
    already present.

    This runs against the real retrieval agent, so it writes to the real
    corpus - the one piece of state in the system with no source to rebuild it
    from - and nothing can remove what it writes: the agent exposes no delete
    tool, deliberately, because every MCP tool is visible to the model.

    It used to append a random suffix per run. That solved the problem of a
    fixed string indexing 0 documents on its second run and failing an exact
    "Indexed 1 document" check, and it did so by writing a brand-new fictional
    document every time. The corpus held 129 of them before anyone looked. The
    retrieval floor kept them away from real questions, which is why nothing
    visibly broke, but a test that grows production data without bound is a
    leak whether or not it is noticed.

    So the check now accepts 0 or 1: the round trip is proven by retrieve
    finding the marker and by the provenance assertions below, which hold either
    way, and the corpus carries exactly one copy of this fixture instead of one
    per run. Proving that a NEW document can be stored belongs to test_store.py,
    which does not need the network or the real corpus to do it.
    """
    marker = "Xylophone Quarks Institute studies imaginary particles (integration fixture)"

    indexed = await registry.call(
        "index_documents", {"texts": [marker], "source": "integration-test"}
    )
    assert re.search(r"Indexed [01] document", indexed), indexed

    found = await registry.call("retrieve", {"query": marker, "k": 3})
    assert marker in found
    # "integration-test" is what this test CLAIMED it was indexing, and the
    # corpus no longer repeats a caller's word back as though it were where the
    # document came from. This text names no origin, so it has none.
    assert "no stated origin" in found
    assert "integration-test" not in found


async def test_retrieve_says_so_when_nothing_matches(registry):
    """
    An empty result must read as empty. A blank-but-successful-looking response
    is what led the model to invent answers when search was stubbed.
    """
    result = await registry.call(
        "retrieve", {"query": "zzzqqq nonexistent gibberish topic", "k": 1}
    )
    assert result.strip()


async def test_search_returns_real_results_with_sources(registry):
    """
    Should return actual web content, not a stub string.

    This one genuinely reaches DuckDuckGo, so it can fail for reasons that are
    not a regression: ddgs scrapes HTML and rate-limits under rapid use, which
    docs/architecture.md lists as a known gap. It failed once during a session
    that had just driven dozens of searches through the eval harness, then
    passed on retry and across three consecutive suite runs. Re-run before
    investigating - and note CI never sees this, because the integration tests
    skip there without agents.

    SearchService now retries once on what looks like throttling, which should
    make this rarer without making it impossible; a sustained rate limit still
    surfaces, deliberately, as a failure rather than as "no results found".
    """
    result = await registry.call("search_web", {"query": "Model Context Protocol"})

    assert "[stub" not in result
    assert "Source: http" in result


async def test_search_results_exclude_advertisements(registry):
    """
    Sponsored results arrive looking like organic ones. Left in, they get
    indexed and cited as evidence - an early run had the model reporting a
    vendor's ebook marketing as a finding.
    """
    result = await registry.call("search_web", {"query": "kubernetes security"})

    for marker in ("/aclick", "duckduckgo.com/y.js", "doubleclick.net"):
        assert marker not in result


async def test_calls_are_routed_to_the_owning_agent(registry):
    """
    Dispatch is a lookup in the ownership map built during discovery.

    This asserted on "[stub analysis]" until analyze_code became real static
    analysis. It kept passing locally because these tests skip without agents
    running, so the staleness only surfaced once they were reachable again.
    """
    result = await registry.call("analyze_code", {"code": "def add(a, b): return a + b"})

    # Clean code, so the report is the no-findings one - which must still say
    # what it checked rather than implying the code is correct.
    assert "No issues found by the checks that were run" in result
    assert "NOT evidence that the code is correct" in result


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
