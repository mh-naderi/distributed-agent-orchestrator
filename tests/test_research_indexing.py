"""
Tests for the research agent indexing its own search results.

The behaviour that matters most here is what happens when the retrieval agent
is NOT available. This is a deliberate coupling between two agents, and the
whole justification for allowing it is that it cannot take the caller down. A
test that only covers the happy path would leave the actual risk unexercised.

No network and no agents: the MCP round trip is faked.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "research_agent"))

import indexer  # noqa: E402

TWO_PARAGRAPHS = "a" + chr(10) + chr(10) + "b"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    """AUTO_INDEX is read at import; tests should not depend on the ambient env."""
    monkeypatch.setattr(indexer, "AUTO_INDEX", True)


def test_search_results_are_handed_to_the_retrieval_agent(monkeypatch):
    seen = {}

    async def fake_index(text, source, url, timeout):
        seen.update(text=text, source=source, url=url, timeout=timeout)
        return "Indexed 2 document(s) from web-search. Corpus now holds 8."

    monkeypatch.setattr(indexer, "_index", fake_index)

    assert indexer.index_results("result one\n\nresult two", source="web-search: mcp") is True
    assert seen["text"] == "result one\n\nresult two"
    assert seen["source"] == "web-search: mcp"


def test_the_text_is_passed_through_unchunked(monkeypatch):
    """
    The retrieval agent splits on blank lines, and search results already
    arrive that way. Chunking here too would be a second opinion about document
    boundaries in the agent that does not own the index.
    """
    seen = {}

    async def fake_index(text, source, url, timeout):
        seen["text"] = text
        return "ok"

    monkeypatch.setattr(indexer, "_index", fake_index)
    indexer.index_results("a\n\nb\n\nc")

    assert seen["text"] == "a\n\nb\n\nc", "the caller pre-chunked what it does not own"


def test_a_dead_retrieval_agent_does_not_break_the_search(monkeypatch):
    """
    The reason this coupling is allowed at all.

    One agent failing must not take another down. If this ever raises, the
    isolation claim the multi-server split rests on stops being true.
    """

    async def exploding(text, source, url, timeout):
        raise ConnectionError("retrieval agent unreachable")

    monkeypatch.setattr(indexer, "_index", exploding)

    assert indexer.index_results("some results") is False


def test_a_timeout_does_not_break_the_search(monkeypatch):
    async def slow(text, source, url, timeout):
        raise TimeoutError("took too long")

    monkeypatch.setattr(indexer, "_index", slow)

    assert indexer.index_results("some results") is False


def test_failure_is_reported_rather_than_swallowed_into_success(monkeypatch):
    """
    Returning False is what lets the caller count the outcome. Best effort that
    reports success regardless is how a corpus stays empty while every
    dashboard looks healthy.
    """

    async def exploding(text, source, url, timeout):
        raise RuntimeError("boom")

    monkeypatch.setattr(indexer, "_index", exploding)
    result = indexer.index_results("results")

    assert result is False
    assert result is not None, "the caller cannot distinguish outcomes"


def test_indexing_can_be_turned_off(monkeypatch):
    called = False

    async def fake_index(text, source, url, timeout):
        nonlocal called
        called = True
        return "ok"

    monkeypatch.setattr(indexer, "_index", fake_index)
    monkeypatch.setattr(indexer, "AUTO_INDEX", False)

    assert indexer.index_results("results") is False
    assert called is False, "AUTO_INDEX=off still made a call"


@pytest.mark.parametrize("empty", ["", "   ", "\n\n"])
def test_empty_results_are_not_indexed(monkeypatch, empty):
    called = False

    async def fake_index(text, source, url, timeout):
        nonlocal called
        called = True
        return "ok"

    monkeypatch.setattr(indexer, "_index", fake_index)

    assert indexer.index_results(empty) is False
    assert called is False, "an empty search result was sent to the index"


async def test_it_works_when_called_from_inside_a_running_event_loop(monkeypatch):
    """
    The regression this file exists for.

    index_results is called from a FastMCP sync tool function, and those run
    ON THE EVENT LOOP THREAD - not, as the first version assumed, on a worker.
    A bare asyncio.run raises "cannot be called from a running event loop"
    there, which the best-effort handler swallowed: searches kept succeeding
    and the corpus silently never grew.

    This test is async, so a loop IS running when index_results is called -
    exactly the condition that was broken in the cluster and invisible here.
    """
    called = {}

    async def fake_index(text, source, url, timeout):
        called["text"] = text
        return "Indexed 3 document(s)."

    monkeypatch.setattr(indexer, "_index", fake_index)

    assert indexer.index_results(TWO_PARAGRAPHS) is True, (
        "indexing failed while an event loop was running - the exact bug this "
        "test exists to catch"
    )
    assert called["text"] == TWO_PARAGRAPHS
