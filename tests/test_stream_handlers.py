"""
The per-node event handlers, called directly.

These decisions used to live as branches inside the streaming generator, mutating
two closure variables between yields, so "does a nudge drop the held answer" could
only be checked by driving the whole endpoint and reading the event stream back.
The state is an object now and each handler is callable, which lets the rules be
stated one at a time.

test_api.py still covers the assembled stream end to end. This is the same logic
examined a node at a time.
"""

import json

from orchestrator.api import (
    NODE_HANDLERS,
    RunState,
    _on_act,
    _on_nudge,
    _on_reason,
    _on_reground,
    _on_truncate,
    _on_unanswered,
)


def events(handler, run, update=None, messages=None):
    return [
        (frame["event"], json.loads(frame["data"]))
        for frame in handler(run, update or {}, messages or [])
    ]


def test_a_run_starts_pessimistic():
    """
    Anything escaping without setting an outcome - an exception, or a client
    navigating away mid-run - must be counted as a failure rather than not
    counted at all.
    """
    assert RunState().outcome == "failed"
    assert RunState().pending_answer is None


# ---------------------------------------------------------------------------
# reason
# ---------------------------------------------------------------------------


def test_a_tool_request_is_announced_immediately():
    run = RunState()

    emitted = events(
        _on_reason,
        run,
        {"iterations": 2},
        [{"role": "assistant", "tool_calls": [{"name": "retrieve", "arguments": {"query": "x"}}]}],
    )

    assert emitted == [("tool_call", {"iteration": 2, "name": "retrieve", "arguments": {"query": "x"}})]
    assert run.pending_answer is None


def test_an_answer_is_held_rather_than_emitted():
    """
    The router has not run yet, so this may still be a narrated tool call the
    loop is about to nudge. Emitting it now would show the page a wrong answer,
    then a nudge, then the real one.
    """
    run = RunState()

    emitted = events(_on_reason, run, {}, [{"role": "assistant", "content": "the answer"}])

    assert emitted == [], "nothing may reach the page yet"
    assert run.pending_answer == "the answer"


# ---------------------------------------------------------------------------
# the two recovery nodes
# ---------------------------------------------------------------------------


def test_a_nudge_drops_the_held_answer_and_says_so():
    run = RunState(pending_answer="I will search for that")

    emitted = events(_on_nudge, run)

    assert run.pending_answer is None, "the held answer was the narration"
    assert emitted[0][0] == "nudge"
    assert "asking it again" in emitted[0][1]["message"]


def test_a_reground_drops_the_held_answer_and_says_so():
    run = RunState(pending_answer="The foundation concluded that grants helped")

    emitted = events(_on_reground, run)

    assert run.pending_answer is None
    assert emitted[0][0] == "reground"
    assert "found nothing" in emitted[0][1]["message"]


# ---------------------------------------------------------------------------
# the two terminal nodes
# ---------------------------------------------------------------------------


def test_an_unanswered_run_is_neither_an_answer_nor_counted_as_one():
    run = RunState(pending_answer='{"name": "retrieve"}')

    emitted = events(_on_unanswered, run, {}, [{"content": "No answer was produced."}])

    assert run.pending_answer is None, "the narration must not be flushed afterwards"
    assert run.outcome == "unanswered"
    assert emitted == [("unanswered", {"content": "No answer was produced."})]


def test_a_stop_notice_is_not_presented_as_a_result():
    run = RunState()

    emitted = events(_on_truncate, run, {}, [{"content": "Stopped after 10 iterations"}])

    assert run.outcome == "truncated"
    assert emitted[0][0] == "truncated"
    assert emitted[0][0] != "answer"


# ---------------------------------------------------------------------------
# act
# ---------------------------------------------------------------------------


def test_every_tool_result_is_reported_in_order():
    run = RunState()

    emitted = events(
        _on_act,
        run,
        {},
        [
            {"name": "retrieve", "content": "first"},
            {"name": "search_web", "content": "second"},
        ],
    )

    assert [name for name, _ in emitted] == ["tool_result", "tool_result"]
    assert [payload["name"] for _, payload in emitted] == ["retrieve", "search_web"]


def test_a_result_without_a_name_still_reports_something():
    emitted = events(_on_act, RunState(), {}, [{"content": "output"}])

    assert emitted[0][1]["name"] == "?"


# ---------------------------------------------------------------------------
# the table itself
# ---------------------------------------------------------------------------


def test_every_graph_node_that_emits_has_a_handler():
    """
    A node added to the graph without one here would run and produce silence,
    which is the failure the truncate node was introduced to fix in the first
    place - the loop stopping with nothing said about it.
    """
    assert set(NODE_HANDLERS) == {
        "reason",
        "act",
        "nudge",
        "reground",
        "unanswered",
        "truncate",
    }
