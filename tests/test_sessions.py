"""
Tests for conversation history and the trimming that bounds it.

Trimming is where the risk lives. Getting it wrong does not raise - Ollama
silently truncates an over-long prompt from the front, so a broken budget shows
up as a model that mysteriously ignores its instructions. Every rule below is
therefore asserted directly rather than inferred from a working run.
"""

import time

import pytest

from orchestrator.sessions import (
    DROPPED_MARKER,
    SessionStore,
    estimate_chars,
    trim,
)

SYSTEM = {"role": "system", "content": "You are an orchestrator."}


def exchange(question: str, tool_output: str, answer: str) -> list[dict]:
    """One complete turn: question, tool call, result, answer."""
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": f"c-{question}", "name": "search_web", "arguments": {}}],
        },
        {
            "role": "tool",
            "name": "search_web",
            "tool_call_id": f"c-{question}",
            "content": tool_output,
        },
        {"role": "assistant", "content": answer},
    ]


# ---------------------------------------------------------------------------
# What must never be sacrificed
# ---------------------------------------------------------------------------


def test_history_within_budget_is_untouched():
    messages = [SYSTEM, *exchange("q", "small result", "a")]

    assert trim(messages, budget=10_000) == messages


def test_the_system_prompt_always_survives():
    """
    Ollama truncates from the FRONT, so an unbounded history silently discards
    the instructions and leaves a model that appears to have forgotten its job.
    """
    messages = [SYSTEM] + exchange("q1", "x" * 5000, "a1") + exchange("q2", "y" * 5000, "a2")

    trimmed = trim(messages, budget=500)

    assert trimmed[0] == SYSTEM
    assert any(m["role"] == "system" for m in trimmed)


def test_the_current_turn_survives_intact():
    """A run that cannot see its own question and results is useless."""
    messages = [SYSTEM] + exchange("old", "x" * 5000, "a1") + exchange("current", "fresh result", "a2")

    trimmed = trim(messages, budget=400)

    assert trimmed[-1]["content"] == "a2"
    current_tool = [m for m in trimmed if m["role"] == "tool" and m["content"] == "fresh result"]
    assert current_tool, "the current turn's tool result was trimmed away"


# ---------------------------------------------------------------------------
# The rule: old tool output goes first
# ---------------------------------------------------------------------------


def test_old_tool_output_is_dropped_before_anything_else():
    messages = [SYSTEM] + exchange("q1", "x" * 4000, "answer one") + exchange("q2", "small", "answer two")

    trimmed = trim(messages, budget=1000)

    texts = [m.get("content") for m in trimmed]
    assert DROPPED_MARKER in texts, "old tool output was not the first thing sacrificed"
    # The conversational thread is what the tool output was summarised into.
    assert "answer one" in texts
    assert "q1" in texts


def test_a_dropped_tool_result_keeps_its_message():
    """
    Truncated in place, never removed.

    A tool result is the other half of an assistant tool call, paired by id.
    Deleting the result while leaving the call behind is a malformed
    conversation - Anthropic rejects an unmatched tool_use, and Ollama accepts
    it and behaves oddly, which is worse.
    """
    messages = [SYSTEM] + exchange("q1", "x" * 4000, "a1") + exchange("q2", "small", "a2")

    trimmed = trim(messages, budget=800)

    call_ids = {
        call["id"]
        for m in trimmed
        for call in m.get("tool_calls", [])
    }
    result_ids = {m["tool_call_id"] for m in trimmed if m["role"] == "tool"}

    assert call_ids == result_ids, f"unpaired tool call/result: {call_ids ^ result_ids}"


def test_whole_turns_go_only_when_blanking_is_not_enough():
    """Pass 2. Turns leave as a unit so no call is orphaned from its result."""
    messages = [SYSTEM]
    for i in range(6):
        messages += exchange(f"question number {i}", "z" * 900, f"answer number {i}")

    trimmed = trim(messages, budget=600)

    assert estimate_chars(trimmed) <= 600
    assert trimmed[0] == SYSTEM
    # The most recent turn is still whole.
    assert trimmed[-1]["content"] == "answer number 5"

    call_ids = {c["id"] for m in trimmed for c in m.get("tool_calls", [])}
    result_ids = {m["tool_call_id"] for m in trimmed if m["role"] == "tool"}
    assert call_ids == result_ids


def test_trimming_actually_gets_under_budget():
    messages = [SYSTEM]
    for i in range(10):
        messages += exchange(f"q{i}", "x" * 2000, f"a{i}")

    trimmed = trim(messages, budget=1500)

    assert estimate_chars(trimmed) <= 1500


def test_trim_does_not_mutate_its_input():
    messages = [SYSTEM] + exchange("q1", "x" * 4000, "a1") + exchange("q2", "s", "a2")
    before = [dict(m) for m in messages]

    trim(messages, budget=500)

    assert messages == before, "trim mutated the caller's history"


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_a_session_remembers_across_runs():
    store = SessionStore()
    store.save("s1", [SYSTEM, {"role": "user", "content": "first"}])

    assert store.get("s1").messages[-1]["content"] == "first"


def test_sessions_are_isolated_from_each_other():
    store = SessionStore()
    store.save("a", [SYSTEM, {"role": "user", "content": "mine"}])
    store.save("b", [SYSTEM, {"role": "user", "content": "theirs"}])

    assert store.get("a").messages[-1]["content"] == "mine"
    assert store.get("b").messages[-1]["content"] == "theirs"


def test_an_unknown_session_starts_empty():
    assert SessionStore().get("never-seen").messages == []


def test_the_store_is_bounded():
    """
    The key is chosen by the client, so an unbounded store is a memory leak
    with a public entry point.
    """
    store = SessionStore(max_sessions=3)
    for i in range(10):
        store.get(f"session-{i}")

    assert len(store) <= 3


def test_the_least_recently_used_session_is_evicted_first():
    """
    No sleeps, because ordering must not depend on the clock.

    This test used to space its calls with time.sleep(0.01) and failed about
    half the time: time.monotonic() has roughly 15ms resolution on Windows, so
    several sessions touched inside one tick shared a timestamp and min() chose
    between them arbitrarily. That was a real defect, not a slow test - a busy
    store could evict the session it had just served. Eviction now orders by a
    counter, which has no resolution to run out of.
    """
    store = SessionStore(max_sessions=2)
    store.save("keep-me", [SYSTEM])
    store.save("old", [SYSTEM])
    store.get("keep-me")          # touch it so "old" is now the stale one
    store.get("new")              # forces an eviction

    assert store.get("keep-me").messages == [SYSTEM]


def test_expired_sessions_are_dropped():
    store = SessionStore(max_sessions=50, ttl=0.01)
    store.save("stale", [SYSTEM, {"role": "user", "content": "old news"}])
    time.sleep(0.05)

    store.get("something-else")  # any access triggers the sweep

    assert store.get("stale").messages == [], "an expired session was still served"


def test_saving_trims_before_storing():
    """Otherwise the budget is only enforced on the way out, one run too late."""
    store = SessionStore()
    huge = [SYSTEM]
    for i in range(10):
        huge += exchange(f"q{i}", "x" * 3000, f"a{i}")

    store.save("s", huge)

    from orchestrator.config import history_budget_chars

    assert estimate_chars(store.get("s").messages) <= history_budget_chars()


@pytest.mark.parametrize("messages", [[], [SYSTEM]])
def test_trimming_degenerate_histories_is_safe(messages):
    assert trim(messages, budget=10) == messages or trim(messages, budget=10) == list(messages)
