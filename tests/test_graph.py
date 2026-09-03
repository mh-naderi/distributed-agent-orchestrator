"""
Tests for the reason -> act -> reason loop.

No LLM and no agent servers involved: the registry and provider are fakes, so
what's under test is purely the orchestration logic - does a tool result get
fed back, does the loop terminate, does the guardrail hold.
"""

import pytest

from orchestrator.graph import (
    MAX_ITERATIONS,
    NO_EVIDENCE,
    REGROUND_PROMPT,
    SYSTEM_PROMPT,
    build_graph,
    looks_like_a_raw_tool_call,
)
from orchestrator.llm import LLMResponse, ToolCall
from tests.conftest import FakeRegistry, ScriptedProvider


def initial_state(task: str = "Research MCP adoption.") -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        "iterations": 0,
    }


async def test_tool_call_result_is_fed_back_to_the_model():
    """The core promise of the loop: the model's next turn can see the result."""
    registry = FakeRegistry(results={"search_web": "MCP is widely adopted."})
    provider = ScriptedProvider(
        [
            LLMResponse(
                content="Let me search.",
                tool_calls=[ToolCall("c1", "search_web", {"query": "MCP adoption"})],
            ),
            LLMResponse(content="MCP adoption is growing."),
        ]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert registry.calls == [("search_web", {"query": "MCP adoption"})]
    assert final["iterations"] == 2
    assert final["messages"][-1]["content"] == "MCP adoption is growing."

    # The second reason step must have been given the tool result.
    second_turn = provider.seen[1]
    tool_messages = [m for m in second_turn if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "MCP is widely adopted."
    assert tool_messages[0]["name"] == "search_web"
    assert tool_messages[0]["tool_call_id"] == "c1"


async def test_history_accumulates_in_order():
    """operator.add on messages should append, never replace."""
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "search_web", {"query": "x"})]),
            LLMResponse(content="done"),
        ]
    )

    final = await build_graph(FakeRegistry(), provider).ainvoke(initial_state())

    assert [m["role"] for m in final["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


async def test_an_answer_with_no_tool_call_is_nudged_once_then_accepted():
    """
    This used to end after a single iteration, and that is what let the
    narration bug through: "no tool call" was read as "finished", so a model
    that said "I need to search the web... let's do that first" ended the run
    with a non-answer.

    The cost is visible here and is the point of the trade: an answer that
    genuinely needed no tool now pays one extra model call before it is
    accepted. It is accepted - the nudge asks, it does not insist.
    """
    registry = FakeRegistry()
    provider = ScriptedProvider(
        [LLMResponse(content="No tools needed."), LLMResponse(content="Still no tools needed.")]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert final["iterations"] == 2, "the model was not given a second chance"
    assert final["nudges"] == 1
    assert registry.calls == []
    assert final["messages"][-1]["content"] == "Still no tools needed."


async def test_parallel_tool_calls_all_execute():
    """One assistant turn may request several tools; each needs a result."""
    registry = FakeRegistry()
    provider = ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("c1", "search_web", {"query": "a"}),
                    ToolCall("c2", "search_web", {"query": "b"}),
                ],
            ),
            LLMResponse(content="done"),
        ]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert registry.calls == [("search_web", {"query": "a"}), ("search_web", {"query": "b"})]
    assert sum(1 for m in final["messages"] if m["role"] == "tool") == 2


async def test_guardrail_stops_a_model_that_never_finishes():
    """Without this the loop would run until something else broke."""

    class NeverStops:
        async def chat(self, messages, tools):
            return LLMResponse(content="", tool_calls=[ToolCall("x", "search_web", {"query": "again"})])

    registry = FakeRegistry()
    final = await build_graph(registry, NeverStops()).ainvoke(initial_state())

    assert final["iterations"] == MAX_ITERATIONS
    # The guardrail is checked after reasoning and before acting, so the last
    # requested tool call is never executed.
    assert len(registry.calls) == MAX_ITERATIONS - 1


async def test_guardrail_leaves_a_usable_answer_rather_than_silence():
    """
    Stopping is not enough - the caller has to be TOLD it was cut short.

    This is the gap the test above missed. It asserted the loop halted, which
    it did, but never looked at what came back. On the truncated path the last
    message was the model's unanswered tool-call request, so content was empty:
    arun() returned '', and the streaming API sent tool_call events then done
    with no answer event - the page stopped mid-run with no explanation.
    """

    class NeverStops:
        async def chat(self, messages, tools):
            return LLMResponse(
                content="", tool_calls=[ToolCall("x", "search_web", {"query": "again"})]
            )

    final = await build_graph(FakeRegistry(), NeverStops()).ainvoke(initial_state())
    last = final["messages"][-1]

    # Not a dangling tool-call request.
    assert not last.get("tool_calls")
    # Something a human or the eval harness can actually read.
    assert last["content"].strip()
    # Flagged, so callers can tell a stop notice from a real answer.
    assert last.get("truncated") is True
    assert str(MAX_ITERATIONS) in last["content"]


async def test_tool_error_is_fed_back_rather_than_raised():
    """A failed tool should keep the loop alive - the model can react to it."""

    class FailingRegistry(FakeRegistry):
        async def call(self, name, arguments):
            self.calls.append((name, arguments))
            return f"Error calling {name}: boom"

    registry = FailingRegistry()
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "search_web", {"query": "x"})]),
            LLMResponse(content="That tool failed, answering directly."),
        ]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    tool_message = next(m for m in final["messages"] if m["role"] == "tool")
    assert "Error calling search_web" in tool_message["content"]
    assert final["messages"][-1]["content"] == "That tool failed, answering directly."


# ---------------------------------------------------------------------------
# Answering from tools that all reported nothing
# ---------------------------------------------------------------------------
# The nudge covers a model that never called a tool. This covers the opposite
# and more dangerous case: it called them, they all said they had nothing, and
# it answered anyway. Every fabrication this project has recorded has that shape.


def _empty(text: str) -> str:
    return f"{NO_EVIDENCE} {text}"


async def test_answering_from_nothing_is_sent_back():
    registry = FakeRegistry(
        results={"retrieve": _empty("No documents are close enough to that query.")}
    )
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "retrieve", {"query": "x"})]),
            LLMResponse(content="The Quazzlemint Foundation concluded that grants helped."),
            LLMResponse(content="I could not find anything about that."),
        ]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert final["regrounds"] == 1
    assert final["messages"][-1]["content"] == "I could not find anything about that."
    assert any(
        m.get("content") == REGROUND_PROMPT for m in final["messages"]
    ), "the model should have been told what the evidence actually was"


async def test_a_real_result_is_not_second_guessed():
    """The guardrail must stay out of the way when a tool actually found something."""
    registry = FakeRegistry(results={"retrieve": "MCP standardises tool access."})
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "retrieve", {"query": "mcp"})]),
            LLMResponse(content="MCP standardises tool access."),
        ]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert final.get("regrounds", 0) == 0


async def test_one_useful_result_among_empty_ones_is_enough():
    """
    Partial evidence is evidence. Firing here would send back an answer that had
    something real behind it, which is a different and worse mistake.
    """
    registry = FakeRegistry(
        results={
            "retrieve": _empty("Nothing close enough."),
            "search_web": "Foundations published annual reports in 2019.",
        }
    )
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "retrieve", {"query": "x"})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "search_web", {"query": "x"})]),
            LLMResponse(content="Foundations published annual reports in 2019."),
        ]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert final.get("regrounds", 0) == 0


async def test_regrounding_happens_at_most_once():
    """
    A model that answers from nothing twice is not going to be talked round by a
    third attempt, and a guardrail that keeps firing is a loop.
    """
    registry = FakeRegistry(results={"retrieve": _empty("Nothing close enough.")})
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "retrieve", {"query": "x"})]),
            LLMResponse(content="The foundation concluded something."),
            LLMResponse(content="The foundation concluded something else."),
        ]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert final["regrounds"] == 1
    assert final["messages"][-1]["content"] == "The foundation concluded something else."


async def test_a_turn_with_no_tool_call_is_the_nudge_s_business():
    """
    Both guardrails must not fire on one failure. Nothing ran, so there is no
    empty evidence to speak of - that is a narration, and the nudge owns it.
    """
    registry = FakeRegistry()
    provider = ScriptedProvider(
        [
            LLMResponse(content="I should search for that."),
            LLMResponse(content="I could not find anything."),
        ]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert final["nudges"] == 1
    assert final.get("regrounds", 0) == 0


async def test_the_guardrails_do_not_fire_on_each_other():
    """
    Regression. Both guardrails address the model as the user, for provider
    portability, so a turn-scoped check that stops at the last user message read
    the guardrail's own prompt as the start of a new turn: after regrounding, an
    honest answer with no tool call looked exactly like a narration and was
    nudged for it. Two recovery attempts on one failure, the second of them
    arguing with a correct answer.
    """
    registry = FakeRegistry(results={"retrieve": _empty("Nothing close enough.")})
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "retrieve", {"query": "x"})]),
            LLMResponse(content="The foundation concluded something."),
            LLMResponse(content="I could not find anything about that."),
        ]
    )

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert final["regrounds"] == 1
    assert final.get("nudges", 0) == 0, "the reground's own prompt started a new turn"


# ---------------------------------------------------------------------------
# A narrated tool call must never be published as an answer
# ---------------------------------------------------------------------------

NARRATION = '{"name": "retrieve", "arguments": {"query": "Quazzlemint Foundation"}}'


async def test_a_tool_call_typed_as_text_does_not_become_the_answer():
    """
    Observed after the nudge had fired and spent its budget: the run ended with
    a JSON tool call as its answer, and the page would have rendered that as the
    result. The nudge asks once; this is what happens when asking did not work.
    """
    provider = ScriptedProvider([LLMResponse(content=NARRATION)] * 3)

    final = await build_graph(FakeRegistry(), provider).ainvoke(initial_state())

    assert final["nudges"] == 1, "it should still be asked once first"
    assert final["messages"][-1].get("unanswered") is True
    assert "not an answer" in final["messages"][-1]["content"]
    assert NARRATION not in final["messages"][-1]["content"]


@pytest.mark.parametrize(
    "content",
    [
        '{"name": "retrieve", "arguments": {}}',
        '[{"name": "search_web", "arguments": {"query": "x"}}]',
        '  {"tool_calls": [{"name": "retrieve"}]}  ',
    ],
)
def test_tool_call_payloads_are_recognised(content):
    assert looks_like_a_raw_tool_call(content)


@pytest.mark.parametrize(
    "content",
    [
        "I could not find anything about that.",
        "",
        "The answer is {not json}",
        '{"total": 42}',                      # JSON, but no tool name
        "I will search for that next.",       # prose narration - the nudge's job
        '{"name": "incomplete',               # malformed
    ],
)
def test_ordinary_answers_are_not_mistaken_for_tool_calls(content):
    """
    A false positive here replaces a real answer with a failure notice, which is
    worse than the bug being fixed. Prose narration is deliberately not caught:
    the nudge covers it, and catching it reliably would need the lexical guessing
    this project keeps getting wrong.
    """
    assert not looks_like_a_raw_tool_call(content)


async def test_a_legitimate_answer_after_a_nudge_still_stands():
    """The nudge working must not be turned into a failure by this path."""
    provider = ScriptedProvider(
        [
            LLMResponse(content="I will search for that."),
            LLMResponse(content="I could not find anything about that."),
        ]
    )

    final = await build_graph(FakeRegistry(), provider).ainvoke(initial_state())

    assert final["nudges"] == 1
    assert final["messages"][-1]["content"] == "I could not find anything about that."
    assert "unanswered" not in final["messages"][-1]
