"""
Tests for the reason -> act -> reason loop.

No LLM and no agent servers involved: the registry and provider are fakes, so
what's under test is purely the orchestration logic - does a tool result get
fed back, does the loop terminate, does the guardrail hold.
"""

from orchestrator.graph import MAX_ITERATIONS, SYSTEM_PROMPT, build_graph
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


async def test_answer_without_tools_ends_after_one_iteration():
    registry = FakeRegistry()
    provider = ScriptedProvider([LLMResponse(content="No tools needed.")])

    final = await build_graph(registry, provider).ainvoke(initial_state())

    assert final["iterations"] == 1
    assert registry.calls == []
    assert final["messages"][-1]["content"] == "No tools needed."


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
