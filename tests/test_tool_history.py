"""
A trace keeps what each tool was asked, not only what it answered.

WHY THIS EXISTS. A run fabricated an answer after calling search_web, and a bug
had just been found in which a query opening with its subject got no coverage
warning. Whether that run's query had that shape could not be checked: the trace
kept the search results and threw the query away. See orchestrator/main.py.

The graph tests below go through build_graph rather than hand-written messages,
so a change to the shape the act node writes breaks this instead of silently
turning every argument into None.
"""

from orchestrator.graph import SYSTEM_PROMPT, build_graph
from orchestrator.llm import LLMResponse, ToolCall
from orchestrator.main import tool_history
from tests.conftest import FakeRegistry, ScriptedProvider


class EchoRegistry(FakeRegistry):
    """Answers with the query it was given, so each output names its request."""

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        return f"results for {arguments.get('query')}"


async def run(provider, registry=None):
    final = await build_graph(registry or EchoRegistry(), provider).ainvoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What did the Quazzlemint Foundation conclude?"},
            ],
            "iterations": 0,
        }
    )
    return tool_history(final["messages"])


async def test_the_query_the_model_sent_is_kept_with_its_results():
    provider = ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "search_web", {"query": "Quazzlemint Foundation 2019 report"})],
            ),
            LLMResponse(content="Nothing was found about the Quazzlemint Foundation."),
        ]
    )

    tools_called, tool_outputs = await run(provider)

    assert tools_called == ["search_web"]
    assert tool_outputs == [
        {
            "name": "search_web",
            "arguments": {"query": "Quazzlemint Foundation 2019 report"},
            "output": "results for Quazzlemint Foundation 2019 report",
        }
    ]


async def test_parallel_calls_each_keep_their_own_arguments():
    """
    Two calls to the same tool in one turn is the case a positional pairing
    would get wrong without any sign of it. The echo makes a mismatch visible:
    each output has to sit next to the query it names.
    """
    provider = ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("c1", "search_web", {"query": "alpha"}),
                    ToolCall("c2", "search_web", {"query": "beta"}),
                ],
            ),
            LLMResponse(content="done"),
        ]
    )

    _, tool_outputs = await run(provider)

    assert len(tool_outputs) == 2
    for entry in tool_outputs:
        assert entry["output"] == f"results for {entry['arguments']['query']}"


async def test_calls_across_turns_are_kept_in_the_order_they_ran():
    provider = ScriptedProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "search_web", {"query": "first"})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "search_web", {"query": "second"})]),
            LLMResponse(content="done"),
        ]
    )

    tools_called, tool_outputs = await run(provider)

    assert tools_called == ["search_web", "search_web"]
    assert [t["arguments"]["query"] for t in tool_outputs] == ["first", "second"]


def test_pairing_is_by_id_not_by_position():
    """Outputs listed out of order must not borrow each other's arguments."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "name": "search_web", "arguments": {"query": "alpha"}},
                {"id": "b", "name": "retrieve", "arguments": {"query": "beta"}},
            ],
        },
        {"role": "tool", "name": "retrieve", "tool_call_id": "b", "content": "B"},
        {"role": "tool", "name": "search_web", "tool_call_id": "a", "content": "A"},
    ]

    _, tool_outputs = tool_history(messages)

    assert tool_outputs == [
        {"name": "retrieve", "arguments": {"query": "beta"}, "output": "B"},
        {"name": "search_web", "arguments": {"query": "alpha"}, "output": "A"},
    ]


def test_an_output_with_no_matching_request_says_so():
    """None, not a neighbour's arguments: a missing fact beats a wrong one."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "a", "name": "search_web", "arguments": {"query": "alpha"}}],
        },
        {"role": "tool", "name": "search_web", "tool_call_id": "zzz", "content": "orphan"},
        {"role": "tool", "name": "search_web", "content": "no id at all"},
    ]

    _, tool_outputs = tool_history(messages)

    assert [t["arguments"] for t in tool_outputs] == [None, None]


def test_a_run_that_called_nothing_has_an_empty_history():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    assert tool_history(messages) == ([], [])
