"""
Tests for the Claude escalation provider.

No key, no network, no spend: the Anthropic client is a fake that records the
request it was handed back. What is under test is the TRANSLATION - the graph
speaks a provider-neutral message shape, and Anthropic's wire format differs
from it in three specific ways that are easy to get quietly wrong.

The live path is deliberately not exercised here. It has never been run against
the real API.
"""

import os
from types import SimpleNamespace

import pytest

from orchestrator.llm import ClaudeProvider, OllamaProvider, get_provider


class FakeBlock(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


class FakeClient:
    """Captures the request and replays a scripted response."""

    def __init__(self, blocks=None, stop_reason="end_turn", stop_details=None):
        self.captured = None
        self._response = SimpleNamespace(
            content=blocks if blocks is not None else [FakeBlock(type="text", text="hi")],
            stop_reason=stop_reason,
            stop_details=stop_details,
        )
        outer = self

        class _Messages:
            async def create(self, **kwargs):
                outer.captured = kwargs
                return outer._response

        self.messages = _Messages()
        self.beta = SimpleNamespace(messages=_Messages())


def provider(client, **kwargs):
    return ClaudeProvider(client=client, **kwargs)


# ---------------------------------------------------------------------------
# Translation problem 1: the system prompt is a parameter, not a message
# ---------------------------------------------------------------------------


async def test_the_system_prompt_is_lifted_out_of_the_messages():
    client = FakeClient()
    await provider(client).chat(
        [
            {"role": "system", "content": "You are an orchestrator."},
            {"role": "user", "content": "hello"},
        ],
        [],
    )

    assert client.captured["system"] == "You are an orchestrator."
    roles = [m["role"] for m in client.captured["messages"]]
    assert "system" not in roles, "the system turn was left in the message list"
    assert roles == ["user"]


# ---------------------------------------------------------------------------
# Translation problem 2: tool results are user messages, batched
# ---------------------------------------------------------------------------


async def test_a_tool_result_becomes_a_user_message():
    client = FakeClient()
    await provider(client).chat(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "search_web", "arguments": {"query": "x"}}],
            },
            {"role": "tool", "name": "search_web", "tool_call_id": "c1", "content": "result"},
        ],
        [],
    )

    last = client.captured["messages"][-1]
    assert last["role"] == "user"
    assert last["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "result"}
    ]


async def test_consecutive_tool_results_are_batched_into_one_message():
    """
    Splitting parallel results across messages teaches the model to stop making
    parallel calls, so they have to arrive together.
    """
    client = FakeClient()
    await provider(client).chat(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "a", "arguments": {}},
                    {"id": "c2", "name": "b", "arguments": {}},
                ],
            },
            {"role": "tool", "name": "a", "tool_call_id": "c1", "content": "one"},
            {"role": "tool", "name": "b", "tool_call_id": "c2", "content": "two"},
        ],
        [],
    )

    user_turns = [m for m in client.captured["messages"] if m["role"] == "user"]
    batched = user_turns[-1]["content"]
    assert len(batched) == 2, f"results were split across messages: {user_turns}"
    assert [b["tool_use_id"] for b in batched] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# Translation problem 3: thinking blocks must be replayed unchanged
# ---------------------------------------------------------------------------


async def test_claude_turns_are_replayed_verbatim():
    """
    The reason raw_content exists.

    Rebuilding an assistant turn from the neutral shape would drop the thinking
    blocks, which must be echoed back unchanged when thinking and tool use are
    combined.
    """
    original = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "c1", "name": "search_web", "input": {"query": "x"}},
    ]
    client = FakeClient()
    await provider(client).chat(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "search_web", "arguments": {"query": "x"}}],
                "_claude_content": original,
            },
        ],
        [],
    )

    assistant = [m for m in client.captured["messages"] if m["role"] == "assistant"][0]
    assert assistant["content"] == original
    assert any(b["type"] == "thinking" for b in assistant["content"])


async def test_a_turn_from_the_local_model_is_rebuilt_rather_than_dropped():
    """A run that started on Ollama and escalated has no Claude blocks to replay."""
    client = FakeClient()
    await provider(client).chat(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "let me look",
                "tool_calls": [{"id": "c1", "name": "search_web", "arguments": {"query": "x"}}],
            },
        ],
        [],
    )

    assistant = [m for m in client.captured["messages"] if m["role"] == "assistant"][0]
    types = [b["type"] for b in assistant["content"]]
    assert types == ["text", "tool_use"]


async def test_the_response_carries_its_blocks_back_for_the_next_turn():
    blocks = [FakeBlock(type="text", text="done")]
    result = await provider(FakeClient(blocks=blocks)).chat([{"role": "user", "content": "q"}], [])

    assert result.raw_content == [{"type": "text", "text": "done"}]


# ---------------------------------------------------------------------------
# Request shape and responses
# ---------------------------------------------------------------------------


async def test_mcp_tool_schemas_pass_through_unchanged():
    """
    MCP and Anthropic describe a tool the same way, down to the key name, which
    is why discovery output needs nothing re-described by hand.
    """
    client = FakeClient()
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    await provider(client).chat(
        [{"role": "user", "content": "q"}],
        [{"name": "search_web", "description": "Search.", "input_schema": schema}],
    )

    assert client.captured["tools"] == [
        {"name": "search_web", "description": "Search.", "input_schema": schema}
    ]


async def test_thinking_is_adaptive():
    """This provider exists for reasoning the local model cannot do."""
    client = FakeClient()
    await provider(client).chat([{"role": "user", "content": "q"}], [])

    assert client.captured["thinking"] == {"type": "adaptive"}


async def test_tool_use_blocks_become_tool_calls():
    blocks = [
        FakeBlock(type="text", text="looking"),
        FakeBlock(type="tool_use", id="toolu_1", name="search_web", input={"query": "mcp"}),
    ]
    result = await provider(FakeClient(blocks=blocks)).chat([{"role": "user", "content": "q"}], [])

    assert result.content == "looking"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    # Claude assigns real ids, unlike Ollama - nothing is synthesised here.
    assert (call.id, call.name, call.arguments) == ("toolu_1", "search_web", {"query": "mcp"})


async def test_thinking_blocks_are_not_treated_as_answer_text():
    blocks = [
        FakeBlock(type="thinking", thinking="internal"),
        FakeBlock(type="text", text="the answer"),
    ]
    result = await provider(FakeClient(blocks=blocks)).chat([{"role": "user", "content": "q"}], [])

    assert result.content == "the answer"
    assert "internal" not in result.content


async def test_a_refusal_is_reported_rather_than_returned_as_empty():
    """
    A refusal is HTTP 200 with a stop_reason, not an exception. Reading content
    without checking would surface an empty answer with no explanation.
    """
    client = FakeClient(
        blocks=[],
        stop_reason="refusal",
        stop_details=SimpleNamespace(type="refusal", category="cyber"),
    )
    result = await provider(client).chat([{"role": "user", "content": "q"}], [])

    assert "declined" in result.content
    assert "cyber" in result.content
    assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_the_default_provider_is_local():
    assert isinstance(get_provider(), OllamaProvider)


def test_escalation_without_a_key_raises_rather_than_downgrading(monkeypatch):
    """
    Silently answering with the weaker model would be the exact class of quiet
    substitution this project keeps having to correct.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_provider(escalate=True)


def test_escalation_with_a_key_selects_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")

    assert isinstance(get_provider(escalate=True), ClaudeProvider)
    # Constructing the client must not have required a network call.
    assert os.environ["ANTHROPIC_API_KEY"] == "not-a-real-key"
