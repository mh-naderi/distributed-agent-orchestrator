"""
Tests for the provider translation layer.

These are the seams where a wrong field name fails at runtime against a live
model rather than at import time, so they're worth pinning down. The shapes
are checked against Ollama's own pydantic models, which means the tests fail
if a future ollama release renames a field - rather than the first real run
failing instead.
"""

import ollama
import pytest
from pydantic import ValidationError

from orchestrator.llm import LLMResponse, OllamaProvider, ToolCall

TOOLS = [
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

HISTORY = [
    {"role": "system", "content": "You are an orchestrator."},
    {"role": "user", "content": "Research MCP."},
    {
        "role": "assistant",
        "content": "Searching.",
        "tool_calls": [{"id": "c1", "name": "search_web", "arguments": {"query": "MCP"}}],
    },
    {"role": "tool", "name": "search_web", "tool_call_id": "c1", "content": "results here"},
]


def test_mcp_schema_passes_straight_through_as_tool_parameters():
    """'Tool schemas are the bridge' - the MCP schema is reused verbatim."""
    converted = OllamaProvider._to_ollama_tools(TOOLS)

    assert converted == [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web.",
                "parameters": TOOLS[0]["input_schema"],
            },
        }
    ]


def test_assistant_tool_calls_are_nested_under_function():
    converted = OllamaProvider._to_ollama_messages(HISTORY)
    assistant = converted[2]

    assert assistant["tool_calls"] == [
        {"function": {"name": "search_web", "arguments": {"query": "MCP"}}}
    ]
    # Our internal call id is not part of Ollama's wire format.
    assert "id" not in assistant


def test_tool_result_carries_tool_name():
    converted = OllamaProvider._to_ollama_messages(HISTORY)
    tool_message = converted[3]

    assert tool_message == {
        "role": "tool",
        "content": "results here",
        "tool_name": "search_web",
    }


def test_translated_messages_validate_against_ollamas_own_model():
    """Catches a field rename in a future ollama release at test time."""
    for message in OllamaProvider._to_ollama_messages(HISTORY):
        ollama._types.Message(**message)


def test_ollama_rejects_non_dict_tool_arguments():
    """
    Documents why llm.py does no argument normalisation.

    A model that emits tool arguments as a JSON string fails inside ollama's
    own validation - Function.arguments is Mapping[str, Any] - so it can never
    reach our code as a string. Any defensive parsing on our side would be
    unreachable. If this test ever starts passing, ollama loosened the type and
    normalisation becomes worth adding back.
    """
    with pytest.raises(ValidationError):
        ollama._types.Message.ToolCall.Function(name="search_web", arguments='{"query": "MCP"}')


async def test_tool_calls_are_converted_to_provider_neutral_objects(monkeypatch):
    """The graph should only ever see ToolCall, never an ollama type."""
    arguments = {"query": "MCP"}

    class FakeResponse:
        message = ollama._types.Message(
            role="assistant",
            content="",
            tool_calls=[
                ollama._types.Message.ToolCall(
                    function=ollama._types.Message.ToolCall.Function(
                        name="search_web", arguments=arguments
                    )
                )
            ],
        )

    provider = OllamaProvider()

    async def fake_chat(**kwargs):
        return FakeResponse()

    monkeypatch.setattr(provider._client, "chat", fake_chat)

    response = await provider.chat(HISTORY, TOOLS)

    assert isinstance(response, LLMResponse)
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.name == "search_web"
    assert call.arguments == arguments
    assert call.id, "every tool call needs an id to correlate its result"


async def test_plain_answer_produces_no_tool_calls(monkeypatch):
    """An empty tool_calls list is how the loop knows the model is finished."""

    class FakeResponse:
        message = ollama._types.Message(role="assistant", content="Final answer.")

    provider = OllamaProvider()

    async def fake_chat(**kwargs):
        return FakeResponse()

    monkeypatch.setattr(provider._client, "chat", fake_chat)

    response = await provider.chat(HISTORY, TOOLS)

    assert response.content == "Final answer."
    assert response.tool_calls == []
