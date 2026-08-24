"""
LLM provider - the "reason" half of the reason -> act loop.

The important idea in this file is that TOOL CALLING IS STRUCTURED INTENT, NOT
EXECUTION. When the model decides a tool is needed it does not run anything and
cannot; it returns data that says "call search_web with these arguments". The
orchestrator reads that, decides whether to honour it, executes it, and feeds
the result back. That split is the entire reason an agent loop exists - if the
model could act directly there would be no loop to write.

This module exists as its own file (rather than calling Ollama inline from the
graph) so that the documented Claude API fallback can be added later as a
second provider implementing the same three-line interface, without the graph
changing at all. Local Ollama for most steps, Claude for harder reasoning, is
the cost strategy in docs/architecture.md.
"""

import uuid
from dataclasses import dataclass, field
from typing import Protocol

import ollama

from orchestrator.config import (
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_THINK,
    ollama_options,
)


# ---------------------------------------------------------------------------
# Provider-neutral shapes
# ---------------------------------------------------------------------------
# The graph only ever sees these, never an Ollama or Anthropic type. That is
# what makes swapping providers a one-file change.


@dataclass
class ToolCall:
    """One tool the model wants run. `id` correlates the call to its result."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """
    One turn from the model: prose, tool calls, or both.

    An empty `tool_calls` is how the loop knows the model is finished - it has
    stopped asking for tools and produced a final answer.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


# Qwen3's chat template appends " /think" or " /no_think" to the LAST USER
# MESSAGE when thinking is explicitly set - which this project does, by
# default, for the timing reasons in config.py. The token is a control
# instruction for the template, but the model sees it as ordinary text at the
# end of the user's words, and copies it into tool arguments when it quotes
# the request back.
#
# Observed, not theorised: asked to review
#     def divide(a, b): return a / b
# the model called analyze_code with
#     {"code": "def divide(a, b): return a / b /no_think"}
# and the analyser duly reported an undefined name 'no_think'. It went
# unnoticed for as long as that tool was a stub that ignored its input.
#
# Stripped here because this module is the boundary: everything above it deals
# in provider-neutral shapes, and a template artifact from one vendor's model
# has no business crossing it. Only a TRAILING token is removed, so code or
# prose that legitimately contains the word is untouched.
_THINK_TOKENS = ("/no_think", "/think")


def _strip_think_token(value):
    """Remove a trailing template control token from a tool argument."""
    if not isinstance(value, str):
        return value
    cleaned = value.rstrip()
    for token in _THINK_TOKENS:
        if cleaned.endswith(token):
            return cleaned[: -len(token)].rstrip()
    return value


class LLMProvider(Protocol):
    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaProvider:
    """
    Local inference via Ollama - free, which is the whole point given the
    no-cloud-budget constraint.

    Caveat worth knowing up front: tool-calling quality varies a lot between
    local models, and a model without tool support will simply answer in prose
    and never emit a tool call. That looks exactly like a broken graph but
    isn't. If tools are never called, check the model before the loop.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        think=OLLAMA_THINK,
    ):
        self.model = model
        self._think = think
        self._client = ollama.AsyncClient(host=host)

    @staticmethod
    def _to_ollama_messages(messages: list[dict]) -> list[dict]:
        """
        Translate the graph's provider-neutral history into Ollama's wire format.

        The graph stores messages in its own shape (see orchestrator/state.py):
        a tool call is {"id", "name", "arguments"}, and a tool result carries
        the name of the tool that produced it. Ollama nests tool calls under a
        "function" key and names the producing tool with "tool_name". Doing the
        conversion here - rather than storing Ollama's shape in state - is what
        keeps a future Claude provider a drop-in.
        """
        converted = []

        for message in messages:
            role = message["role"]

            if role == "assistant" and message.get("tool_calls"):
                converted.append(
                    {
                        "role": "assistant",
                        "content": message.get("content", ""),
                        "tool_calls": [
                            {
                                "function": {
                                    "name": call["name"],
                                    "arguments": call["arguments"],
                                }
                            }
                            for call in message["tool_calls"]
                        ],
                    }
                )
            elif role == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "content": message["content"],
                        "tool_name": message.get("name", ""),
                    }
                )
            else:
                converted.append({"role": role, "content": message.get("content", "")})

        return converted

    @staticmethod
    def _to_ollama_tools(tools: list[dict]) -> list[dict]:
        """
        Translate MCP tool schemas into the tool format Ollama expects.

        This is the concrete form of "tool schemas are the bridge": the JSON
        Schema that came back from an agent's list_tools() drops straight into
        the model's `parameters` field. Nothing is re-described by hand, so
        adding a tool to an agent automatically makes it visible to the model.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        response = await self._client.chat(
            model=self.model,
            messages=self._to_ollama_messages(messages),
            tools=self._to_ollama_tools(tools),
            think=self._think,
            # Bounded context and prompt unloading, so two models don't sit in
            # 4GB of VRAM at once - see orchestrator/config.py.
            options=ollama_options(),
            keep_alive=OLLAMA_KEEP_ALIVE,
        )

        message = response.message
        tool_calls = []

        for call in message.tool_calls or []:
            # ollama types Function.arguments as Mapping[str, Any] and validates
            # it on the way in, so a model that emits arguments as a JSON string
            # fails inside the ollama client before reaching us. Nothing to
            # normalise here - dict() is just to detach from ollama's mapping.
            arguments = call.function.arguments

            tool_calls.append(
                ToolCall(
                    # Ollama doesn't assign call ids the way the Anthropic and
                    # OpenAI APIs do, so synthesise one. The loop needs some
                    # handle to match a result back to its call.
                    id=uuid.uuid4().hex[:8],
                    name=call.function.name,
                    arguments={k: _strip_think_token(v) for k, v in arguments.items()},
                )
            )

        return LLMResponse(content=message.content or "", tool_calls=tool_calls)


def get_provider() -> LLMProvider:
    """
    Single place to choose the provider.

    When the Claude fallback lands, the escalation rule goes here - everything
    upstream keeps talking to the LLMProvider interface.
    """
    return OllamaProvider()
