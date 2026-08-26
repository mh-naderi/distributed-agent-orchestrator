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

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import anthropic
import ollama

from orchestrator.config import (
    CLAUDE_FALLBACKS,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_THINK,
    ollama_options,
)

logger = logging.getLogger(__name__)


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

    # Provider-opaque original blocks, carried so a provider that needs to
    # replay its own output verbatim can. Claude needs this: with thinking and
    # tool use combined, thinking blocks must be echoed back unchanged, and
    # rebuilding them from `content` would lose them. Ollama leaves it None and
    # nothing else looks at it.
    raw_content: list | None = None


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


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class ClaudeProvider:
    """
    The escalation path. docs/architecture.md calls this "the real answer for
    anything sustained" - a 4GB laptop GPU is under-specified for this workload,
    and the small local model's measured cost is tool-selection accuracy.

    It implements the same three-line LLMProvider interface as OllamaProvider,
    which is the whole reason this module was split out in the first place. The
    graph does not know which one it is talking to.

    THREE TRANSLATION PROBLEMS, none of which Ollama has:

    1. The system prompt is a top-level parameter, not a message. The graph
       stores it as messages[0] with role "system", so it is lifted out here.
    2. Tool results are USER messages containing tool_result blocks, and
       consecutive results must be batched into ONE message. Splitting them
       across several messages teaches the model to stop making parallel calls.
    3. Thinking blocks have to be echoed back unchanged on the next turn when
       thinking and tool use are combined. The graph's neutral message format
       has nowhere to put them, so the raw content blocks ride along under
       _claude_content - a key this provider writes and reads and every other
       provider ignores. Reconstructing the blocks from the neutral shape would
       lose the thinking, which is exactly what must not happen.
    """

    def __init__(
        self,
        model: str = CLAUDE_MODEL,
        max_tokens: int = CLAUDE_MAX_TOKENS,
        client=None,
    ):
        self.model = model
        self._max_tokens = max_tokens
        # Constructed lazily so importing this module never requires a key.
        self._client = client or anthropic.AsyncAnthropic()

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        """Lift the system prompt out of the message list."""
        system = "\n\n".join(
            m.get("content", "") for m in messages if m["role"] == "system"
        )
        return system, [m for m in messages if m["role"] != "system"]

    @classmethod
    def _to_claude_messages(cls, messages: list[dict]) -> list[dict]:
        """
        Translate the graph's history into Anthropic's wire format.

        Walks with an index rather than a for loop because runs of tool results
        collapse into a single user message - see problem 2 above.
        """
        converted: list[dict] = []
        index = 0

        while index < len(messages):
            message = messages[index]
            role = message["role"]

            if role == "tool":
                # Consume every consecutive tool result into one user message.
                results = []
                while index < len(messages) and messages[index]["role"] == "tool":
                    result = messages[index]
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": result.get("tool_call_id", ""),
                            "content": result.get("content", ""),
                        }
                    )
                    index += 1
                converted.append({"role": "user", "content": results})
                continue

            if role == "assistant":
                # Replay the original blocks when this turn came from Claude, so
                # thinking survives. Otherwise rebuild from the neutral shape -
                # which is the case when a run started on Ollama and escalated.
                raw = message.get("_claude_content")
                if raw is not None:
                    converted.append({"role": "assistant", "content": raw})
                else:
                    blocks: list[dict] = []
                    if message.get("content"):
                        blocks.append({"type": "text", "text": message["content"]})
                    for call in message.get("tool_calls", []):
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": call["id"],
                                "name": call["name"],
                                "input": call["arguments"],
                            }
                        )
                    if blocks:
                        converted.append({"role": "assistant", "content": blocks})
                index += 1
                continue

            converted.append({"role": role, "content": message.get("content", "")})
            index += 1

        return converted

    @staticmethod
    def _to_claude_tools(tools: list[dict]) -> list[dict]:
        """
        MCP tool schemas need almost no translation.

        MCP and Anthropic both describe a tool as a name, a description and a
        JSON Schema for its arguments, and the key is even spelled input_schema
        in both. That is not a coincidence - it is the same idea in both
        protocols, and it is why discovery output can be handed to the model
        without anything re-describing it by hand.
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
            }
            for tool in tools
        ]

    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        system, rest = self._split_system(messages)

        request = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": self._to_claude_messages(rest),
            "tools": self._to_claude_tools(tools),
            # Adaptive rather than a fixed token budget: this provider exists
            # precisely for the reasoning the local model cannot do, so letting
            # the model decide how much to think is the point of using it.
            "thinking": {"type": "adaptive"},
        }

        if CLAUDE_FALLBACKS:
            # Safety classifiers can decline a request; this routes such a turn
            # to another model by refusal category rather than failing the run.
            request["betas"] = ["server-side-fallback-2026-07-01"]
            request["fallbacks"] = "default"
            response = await self._client.beta.messages.create(**request)
        else:
            response = await self._client.messages.create(**request)

        # Checked before reading content: a refusal returns HTTP 200 with a
        # stop_reason rather than raising, so treating it as a normal response
        # would surface an empty answer with no explanation.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return LLMResponse(
                content=(
                    "The model declined this request"
                    + (f" ({category})" if category else "")
                    + "."
                )
            )

        content_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    # Claude assigns real tool-call ids, so unlike the Ollama
                    # provider there is nothing to synthesise.
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return LLMResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            raw_content=[b.model_dump() for b in response.content],
        )


def get_provider(escalate: bool = False) -> LLMProvider:
    """
    Single place to choose the provider.

    Escalation is MANUAL, per request. The alternatives were an iteration-based
    rule (escalate after N cycles without an answer) or a failure-based one, and
    both were left for later on purpose: a three-case evaluation suite cannot
    tell whether such a heuristic helps, and an automatic rule that spends money
    on a guess is worse than a switch somebody chose to flip. The seam is here
    when there is evidence to build one on.

    Missing credentials raise rather than falling back to the local model. A
    request that asked for the better model and quietly got the weaker one is
    the exact class of silent substitution this project keeps having to correct.
    """
    if not escalate:
        return OllamaProvider()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "escalation requested but ANTHROPIC_API_KEY is not set - refusing to "
            "silently answer with the local model instead"
        )

    logger.info("escalating this run to %s", CLAUDE_MODEL)
    return ClaudeProvider()
