"""
Conversation history, and the trimming that keeps it inside the context window.

WHY THIS EXISTS. Every run started from an empty history, so a follow-up
question could not see what the previous one found. "Now summarise that" had
nothing to refer to.

WHY IT IS MOSTLY ABOUT TRIMMING. Holding history is three lines; holding it
against a 4096-token context window is the actual problem. One measured
`search_web` result was 2111 characters - roughly 500 tokens - so two searches
and the system prompt very nearly fill the window on their own. Overflow is not
an error: Ollama silently truncates from the front, which would quietly discard
the system prompt and leave the model with no instructions and no explanation.
That is the exact class of failure this project keeps having to correct, so the
budget is enforced here rather than discovered downstream.

THE RULE. Tool results are by far the largest items in a history and the
assistant's own answer already says what they contained, so they are what gets
sacrificed first:

  1. The system prompt is never touched.
  2. The CURRENT turn is never touched - a run must always see the results it
     just received.
  3. Over budget, older tool results are TRUNCATED IN PLACE, oldest first.
  4. Still over budget, whole oldest turns are dropped.
  5. As a last resort the largest remaining tool result is cut to a prefix.
     The current turn is protected from 3 and 4, but nothing can be exempt
     from the budget itself - one oversized result would otherwise blow the
     window alone, which is the failure this function exists to prevent.

WHY TRUNCATE RATHER THAN DELETE. A tool result is the other half of an
assistant tool call, and the two are paired by id. Removing the result while
leaving the call behind produces a malformed conversation - Anthropic rejects an
unmatched tool_use outright, and Ollama accepts it and behaves strangely, which
is worse. Replacing the CONTENT with a short marker reclaims essentially all of
the tokens while leaving every pair intact.
"""

import logging
import time
from dataclasses import dataclass, field

from orchestrator.config import (
    MAX_SESSIONS,
    SESSION_TTL,
    history_budget_chars,
)

logger = logging.getLogger(__name__)

DROPPED_MARKER = "[earlier tool output omitted to stay within the context window]"
TRUNCATED_SUFFIX = " [...truncated to fit the context window]"


def estimate_chars(messages: list[dict]) -> int:
    """
    Size a history in characters.

    Characters rather than tokens on purpose. Counting real tokens would mean
    either shipping a tokenizer for a model that can be swapped by an
    environment variable, or paying a round trip per trim. Characters are a
    stable proxy, and the budget derived from them is set well below the true
    limit precisely because this is an estimate - see history_budget_chars.
    """
    total = 0
    for message in messages:
        total += len(str(message.get("content", "")))
        for call in message.get("tool_calls", []):
            total += len(str(call.get("arguments", "")))
    return total


def _turns(messages: list[dict]) -> list[list[dict]]:
    """
    Group a history into turns, each beginning with a user message.

    Turns are the unit of dropping because an assistant tool call and its
    results have to leave together or not at all.
    """
    turns: list[list[dict]] = []
    for message in messages:
        if message["role"] == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def trim(messages: list[dict], budget: int | None = None) -> list[dict]:
    """
    Bring a history within budget, preserving as much meaning per character as
    possible. Returns a new list; the input is not mutated.
    """
    budget = budget if budget is not None else history_budget_chars()

    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]

    if estimate_chars(system + rest) <= budget:
        return list(messages)

    turns = _turns(rest)

    # Pass 1: blank out tool output in every turn except the current one.
    for turn in turns[:-1]:
        for index, message in enumerate(turn):
            if message["role"] == "tool" and message.get("content") != DROPPED_MARKER:
                turn[index] = {**message, "content": DROPPED_MARKER}
        if estimate_chars(system + [m for t in turns for m in t]) <= budget:
            break

    # Pass 2: still too big, so drop whole oldest turns. The last turn always
    # survives - a run that cannot see its own question is useless.
    while len(turns) > 1 and estimate_chars(
        system + [m for t in turns for m in t]
    ) > budget:
        dropped = turns.pop(0)
        logger.info("history over budget: dropped a turn of %d message(s)", len(dropped))

    surviving = system + [m for t in turns for m in t]

    # Pass 3: the current turn is protected from passes 1 and 2, but it cannot
    # be exempt from the budget itself - a single large tool result would
    # otherwise blow the window on its own, and the consequence of going over
    # is exactly what this function exists to prevent. So as a last resort the
    # biggest remaining tool result is cut to fit, keeping a PREFIX rather than
    # replacing it wholesale: half a result the model can read beats a marker
    # saying one existed.
    while estimate_chars(surviving) > budget:
        tools = [
            (index, message)
            for index, message in enumerate(surviving)
            if message["role"] == "tool" and len(str(message.get("content", ""))) > 0
        ]
        if not tools:
            # Nothing left that may be cut. A history of nothing but a system
            # prompt and a question can legitimately exceed a small budget,
            # and mangling either of those would be worse than being over.
            logger.warning(
                "history still over budget (%d > %d) with nothing safe left to trim",
                estimate_chars(surviving),
                budget,
            )
            break

        index, largest = max(tools, key=lambda pair: len(str(pair[1]["content"])))
        over = estimate_chars(surviving) - budget
        content = str(largest["content"])
        keep = max(0, len(content) - over - len(TRUNCATED_SUFFIX))
        surviving[index] = {
            **largest,
            "content": (content[:keep] + TRUNCATED_SUFFIX) if keep else DROPPED_MARKER,
        }

    return surviving


@dataclass
class Session:
    """One conversation. `messages` is the graph's neutral history format."""

    messages: list[dict] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()


class SessionStore:
    """
    In-memory conversations, bounded by count and by age.

    Bounded on purpose. An unbounded dict keyed by a value the client chooses is
    a memory leak with a public entry point: anyone can mint session ids faster
    than they expire. Eviction is oldest-first once MAX_SESSIONS is reached.

    In memory, so restarting the orchestrator forgets every conversation. That
    is a real limitation rather than an oversight - persisting them would mean
    giving the orchestrator the durable state that, by design, only the
    retrieval agent has.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl: float = SESSION_TTL):
        self._sessions: dict[str, Session] = {}
        self._max = max_sessions
        self._ttl = ttl

    def __len__(self) -> int:
        return len(self._sessions)

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, session in self._sessions.items()
            if now - session.last_used > self._ttl
        ]
        for key in expired:
            del self._sessions[key]

        while len(self._sessions) > self._max:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].last_used)
            del self._sessions[oldest]
            logger.info("session store full; evicted the least recently used")

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = Session()
            self._sessions[session_id] = session
            self._evict()
        session.touch()
        return session

    def save(self, session_id: str, messages: list[dict]) -> None:
        session = self.get(session_id)
        session.messages = trim(messages)

    def clear(self) -> None:
        self._sessions.clear()
