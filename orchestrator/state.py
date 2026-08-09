"""
Shared state that flows through every node in the LangGraph graph.

This is the object we discussed: it accumulates conversation history and
tool results as the reason -> act -> reason loop runs, so each step can
see everything that happened before it.
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # add_messages is a LangGraph helper that appends new messages to the
    # list rather than overwriting it, so history accumulates across the loop.
    messages: Annotated[list, add_messages]
    # Tracks how many reason/act cycles we've done, so we can enforce a
    # max-iteration guardrail (see docs/architecture.md - agents that loop
    # forever are a real failure mode, not a hypothetical one).
    iterations: int
