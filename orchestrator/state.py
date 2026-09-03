"""
Shared state that flows through every node in the LangGraph graph.

This is the object we discussed: it accumulates conversation history and
tool results as the reason -> act -> reason loop runs, so each step can
see everything that happened before it.

A LangGraph node doesn't mutate this dict - it returns a *partial* one, and
LangGraph merges the result in. How each key merges is decided by its reducer,
which is what the Annotated[...] below declares.
"""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    # operator.add on a list means "append", so each node returning
    # {"messages": [msg]} adds to the history instead of replacing it.
    #
    # LangGraph also ships add_messages, which does the same job but coerces
    # entries into LangChain message objects (HumanMessage, AIMessage, ...).
    # That's the right reducer when the graph talks to an LLM through a
    # LangChain chat model. Here the orchestrator calls Ollama directly (see
    # orchestrator/llm.py), so messages are provider-neutral dicts and the
    # coercion would just get in the way of round-tripping tool calls.
    messages: Annotated[list, operator.add]

    # Tracks how many reason/act cycles we've done, so we can enforce a
    # max-iteration guardrail (see docs/architecture.md - agents that loop
    # forever are a real failure mode, not a hypothetical one).
    #
    # No reducer, so the value a node returns simply replaces the old one.
    iterations: int

    # How many times this run has been nudged after the model described a tool
    # call instead of making one. Bounded to one, and tracked in state rather
    # than in a closure because the nudge node has to know whether it already
    # fired - two nudges in a row would be a loop, not a recovery.
    #
    # No reducer, so the value a node returns replaces the old one.
    nudges: int

    # How many times this run has been sent back after answering from tools that
    # all reported having nothing. Separate from nudges because it is a different
    # failure with a different remedy: a nudge means no tool ran, this means they
    # ran and came back empty. Bounded to one for the same reason.
    #
    # No reducer, so the value a node returns replaces the old one.
    regrounds: int
