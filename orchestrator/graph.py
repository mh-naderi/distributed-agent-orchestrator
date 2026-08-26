"""
The orchestrator graph: the reason -> act -> reason loop we discussed.

This file wires together:
  - a "reason" node: calls the LLM, which decides whether to call a tool
    or return a final answer
  - an "act" node: executes whatever tool the LLM asked for, via MCP
  - a conditional edge that routes between them based on the LLM's decision

The shape of the loop is the whole idea. The LLM never executes anything - it
returns structured intent ("call search_web with this query"), the act node
runs it against the right agent, the result is appended to the conversation,
and the LLM reasons again now that it knows more. Repeat until it stops asking
for tools. Everything else in this project is infrastructure around that.

The nodes are async because both halves are I/O: an HTTP call to Ollama and an
MCP call over the network. LangGraph runs async nodes natively - the graph is
driven with ainvoke() instead of invoke().
"""

import logging

from langgraph.graph import END, StateGraph

from orchestrator.llm import LLMProvider
from orchestrator.mcp_client import MCPToolRegistry
from orchestrator.state import AgentState

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10  # guardrail against infinite loops - see project notes

SYSTEM_PROMPT = """You are an orchestrator. You answer by calling tools, not from memory.

RULES:

1. If the request contains code, call analyze_code on that code. Always.
2. If the request needs external facts, call retrieve first. It searches a
   stored index that persists between runs and may already have the answer.
3. Only if retrieve returns nothing useful, call search_web.
4. After search_web, call index_documents with the results, so the next
   question is cheaper. This does not help your current answer - do it anyway.
5. Call one tool at a time and read its result before deciding the next step.
6. Stop calling tools once you can answer, then answer.

Use only what the tools returned. If they returned nothing useful, say so.
Never fill a gap with your own knowledge."""


def build_graph(registry: MCPToolRegistry, provider: LLMProvider):
    """
    Build and compile the graph.

    The registry and provider are passed in rather than constructed here so
    the graph has no opinion about which LLM is behind it or which agents are
    reachable. That's what lets the Claude fallback and any future agent slot
    in without touching this file - and it makes the nodes testable with fakes.
    """

    async def call_llm(state: AgentState) -> dict:
        """
        The 'reason' node. Sends the task and history so far to the LLM along
        with the tools discovered from the MCP servers, and gets back either
        tool call requests or a final answer.

        Returns a *partial* state. LangGraph merges it using the reducers
        declared in state.py: the message list is appended to, iterations is
        replaced.
        """
        response = await provider.chat(state["messages"], registry.tools)

        message: dict = {"role": "assistant", "content": response.content}

        # Carried opaquely for providers that must replay their own output
        # verbatim. Claude needs it: with thinking and tool use combined, its
        # thinking blocks have to be echoed back unchanged on the next turn,
        # and rebuilding them from `content` would silently drop them. Nothing
        # in the graph reads this - it only has to survive the round trip.
        if response.raw_content is not None:
            message["_claude_content"] = response.raw_content
        if response.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in response.tool_calls
            ]
            logger.info(
                "iteration %d: model requested %s",
                state["iterations"] + 1,
                [call.name for call in response.tool_calls],
            )
        else:
            logger.info("iteration %d: model produced a final answer", state["iterations"] + 1)

        return {"messages": [message], "iterations": state["iterations"] + 1}

    async def call_mcp_tool(state: AgentState) -> dict:
        """
        The 'act' node. Reads the tool calls the LLM just requested, executes
        each against whichever agent owns it, and appends the results so the
        next 'reason' step can see them.

        Note there's no routing logic here: the registry already learned which
        agent owns which tool during discovery, so dispatch is a lookup. A
        fourth agent would need no change to this function.
        """
        last_message = state["messages"][-1]
        results = []

        for call in last_message.get("tool_calls", []):
            output = await registry.call(call["name"], call["arguments"])
            results.append(
                {
                    "role": "tool",
                    "name": call["name"],
                    "tool_call_id": call["id"],
                    "content": output,
                }
            )

        return {"messages": results}

    async def note_truncation(state: AgentState) -> dict:
        """
        Terminal node for the guardrail path.

        Without this the loop stopped correctly and said nothing about it. The
        last message on the truncated path is the model's unanswered tool-call
        request, so content was empty: arun() returned '', and the streaming
        API emitted tool_call events followed by done with no answer event at
        all - the UI simply stopped mid-run. The guardrail exists for a real
        failure mode, and its firing was the one thing it never reported.

        A router cannot do this: should_continue returns a string and cannot
        add to state. Hence a node.
        """
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"Stopped after {MAX_ITERATIONS} iterations without reaching "
                        "an answer. This is not a final answer: the loop was still "
                        "requesting tools when it hit the max-iteration guardrail, "
                        "and the last requested tool calls were not executed."
                    ),
                    "truncated": True,
                }
            ]
        }

    def should_continue(state: AgentState) -> str:
        """
        Conditional routing. Reads the LLM's last decision and the iteration
        guardrail to decide whether to act again or end.

        The guardrail is checked first and deliberately: a model that keeps
        requesting tools forever is a real failure mode, and without this the
        loop would happily run until something else broke.
        """
        if state["iterations"] >= MAX_ITERATIONS:
            logger.warning("hit MAX_ITERATIONS (%d), stopping", MAX_ITERATIONS)
            return "truncated"

        if state["messages"][-1].get("tool_calls"):
            return "continue"

        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("reason", call_llm)
    graph.add_node("act", call_mcp_tool)
    graph.add_node("truncate", note_truncation)

    graph.set_entry_point("reason")
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {"continue": "act", "truncated": "truncate", "end": END},
    )
    graph.add_edge("act", "reason")
    graph.add_edge("truncate", END)

    return graph.compile()
