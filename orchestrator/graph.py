"""
The orchestrator graph: the reason -> act -> reason loop we discussed.

This file wires together:
  - a "reason" node: calls the LLM, which decides whether to call a tool
    or return a final answer
  - an "act" node: executes whatever tool the LLM asked for, via MCP
  - a conditional edge that routes between them based on the LLM's decision

TODO(week 1, day 5-6): fill in call_llm() and call_mcp_tool() with real
MCP client connections to the three agent servers. This skeleton defines
the shape of the graph so it can be tested with stub responses first.
"""

from langgraph.graph import StateGraph, END
from orchestrator.state import AgentState

MAX_ITERATIONS = 10  # guardrail against infinite loops - see project notes


def call_llm(state: AgentState) -> AgentState:
    """
    The 'reason' node. Sends current state (task + history so far) to the
    LLM along with the list of available tools (discovered from the MCP
    servers at startup), and gets back either a tool call request or a
    final answer.

    TODO: replace with a real LLM call (Ollama for most steps, Claude API
    fallback for harder reasoning - see architecture notes on cost).
    """
    state["iterations"] += 1
    # STUB: real implementation calls the LLM with state["messages"] +
    # tool schemas, and appends its response (either a tool_call or a
    # final answer) to state["messages"].
    return state


def call_mcp_tool(state: AgentState) -> AgentState:
    """
    The 'act' node. Reads the tool call the LLM just requested, connects
    to the corresponding MCP server, executes it, and appends the result
    back into state so the next 'reason' step can see it.

    TODO: replace with real MCP client calls to research_agent,
    summarizer_agent, and code_analysis_agent.
    """
    # STUB: real implementation inspects the last message for a tool_call,
    # dispatches to the right MCP server, and appends the tool result.
    return state


def should_continue(state: AgentState) -> str:
    """
    Conditional routing function. Reads the LLM's last decision and the
    iteration guardrail to decide whether to act again or end.
    """
    if state["iterations"] >= MAX_ITERATIONS:
        return "end"
    # TODO: check the LLM's last message - if it requested a tool call,
    # return "continue"; if it produced a final answer, return "end".
    return "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("reason", call_llm)
    graph.add_node("act", call_mcp_tool)

    graph.set_entry_point("reason")
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {"continue": "act", "end": END},
    )
    graph.add_edge("act", "reason")

    return graph.compile()
