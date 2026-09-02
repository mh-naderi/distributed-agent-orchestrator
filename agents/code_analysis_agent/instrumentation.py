"""
Tool metrics, recorded at the MCP boundary rather than inside each tool.

WHY NOT IN THE TOOL BODIES. The obvious place to count a tool call is the tool
itself, and that is where these counters used to live. It has a blind spot that
took a measurement to see: FastMCP validates arguments against the schema
derived from the function signature BEFORE calling the function, so a call like
retrieve(k="abc") is rejected upstream of the tool body. The try/except that was
supposed to record the failure is inside the function that never runs.

The effect was not that such calls were miscategorised - they were invisible.
Neither status="success" nor status="error" moved, so sum(tool_calls_total)
undercounted real traffic and the error panel could not show this class of
failure at all. Verified before the change: a schema-invalid call raised
ToolError to the client and left every sample untouched.

_setup_handlers registers FastMCP.call_tool as the protocol handler for tool
requests, and validation happens below it, inside tool.run(). Overriding that
one method therefore sees every call - valid, invalid, and unknown - and gets
the tool name from the request instead of a hardcoded label.

Two details that are not incidental:

- Exactly one increment per call. The per-tool version could record success and
  then fail in code that ran afterwards, counting a single call twice. Wrapping
  the whole call makes that impossible rather than merely unlikely.

- The tool name is client-supplied, and it becomes a Prometheus label. An
  unregistered name is recorded as "unknown" rather than passed through: a
  label whose values come from the caller is an unbounded-cardinality hole, and
  a confused client looping over made-up tool names should not be able to grow
  the metric store.

This file is duplicated verbatim into each agent directory. Each image is built
from its own agent directory as context, so a shared repo-root module would not
be copied in without widening every build context - the same reason each agent
carries its own requirements.txt. test_instrumentation.py asserts the copies
are byte-identical, so the duplication cannot drift silently.
"""

import time
from typing import Any

from mcp.server.fastmcp import FastMCP
from prometheus_client import Counter, Histogram

# Counter: only ever goes up (total calls, total errors)
# Histogram: a distribution, so p50/p95/p99 latency is available in Grafana
#            rather than only an average.
TOOL_CALLS = Counter(
    "tool_calls_total",
    "Total number of tool calls",
    ["tool_name", "status"],  # labels let you slice metrics in Grafana
)
TOOL_LATENCY = Histogram(
    "tool_call_duration_seconds",
    "Tool call latency in seconds",
    ["tool_name"],
)


class InstrumentedMCP(FastMCP):
    """A FastMCP that counts every tool call, including the ones it rejects."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        # Only a registered name is used as a label value; see the note above.
        # _tool_manager is internal to FastMCP, but so is this override - the
        # alternative is awaiting list_tools() on every single call.
        label = name if self._tool_manager.get_tool(name) else "unknown"

        start = time.time()
        try:
            result = await super().call_tool(name, arguments)
        except Exception:
            TOOL_CALLS.labels(tool_name=label, status="error").inc()
            raise
        else:
            TOOL_CALLS.labels(tool_name=label, status="success").inc()
            return result
        finally:
            TOOL_LATENCY.labels(tool_name=label).observe(time.time() - start)
