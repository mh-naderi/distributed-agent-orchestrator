"""
Research Agent - MCP Server

This is a complete, working example of an MCP server. It exposes one tool,
search_web, that any MCP client (our LangGraph orchestrator) can discover
and call.

Pattern used throughout this project:
  - The @mcp.tool() function is a THIN ADAPTER. It does not contain business
    logic. It validates input, calls into a separate service module, and
    returns the result. This keeps protocol-handling code separate from
    the actual work, so you can test/replace the logic without touching
    the MCP wiring.
  - Every tool call is instrumented with Prometheus metrics (latency +
    success/failure counts), because "did it work and how long did it take"
    is the minimum observability you need for a distributed system.
"""

import time
from mcp.server.fastmcp import FastMCP
from prometheus_client import Counter, Histogram, start_http_server

# ---------------------------------------------------------------------------
# Metrics setup
# ---------------------------------------------------------------------------
# Counter: a number that only goes up (total calls, total errors)
# Histogram: tracks a distribution of values (so you can compute p50/p95/p99
#            latency later in Grafana, not just an average)
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

# Prometheus scrapes this port for metrics (separate from the MCP protocol
# port). Exposed on 9100 by convention for this project - each agent gets
# its own metrics port so Prometheus can tell them apart.
METRICS_PORT = 9100

mcp = FastMCP("research-agent")


# ---------------------------------------------------------------------------
# Business logic - kept separate from the MCP tool wrapper on purpose.
# TODO(week 1, day 3-4): replace this stub with a real web search call
# (e.g. a search API). Keeping it as a stub for now so the orchestrator
# and the graph can be built and tested end-to-end before wiring up a
# real external dependency.
# ---------------------------------------------------------------------------
class SearchService:
    def run(self, query: str) -> str:
        # STUB: replace with a real search API call.
        return f"[stub result] Top findings for '{query}': ..."


search_service = SearchService()


# ---------------------------------------------------------------------------
# MCP tool definition - the thin adapter
# ---------------------------------------------------------------------------
@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for information relevant to the query and return
    a summary of findings."""
    start = time.time()
    try:
        result = search_service.run(query)
        TOOL_CALLS.labels(tool_name="search_web", status="success").inc()
        return result
    except Exception:
        TOOL_CALLS.labels(tool_name="search_web", status="error").inc()
        raise
    finally:
        TOOL_LATENCY.labels(tool_name="search_web").observe(time.time() - start)


if __name__ == "__main__":
    # Start the metrics endpoint (Prometheus will scrape http://<pod>:9100/metrics)
    start_http_server(METRICS_PORT)
    # Start the MCP server itself. Using SSE transport (HTTP-based) rather
    # than stdio, because this needs to be reachable over the Kubernetes
    # cluster network, not spawned as a local subprocess.
    mcp.run(transport="sse")
