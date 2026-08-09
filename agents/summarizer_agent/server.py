"""
Summarizer Agent - MCP Server

Same pattern as research_agent/server.py (see that file for the fully
annotated version). This one exposes a summarize tool.

TODO(week 1, day 3-4): wire summarize() to an actual LLM call (Ollama
locally, or the Claude API for harder cases - see project notes on the
sync-vs-worker-queue decision: this agent is fast, so it stays synchronous).
"""

import time
from mcp.server.fastmcp import FastMCP
from prometheus_client import Counter, Histogram, start_http_server

TOOL_CALLS = Counter("tool_calls_total", "Total tool calls", ["tool_name", "status"])
TOOL_LATENCY = Histogram("tool_call_duration_seconds", "Tool call latency", ["tool_name"])
METRICS_PORT = 9101  # each agent gets its own port

mcp = FastMCP("summarizer-agent")


class SummarizerService:
    def run(self, text: str) -> str:
        # STUB: replace with an Ollama or Claude API call.
        return f"[stub summary] {text[:80]}..."


summarizer_service = SummarizerService()


@mcp.tool()
def summarize(text: str) -> str:
    """Summarize the given text into a concise, well-structured summary."""
    start = time.time()
    try:
        result = summarizer_service.run(text)
        TOOL_CALLS.labels(tool_name="summarize", status="success").inc()
        return result
    except Exception:
        TOOL_CALLS.labels(tool_name="summarize", status="error").inc()
        raise
    finally:
        TOOL_LATENCY.labels(tool_name="summarize").observe(time.time() - start)


if __name__ == "__main__":
    start_http_server(METRICS_PORT)
    mcp.run(transport="sse")
