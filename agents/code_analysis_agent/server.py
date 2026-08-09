"""
Code Analysis Agent - MCP Server

Same pattern as research_agent/server.py. This agent is the one flagged
in project notes as the candidate for the async worker-queue pattern
later (Redis/RabbitMQ-backed), since code analysis can genuinely be slow.

MVP: stays synchronous like the other two agents. The worker-queue
version is a documented stretch goal, not required for the core build -
see docs/architecture.md.

TODO(week 1, day 3-4): wire analyze_code() to real static analysis +
LLM reasoning.
TODO(stretch goal, week 2+): convert this agent specifically to the
async worker pattern and document the before/after in the README.
"""

import time
from mcp.server.fastmcp import FastMCP
from prometheus_client import Counter, Histogram, start_http_server

TOOL_CALLS = Counter("tool_calls_total", "Total tool calls", ["tool_name", "status"])
TOOL_LATENCY = Histogram("tool_call_duration_seconds", "Tool call latency", ["tool_name"])
METRICS_PORT = 9102

mcp = FastMCP("code-analysis-agent")


class CodeAnalysisService:
    def run(self, code: str) -> str:
        # STUB: replace with static analysis + LLM review.
        return f"[stub analysis] Reviewed {len(code)} chars of code, no issues found (stub)."


code_analysis_service = CodeAnalysisService()


@mcp.tool()
def analyze_code(code: str) -> str:
    """Analyze the given code snippet for issues, style problems, or
    potential bugs."""
    start = time.time()
    try:
        result = code_analysis_service.run(code)
        TOOL_CALLS.labels(tool_name="analyze_code", status="success").inc()
        return result
    except Exception:
        TOOL_CALLS.labels(tool_name="analyze_code", status="error").inc()
        raise
    finally:
        TOOL_LATENCY.labels(tool_name="analyze_code").observe(time.time() - start)


if __name__ == "__main__":
    start_http_server(METRICS_PORT)
    mcp.run(transport="sse")
