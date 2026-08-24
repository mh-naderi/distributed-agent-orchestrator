"""
Code Analysis Agent - MCP Server

Same pattern as research_agent/server.py. This agent is the one flagged
in project notes as the candidate for the async worker-queue pattern
later (Redis/RabbitMQ-backed), since code analysis can genuinely be slow.

MVP: stays synchronous like the other two agents. The worker-queue
version is a documented stretch goal, not required for the core build -
see docs/architecture.md.

analyze_code runs real static analysis - see analysis.py, which also records
why it deliberately reports less than it could. An LLM review pass on top of
the mechanical findings remains a possible extension, but it would make this
agent depend on Ollama the way the retrieval agent does, and static analysis
is the fast half.
TODO(stretch goal, week 2+): convert this agent specifically to the
async worker pattern and document the before/after in the README.
"""

import os
import time
from analysis import report
from mcp.server.fastmcp import FastMCP
from prometheus_client import Counter, Histogram, start_http_server

TOOL_CALLS = Counter("tool_calls_total", "Total tool calls", ["tool_name", "status"])
TOOL_LATENCY = Histogram("tool_call_duration_seconds", "Tool call latency", ["tool_name"])
METRICS_PORT = 9102
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

mcp = FastMCP("code-analysis-agent", host="0.0.0.0", port=MCP_PORT)


class CodeAnalysisService:
    """
    Thin seam over analysis.report, kept so server.py matches the shape of the
    other agents: protocol wiring here, logic in its own module.
    """

    def run(self, code: str) -> str:
        return report(code)

code_analysis_service = CodeAnalysisService()


@mcp.tool()
def analyze_code(code: str) -> str:
    """Run static analysis on a Python snippet.

    Reports syntax errors, undefined or unused names, mutable default
    arguments, bare or silenced excepts, == comparisons against None/True/
    False, division by an unguarded parameter, and unreachable code.

    Does NOT execute the code and does not check logic, performance or
    security. A clean result means these checks found nothing; it is not
    evidence that the code is correct.
    """
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
    mcp.run(transport="streamable-http")
