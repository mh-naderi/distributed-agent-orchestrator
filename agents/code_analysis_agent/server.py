"""
Code Analysis Agent - MCP Server

Same pattern as research_agent/server.py. This agent is the one flagged
in project notes as the candidate for the async worker-queue pattern
later (Redis/RabbitMQ-backed), since code analysis can genuinely be slow.

MVP: stays synchronous like the other two agents. The worker-queue
version is a documented stretch goal, not required for the core build -
see docs/architecture.md.

TODO: wire analyze_code() to real static analysis (ast/pyflakes) + LLM
reasoning. Until then the tool raises NotImplementedError on purpose - see
CodeAnalysisService.run for why a failing tool beats a reassuring one.
TODO(stretch goal, week 2+): convert this agent specifically to the
async worker pattern and document the before/after in the README.
"""

import os
import time
from mcp.server.fastmcp import FastMCP
from prometheus_client import Counter, Histogram, start_http_server

TOOL_CALLS = Counter("tool_calls_total", "Total tool calls", ["tool_name", "status"])
TOOL_LATENCY = Histogram("tool_call_duration_seconds", "Tool call latency", ["tool_name"])
METRICS_PORT = 9102
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

mcp = FastMCP("code-analysis-agent", host="0.0.0.0", port=MCP_PORT)


class CodeAnalysisService:
    def run(self, code: str) -> str:
        """
        Not implemented yet - and it FAILS rather than returning a pleasant
        placeholder, which is the whole point.

        This previously returned
            "[stub analysis] Reviewed N chars of code, no issues found (stub)."
        A well-formed success carrying no information is undetectable
        downstream: the orchestrating model has been told to use only what the
        tools returned, so it faithfully relayed the stub and reported
        "No issues were found in the provided code." for
            def divide(a, b): return a / b
        - an unhandled ZeroDivisionError. The model behaved correctly; the tool
        lied to it.

        This is the same failure that got the research agent's stub deleted -
        see 'Grounding, and why the stubs had to go' in docs/architecture.md.
        Raising makes the gap visible in three places at once: the model sees
        an error and can say it could not analyse the code, the loop stays
        alive because MCPToolRegistry.call feeds tool errors back as text
        rather than raising, and tool_calls_total{status="error"} finally
        becomes reachable for this agent.
        """
        raise NotImplementedError(
            "analyze_code is not implemented: this agent performs no static "
            "analysis yet. Nothing was checked. Do not conclude that the code "
            "is correct or free of problems - report that analysis was "
            "unavailable."
        )


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
    mcp.run(transport="streamable-http")
