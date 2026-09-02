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
from analysis import report
from evaluator import report as evaluate_report
from instrumentation import InstrumentedMCP
from prometheus_client import start_http_server

METRICS_PORT = 9102
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

# stateless_http=True: no Mcp-Session-Id is issued, so every request stands
# alone and any replica can serve any of them.
#
# This was MEASURED, not assumed. With two replicas behind the Service the
# session handshake landed on one pod and the next request round-robined to
# the other, which had never seen that session id, answered 404, and the
# client raised McpError: Session terminated - three attempts out of three.
# At one replica the same code succeeded three out of three.
#
# The distinction that caused it is worth keeping: these agents are
# stateless, but the TRANSPORT was not. Holding no state does not make a
# service horizontally scalable if the protocol in front of it is
# session-oriented. Nothing is lost here - the tools are pure functions, and
# the retrieval agent keeps its state in sqlite rather than in a session.
mcp = InstrumentedMCP(
    "code-analysis-agent",
    host="0.0.0.0",
    port=MCP_PORT,
    stateless_http=True,
)


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
    result = code_analysis_service.run(code)
    return result


@mcp.tool()
def evaluate_expression(expression: str) -> str:
    """Evaluate a single Python EXPRESSION and return its value.

    Use this for arithmetic and simple data questions - "what is 17 * 23",
    "what is the sum of these numbers" - rather than working it out yourself.

    This is a restricted subset, not a Python interpreter. Arithmetic,
    comparisons, boolean logic, list/tuple/dict/set literals, indexing and a
    dozen built-in functions are available. Imports, attribute access,
    assignment, loops, comprehensions and function definitions are not, and
    asking for them returns a refusal that lists what is available instead.
    """
    result = evaluate_report(expression)
    # A refusal is a successful CALL that returns a refusal - the tool did
    # exactly its job. Counting it as an error would make the error rate
    # measure how often the model asks for too much, which is a different
    # question from whether this agent is healthy.
    return result


if __name__ == "__main__":
    start_http_server(METRICS_PORT)
    mcp.run(transport="streamable-http")
