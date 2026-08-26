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

import os
import time
from ddgs import DDGS
from mcp.server.fastmcp import FastMCP
from prometheus_client import Counter, Histogram, start_http_server

# ---------------------------------------------------------------------------
# Metrics setup
# ---------------------------------------------------------------------------
# Counter: a number that only goes up (total calls, total errors)
# Histogram: tracks a distribution of values (so you can compute p50/p95/p99
#            latency later in Grafana, not just an average)
#
# Known gap worth understanding before building dashboards on these: FastMCP
# validates tool arguments against the schema and rejects bad input BEFORE
# calling the decorated function, so a schema-invalid call never reaches the
# code below and never increments status="error". Only failures raised inside
# the tool body are counted here.
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

# MCP protocol port. Defaults to 8000 (what the k8s Service targets), but is
# overridable so all three agents can run side by side on one machine during
# local development - in the cluster each agent gets its own pod and can keep
# the same port, but on a laptop they'd collide.
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

# Bind 0.0.0.0, not FastMCP's 127.0.0.1 default: a process inside a container
# that listens only on loopback is unreachable from outside the pod, so
# kubectl port-forward and Service traffic would both fail.
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
mcp = FastMCP(
    "research-agent",
    host="0.0.0.0",
    port=MCP_PORT,
    stateless_http=True,
)


# ---------------------------------------------------------------------------
# Business logic - kept separate from the MCP tool wrapper on purpose.
# ---------------------------------------------------------------------------
# Why this stopped being a stub: while it returned a canned string, the tool
# call still *looked* successful, so the orchestrating model received a
# well-formed result containing no information and confidently invented an
# answer around it (two runs produced two different fictional expansions of
# "MCP", one with a fabricated statistic). An empty result shaped like a good
# one is worse than an error, because nothing downstream can detect it.
#
# DuckDuckGo via ddgs needs no API key, which keeps the project inside its
# no-cloud-budget constraint. The tradeoff is that ddgs scrapes HTML rather than
# calling a supported API: it rate-limits, and under rapid use it starts
# returning throttled responses. That's handled rather than hidden - failures
# propagate, which is what finally makes tool_calls_total{status="error"}
# reachable. Swapping in a keyed search API later means changing this class only.
MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "5"))
MIN_SECONDS_BETWEEN_SEARCHES = float(os.environ.get("SEARCH_MIN_INTERVAL", "1.0"))

# Sponsored results come back looking exactly like organic ones, but their URLs
# are ad-network redirects. Left in, they get indexed and cited as evidence: an
# early run had the model reporting a security vendor's ebook marketing as a
# finding about MCP adoption. A model cannot tell an advertisement from a source,
# so the filtering has to happen here.
AD_URL_MARKERS = (
    "/aclick",
    "duckduckgo.com/y.js",
    "googleadservices.com",
    "doubleclick.net",
    "/aclk?",
)


def _is_advertisement(url: str) -> bool:
    return any(marker in url.lower() for marker in AD_URL_MARKERS)


class SearchService:
    def __init__(self):
        self._last_search_at = 0.0

    def _throttle(self) -> None:
        """
        Space out requests. DuckDuckGo throttles automated traffic, and an agent
        loop can fire several searches in a few seconds - which is exactly the
        pattern that trips it.
        """
        elapsed = time.time() - self._last_search_at
        if elapsed < MIN_SECONDS_BETWEEN_SEARCHES:
            time.sleep(MIN_SECONDS_BETWEEN_SEARCHES - elapsed)
        self._last_search_at = time.time()

    def run(self, query: str) -> str:
        self._throttle()
        # Over-fetch a little, since some results get dropped as ads.
        results = DDGS().text(query, max_results=MAX_RESULTS + 3)
        organic = [r for r in results if not _is_advertisement(r["href"])][:MAX_RESULTS]

        if not organic:
            # Not an error - the search worked and found nothing usable. Say so
            # plainly so the model doesn't read silence as permission to invent.
            return f"No search results found for '{query}'."

        # Results are separated by a blank line, which is also the boundary the
        # retrieval agent chunks on - so passing this straight to index_documents
        # stores one document per result rather than one blob.
        # The URL is included so indexed documents carry their provenance and a
        # reader of the final answer can check it.
        return "\n\n".join(
            f"{r['title']}\n{r['body']}\nSource: {r['href']}" for r in organic
        )


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
    # Start the MCP server itself. Using an HTTP-based transport rather than
    # stdio, because this needs to be reachable over the Kubernetes cluster
    # network, not spawned as a local subprocess.
    #
    # Specifically streamable-http, not sse: MCP's original HTTP transport
    # (HTTP+SSE) used two endpoints - one to POST requests, a separate
    # Server-Sent Events stream for responses. Streamable HTTP replaced it in
    # spec revision 2025-03-26 with a single /mcp endpoint that upgrades to a
    # stream only when the server needs to push. SSE reached end-of-life on
    # 2026-04-01; the SDK still ships it for backwards compatibility only.
    mcp.run(transport="streamable-http")
