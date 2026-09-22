"""
Reader Agent - MCP Server

The fourth agent, and the first added since the tool-ownership map was built.
Nothing in the orchestrator knows it exists: discovery asks each configured URL
what it offers, and dispatch is a lookup in what came back. The only orchestrator
change is one line in AGENT_URLS.

It exposes `fetch_page`, which reads ONE url and returns its text. The gap it
fills is measured elsewhere in this project: `search_web` returns snippets and
`retrieve` returns stored fragments, so an answer has never been built from a
page the system actually read. The documented fabrication path is answers
assembled from "real documents about a different subject", which thin snippets
make easy.

Stateless, so a Deployment. See agents/reader_agent/fetch.py for the rules about
what it will and will not fetch, which are the substance of this agent.
"""

import logging
import os

from instrumentation import InstrumentedMCP
from prometheus_client import Counter, start_http_server

from fetch import MAX_CHARS, FetchRefused, fetch

logger = logging.getLogger(__name__)

METRICS_PORT = 9104  # 9100 research, 9101 retrieval, 9102 code-analysis, 9103 orchestrator
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

# Why a fetch did not produce a page, which tool_calls_total cannot say: a
# refusal is a successful call. "refused" rising is the model asking for
# addresses it should not have - worth seeing, and invisible in an error rate.
FETCH_OUTCOMES = Counter(
    "reader_fetch_outcomes_total",
    "Outcome of fetch_page calls",
    ["outcome"],
)

mcp = InstrumentedMCP(
    "reader-agent",
    host="0.0.0.0",
    port=MCP_PORT,
    stateless_http=True,
)

# Stated in every successful result, in the same spirit as analyze_code's report:
# a tool that implies more than it knows is how this project's first stub misled
# the model. A page read is not a page understood, and it is certainly not a
# page verified.
LIMITS = (
    "Not checked: markup was stripped to text, so navigation and boilerplate may "
    "be mixed in; JavaScript was not executed, so anything the browser would have "
    "rendered is absent; and nothing here verifies what the page claims."
)


@mcp.tool()
def fetch_page(url: str) -> str:
    """Fetch one web page by URL and return its text. Use after search_web when
    a result looks like it answers the question and the snippet is too short to
    tell. Takes a single http or https URL; returns the page's text, or a
    refusal saying why it was not fetched."""
    try:
        page = fetch(url)
    except FetchRefused as refusal:
        # A refusal is a RESULT, not a failure: the tool did its job. Returning
        # it as text keeps it inside the loop as feedback the model can act on,
        # and keeps the error rate meaning "this tool broke".
        FETCH_OUTCOMES.labels(outcome="refused").inc()
        logger.info("fetch refused for %r: %s", url, refusal)
        return f"Could not fetch that page. {refusal}"

    FETCH_OUTCOMES.labels(outcome="fetched").inc()
    logger.info("fetched %s (%d chars, truncated=%s)", page.url, len(page.text), page.truncated)

    if not page.text.strip():
        # An empty page is a fact about the page, and saying "here is the text:"
        # followed by nothing invites the model to fill the gap itself.
        return (
            f"{page.url} returned no readable text. That is not evidence the topic "
            "is absent - the page may render its content with JavaScript, which "
            "this tool does not run."
        )

    header = f"Text of {page.url} ({page.content_type}, {len(page.text)} characters"
    header += f", truncated at {MAX_CHARS}).\n\n" if page.truncated else ").\n\n"
    return header + page.text + "\n\n" + LIMITS


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    start_http_server(METRICS_PORT)
    logger.info("reader agent: MCP on %s, metrics on %s", MCP_PORT, METRICS_PORT)
    mcp.run(transport="streamable-http")
