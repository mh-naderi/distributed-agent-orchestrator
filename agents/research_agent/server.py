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

import logging
import os
import time
from dataclasses import dataclass

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException
from coverage import unmentioned_terms
from indexer import index_results
from instrumentation import InstrumentedMCP
from prometheus_client import Counter, start_http_server

# ---------------------------------------------------------------------------
# Metrics setup
# ---------------------------------------------------------------------------
# Counter: a number that only goes up; Histogram: a distribution, so p95
# latency is available in Grafana rather than only an average. Both live in
# instrumentation.py now, recorded at the MCP boundary - see that file for why
# counting inside the tool body could not see rejected calls at all.
# Search results are indexed into the retrieval agent after a successful
# search - see indexer.py for why the side effect lives here rather than in
# the orchestrator. It is best effort, so it needs its own counter: indexing
# that fails quietly is how a corpus stays empty while every dashboard looks
# healthy.
RESULTS_INDEXED = Counter(
    "search_results_indexed_total",
    "Search results handed to the retrieval agent, by outcome",
    ["status"],
)

# Why a search produced no usable results, which tool_calls_total cannot say.
# A rate limit and a genuinely empty search both used to look like one call;
# only this counter distinguishes "we were throttled" from "nothing matched",
# and the first is the one that should never reach an answer as a finding.
SEARCH_OUTCOMES = Counter(
    "search_outcomes_total",
    "Web searches by outcome",
    # results | results_missing_terms | only_sponsored | no_results
    # | rate_limited | failed
    ["outcome"],
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
mcp = InstrumentedMCP(
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

# One extra attempt by default. See SearchService._fetch for why it is not more.
SEARCH_MAX_ATTEMPTS = int(os.environ.get("SEARCH_MAX_ATTEMPTS", "2"))
SEARCH_RETRY_BACKOFF = float(os.environ.get("SEARCH_RETRY_BACKOFF", "2.0"))

# A tool says so when it has nothing to offer, rather than leaving the caller to
# recognise the prose. The orchestrator refuses to end a run on an answer built
# from nothing but these, and it must not do that by pattern-matching English:
# this project has twice shipped a lexical matcher that missed a rephrasing.
# The marker is the contract, the sentence after it is for the model.
NO_EVIDENCE = "[no-evidence]"

logger = logging.getLogger(__name__)

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


class SearchUnavailable(RuntimeError):
    """
    The search backend failed to answer.

    A distinct type because the distinction is the whole point: "the search
    failed" and "the web contains nothing about this" are different facts, and
    only one of them is evidence. Collapsing them is how a model ends up
    reporting an absence it never established.
    """


# ddgs defines RatelimitException but never raises it - every failure arrives as
# a generic DDGSException carrying the underlying engine error, so throttling has
# to be recognised from the message. These markers are a best effort, and the
# design deliberately does not depend on them being complete: an unrecognised
# failure is still reported as a failure, never as absence. Getting the
# classification wrong costs a retry, not correctness.
THROTTLE_MARKERS = (
    "ratelimit",
    "rate limit",
    "429",
    "too many requests",
    "403",
    "forbidden",
    "blocked",
    "captcha",
)

# ddgs raises this rather than returning an empty list when nothing matched.
NO_RESULTS_MARKER = "no results found"


def _looks_throttled(exc: Exception) -> bool:
    if isinstance(exc, TimeoutException):
        return True
    return any(marker in str(exc).lower() for marker in THROTTLE_MARKERS)


@dataclass(frozen=True)
class SearchOutcome:
    """
    What a search produced, and whether it is worth storing.

    `indexable` exists because search_web feeds its own output to the retrieval
    agent. Before this, a message like "no search results found" was indexed as
    though it were a document, so a later retrieve could return the record of a
    failed search as evidence - polluting the corpus that retrieval-first
    routing now depends on. Only real results are worth keeping.
    """

    text: str
    indexable: bool
    # What goes to the corpus, when that differs from what the model is shown.
    # The coverage note is commentary about the results, not a document: it was
    # briefly indexed along with them, and because it repeats the words of the
    # query it then came back as the NEAREST match to that query - a note about
    # finding nothing, stored as evidence, ranking first. Defaults to `text` so
    # every other outcome is unaffected.
    storable: str | None = None

    def to_store(self) -> str:
        return self.text if self.storable is None else self.storable


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

    def _fetch(self, query: str) -> list[dict]:
        """
        Call ddgs, retrying once on what looks like throttling.

        Retrying a rate limit is in tension with being rate-limited - the remedy
        for "too many requests" is not another request. It is bounded to one
        extra attempt for an observed reason: the live integration test failed
        once during a session that had just driven dozens of searches through
        the eval harness, then passed on retry. One spaced-out retry covers
        that; more would be arguing with the server. The whole budget stays well
        inside MCP_HTTP_TIMEOUT, which is 30 seconds.
        """
        last_error: Exception = DDGSException("search was never attempted")

        for attempt in range(1, SEARCH_MAX_ATTEMPTS + 1):
            self._throttle()
            try:
                return DDGS().text(query, max_results=MAX_RESULTS + 3)
            except DDGSException as exc:
                last_error = exc
                if not _looks_throttled(exc) or attempt == SEARCH_MAX_ATTEMPTS:
                    break
                logger.warning("search throttled (attempt %s), backing off: %s", attempt, exc)
                time.sleep(SEARCH_RETRY_BACKOFF * attempt)

        raise last_error

    def run(self, query: str) -> SearchOutcome:
        try:
            results = self._fetch(query)
        except DDGSException as exc:
            return self._failure(query, exc)

        # Over-fetch a little, since some results get dropped as ads.
        organic = [r for r in results if not _is_advertisement(r["href"])][:MAX_RESULTS]

        if not organic:
            # The search worked; everything it returned was an advertisement.
            # Say exactly that. The old wording here was "no search results
            # found", which told the model the topic had nothing written about
            # it - a different and unsupported claim.
            SEARCH_OUTCOMES.labels(outcome="only_sponsored").inc()
            return SearchOutcome(
                text=(
                    f"{NO_EVIDENCE} "
                    f"Every result for '{query}' was a sponsored link, so there is "
                    f"nothing citable. This is not evidence that the topic is absent "
                    f"from the web - the search was not usable. Try the retrieve tool "
                    f"or a differently worded query."
                ),
                indexable=False,
            )

        SEARCH_OUTCOMES.labels(outcome="results").inc()
        # Results are separated by a blank line, which is also the boundary the
        # retrieval agent chunks on - so passing this straight to index_documents
        # stores one document per result rather than one blob.
        # The URL is included so indexed documents carry their provenance and a
        # reader of the final answer can check it.
        body = "\n\n".join(
            f"{r['title']}\n{r['body']}\nSource: {r['href']}" for r in organic
        )

        results_only = body
        missing = unmentioned_terms(query, body)
        if missing:
            SEARCH_OUTCOMES.labels(outcome="results_missing_terms").inc()
            # Appended after the results, not before: it is a note about them,
            # and putting it first would read as a refusal to show them.
            body += (
                "\n\nNote: none of these results mention "
                + ", ".join(missing)
                + ". They were the closest matches, not necessarily results about "
                "it. Do not describe them as though they were, and say so if that "
                "is all there is."
            )

        return SearchOutcome(text=body, indexable=True, storable=results_only)

    def _failure(self, query: str, exc: DDGSException) -> SearchOutcome:
        """
        Turn a backend failure into something the model can act on.

        Genuine emptiness is reported as emptiness. Everything else is reported
        as a failed lookup that says so, because the one thing the model must
        not take from a broken search is permission to answer from memory.
        """
        if _looks_throttled(exc):
            SEARCH_OUTCOMES.labels(outcome="rate_limited").inc()
            logger.warning("search rate-limited for %r: %s", query, exc)
            raise SearchUnavailable(
                f"{NO_EVIDENCE} "
                f"Web search is rate-limited right now, so '{query}' could not be "
                f"looked up. This is a transport failure, not a finding: it says "
                f"nothing about whether the information exists. Try the retrieve "
                f"tool, or state that you could not check - do not answer from "
                f"memory as though the search had succeeded."
            ) from exc

        if NO_RESULTS_MARKER in str(exc).lower():
            SEARCH_OUTCOMES.labels(outcome="no_results").inc()
            return SearchOutcome(
                text=(
                    f"{NO_EVIDENCE} "
                    f"The search ran and matched nothing for '{query}'. That is a "
                    f"result about this query's wording, not proof the subject does "
                    f"not exist - consider rephrasing before concluding anything."
                ),
                indexable=False,
            )

        SEARCH_OUTCOMES.labels(outcome="failed").inc()
        logger.warning("search failed for %r: %s", query, exc)
        raise SearchUnavailable(
            f"{NO_EVIDENCE} "
            f"Web search failed for '{query}' ({exc}). The lookup did not happen, "
            f"so nothing follows from it about what is or is not true. Try the "
            f"retrieve tool, or say the check could not be completed."
        ) from exc


search_service = SearchService()


# ---------------------------------------------------------------------------
# MCP tool definition - the thin adapter
# ---------------------------------------------------------------------------
@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for information relevant to the query and return
    a summary of findings."""
    outcome = search_service.run(query)

    # Housekeeping, not part of the answer. The model was asked to do this
    # via the system prompt and reliably would not, because indexing pays
    # off on the NEXT run and costs tokens on this one. Doing it here means
    # the corpus grows without the orchestrator hardcoding a search/index
    # pairing - and it cannot fail this call, only be counted.
    #
    # Only real results are stored. A message explaining that a search found
    # nothing is not a document, and indexing it meant a later retrieve could
    # return the record of a failed search as though it were evidence.
    #
    # The label is "web-search", not the query. Passing the query made every
    # stored document claim to be ABOUT what was asked, so a search for something
    # that does not exist filed real documents under its name and a later
    # retrieve returned them as evidence. Each result carries its own "Source:"
    # line, and the retrieval agent prefers that - see provenance_of in store.py.
    if outcome.indexable:
        stored = index_results(outcome.to_store(), source="web-search")
        RESULTS_INDEXED.labels(status="stored" if stored else "skipped").inc()
    else:
        RESULTS_INDEXED.labels(status="not_indexable").inc()

    return outcome.text


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
