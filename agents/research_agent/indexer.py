"""
The research agent's client for the retrieval agent.

WHY THIS EXISTS. `index_documents` was never being called in real use. The
system prompt asks the model to index search results after searching, and a
small local model reliably declines - correctly, from its own point of view,
since indexing helps the NEXT run and does nothing for the answer in progress.
So the corpus never grew, and the retrieval agent's persistence story was
hollow: a durable index that nothing ever wrote to.

WHY HERE AND NOT IN THE ORCHESTRATOR. The other fix is to have the orchestrator
call `index_documents` itself after every `search_web`. docs/architecture.md
rejects that, and the reason is worth repeating: it hardcodes a specific pair of
tools into the router, which trades away the routing generality that the
tool-ownership map exists to provide. The orchestrator would stop being a
general agent loop and start being a pipeline with one branch special-cased.

Putting the side effect in the agent that PRODUCED the data keeps the
orchestrator generic. Nothing about the loop changes; a tool simply does its own
housekeeping.

WHAT IT COSTS. This is a genuine coupling, and it should be named rather than
glossed over. The research agent now has an opinion about another agent's
existence, where before it had none. Two things keep that honest:

  - It is BEST EFFORT. If the retrieval agent is unreachable, or slow, or
    broken, `search_web` still returns its results. One agent failing must not
    take another down - that isolation is a claim this project makes, and it is
    only true if the client is written to hold it up.
  - It is COUNTED. Best effort that fails silently is how a corpus stays empty
    while everything looks healthy. Successes and failures are both metrics, so
    "indexing is quietly broken" is a visible state rather than an invisible
    one.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

logger = logging.getLogger(__name__)

# Defaults to the local development port, matching the orchestrator's own
# defaults; the Kubernetes manifest points this at the Service DNS name.
RETRIEVAL_AGENT_URL = os.environ.get(
    "RETRIEVAL_AGENT_URL", "http://localhost:18001/mcp"
)

# Kept deliberately short. This work is a side effect the caller is not waiting
# on for correctness, so it must never be the reason a search feels slow. If the
# retrieval agent cannot answer in this long, skipping is the right outcome.
INDEX_TIMEOUT = float(os.environ.get("RESEARCH_INDEX_TIMEOUT", "20"))

_auto = os.environ.get("RESEARCH_AUTO_INDEX", "on").strip().lower()
AUTO_INDEX = _auto not in ("off", "false", "0", "no")


async def _index(text: str, source: str, url: str, timeout: float) -> str:
    http_client = create_mcp_http_client(timeout=timeout)
    async with streamable_http_client(url, http_client=http_client) as (
        read_stream,
        write_stream,
        _session_id,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=timeout),
        ) as session:
            await session.initialize()
            result = await session.call_tool(
                "index_documents", {"texts": [text], "source": source}
            )
            return "\n".join(
                block.text
                for block in result.content
                if getattr(block, "type", None) == "text"
            )


def index_results(text: str, source: str = "web-search") -> bool:
    """
    Store search results in the retrieval agent's index. Never raises.

    Returns whether the documents were stored, so the caller can count both
    outcomes. The text is passed through as-is: the retrieval agent splits on
    blank lines, and the search results are already separated that way, so each
    result becomes its own document - which is what makes the stored vectors
    mean anything (see the chunking note in the retrieval agent's store).

    The coroutine runs on a THREAD OF ITS OWN, which is not fussiness.
    An earlier version called asyncio.run directly, on the assumption that
    FastMCP dispatches sync tool functions to a worker thread. It does not -
    they execute on the event loop thread, and asyncio.run raises
    "cannot be called from a running event loop" there. The search still
    succeeded and the corpus silently stayed empty; the skipped counter and a
    log line are what surfaced it. Running in a private thread with its own
    loop works whether or not a loop is already running.
    """
    if not AUTO_INDEX or not text or not text.strip():
        return False

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(
                asyncio.run,
                _index(text, source, RETRIEVAL_AGENT_URL, INDEX_TIMEOUT),
            ).result(timeout=INDEX_TIMEOUT + 5)
    except Exception as exc:
        # Deliberately swallowed. The search succeeded; the housekeeping did
        # not. Failing the tool here would let one agent's outage take out
        # another's, which is precisely the failure isolation this project
        # claims the multi-server split buys.
        logger.warning("indexing search results failed (continuing): %s", exc)
        return False

    logger.info("indexed search results: %s", response)
    return True
