"""
Retrieval Agent - MCP Server

Same pattern as research_agent/server.py (see that file for the fully annotated
version): thin @mcp.tool() adapters over a separate service, with Prometheus
instrumentation on every call.

This agent replaced the summarizer. The summarizer took `text` as an argument,
which meant the orchestrator had to hold the document and pass it by value - so
delegating saved nothing that the orchestrator couldn't do itself, and in
practice the model correctly declined to call it. Retrieval takes a *query* and
returns text the orchestrator has never seen, which is a capability the
orchestrator genuinely lacks.

It is also the only stateful service in the system. Everything else can be
killed and restarted with no consequence; this one owns an index that has to
survive. That's what makes the multi-server split load-bearing - see
docs/architecture.md.
"""

import os

import ollama
from instrumentation import InstrumentedMCP
from prometheus_client import Gauge, start_http_server

from store import VectorStore, chunk


# A stateless agent's metrics are all about flow - how many calls, how fast. A
# stateful one also has a *size*, and "how big is the corpus" is the first thing
# you want on a dashboard when retrieval quality changes unexpectedly.
DOCUMENTS_INDEXED = Gauge("retrieval_documents_total", "Documents currently in the index")

METRICS_PORT = 9101  # inherited from the summarizer agent it replaced
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

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
    "retrieval-agent",
    host="0.0.0.0",
    port=MCP_PORT,
    stateless_http=True,
)

_ollama = ollama.Client(host=OLLAMA_HOST)


def embed(texts: list[str]) -> list[list[float]]:
    """
    Turn texts into vectors via Ollama.

    One batched call rather than one per text: the HTTP round trip dominates.

    num_ctx is set explicitly because Ollama serves nomic-embed-text with a
    2048-token window by default even though the model supports 8192 - and it
    truncates silently, so a long document would be embedded from its first
    fragment only, with no error to tell you.
    """
    response = _ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
        options={"num_ctx": 8192},
    )
    return [list(vector) for vector in response.embeddings]


store = VectorStore(embed)
DOCUMENTS_INDEXED.set(store.count())


@mcp.tool()
def index_documents(texts: list[str], source: str = "unknown") -> str:
    """Store documents in the vector index so they can be retrieved later by
    meaning. Pass the text of search results or any other material worth
    remembering, and a short label for where it came from. Text separated by
    blank lines is stored as separate documents."""
    count = store.index(chunk(texts), source)
    DOCUMENTS_INDEXED.set(store.count())
    return f"Indexed {count} document(s) from {source}. Corpus now holds {store.count()}."


@mcp.tool()
def retrieve(query: str, k: int = 5) -> str:
    """Search the stored corpus for documents relevant to the query, by meaning
    rather than keyword.

    TRY THIS BEFORE search_web. The corpus persists between runs and already
    holds what previous searches found, so it frequently answers the question
    with no network round trip. Only if it returns nothing relevant should the
    web be searched.

    The description carries this instruction rather than leaving it to the
    system prompt alone. That was measured: with four tools the model chose
    retrieve for a corpus question 4 times out of 4, and after a fifth tool was
    added it chose search_web 4 times out of 5 - the prompt rule was competing
    with five tool descriptions and losing. Guidance about WHEN to use a tool
    belongs next to the tool.
    """
    hits = store.retrieve(query, k)

    if not hits:
        # Say so explicitly. An empty result that reads like a successful
        # answer is how a model ends up inventing one - exactly what the
        # stubbed search tool caused before it was made real.
        return (
            "No documents in the index match that query. "
            "The index may be empty - try searching the web and indexing the results first."
        )

    return "\n\n".join(
        f"[{i + 1}] (source: {hit['source']}, distance: {hit['distance']:.3f})\n{hit['text']}"
        for i, hit in enumerate(hits)
    )


if __name__ == "__main__":
    start_http_server(METRICS_PORT)
    mcp.run(transport="streamable-http")
