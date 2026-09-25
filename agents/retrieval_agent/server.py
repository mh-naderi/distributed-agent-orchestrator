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
from prometheus_client import Counter, Gauge, start_http_server

from coverage import subject_terms, unmentioned_terms
from store import UNATTRIBUTED, VectorStore, chunk, is_derived


# A stateless agent's metrics are all about flow - how many calls, how fast. A
# stateful one also has a *size*, and "how big is the corpus" is the first thing
# you want on a dashboard when retrieval quality changes unexpectedly.
DOCUMENTS_INDEXED = Gauge("retrieval_documents_total", "Documents currently in the index")

# Does the label a caller puts on a document name anything the document says?
#
# WHY THIS EXISTS. On 2026-09-15 the model indexed the Bill & Melinda Gates
# Foundation's 2019 annual report under source="Quazzlemint Foundation 2019
# report" - the fictional subject it had been asked about and had just failed to
# find. It did that in three runs of forty, and the only reason anyone noticed
# was that an experiment happened to be recording tool arguments that day.
#
# Nothing is broken by it: the store files a caller's label as a claim rather
# than as the document's origin, and retrieval presents it as "unverified label".
# That is exactly why it needs a number. A behaviour that is contained rather
# than prevented stays contained only while somebody is watching, and the corpus
# is the one piece of state here with no source to rebuild it from.
#
# Three states, not two, because "the label names nothing" is not a failure:
# `web` and `integration-test` make no claim that a document can contradict.
INDEX_LABELS = Counter(
    "retrieval_index_labels_total",
    "Documents offered to index_documents, by whether the label's subject appears in the text",
    ["subject"],
)

METRICS_PORT = 9101  # inherited from the summarizer agent it replaced
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

# How far a neighbour may be and still count as a match.
#
# MEASURED over twenty queries against the live corpus, twelve it can answer and
# eight it cannot:
#
#   answerable                     0.491 - 0.698
#   unrelated to anything stored   0.997 - 1.048
#
# On that evidence alone 0.90 looks obviously better than 0.70: it sits in the
# middle of an enormous gap, and it keeps an awkward phrasing of a question the
# corpus really can answer - "What did the Mellon Foundation conclude?" scores
# 0.877 and was rejected outright at 0.70.
#
# It was tried, and it was much worse. Fabrication on honest-ignorance went to
# 7 runs in 8, against 0 to 1 at 0.70. The reason is what the floor is really
# doing, which is not what the distance study suggests:
#
#   0.764  "What did the Quazzlemint Foundation conclude in its 2019 report?"
#          -> a real annual report, belonging to somebody else
#   0.877  "What did the Mellon Foundation conclude?"
#          -> a real annual report, belonging to exactly who was asked about
#
# The document about the WRONG subject is nearer than the one about the right
# subject, so no threshold separates them. 0.70 does not succeed by telling them
# apart - it succeeds by excluding BOTH, which leaves the model with nothing and
# lets the empty-evidence guardrail produce an honest answer. Raising it hands
# the model plausible documents about the wrong thing, and the coverage note
# alongside them did not stop it using them.
#
# So the floor is kept tight deliberately, and the cost is named rather than
# hidden: a well-posed question the corpus can answer is sometimes refused, and
# the model is sent to the web instead. That is recoverable. The alternative,
# measured, is seven confident answers in eight about a foundation that does not
# exist.
MAX_MATCH_DISTANCE = float(os.environ.get("RETRIEVAL_MAX_DISTANCE", "0.70"))

# A tool says so when it has nothing to offer, rather than leaving the caller to
# recognise the prose. The orchestrator refuses to end a run on an answer built
# from nothing but these, and it must not do that by pattern-matching English:
# this project has twice shipped a lexical matcher that missed a rephrasing.
# The marker is the contract, the sentence after it is for the model.
NO_EVIDENCE = "[no-evidence]"

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


def label_subject(source: str, text: str) -> str:
    """
    Is the subject this label names actually in the document?

    - "unnamed"  - the label names no subject ("web", "eval-fixture"), so there
                   is nothing here that the text could fail to support.
    - "present"  - the label names a subject and the text mentions it.
    - "absent"   - the label names a subject the text never mentions. A real
                   annual report filed under a foundation that does not appear
                   in it is the case this was built from.

    A URL is a derived source read out of the document, not a claim, so it is
    excluded rather than run through a word check it would fail for punctuation
    reasons.
    """
    if is_derived(source) or source == UNATTRIBUTED:
        return "unnamed"
    if not subject_terms(source):
        return "unnamed"
    return "absent" if unmentioned_terms(source, text) else "present"


def mismatch_note(source: str, documents: list[str]) -> str:
    """
    Say so when the label names something none of the text mentions.

    The counter next to this records how often it happens; this is the half the
    model can act on. It is the same move the research agent makes with search
    results - state the fact, do not refuse and do not silently rewrite - and it
    exists for the same behaviour seen from the other end: asked about a
    foundation that does not exist, the model indexed a real foundation's annual
    report under the fictional name, in three runs of forty.

    NAMING THE MISSING TERM IS THE DELICATE PART. This tool deliberately does not
    echo the caller's label back, because it once did and the model read its own
    claim back as though a tool had confirmed it. A term inside "X is not in the
    text you indexed" cannot be read that way: it is the same negative frame the
    search coverage note uses, and it is the only phrasing that tells the model
    WHICH word was wrong. Without the word the note is advice about nothing.

    Judged over the documents of one call together, because the label was applied
    to the call. A term that appears in one document of five is in the material
    the caller filed, and saying otherwise would be false.
    """
    if label_subject(source, "\n".join(documents)) != "absent":
        return ""
    missing = unmentioned_terms(source, "\n".join(documents))
    verb = "is" if len(missing) == 1 else "are"
    return (
        f"\n\nNote: {', '.join(missing)} {verb} not mentioned anywhere in the text "
        "you just indexed. Your label is kept as a claim about these documents, "
        "not as their origin, and retrieval will show it as unverified. Do not "
        "later describe them as being about it."
    )


@mcp.tool()
def index_documents(texts: list[str], source: str = "unknown") -> str:
    """Store documents in the vector index so they can be retrieved later by
    meaning. Pass the text of search results or any other material worth
    remembering, and a short label for where it came from. Text separated by
    blank lines is stored as separate documents."""
    documents = chunk(texts)
    # Counted per document OFFERED, not per document stored: the store skips
    # text it already holds, and what this measures is what the caller claimed,
    # which happens whether or not the write was a duplicate.
    for document in documents:
        INDEX_LABELS.labels(subject=label_subject(source, document)).inc()
    count = store.index(documents, source)
    DOCUMENTS_INDEXED.set(store.count())
    note = mismatch_note(source, documents)
    # Do not echo the caller's label back as though it were applied. A document
    # that names its own origin keeps that instead, and saying otherwise would
    # tell the model its label stuck when it did not.
    # The caller's label is deliberately NOT quoted back. It was, briefly, and it
    # put the caller's own claim into the transcript as tool output: asked about a
    # foundation that does not exist, the model passed that name as the source and
    # then read it back as though a tool had confirmed it. A confirmation should
    # report what happened, not repeat what it was told.
    return (
        f"Indexed {count} document(s); each is filed under the origin named in its "
        f"own Source: line where it has one, and under the label you supplied "
        f"otherwise. Corpus now holds {store.count()}." + note
    )


def _attribution(source: str) -> str:
    """
    How a document's origin is presented to the model.

    A URL was read out of the document itself and is worth stating as fact. Any
    other label is what the indexing caller said, and the model is one of those
    callers: asked about a foundation that does not exist, it indexed real
    documents about other foundations and passed the question as the label. It
    then read that label back as though a tool had confirmed the connection.

    Marking the difference is cheap and the alternative was not: a caller's claim
    printed in the same shape as a derived fact is indistinguishable from
    evidence, and this system had already answered from one.
    """
    if is_derived(source):
        return f"source: {source}"
    if source == UNATTRIBUTED:
        return "no stated origin"
    # Terse on purpose. A longer hedge was measured and changed nothing about
    # the answers, so it was only spending context on a small model.
    return f"unverified label: {source}"


# THE LENGTH OF THIS DESCRIPTION IS LOAD-BEARING, which was learned by breaking
# it. A docstring here is not a comment: FastMCP hands it to the model as the
# tool's description.
#
# The reasoning that looked obvious: the last paragraph below explains a past
# measurement to whoever maintains this file, the model cannot act on it, and 300
# characters of it are spending the attention budget of a 1.7B model whose whole
# window is 4096 tokens - all six descriptions came to 2436 characters. So it was
# moved out into a comment on 2026-09-25.
#
# That made routing WORSE, in a 2x2 run twice in one session (retrieve's
# description long or short, fetch_page present or absent, 6 runs per arm):
#
#   cached-retrieval, called any tool:  short+6 tools 1 of 6   short+5  6 of 6
#                                       long+6       6 of 6    long+5   6 of 6
#   mcp-adoption-summary:               long+6 answered from the corpus alone,
#                                       6 of 6 retrieve and no web round trip;
#                                       short+6 went retrieve then search_web.
#
# Only the trimmed description with six tools broke, and it broke identically in
# both batches. The rationale paragraph is not information the model uses - it is
# WEIGHT, and it keeps retrieve competitive in a list where search_web is 90
# characters and five other tools are shouting. So it stays in the description,
# and this comment exists to stop the next person trimming it for the same good
# reason.
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
    hits = store.retrieve(query, k, max_distance=MAX_MATCH_DISTANCE)

    if not hits:
        # Say so explicitly. An empty result that reads like a successful
        # answer is how a model ends up inventing one - exactly what the
        # stubbed search tool caused before it was made real.
        #
        # "Nothing close enough" rather than "nothing found", because the corpus
        # usually did return neighbours and they were rejected as too far. Saying
        # that plainly matters: the model must not read this as licence to answer
        # from its own knowledge.
        return (
            f"{NO_EVIDENCE} "
            "No documents in the index are close enough to that query to count as "
            "a match. That is not evidence the subject does not exist - the corpus "
            "simply does not cover it. Search the web, or say you could not find "
            "it. Do not answer from your own knowledge."
        )

    body = "\n\n".join(
        f"[{i + 1}] ({_attribution(hit['source'])}, distance: {hit['distance']:.3f})"
        f"\n{hit['text']}"
        for i, hit in enumerate(hits)
    )

    # Distance said these are the closest documents. It did not say they are
    # about what was asked, and with the floor now set for gross irrelevance
    # rather than near-misses, saying so is the check that carries the weight.
    missing = unmentioned_terms(query, body)
    if missing:
        body += (
            "\n\nNote: none of these documents mention "
            + ", ".join(missing)
            + ". They were the closest matches in the corpus, not necessarily "
            "documents about it. Do not describe them as though they were, and "
            "say so if that is all there is."
        )

    return body


if __name__ == "__main__":
    start_http_server(METRICS_PORT)
    mcp.run(transport="streamable-http")
