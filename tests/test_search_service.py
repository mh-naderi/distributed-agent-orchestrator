"""
Tests for how the research agent reports a search that did not work.

The distinction under test is the one the whole agent exists to protect: "the
search failed" and "there is nothing to find" are different facts, and only the
second is evidence. ddgs makes them easy to conflate - it never returns an empty
list, raising a generic DDGSException instead, so a rate limit and an empty
result arrive in the same shape and the difference has to be recovered from a
message.

No network. ddgs is faked, because the behaviour worth pinning is what this code
does with each failure, not whether DuckDuckGo is reachable. The one test that
does hit the network lives in test_mcp_integration.py and is skipped without
agents.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from ddgs.exceptions import DDGSException, TimeoutException

AGENT_DIR = Path(__file__).resolve().parents[1] / "agents" / "research_agent"
sys.path.insert(0, str(AGENT_DIR))

# Loaded under an explicit name rather than `import server`: another test module
# already imports a different agent's server.py, and Python caches modules by
# name - a plain import here would silently hand back the code-analysis agent.
_spec = importlib.util.spec_from_file_location("research_server", AGENT_DIR / "server.py")
research_server = importlib.util.module_from_spec(_spec)
sys.modules["research_server"] = research_server
_spec.loader.exec_module(research_server)

SearchUnavailable = research_server.SearchUnavailable


ORGANIC = [
    {"title": "A", "body": "body a", "href": "https://example.com/a"},
    {"title": "B", "body": "body b", "href": "https://example.org/b"},
]
SPONSORED = [
    {"title": "Ad", "body": "buy", "href": "https://duckduckgo.com/y.js?ad=1"},
    {"title": "Ad2", "body": "buy", "href": "https://example.com/aclick?x=1"},
]


class FakeDDGS:
    """Stands in for ddgs. Records attempts, replays a scripted sequence."""

    calls = 0
    script = None  # list of results-or-exception, one per attempt

    def __init__(self):
        pass

    def text(self, query, max_results=None):
        step = type(self).script[min(type(self).calls, len(type(self).script) - 1)]
        type(self).calls += 1
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    """Point the service at the fake and remove every real sleep."""
    FakeDDGS.calls = 0
    FakeDDGS.script = [ORGANIC]
    monkeypatch.setattr(research_server, "DDGS", FakeDDGS)
    monkeypatch.setattr(research_server.time, "sleep", lambda _seconds: None)
    return FakeDDGS


@pytest.fixture
def service():
    return research_server.SearchService()


def outcomes(label):
    counter = research_server.SEARCH_OUTCOMES.labels(outcome=label)
    return counter._value.get() or 0.0


# ---------------------------------------------------------------------------
# A failed lookup must never read as an absence
# ---------------------------------------------------------------------------


def test_a_rate_limit_raises_rather_than_reporting_nothing_found(service, fake_search):
    """
    The failure mode this guards against: the model is told "no results", treats
    it as established absence, and answers from memory. A rate limit says
    nothing whatsoever about what exists.
    """
    fake_search.script = [DDGSException("DuckDuckGo: 202 Ratelimit")]
    before = outcomes("rate_limited")

    with pytest.raises(SearchUnavailable) as raised:
        service.run("anything")

    message = str(raised.value).lower()
    assert "rate-limited" in message
    assert "not a finding" in message
    assert "no results" not in message, "must not phrase a failure as an absence"
    assert outcomes("rate_limited") == before + 1


def test_an_unrecognised_failure_is_still_reported_as_a_failure(service, fake_search):
    """
    The throttle markers are a guess at strings this project has not observed.
    Correctness must not depend on them: anything unclassified is still a failed
    lookup, never an answer about the world.
    """
    fake_search.script = [DDGSException("something nobody predicted")]
    before = outcomes("failed")

    with pytest.raises(SearchUnavailable) as raised:
        service.run("anything")

    assert "did not happen" in str(raised.value)
    assert outcomes("failed") == before + 1


def test_a_genuinely_empty_search_is_reported_as_empty(service, fake_search):
    """
    ddgs signals "nothing matched" with an exception too. This one IS a result,
    so it is returned rather than raised - but still hedged, because an empty
    search says something about the query wording, not about the subject.
    """
    fake_search.script = [DDGSException("No results found.")]
    before = outcomes("no_results")

    outcome = service.run("obscure phrase")

    assert "matched nothing" in outcome.text
    assert "not proof" in outcome.text
    assert outcome.indexable is False
    assert outcomes("no_results") == before + 1


# ---------------------------------------------------------------------------
# Retry, bounded
# ---------------------------------------------------------------------------


def test_a_throttled_search_is_retried_once_and_can_succeed(service, fake_search):
    fake_search.script = [DDGSException("429 Too Many Requests"), ORGANIC]

    outcome = service.run("q")

    assert fake_search.calls == 2
    assert outcome.indexable is True
    assert "example.com" in outcome.text


def test_a_timeout_counts_as_throttling(service, fake_search):
    fake_search.script = [TimeoutException("timed out"), ORGANIC]

    service.run("q")

    assert fake_search.calls == 2


def test_a_failure_that_is_not_throttling_is_not_retried(service, fake_search):
    """Retrying a malformed-query error just spends a second to fail again."""
    fake_search.script = [DDGSException("bad request")]

    with pytest.raises(SearchUnavailable):
        service.run("q")

    assert fake_search.calls == 1


def test_retries_are_bounded(service, fake_search, monkeypatch):
    monkeypatch.setattr(research_server, "SEARCH_MAX_ATTEMPTS", 2)
    fake_search.script = [DDGSException("ratelimit")]

    with pytest.raises(SearchUnavailable):
        service.run("q")

    assert fake_search.calls == 2, "must not keep hammering a server asking us to stop"


# ---------------------------------------------------------------------------
# Sponsored results
# ---------------------------------------------------------------------------


def test_sponsored_results_are_filtered_out(service, fake_search):
    fake_search.script = [SPONSORED[:1] + ORGANIC]

    outcome = service.run("q")

    assert "duckduckgo.com/y.js" not in outcome.text
    assert "example.com/a" in outcome.text


def test_an_all_sponsored_page_does_not_claim_the_topic_is_absent(service, fake_search):
    """
    This branch used to say "No search results found", which is a claim about
    the world. What actually happened is that every result was an ad.
    """
    fake_search.script = [SPONSORED]
    before = outcomes("only_sponsored")

    outcome = service.run("laptops")

    assert "sponsored" in outcome.text
    assert "not evidence" in outcome.text
    assert outcome.indexable is False
    assert outcomes("only_sponsored") == before + 1


def test_real_results_carry_their_source(service, fake_search):
    outcome = service.run("q")

    assert outcome.indexable is True
    assert outcome.text.count("Source: ") == 2
    assert "\n\n" in outcome.text, "blank line is the retrieval agent's chunk boundary"


# ---------------------------------------------------------------------------
# What reaches the corpus
# ---------------------------------------------------------------------------


def test_only_real_results_are_indexed(fake_search, monkeypatch):
    """
    The corpus pollution this prevents: a message explaining that a search found
    nothing was previously stored as a document, so a later retrieve could
    return the record of a failed search as though it were evidence - in a
    system whose routing now tries retrieve first.
    """
    indexed = []
    monkeypatch.setattr(research_server, "index_results", lambda text, source: indexed.append(text) or True)

    fake_search.script = [SPONSORED]
    research_server.search_web("laptops")
    assert indexed == [], "a failed search must not become a document"

    fake_search.script = [ORGANIC]
    research_server.search_web("q")
    assert len(indexed) == 1
    assert "Source: " in indexed[0]


def test_a_rate_limit_reaches_the_caller_from_the_tool(fake_search, monkeypatch):
    """The tool must propagate, so the boundary counts an error and the
    orchestrator relays the message to the model rather than a bare failure."""
    monkeypatch.setattr(research_server, "index_results", lambda text, source: True)
    fake_search.script = [DDGSException("ratelimit")]

    with pytest.raises(SearchUnavailable):
        research_server.search_web("q")


# ---------------------------------------------------------------------------
# How a document's origin is presented to the model
# ---------------------------------------------------------------------------

_ret_spec = importlib.util.spec_from_file_location(
    "retrieval_server",
    Path(__file__).resolve().parents[1] / "agents" / "retrieval_agent" / "server.py",
)
retrieval_server = importlib.util.module_from_spec(_ret_spec)
sys.modules["retrieval_server"] = retrieval_server
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "retrieval_agent"))
_ret_spec.loader.exec_module(retrieval_server)


def test_a_derived_url_is_presented_as_a_source():
    assert retrieval_server._attribution("https://example.org/a.pdf") == (
        "source: https://example.org/a.pdf"
    )


@pytest.mark.parametrize("label", ["Quazzlemint Foundation 2019 report", "integration-test"])
def test_an_asserted_label_is_marked_unverified(label):
    """
    What the model reads is the whole point. A caller's claim printed in the same
    shape as a derived fact is indistinguishable from evidence, and this system
    had already answered from one.
    """
    rendered = retrieval_server._attribution(label)

    assert rendered == f"unverified label: {label}"
    assert not rendered.startswith("source:")


# ---------------------------------------------------------------------------
# Results that are not about what was asked
# ---------------------------------------------------------------------------
# The last fabrication path. The corpus fixes and the empty-evidence guardrail
# both leave it untouched, because the search genuinely succeeds - it just
# returns documents about something else, and the model answers from them.


@pytest.mark.parametrize(
    "query, results, expected",
    [
        (
            "What did the Quazzlemint Foundation conclude in its 2019 report?",
            "Annual Report 2019 - Mellon Foundation grants supported work",
            ["Quazzlemint"],
        ),
        # Every capitalised word is present, so there is nothing to say.
        (
            "Who won the 2018 FIFA World Cup final?",
            "The 2018 FIFA World Cup final: France beat Croatia.",
            [],
        ),
        # The first word is capitalised by sentence position, not because it
        # names anything - reporting it would be noise on every single query.
        ("What is quantum tunnelling?", "Tunnelling is a quantum effect.", []),
        # Lower-case queries name nothing in particular.
        ("what is kubernetes", "Docker and containers", []),
        # Case-insensitive matching: the result names it differently.
        ("Tell me about Kubernetes", "kubernetes orchestrates containers", []),
    ],
)
def test_only_unmatched_proper_nouns_are_reported(query, results, expected):
    assert research_server.unmentioned_terms(query, results) == expected


def test_a_term_is_reported_once_however_often_it_is_asked(service, fake_search):
    assert research_server.unmentioned_terms("Zorbulon and Zorbulon", "nothing") == [
        "Zorbulon"
    ]


def test_the_note_is_appended_to_real_results(service, fake_search):
    """
    Appended, not substituted. The results are real and worth having; what is
    added is the fact that they do not mention what was asked about.
    """
    fake_search.script = [
        [{"title": "Mellon 2019", "body": "grants", "href": "https://example.org/m"}]
    ]

    outcome = service.run("What did the Quazzlemint Foundation conclude?")

    assert "Mellon 2019" in outcome.text, "the results must still be there"
    assert "none of these results mention Quazzlemint" in outcome.text
    assert outcome.indexable is True


def test_no_note_when_the_results_are_about_the_subject(service, fake_search):
    """
    A note on every search would be noise, and noise is what a model learns to
    skip. It has to mean something when it appears.
    """
    fake_search.script = [
        [{"title": "Kubernetes", "body": "orchestration", "href": "https://example.org/k"}]
    ]

    outcome = service.run("Tell me about Kubernetes")

    assert "Note:" not in outcome.text


def test_the_note_is_counted_separately(service, fake_search):
    """Worth a metric: a rising line here means searches are drifting off-subject."""
    fake_search.script = [
        [{"title": "Mellon", "body": "grants", "href": "https://example.org/m"}]
    ]
    before = outcomes("results_missing_terms")

    service.run("What did the Quazzlemint Foundation conclude?")

    assert outcomes("results_missing_terms") == before + 1


def test_both_agents_carry_an_identical_coverage_module():
    """
    Duplicated rather than shared, the same trade as instrumentation.py: each
    image builds from its own directory. Duplication is only acceptable if drift
    is caught.
    """
    copies = sorted((Path(__file__).resolve().parents[1] / "agents").glob("*/coverage.py"))

    assert len(copies) == 2, [str(c) for c in copies]
    assert copies[0].read_bytes() == copies[1].read_bytes(), (
        f"{copies[1]} has drifted from {copies[0]} - copy it across rather than editing one"
    )


def test_the_coverage_note_is_shown_but_not_stored(service, fake_search):
    """
    The note is commentary about the results, not a document. It was briefly
    indexed along with them, and because it repeats the words of the query it
    came back as the NEAREST match to that same query - a note about finding
    nothing, stored as evidence, ranking first. It corrupted a threshold study
    before it was noticed.
    """
    fake_search.script = [
        [{"title": "Mellon 2019", "body": "grants", "href": "https://example.org/m"}]
    ]

    outcome = service.run("What did the Quazzlemint Foundation conclude?")

    assert "none of these results mention" in outcome.text, "the model must see it"
    assert "none of these results mention" not in outcome.to_store(), "the corpus must not"
    assert "Mellon 2019" in outcome.to_store(), "the results themselves are still stored"


def test_an_outcome_without_a_note_stores_what_it_shows(service, fake_search):
    outcome = service.run("q")

    assert outcome.to_store() == outcome.text
