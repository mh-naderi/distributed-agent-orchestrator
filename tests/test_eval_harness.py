"""
Tests for the evaluation harness itself.

Worth testing because the harness is what the README's published numbers come
from. A harness that measures the wrong thing produces confident, wrong
documentation - which is exactly what happened: cached-retrieval carried a
plain-text note saying "Run after mcp-adoption-summary", nothing enforced it,
and the case ended up measuring whether an unrelated case had happened to
index something.

No agents and no model here: seeding is exercised against a fake registry.
"""

import json
from pathlib import Path

import pytest

import eval.run_eval as run_eval

CASES_PATH = Path(__file__).resolve().parents[1] / "eval" / "test_cases.json"


class FakeRegistry:
    """Records index_documents calls instead of performing them."""

    def __init__(self, tools=("index_documents", "retrieve")):
        self.tools = [{"name": name} for name in tools]
        self.calls = []

    async def discover(self):
        return self.tools

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        return f"indexed {len(arguments.get('texts', []))} document(s)"


@pytest.fixture
def registry(monkeypatch):
    fake = FakeRegistry()
    monkeypatch.setattr(run_eval, "MCPToolRegistry", lambda: fake)
    return fake


def test_a_case_without_seed_documents_does_nothing(registry):
    assert run_eval.seed_corpus({"id": "x"}) is None
    assert registry.calls == []


def test_seed_documents_are_indexed_before_the_case_runs(registry):
    result = run_eval.seed_corpus(
        {
            "id": "x",
            "seed_documents": ["one", "two"],
            "seed_source": "eval-fixture",
        }
    )

    assert registry.calls == [
        ("index_documents", {"texts": ["one", "two"], "source": "eval-fixture"})
    ]
    assert "2 document(s)" in result


def test_seed_source_defaults_so_fixtures_stay_identifiable(registry):
    run_eval.seed_corpus({"id": "x", "seed_documents": ["one"]})

    _, arguments = registry.calls[0]
    assert arguments["source"] == "eval-fixture"


def test_seeding_fails_loudly_when_the_retrieval_agent_is_missing(monkeypatch):
    """
    Silence here would be the original bug again.

    If seeding quietly no-ops, the case runs against whatever the corpus
    happens to hold and reports a score as though the fixture were present.
    """
    monkeypatch.setattr(run_eval, "MCPToolRegistry", lambda: FakeRegistry(tools=()))

    with pytest.raises(RuntimeError, match="not reachable"):
        run_eval.seed_corpus({"id": "x", "seed_documents": ["one"]})


def test_a_seeding_failure_is_reported_as_a_case_error(monkeypatch):
    """run_case must not let a broken fixture look like a model failure."""
    monkeypatch.setattr(run_eval, "MCPToolRegistry", lambda: FakeRegistry(tools=()))

    result = run_eval.run_case({"id": "x", "seed_documents": ["one"], "task": "t"})

    assert "seeding failed" in result["error"]


# ---------------------------------------------------------------------------
# The case definitions themselves
# ---------------------------------------------------------------------------


def test_no_case_relies_on_an_unenforced_ordering_note():
    """
    The specific regression: a dependency expressed only as prose.

    cached-retrieval declared its dependency in a "note" field that the harness
    never read. If a case needs the corpus primed, it must say so in
    seed_documents, where something actually acts on it.
    """
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    for case in cases:
        note = (case.get("note") or "").lower()
        assert "run after" not in note, (
            f"{case['id']} declares an ordering dependency in prose; "
            f"use seed_documents instead"
        )


def test_the_retrieval_case_seeds_what_it_asks_about():
    """A retrieval case whose fixture omits the answer measures nothing."""
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    case = next(c for c in cases if c["id"] == "cached-retrieval")

    assert case.get("seed_documents"), "cached-retrieval must seed its own corpus"

    seeded = " ".join(case["seed_documents"]).lower()
    for keyword in case["must_contain"]:
        assert keyword.lower() in seeded, (
            f"the case requires {keyword!r} in the answer but never seeds it"
        )


# ---------------------------------------------------------------------------
# The signals that catch a confidently wrong answer
# ---------------------------------------------------------------------------


def test_a_forbidden_phrase_is_reported():
    """
    The regression guard. "No issues were found in the provided code" was a
    fluent, confident, wrong answer that every signal of the day passed.
    """
    case = {"must_contain": [], "must_not_contain": ["no issues were found"]}

    signals = run_eval.check_automated_signals(
        case, "No issues were found in the provided code.", ["analyze_code"]
    )

    assert signals["forbidden_phrases"] == ["no issues were found"]


def test_forbidden_matching_ignores_case():
    case = {"must_contain": [], "must_not_contain": ["NO ISSUES"]}

    signals = run_eval.check_automated_signals(case, "no issues here", [])

    assert signals["forbidden_phrases"] == ["NO ISSUES"]


def test_a_clean_answer_has_no_forbidden_phrases():
    case = {"must_contain": ["zero"], "must_not_contain": ["no issues found"]}

    signals = run_eval.check_automated_signals(
        case, "It divides by b without checking for zero.", ["analyze_code"]
    )

    assert signals["forbidden_phrases"] == []
    assert signals["keyword_match"] is True


def test_cases_without_the_key_are_unaffected():
    """Existing cases must keep working untouched."""
    signals = run_eval.check_automated_signals(
        {"must_contain": []}, "anything at all", []
    )

    assert signals["forbidden_phrases"] == []


# ---------------------------------------------------------------------------
# The case definitions
# ---------------------------------------------------------------------------


def test_the_regression_case_forbids_the_sentence_that_caused_it():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    case = next(c for c in cases if c["id"] == "code-review-finds-a-real-bug")

    forbidden = " ".join(case["must_not_contain"]).lower()
    assert "no issues" in forbidden, (
        "the case meant to guard the false-negative regression does not forbid it"
    )


def test_the_ignorance_case_budgets_unsupported_claims():
    """
    Grounding cannot discriminate here - the rubric scores honest ignorance as
    a 5 - so without a claim budget this case would measure nothing.
    """
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    case = next(c for c in cases if c["id"] == "honest-ignorance")

    assert case.get("max_unsupported_claims") == 0


def test_every_case_declares_why_it_exists():
    """A case whose purpose is not written down cannot be judged stale later."""
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    missing = [c["id"] for c in cases if not c.get("why")]
    assert missing == [], f"cases without a stated purpose: {missing}"
