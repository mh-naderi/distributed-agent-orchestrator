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
