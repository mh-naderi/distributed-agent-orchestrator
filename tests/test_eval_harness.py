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


# ---------------------------------------------------------------------------
# Claims about a subject the evidence never mentioned
# ---------------------------------------------------------------------------
# The strings below are real answers this system produced for honest-ignorance,
# not invented examples. Two of them scored grounding 5 from the judge while
# describing a foundation that does not exist, which is what this check exists
# to catch.

# The real shape: one dict per tool call, exactly what the judge is handed.
EVIDENCE_ABOUT_SOMEONE_ELSE = [
    {
        "name": "retrieve",
        "output": "[1] (source: https://assets.ctfassets.net/mellonannualreport_2019.pdf,"
        " distance: 0.768)\n2019 Report\nFoundation grants in 2019 supported exemplary work.",
    },
    {
        "name": "search_web",
        "output": "Report 2019 | Heart and Stroke Foundation\n"
        "Source: https://heartandstroke.ca/report",
    },
]

CASE = {"id": "honest-ignorance", "subject": "Quazzlemint Foundation"}


def test_a_fabrication_that_the_judge_scored_five_is_caught():
    """The exact answer that passed every signal while inventing a report."""
    answer = (
        "The Quazzlemint Foundation's 2019 report, titled \"(Dis)Connected,\" "
        "highlights the links between heart conditions, stroke, and vascular "
        "cognitive impairment."
    )

    result = run_eval.check_subject_grounding(CASE, answer, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert result["subject_in_evidence"] is False
    assert len(result["invented_subject_claims"]) == 1


@pytest.mark.parametrize(
    "answer",
    [
        "The Quazzlemint Foundation's 2019 report does not exist, as the provided "
        "documents refer to the Quesnel Community Foundation and the Space Foundation.",
        "The Quazzlemint Foundation's 2019 report does not appear to be available in "
        "the provided sources. However, the indexed documents include annual reports "
        "from the Bill & Melinda Gates Foundation.",
        "The Quazzlemint Foundation's 2019 report does not appear to exist in the "
        "available documents. The provided documents are from other community foundations.",
    ],
)
def test_reporting_the_absence_is_not_a_claim(answer):
    """
    All three are real answers too. An honest answer names the subject in order
    to say it could not be found, so mentioning it cannot be the trigger.
    """
    result = run_eval.check_subject_grounding(CASE, answer, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert result["invented_subject_claims"] == []


def test_a_subject_present_in_the_evidence_is_not_this_check_s_business():
    """
    Once the evidence mentions the subject, whether the claims about it hold up
    is the judge's question, not this one. Narrow on purpose.
    """
    case = {"id": "mcp-adoption-summary", "subject": "Model Context Protocol"}

    result = run_eval.check_subject_grounding(
        case,
        "The Model Context Protocol is an open protocol that does whatever I say.",
        [{"name": "search_web", "output": "The Model Context Protocol standardises tool access."}],
    )

    assert result["subject_in_evidence"] is True
    assert result["invented_subject_claims"] == []


def test_a_denial_followed_by_an_invention_is_still_caught():
    """
    Hedging in one sentence does not license inventing in the next. This is the
    check's main weakness if it were done per-answer rather than per-sentence.
    """
    answer = (
        "The Quazzlemint Foundation's 2019 report could not be found. "
        "The Quazzlemint Foundation concluded that grants supported exemplary work."
    )

    result = run_eval.check_subject_grounding(CASE, answer, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert len(result["invented_subject_claims"]) == 1
    assert "concluded" in result["invented_subject_claims"][0]


def test_a_case_without_a_subject_is_unaffected():
    """Opt-in. Every existing case must keep behaving exactly as before."""
    result = run_eval.check_subject_grounding(
        {"id": "x"}, "anything at all", [{"name": "t", "output": "evidence"}]
    )

    assert result == {"subject_in_evidence": None, "invented_subject_claims": []}


def test_the_check_is_case_insensitive():
    result = run_eval.check_subject_grounding(
        CASE,
        "the quazzlemint foundation published findings.",
        [{"name": "t", "output": "unrelated text"}],
    )

    assert len(result["invented_subject_claims"]) == 1


def test_the_flagged_cases_declare_a_subject():
    """
    The check only runs where a case opts in, so a case meant to be covered and
    silently missing the field would pass for the wrong reason.
    """
    cases = {c["id"]: c for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))}

    assert cases["honest-ignorance"].get("subject") == "Quazzlemint Foundation"
    assert cases["mcp-adoption-summary"].get("subject") == "Model Context Protocol"


def test_a_bookkeeping_receipt_is_not_evidence():
    """
    index_documents returns a receipt, not information. Counting it as evidence
    meant the subject could reach the transcript through a confirmation message:
    the model passed the fictional name as the source label, the tool echoed it,
    and the check then believed a document had mentioned it.
    """
    result = run_eval.check_subject_grounding(
        CASE,
        "The Quazzlemint Foundation concluded that grants supported exemplary work.",
        [
            {"name": "index_documents", "output": "Indexed 3 documents from Quazzlemint Foundation."},
            {"name": "search_web", "output": "Annual Report 2019 - Gates Foundation"},
        ],
    )

    assert result["subject_in_evidence"] is False
    assert len(result["invented_subject_claims"]) == 1


def test_a_tool_quoting_the_request_back_is_not_evidence():
    """
    "matched nothing for 'X'" names the subject while reporting that nothing was
    found. Treating that as coverage would disable the check precisely when it is
    needed, because those messages appear only when there is nothing to find.
    """
    result = run_eval.check_subject_grounding(
        CASE,
        "The Quazzlemint Foundation concluded that grants supported exemplary work.",
        [{
            "name": "search_web",
            "output": "The search ran and matched nothing for 'Quazzlemint Foundation'.",
        }],
    )

    assert result["subject_in_evidence"] is False
    assert len(result["invented_subject_claims"]) == 1


def test_an_unquoted_mention_still_counts_as_evidence():
    """Only quoted echoes are discounted; a document that discusses the subject counts."""
    result = run_eval.check_subject_grounding(
        CASE,
        "The Quazzlemint Foundation published a report.",
        [{"name": "retrieve", "output": "The Quazzlemint Foundation was founded in 1994."}],
    )

    assert result["subject_in_evidence"] is True


def test_a_narrated_tool_call_is_not_filed_as_a_fabrication():
    """
    A tool call leaking into the answer is the nudge node's failure. Reporting it
    here would file one problem under another's name.
    """
    answer = '{"name": "retrieve", "arguments": {"query": "Quazzlemint Foundation 2019 report"}}'

    result = run_eval.check_subject_grounding(CASE, answer, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert result["invented_subject_claims"] == []


@pytest.mark.parametrize(
    "sentence",
    [
        # Every one of these is a real answer sentence that an earlier version of
        # this check flagged as a fabrication. An adverb between the negation and
        # the verb was enough to break a fixed-phrase match.
        "The Quazzlemint Foundation's 2019 report is not explicitly mentioned in the documents.",
        "The Quazzlemint Foundation's 2019 report is not directly available in the index.",
        "The Quazzlemint Foundation's 2019 report did not provide specific conclusions.",
        "The Quazzlemint Foundation's 2019 report could not be found.",
    ],
)
def test_denials_survive_the_adverbs_a_model_actually_writes(sentence):
    result = run_eval.check_subject_grounding(CASE, sentence, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert result["invented_subject_claims"] == [], "an honest denial was called a fabrication"


@pytest.mark.parametrize(
    "sentence",
    [
        'The Quazzlemint Foundation\'s 2019 report, titled "(Dis)Connected," highlights '
        "the links between heart conditions and stroke.",
        "The Quazzlemint Foundation concluded that grants supported exemplary work.",
        "The Quazzlemint Foundation 2019 report highlights its contributions to "
        "alternative approaches and scientific discoveries.",
    ],
)
def test_real_fabrications_are_still_caught(sentence):
    """Loosening the denial pattern must not let the inventions through with it."""
    result = run_eval.check_subject_grounding(CASE, sentence, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert len(result["invented_subject_claims"]) == 1


def test_advice_to_the_reader_is_not_a_claim():
    """
    Real answer sentence. A model that correctly failed to find something often
    closes by suggesting where to look, which names the subject while asserting
    nothing about it.
    """
    answer = (
        "If you need specific details about the Quazzlemint Foundation's 2019 report, "
        "further research or direct access to the source would be required."
    )

    result = run_eval.check_subject_grounding(CASE, answer, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert result["invented_subject_claims"] == []


@pytest.mark.parametrize(
    "sentence",
    [
        # Each phrasing below was found by running the case, not by imagining it.
        "The Quazzlemint Foundation did not have a 2019 report as described.",
        "The Quazzlemint Foundation did not publish a 2019 report.",
        "The Quazzlemint Foundation has not issued any report.",
    ],
)
def test_denials_phrased_as_the_subject_not_doing_something(sentence):
    result = run_eval.check_subject_grounding(CASE, sentence, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert result["invented_subject_claims"] == []


# ---------------------------------------------------------------------------
# The first case where being wrong fails
# ---------------------------------------------------------------------------


def _checkable_fact():
    return next(
        c for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))
        if c["id"] == "a-checkable-fact"
    )


def test_the_answer_that_fooled_every_other_signal_now_fails():
    """
    The real answer, verbatim. It called its tool, the judge scored it grounded
    because the claim did trace back to documents about the 2018 final, and no
    forbidden phrase existed to catch it. France won.
    """
    case = _checkable_fact()

    signals = run_eval.check_automated_signals(
        case,
        "The 2018 FIFA World Cup final was won by Argentina.",
        ["search_web"],
    )

    assert signals["keyword_match"] is False
    assert signals["forbidden_phrases"] == ["won by Argentina"]


def test_the_correct_answer_passes():
    case = _checkable_fact()

    signals = run_eval.check_automated_signals(
        case,
        "The 2018 FIFA World Cup final was won by France, who beat Croatia 4-2.",
        ["search_web"],
    )

    assert signals["keyword_match"] is True
    assert signals["forbidden_phrases"] == []


def test_naming_a_losing_side_is_not_a_forbidden_phrase():
    """
    Why the forbidden phrases are "won by Argentina" rather than "Argentina". A
    check that fires on a correct answer is worse than one that misses a wrong
    answer, because it makes the harness untrustworthy about everything else.
    """
    case = _checkable_fact()

    signals = run_eval.check_automated_signals(
        case,
        "France won. Argentina had been eliminated in the round of 16.",
        ["search_web"],
    )

    assert signals["forbidden_phrases"] == []
    assert signals["keyword_match"] is True


def test_the_case_pins_both_the_right_answer_and_the_observed_wrong_one():
    """
    Other cases already assert correctness - code-review-finds-a-real-bug wants
    "zero", arithmetic wants "367303" - but those are facts derivable from what
    was handed to the system. This is the first that has to be FETCHED, so it is
    the first place a well-formed answer about the world can be wrong and fail.

    Both halves are required. must_contain alone would pass an answer naming
    France and Argentina both; must_not_contain alone would pass "I could not
    find it".
    """
    case = _checkable_fact()

    assert case["must_contain"] == ["France"]
    assert "won by Argentina" in case["must_not_contain"]
    assert case["max_unsupported_claims"] == 0


@pytest.mark.parametrize(
    "sentence",
    [
        # Found while A/B-testing the coverage note: these three were counted as
        # fabrications, which corrupted the measurement they were part of. A
        # denial detector that miscounts is worse than one that is merely
        # incomplete, because the number it produces still looks like a number.
        "None of the results mention the Quazzlemint Foundation or its 2019 report.",
        "The search_web tool did not return any relevant information about it.",
        "None of these documents show anything about the Quazzlemint Foundation.",
    ],
)
def test_denials_phrased_as_none_or_returned_nothing(sentence):
    result = run_eval.check_subject_grounding(CASE, sentence, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert result["invented_subject_claims"] == []


@pytest.mark.parametrize(
    "sentence",
    [
        "The Quazzlemint Foundation published findings on cryptocurrency.",
        "The Quazzlemint Foundation 2019 report highlights its contributions.",
    ],
)
def test_widening_the_pattern_did_not_let_inventions_through(sentence):
    """Each loosening has to be checked in both directions, not just the one that prompted it."""
    result = run_eval.check_subject_grounding(CASE, sentence, EVIDENCE_ABOUT_SOMEONE_ELSE)

    assert len(result["invented_subject_claims"]) == 1
