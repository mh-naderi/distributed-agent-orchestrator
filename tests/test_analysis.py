"""
Tests for the code-analysis agent's static analysis.

Pure functions over strings - no MCP, no cluster, no model. That is the payoff
of keeping the logic in analysis.py rather than inside the tool wrapper.

The FALSE POSITIVE tests carry the most weight here. This agent feeds an
orchestrating model that has been instructed to trust what its tools return, so
an invented finding is reported to a person as fact. Missing a real bug is a
disappointment; inventing one is the failure this project already had once, in
the opposite direction, when the stub implied code was clean.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "code_analysis_agent"))

from analysis import CHECK_NAMES, MAX_CODE_BYTES, analyse, report  # noqa: E402


def messages(code: str) -> list[str]:
    return [f.message for f in analyse(code)]


# ---------------------------------------------------------------------------
# Precision: correct code must come back clean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, code",
    [
        (
            "division guarded by an equality check",
            "def d(a, b):\n    if b == 0:\n        raise ValueError\n    return a / b\n",
        ),
        (
            "division guarded by truthiness",
            "def d(a, b):\n    if not b:\n        return None\n    return a / b\n",
        ),
        ("division by a constant", "def half(a):\n    return a / 2\n"),
        (
            "division by a local, not a parameter",
            "def mean(xs):\n    n = len(xs)\n    return sum(xs) / n\n",
        ),
        (
            "ordinary idiomatic code",
            "import os\n\n\ndef get(name):\n    value = os.environ.get(name)\n"
            "    if value is None:\n        return ''\n    return value.strip()\n",
        ),
        ("immutable default argument", "def f(x=()):\n    return x\n"),
        (
            "except that actually handles",
            "def f():\n    try:\n        return 1\n    except ValueError:\n"
            "        return 0\n",
        ),
    ],
)
def test_correct_code_produces_no_findings(label, code):
    assert analyse(code) == [], f"false positive on {label}: {messages(code)}"


# ---------------------------------------------------------------------------
# Recall: the patterns we claim to catch
# ---------------------------------------------------------------------------


def test_the_case_that_started_this():
    """
    The exact snippet the stub called clean.

    Note what the finding says: that the parameter is never compared against
    zero. That is a fact about the code. It deliberately does not assert the
    function "will crash", which depends on inputs this tool never sees.
    """
    found = messages("def divide(a, b): return a / b")

    assert len(found) == 1
    assert "divides by parameter 'b'" in found[0]
    assert "never compared against zero" in found[0]


def test_syntax_error_is_a_result_not_an_exception():
    """
    "Your code does not parse" is a successful analysis.

    It must not raise: the tool wrapper counts raised exceptions as
    tool_calls_total{status="error"}, which would file a fact about the
    submitted code as a failure of this agent.
    """
    found = analyse("def f(:\n")

    assert len(found) == 1
    assert found[0].source == "syntax"
    assert "syntax error" in found[0].message


def test_mutable_default_argument():
    assert any("mutable default argument" in m for m in messages("def f(x=[]):\n    return x\n"))


def test_bare_except_and_silenced_exception():
    found = messages("try:\n    pass\nexcept:\n    pass\n")

    assert any("bare 'except:'" in m for m in found)
    assert any("silently discarded" in m for m in found)


def test_equality_against_none():
    assert any("'is' is the identity test" in m for m in messages("def f(x):\n    return x == None\n"))


def test_unreachable_code():
    assert any("unreachable" in m for m in messages("def f():\n    return 1\n    print('x')\n"))


def test_pyflakes_findings_are_included():
    """The pyflakes layer, which covers names rather than patterns."""
    found = analyse("def f():\n    return undefined_thing\n")

    assert len(found) == 1
    assert found[0].source == "pyflakes"
    assert "undefined name" in found[0].message


def test_no_duplicate_finding_for_assert_on_a_tuple():
    """
    pyflakes already reports this, so there is no AST rule for it.

    An earlier version had both and reported one problem twice, which reads as
    two problems.
    """
    found = analyse("def f(x):\n    assert (x > 0, 'must be positive')\n")

    assert len(found) == 1
    assert found[0].source == "pyflakes"


# ---------------------------------------------------------------------------
# The report, which is what the model actually reads
# ---------------------------------------------------------------------------


def test_a_clean_report_refuses_to_claim_correctness():
    """
    The whole reason this agent was rewritten.

    A clean result that reads as "no issues" invites exactly the answer the
    stub produced. It has to say what it checked and what it cannot know.
    """
    text = report("def add(a, b):\n    return a + b\n")

    assert "No issues found by the checks that were run" in text
    assert "NOT evidence" in text
    for name in CHECK_NAMES:
        assert name in text, f"clean report omits the {name!r} check it claims"


def test_a_report_with_findings_lists_line_numbers():
    text = report("def f(x=[]):\n    return x == None\n")

    assert "Found 2 issues" in text
    assert "line 1" in text and "line 2" in text
    # The disclaimer rides along even when problems were found - a found issue
    # says nothing about what was not looked for.
    assert "static analysis only" in text


@pytest.mark.parametrize("bad", ["", "   \n  "])
def test_empty_input_is_rejected(bad):
    """Raises, because an empty snippet is a caller error, not a finding."""
    with pytest.raises(ValueError, match="no code supplied"):
        analyse(bad)


def test_oversized_input_is_rejected():
    with pytest.raises(ValueError, match="larger than"):
        analyse("x = 1\n" * (MAX_CODE_BYTES // 2))
