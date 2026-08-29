"""
Tests for the restricted expression evaluator.

The refusal tests carry the weight. A whitelist that rejects `import os` proves
almost nothing - that is the easy case. What decides whether this is safe is the
expressions built entirely from ALLOWED pieces: attribute chains that reach
object internals, whitelisted operators that never return, and whitelisted
operators that allocate a gigabyte.

Anything that hangs or exhausts memory here fails as a timeout rather than an
assertion, so these are written to complete instantly when the guard works.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "code_analysis_agent"))

from evaluator import (  # noqa: E402
    MAX_POW_EXPONENT,
    MAX_SEQUENCE_LEN,
    EvaluationError,
    evaluate,
    report,
)


# ---------------------------------------------------------------------------
# It has to actually work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("1 + 1", 2),
        ("2 * (3 + 4)", 14),
        ("10 / 4", 2.5),
        ("10 // 4", 2),
        ("10 % 3", 1),
        ("2 ** 10", 1024),
        ("-5", -5),
        ("not False", True),
        ("1 < 2 < 3", True),
        ("1 < 2 > 5", False),
        ("True and False", False),
        ("True or False", True),
        ("'a' + 'b'", "ab"),
        ("'ab' * 3", "ababab"),
        ("[1, 2, 3][1]", 2),
        ("[1, 2, 3, 4][1:3]", [2, 3]),
        ("{'a': 1}['a']", 1),
        ("(1, 2)[0]", 1),
        ("5 if 1 < 2 else 9", 5),
        ("len([1, 2, 3])", 3),
        ("sum([1, 2, 3])", 6),
        ("max(3, 7)", 7),
        ("min([4, 2, 9])", 2),
        ("abs(-3)", 3),
        ("round(2.567, 2)", 2.57),
        ("sorted([3, 1, 2])", [1, 2, 3]),
        ("int('42')", 42),
        ("str(42)", "42"),
        ("divmod(7, 2)", (3, 1)),
    ],
)
def test_ordinary_expressions_evaluate(expression, expected):
    assert evaluate(expression) == expected


def test_short_circuit_means_the_dead_branch_never_runs():
    """`and` must not evaluate the right side, or a guarded expression breaks."""
    assert evaluate("False and (1 / 0)") is False
    assert evaluate("True or (1 / 0)") is True


# ---------------------------------------------------------------------------
# The escape that a naive whitelist misses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "().__class__",
        "().__class__.__bases__[0].__subclasses__()",
        "''.join",
        "(1).bit_length()",
        "[].append",
    ],
)
def test_attribute_access_is_refused(expression):
    """
    The classic sandbox escape is a chain of attributes from a harmless literal
    to something that can open a file. Rejecting Attribute outright removes the
    entire class rather than trying to blacklist the destinations.
    """
    with pytest.raises(EvaluationError, match="attribute access"):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression, reason",
    [
        ("__import__('os')", "not one of the allowed"),
        ("open('/etc/passwd')", "not one of the allowed"),
        ("eval('1')", "not one of the allowed"),
        ("exec('x=1')", "not one of the allowed"),
        ("globals()", "not one of the allowed"),
        ("vars()", "not one of the allowed"),
        ("getattr(1, 'real')", "not one of the allowed"),
    ],
)
def test_dangerous_builtins_are_not_reachable(expression, reason):
    with pytest.raises(EvaluationError, match=reason):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "lambda: 1",
        "[x for x in [1, 2]]",
        "{x for x in [1]}",
        "{x: 1 for x in [1]}",
        "(x for x in [1])",
        "f'{1}'",
        "(y := 2)",
    ],
)
def test_constructs_that_could_loop_or_bind_are_refused(expression):
    """
    Comprehensions loop, lambdas defer, walrus binds. None are needed to
    evaluate an arithmetic expression, and each would need its own budget.
    """
    with pytest.raises(EvaluationError):
        evaluate(expression)


def test_statements_are_rejected_by_the_parser():
    """
    Parsing in eval mode means assignment, loops, imports and definitions are
    not merely un-whitelisted - they are not expressions at all.
    """
    for statement in ["import os", "x = 1", "def f(): pass", "while True: pass"]:
        with pytest.raises(EvaluationError, match="syntax error"):
            evaluate(statement)


# ---------------------------------------------------------------------------
# Whitelisted operators that are still weapons
# ---------------------------------------------------------------------------


def test_an_exponent_bomb_is_refused_before_it_runs():
    """
    `10 ** 10 ** 10` uses nothing but allowed operators and never returns. The
    check has to happen BEFORE the operation - there is no result to inspect
    afterwards.
    """
    started = time.monotonic()
    with pytest.raises(EvaluationError, match="exponent"):
        evaluate("10 ** 10 ** 10")
    assert time.monotonic() - started < 1.0, "the guard did not fire before the work"


def test_a_string_repetition_bomb_is_refused_before_it_allocates():
    started = time.monotonic()
    with pytest.raises(EvaluationError, match="sequence longer than"):
        evaluate("'a' * 1000000000")
    assert time.monotonic() - started < 1.0


def test_a_list_repetition_bomb_is_refused():
    with pytest.raises(EvaluationError, match="sequence longer than"):
        evaluate("[0] * 1000000000")


def test_a_large_but_legal_exponent_still_works():
    """The guard must bound the bomb without banning arithmetic."""
    assert evaluate(f"2 ** {MAX_POW_EXPONENT}") == 2**MAX_POW_EXPONENT


def test_growth_by_repeated_multiplication_is_bounded():
    """
    No single operation here is suspicious; the value grows across several. The
    size check is on results for exactly this reason.
    """
    with pytest.raises(EvaluationError):
        evaluate("('a' * 10000) * 10000")


def test_a_sequence_at_the_limit_is_allowed():
    assert len(evaluate(f"'a' * {MAX_SEQUENCE_LEN}")) == MAX_SEQUENCE_LEN


# ---------------------------------------------------------------------------
# Ordinary errors stay ordinary
# ---------------------------------------------------------------------------


def test_division_by_zero_is_reported_not_raised():
    with pytest.raises(EvaluationError, match="division by zero"):
        evaluate("1 / 0")


def test_an_index_error_is_reported_readably():
    with pytest.raises(EvaluationError, match="IndexError"):
        evaluate("[1, 2][9]")


@pytest.mark.parametrize("blank", ["", "   "])
def test_an_empty_expression_is_refused(blank):
    with pytest.raises(EvaluationError, match="no expression"):
        evaluate(blank)


def test_an_undefined_name_says_so():
    with pytest.raises(EvaluationError, match="not defined here"):
        evaluate("some_variable + 1")


# ---------------------------------------------------------------------------
# The report, which is what the model reads
# ---------------------------------------------------------------------------


def test_a_result_states_what_the_evaluator_is():
    """
    Same discipline as analyze_code: say what was done, so the model does not
    infer that arbitrary code ran.
    """
    text = report("2 + 2")

    assert "= 4" in text
    assert "restricted expression subset" in text
    assert "no imports" in text


def test_a_refusal_says_what_is_available_instead():
    """A bare refusal invites the model to retry the same thing."""
    text = report("__import__('os')")

    assert "Refused:" in text
    assert "arithmetic on numbers" in text
    assert "not arbitrary code" in text
