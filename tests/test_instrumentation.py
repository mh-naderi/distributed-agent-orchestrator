"""
Tests for tool metrics recorded at the MCP boundary.

The interesting cases here are the calls that never reach a tool function.
Counting inside the tool body looks correct and passes any test that calls the
tool normally - the blind spot only appears when FastMCP rejects the arguments
first, which is why it survived until someone asked the error panel a question
it could not answer.

The code-analysis agent stands in for all three: its tools are pure, so this
needs no network, no Ollama and no cluster.
"""

import asyncio
import sys
from pathlib import Path

import pytest

AGENTS = Path(__file__).resolve().parents[1] / "agents"
sys.path.insert(0, str(AGENTS / "code_analysis_agent"))

import server  # noqa: E402
from instrumentation import TOOL_CALLS, TOOL_LATENCY  # noqa: E402


def calls(tool_name, status):
    """Current value of tool_calls_total for one label pair, 0 if never set."""
    value = TOOL_CALLS.labels(tool_name=tool_name, status=status)._value.get()
    return value or 0.0


def latency_count(tool_name):
    return TOOL_LATENCY.labels(tool_name=tool_name)._sum.get() is not None and sum(
        s.value
        for m in TOOL_LATENCY.collect()
        for s in m.samples
        if s.name.endswith("_count") and s.labels.get("tool_name") == tool_name
    )


def call(name, arguments):
    return asyncio.run(server.mcp.call_tool(name, arguments))


# ---------------------------------------------------------------------------
# The blind spot this replaced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments, why",
    [
        ({"expression": 12345}, "wrong type for a declared parameter"),
        ({}, "required parameter missing"),
    ],
)
def test_a_call_rejected_by_validation_is_counted_as_an_error(arguments, why):
    """
    FastMCP validates against the signature-derived schema before calling the
    function, so none of these reach the tool body. When the counters lived in
    that body they recorded nothing at all - not an error, not even a call.
    """
    before = calls("evaluate_expression", "error")

    with pytest.raises(Exception):
        call("evaluate_expression", arguments)

    assert calls("evaluate_expression", "error") == before + 1, why


def test_an_unexpected_argument_is_accepted_rather_than_rejected():
    """
    Characterising real behaviour, not asserting it is ideal. Pydantic ignores
    fields that are not in the schema, so a model that invents a parameter gets
    a successful call rather than a signal that it guessed. Worth knowing before
    reading the error rate as "how often the model called tools wrongly" - this
    kind of mistake is invisible there by design, not by oversight.
    """
    before = calls("evaluate_expression", "success")

    result = call("evaluate_expression", {"expression": "1 + 1", "nonsense": True})

    assert "= 2" in str(result)
    assert calls("evaluate_expression", "success") == before + 1


def test_an_unknown_tool_name_is_not_used_as_a_label():
    """
    The tool name comes from the client and becomes a Prometheus label. Passing
    an unregistered name straight through would let a confused caller create
    unbounded label cardinality, so it is recorded as "unknown".
    """
    before_unknown = calls("unknown", "error")
    before_named = calls("definitely_not_a_tool", "error")

    with pytest.raises(Exception):
        call("definitely_not_a_tool", {})

    assert calls("unknown", "error") == before_unknown + 1
    assert calls("definitely_not_a_tool", "error") == before_named


# ---------------------------------------------------------------------------
# The ordinary paths still behave
# ---------------------------------------------------------------------------


def test_a_successful_call_is_counted_once():
    """
    Once, not twice. The per-tool version incremented success partway through
    the body and could still fail afterwards, counting one call under both
    labels. Wrapping the whole call makes that unrepresentable.
    """
    before_ok = calls("evaluate_expression", "success")
    before_err = calls("evaluate_expression", "error")

    call("evaluate_expression", {"expression": "2 + 2"})

    assert calls("evaluate_expression", "success") == before_ok + 1
    assert calls("evaluate_expression", "error") == before_err


def test_a_refusal_is_a_successful_call():
    """
    The evaluator refusing `__import__` is the tool doing its job. Counting it
    as an error would make the error rate measure how often the model overreaches
    rather than whether the agent is healthy.
    """
    before = calls("evaluate_expression", "success")

    call("evaluate_expression", {"expression": "__import__('os')"})

    assert calls("evaluate_expression", "success") == before + 1


def test_a_failure_inside_the_tool_is_still_counted(monkeypatch):
    before = calls("analyze_code", "error")

    def boom(code):
        raise RuntimeError("analysis exploded")

    monkeypatch.setattr(server.code_analysis_service, "run", boom)

    with pytest.raises(Exception):
        call("analyze_code", {"code": "x = 1"})

    assert calls("analyze_code", "error") == before + 1


def test_latency_is_observed_even_for_a_rejected_call():
    """A call that fails validation still took time; the histogram should say so."""
    before = latency_count("evaluate_expression")

    with pytest.raises(Exception):
        call("evaluate_expression", {})

    assert latency_count("evaluate_expression") == before + 1


# ---------------------------------------------------------------------------
# The duplication is deliberate, so it has to be verified
# ---------------------------------------------------------------------------


def test_every_agent_carries_an_identical_copy():
    """
    Each image is built from its own agent directory as context, so this module
    is duplicated rather than shared - the same trade already made for
    requirements.txt. Duplication is only acceptable if drift is caught, which
    is what this asserts.
    """
    copies = sorted(AGENTS.glob("*/instrumentation.py"))

    assert len(copies) == 3, f"expected one per agent, found {[str(c) for c in copies]}"

    contents = {c: c.read_bytes() for c in copies}
    first = copies[0]
    for other in copies[1:]:
        assert contents[other] == contents[first], (
            f"{other} has drifted from {first} - copy it across rather than editing one"
        )


def test_each_agent_uses_the_instrumented_server():
    """A copied file that no server actually constructs would pass every test above."""
    for name in ("research_agent", "retrieval_agent", "code_analysis_agent"):
        source = (AGENTS / name / "server.py").read_text(encoding="utf-8")
        assert "InstrumentedMCP(" in source, f"{name} still builds a plain FastMCP"
        assert "TOOL_CALLS.labels" not in source, (
            f"{name} still counts inside a tool body, which cannot see rejected calls"
        )
