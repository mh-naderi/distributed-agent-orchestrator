"""
Tests for the measurement harness.

Only the reduction is tested, not the running: what makes a measurement
trustworthy is that the numbers it prints follow from the runs it did, and that
is the part which can be checked without a model. The rest is the whole system,
which the eval suite already exercises.
"""

import pytest

from eval import experiment


def run(tools, invented=None, seconds=1.0):
    return {"tools": tools, "invented": invented or [], "seconds": seconds}


def test_fabrications_are_counted_not_averaged():
    summary = experiment.summarise(
        [run(["retrieve"]), run(["retrieve"], ["it concluded X"]), run(["retrieve"])]
    )

    assert summary["runs"] == 3
    assert summary["fabricated"] == 1


def test_the_path_taken_is_reported():
    """
    Which route the loop took is often the finding rather than a detail. A change
    that stops the model calling search_web at all looks like a fabrication fix
    if only the totals are read - that happened, and the path counts are what
    made it visible.
    """
    summary = experiment.summarise(
        [
            run(["retrieve"]),
            run(["retrieve"]),
            run(["retrieve", "search_web"]),
            run([]),
        ]
    )

    assert summary["paths"] == {
        "retrieve": 2,
        "retrieve,search_web": 1,
        "(no tools)": 1,
    }


def test_paths_are_ordered_by_how_often_they_happened():
    summary = experiment.summarise([run(["a"]), run(["b"]), run(["b"])])

    assert list(summary["paths"]) == ["b", "a"]


def test_a_run_with_no_tools_is_its_own_path_not_a_missing_one():
    summary = experiment.summarise([run([])])

    assert summary["paths"] == {"(no tools)": 1}
    assert summary["fabricated"] == 0


def test_timing_is_a_median_because_one_slow_run_is_not_the_story():
    summary = experiment.summarise(
        [run(["a"], seconds=5.0), run(["a"], seconds=6.0), run(["a"], seconds=60.0)]
    )

    assert summary["median_seconds"] == 6.0


def test_missing_timings_do_not_break_the_summary():
    summary = experiment.summarise([{"tools": ["a"], "invented": [], "seconds": None}])

    assert summary["median_seconds"] is None


def test_an_unknown_case_exits_rather_than_measuring_nothing():
    """A typo in --case must not produce a confident zero."""
    with pytest.raises(SystemExit):
        experiment._case("no-such-case")

# ---------------------------------------------------------------------------
# A measurement has to say what it is a measurement of
# ---------------------------------------------------------------------------
# Failed runs are dropped rather than counted, so the denominator shrinks
# silently unless the report is told how many were asked for. This is the
# project's own failure mode turning up in the instrument: a result that reads
# as clean because nothing was measured.


def test_a_summary_says_how_many_runs_were_lost(capsys):
    """
    Eight asked for, two completed. Printing "0/2" alone reads as a clean
    result on a small sample rather than a measurement that mostly did not
    happen.
    """
    runs = [
        {"tools": ["retrieve"], "invented": [], "seconds": 1.0},
        {"tools": ["retrieve"], "invented": [], "seconds": 1.0},
    ]
    experiment._report("case", runs, attempted=8)

    out = capsys.readouterr().out
    assert "0/2" in out
    assert "6 of 8 runs failed" in out
    assert "NOT counted" in out


def test_a_summary_with_nothing_lost_stays_quiet(capsys):
    """The caveat must not appear when every run completed."""
    runs = [{"tools": ["retrieve"], "invented": [], "seconds": 1.0}] * 4
    experiment._report("case", runs, attempted=4)

    out = capsys.readouterr().out
    assert "0/4" in out
    assert "failed" not in out


def test_every_run_failing_refuses_to_report_a_result(capsys):
    """
    The case that prompted this. Four repeats against a host where Ollama had
    moved ports printed "fabricated 0/0" - which is not a result, but reads
    like one.
    """
    experiment._report("case", [], attempted=4)

    out = capsys.readouterr().out
    assert "NOTHING MEASURED" in out
    assert "all 4 run(s) failed" in out
    # ...and it must not print a fraction that could be quoted.
    assert "0/0" not in out


def test_a_report_without_an_attempted_count_is_unchanged(capsys):
    """ab() knows every run completed, because it does not catch errors."""
    runs = [{"tools": ["search_web"], "invented": [], "seconds": None}] * 3
    experiment._report("with note", runs)

    out = capsys.readouterr().out
    assert "0/3" in out
    assert "failed" not in out and "NOTHING MEASURED" not in out
