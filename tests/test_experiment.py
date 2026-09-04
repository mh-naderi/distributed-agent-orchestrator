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
