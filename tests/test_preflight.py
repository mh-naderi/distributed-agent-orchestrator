"""
The pre-flight check: parse Windows' reserved port ranges, and decide go / warn /
refuse from commit charge and those ranges.

Only the pure logic is tested, and on any OS. The probes read Windows APIs and
netsh, so they were checked by hand against an independent reading instead:
the script and Win32_OperatingSystem both reported 24.9 of 31.4 GB committed on
2026-09-21.
"""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "preflight", Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
)
preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preflight)

GB = 2**30

# Real output, 2026-09-14 - the morning Ollama's default port 11434 sat inside
# 11344-11443. Trimmed, with the banner, the dashes, an administered range and
# the footnote kept, because those are what a parser gets wrong.
NETSH_2026_09_14 = """
Protocol tcp Port Exclusion Ranges

Start Port    End Port
----------    --------
      1036        1135
      2869        2869
     11244       11343
     11344       11443
     11444       11543
     28385       28385
     50000       50059     *

* - Administered port exclusions.
"""


def memory(percent: float) -> "preflight.Memory":
    limit = 32 * GB
    return preflight.Memory(
        commit_used=int(limit * percent / 100), commit_limit=limit, physical_free=2 * GB
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_only_range_lines_are_read():
    ranges = preflight.parse_excluded_ranges(NETSH_2026_09_14)

    assert ranges == [
        (1036, 1135),
        (2869, 2869),
        (11244, 11343),
        (11344, 11443),
        (11444, 11543),
        (28385, 28385),
        (50000, 50059),
    ]


def test_an_administered_range_blocks_a_bind_like_any_other():
    """The trailing * is a label, not an exemption."""
    ranges = preflight.parse_excluded_ranges(NETSH_2026_09_14)

    assert preflight.reserved([50010], ranges) == [50010]


def test_the_morning_ollamas_default_port_was_taken():
    ranges = preflight.parse_excluded_ranges(NETSH_2026_09_14)

    assert preflight.reserved([11434, 18434], ranges) == [11434]


@pytest.mark.parametrize("port, blocked", [(2868, False), (2869, True), (2870, False)])
def test_range_bounds_are_inclusive(port, blocked):
    """A one-port range is common in real output, and an off-by-one misses it."""
    ranges = preflight.parse_excluded_ranges(NETSH_2026_09_14)

    assert bool(preflight.reserved([port], ranges)) is blocked


# ---------------------------------------------------------------------------
# Deciding
# ---------------------------------------------------------------------------


def test_a_normal_day_is_a_go():
    """79-81% is where this machine sits with the whole stack running."""
    status, lines = preflight.assess(memory(80), ranges=[])

    assert status == "ok"
    assert all(line.startswith("ok") for line in lines)


def test_tight_memory_warns_but_does_not_stop_anything():
    status, lines = preflight.assess(memory(87), ranges=[])

    assert status == "warn"
    assert any(line.startswith("WARN") and "memory" in line for line in lines)


def test_the_morning_of_2026_09_20_is_refused():
    """96% of the commit limit with nothing of this project running."""
    status, lines = preflight.assess(memory(96), ranges=[])

    assert status == "refuse"
    assert any(line.startswith("REFUSE") and "memory" in line for line in lines)


def test_every_memory_line_carries_its_numbers():
    """'Memory is tight' is not something anybody can check."""
    _, lines = preflight.assess(memory(87), ranges=[])
    line = next(l for l in lines if "memory" in l)

    assert "of 32.0 GB" in line and "(87%)" in line and "physical free" in line


@pytest.mark.parametrize("port", sorted(preflight.FIXED_PORTS))
def test_a_reserved_port_baked_into_the_node_refuses_even_with_memory_to_spare(port):
    """
    The API server port was taken overnight once, and a container's port
    mappings cannot be changed - the cluster could not start at all.
    """
    status, lines = preflight.assess(memory(50), ranges=[(port - 5, port + 5)])

    assert status == "refuse"
    assert any(line.startswith("REFUSE") and str(port) in line for line in lines)


@pytest.mark.parametrize("port", sorted(preflight.FLEXIBLE_PORTS))
def test_a_reserved_port_that_can_move_only_warns(port):
    status, lines = preflight.assess(memory(50), ranges=[(port, port)])

    assert status == "warn"
    assert any(line.startswith("WARN") and str(port) in line for line in lines)


def test_the_worst_finding_wins():
    """A warning found after a refusal must not downgrade it."""
    fixed = min(preflight.FIXED_PORTS)
    flexible = min(preflight.FLEXIBLE_PORTS)

    status, _ = preflight.assess(memory(87), ranges=[(fixed, fixed), (flexible, flexible)])

    assert status == "refuse"


def test_a_clean_port_check_says_how_much_it_checked():
    """A bare 'ok' would read the same whether it looked at nine ports or none."""
    _, lines = preflight.assess(memory(50), ranges=[(1, 2), (3, 4)])
    line = next(l for l in lines if "ports" in l)

    assert "none of the 9" in line and "2 reserved range(s)" in line


def test_only_a_refusal_fails_the_exit_code(monkeypatch):
    """Warnings are advice; exit 1 is reserved for 'do not start'."""
    monkeypatch.setattr(preflight.sys, "platform", "win32")
    monkeypatch.setattr(preflight, "read_reserved_ranges", lambda: [])

    monkeypatch.setattr(preflight, "read_memory", lambda: memory(87))
    assert preflight.main([]) == 0

    monkeypatch.setattr(preflight, "read_memory", lambda: memory(96))
    assert preflight.main([]) == 1


def test_another_os_says_nothing_was_checked(monkeypatch, capsys):
    monkeypatch.setattr(preflight.sys, "platform", "linux")

    assert preflight.main([]) == 0
    assert "nothing was checked" in capsys.readouterr().out
