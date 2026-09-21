"""
Check the machine before starting the cluster and local inference.

    .venv/Scripts/python.exe scripts/preflight.py

Exit 0 means go (possibly with warnings); exit 1 means do not start.

WHY THIS EXISTS. Two things have repeatedly stopped this system before any code
ran, and both were only ever checked by hand when somebody remembered to:

- MEMORY. On 2026-09-20 the machine reached 96% of its commit limit with nothing
  of this project running, and 89% on the 21st. Starting the kind node and
  loading a model adds commit on top. The cluster stopped mid-session on
  2026-09-17 with 2.3 GB of physical memory free; whether pressure caused that
  stop is NOT established - see docs/RUNBOOK.md - but the pressure is measured,
  and the one piece of state here that cannot be rebuilt lives in a sqlite file
  on that node.

- RESERVED PORTS. Windows reserves blocks of TCP ports and redraws them on boot.
  A reserved port cannot be bound by anything. Ollama's default and its fallback
  were both taken on one morning, and the kind API server's port was taken on
  another, which left a cluster that could not start at all.

The check reads COMMIT CHARGE rather than free physical memory. Free physical
memory is mostly a cache the OS will hand back; commit is what a process has
been promised, and a machine at its commit limit refuses allocations regardless
of how much it could page out.
"""

import argparse
import ctypes
import re
import subprocess
import sys
from dataclasses import dataclass

# Chosen, not calibrated - there is one measured failure and it was never shown
# to be caused by memory. The observed working range with the whole stack up is
# 79-81% of the commit limit, and starting the cluster plus the model's first
# load added about 2 points on each of the last two starts. 85% leaves a margin
# above normal; 90% is where a start would land within reach of the 96% seen
# on 2026-09-20.
WARN_PERCENT = 85.0
REFUSE_PERCENT = 90.0

# Ports baked into the kind node container when it was created. A container's
# port mappings cannot be changed afterwards, so if either is reserved the
# cluster CANNOT start - this is a refusal, not advice.
FIXED_PORTS = {
    18443: "kind API server (kind-cluster.yaml apiServerPort)",
    18080: "ingress host port (kind-cluster.yaml extraPortMappings)",
}

# Ports this project binds on the host at runtime. Any of these can be moved,
# so a reservation is a warning with the fix in it rather than a refusal.
FLEXIBLE_PORTS = {
    18434: "Ollama (OLLAMA_HOST)",
    18000: "port-forward: research agent",
    18001: "port-forward: retrieval agent",
    18002: "port-forward: code-analysis agent",
    19090: "port-forward: Prometheus",
    19101: "port-forward: retrieval agent metrics",
    13000: "port-forward: Grafana",
}


@dataclass
class Memory:
    commit_used: int
    commit_limit: int
    physical_free: int

    @property
    def commit_percent(self) -> float:
        return 100.0 * self.commit_used / self.commit_limit


# ---------------------------------------------------------------------------
# Pure logic - tested on any OS
# ---------------------------------------------------------------------------

_RANGE_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s*\*?\s*$")


def parse_excluded_ranges(text: str) -> list[tuple[int, int]]:
    """
    The start-end pairs from `netsh interface ipv4 show excludedportrange`.

    Only lines that are exactly two port numbers count. The header, the dashes
    and the footnote are all skipped by shape rather than by position, so a
    change in netsh's banner text cannot shift what is read as a range. A
    trailing * marks an administered exclusion, which blocks a bind just the
    same.
    """
    ranges = []
    for line in text.splitlines():
        match = _RANGE_LINE.match(line)
        if match:
            ranges.append((int(match.group(1)), int(match.group(2))))
    return ranges


def reserved(ports, ranges: list[tuple[int, int]]) -> list[int]:
    """Which of these ports fall inside a reserved range. Bounds are inclusive."""
    return sorted(p for p in ports if any(lo <= p <= hi for lo, hi in ranges))


def assess(
    memory: Memory,
    ranges: list[tuple[int, int]],
    warn_percent: float = WARN_PERCENT,
    refuse_percent: float = REFUSE_PERCENT,
) -> tuple[str, list[str]]:
    """
    Decide go / warn / refuse, and say why in lines a person can act on.

    Every line carries its own numbers. "Memory is tight" is not something
    anybody can check; "commit 27.9 of 31.4 GB (89%)" is.
    """
    lines = []
    status = "ok"

    def worst(new):
        order = {"ok": 0, "warn": 1, "refuse": 2}
        return new if order[new] > order[status] else status

    pct = memory.commit_percent
    figures = (
        f"commit {memory.commit_used / 2**30:.1f} of {memory.commit_limit / 2**30:.1f} GB "
        f"({pct:.0f}%), {memory.physical_free / 2**30:.1f} GB physical free"
    )
    if pct >= refuse_percent:
        status = worst("refuse")
        lines.append(
            f"REFUSE  memory: {figures}. At or above {refuse_percent:.0f}%. Close "
            "applications - the WSL VM, other VMs and browsers are the usual "
            "weight - and re-run this."
        )
    elif pct >= warn_percent:
        status = worst("warn")
        lines.append(
            f"WARN    memory: {figures}. Above {warn_percent:.0f}%; starting the "
            "cluster and a model adds about 2 points. Consider closing something first."
        )
    else:
        lines.append(f"ok      memory: {figures}.")

    blocked_fixed = reserved(FIXED_PORTS, ranges)
    for port in blocked_fixed:
        status = worst("refuse")
        lines.append(
            f"REFUSE  port {port} is reserved by Windows - {FIXED_PORTS[port]}. The "
            "cluster cannot start: this port is baked into the node container. See "
            "'Windows reserves TCP ports' in docs/RUNBOOK.md."
        )

    blocked_flexible = reserved(FLEXIBLE_PORTS, ranges)
    for port in blocked_flexible:
        status = worst("warn")
        lines.append(
            f"WARN    port {port} is reserved by Windows - {FLEXIBLE_PORTS[port]}. "
            "Pick a free port for it before starting."
        )

    if not blocked_fixed and not blocked_flexible:
        checked = len(FIXED_PORTS) + len(FLEXIBLE_PORTS)
        lines.append(
            f"ok      ports: none of the {checked} this project binds is reserved "
            f"({len(ranges)} reserved range(s) on this boot)."
        )

    return status, lines


# ---------------------------------------------------------------------------
# Probes - Windows only
# ---------------------------------------------------------------------------


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def read_memory() -> Memory:
    """
    Commit charge and limit from GlobalMemoryStatusEx.

    Despite the name, ullTotalPageFile is the system COMMIT LIMIT (physical
    memory plus page files), and ullAvailPageFile is how much of it is still
    unpromised - the same figures Task Manager shows as "Committed".
    """
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return Memory(
        commit_used=status.ullTotalPageFile - status.ullAvailPageFile,
        commit_limit=status.ullTotalPageFile,
        physical_free=status.ullAvailPhys,
    )


def read_reserved_ranges() -> list[tuple[int, int]]:
    result = subprocess.run(
        ["netsh", "interface", "ipv4", "show", "excludedportrange", "protocol=tcp"],
        capture_output=True, text=True, check=True,
    )
    return parse_excluded_ranges(result.stdout)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--warn-percent", type=float, default=WARN_PERCENT)
    parser.add_argument("--refuse-percent", type=float, default=REFUSE_PERCENT)
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        # Nothing here is wrong on another OS - these checks just do not apply.
        # Saying so beats a silent pass that reads as "checked and fine".
        print("preflight: these checks are Windows-specific; nothing was checked.")
        return 0

    status, lines = assess(
        read_memory(), read_reserved_ranges(), args.warn_percent, args.refuse_percent
    )
    for line in lines:
        print(line)
    verdict = {"ok": "GO", "warn": "GO, with warnings above", "refuse": "DO NOT START"}[status]
    print(f"\npreflight: {verdict}")
    return 1 if status == "refuse" else 0


if __name__ == "__main__":
    sys.exit(main())
