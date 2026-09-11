"""
A metric nothing looks at is a metric nobody will look at.

WHY THIS EXISTS. Instrumenting something is cheap and satisfying, and putting
it on a surface is neither, so metrics accumulate faster than places to see
them. Seven of fourteen had ended up charted nowhere, including
orchestrator_regrounds_total - the fabrication guardrail, the single signal this
project is organised around, alerted on and invisible.

That combination is the one worth preventing. An alert says a line was crossed;
the dashboard is where you work out why. MostRunsAnsweredFromNothing could fire
with nothing to open.

The worst case was orchestrator_tools_discovered, which appeared on no surface
at all while metrics.py said of it: discovery is best-effort by design, an
unreachable agent is logged and the run proceeds with a partial toolset, and
"that tolerance is only safe if the degradation is visible somewhere". It was
not. Both discovery alerts fire only on total failure, so losing one agent out
of three showed up nowhere.

Checked by reading, like the alert rules and the agent images: this cannot say
whether a panel is legible, only whether the metric reaches a surface at all.
"""

import ast
import json
import re
from pathlib import Path

import pytest
import yaml
from prometheus_client.utils import floatToGoString

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "k8s" / "grafana.yaml"
ALERTS = ROOT / "k8s" / "prometheus.yaml"

# Metrics deliberately on no surface, and why. Being here is a decision; being
# absent from here and from both surfaces is an oversight, and the test cannot
# tell those apart unless the decision is written down. Empty on purpose - every
# metric this repo declares currently reaches a surface, and the map exists so
# that stays a choice rather than an accident.
EXEMPT_METRICS: dict[str, str] = {}


def dashboard() -> dict:
    docs = [d for d in yaml.safe_load_all(DASHBOARD.read_text(encoding="utf-8")) if d]
    config = next(d for d in docs if d["metadata"]["name"] == "grafana-dashboard")
    return json.loads(config["data"]["agents.json"])


def declared_metrics() -> set[str]:
    """Every Counter, Histogram or Gauge this repo creates."""
    pattern = re.compile(
        r"(?:Counter|Histogram|Gauge|Summary)\(\s*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']"
    )
    found = set()
    for path in list((ROOT / "orchestrator").rglob("*.py")) + list(
        (ROOT / "agents").rglob("*.py")
    ):
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    return found


def charted_metrics() -> set[str]:
    """Metric names appearing in any panel's queries."""
    expressions = " ".join(
        target.get("expr", "")
        for panel in dashboard()["panels"]
        for target in panel.get("targets", [])
    )
    # Histogram panels query the derived _bucket series; the base name is what
    # the code declares.
    names = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", expressions))
    return {re.sub(r"_(bucket|count|sum)$", "", n) for n in names}


def alerted_metrics() -> set[str]:
    return set(re.findall(r"\b[a-z_][a-z0-9_]*\b", ALERTS.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# The check that matters
# ---------------------------------------------------------------------------


def test_every_metric_reaches_a_surface():
    declared = declared_metrics()
    assert declared, "found no metrics; the extractor is broken"

    visible = charted_metrics() | alerted_metrics() | set(EXEMPT_METRICS)
    invisible = declared - visible

    assert not invisible, (
        f"{sorted(invisible)} are declared but appear on no dashboard panel, in "
        "no alert rule, and in no exemption. Instrumenting something and then "
        "never looking at it is how orchestrator_regrounds_total ended up being "
        "the most important signal in the project and charted nowhere."
    )


def test_the_guardrail_signals_are_charted():
    """
    Not merely visible somewhere - on the dashboard specifically. These two are
    what the project is about, they describe different failures, and an alert on
    either is unreadable without the line behind it.
    """
    charted = charted_metrics()
    for metric in ("orchestrator_regrounds_total", "orchestrator_nudges_total"):
        assert metric in charted, f"{metric} is not on the dashboard"


def test_exemptions_name_real_metrics():
    stale = set(EXEMPT_METRICS) - declared_metrics()
    assert not stale, f"EXEMPT_METRICS lists {sorted(stale)}, which nothing declares"


def test_exemptions_are_justified():
    for metric, reason in EXEMPT_METRICS.items():
        assert len(reason.split()) >= 3, f"{metric} is exempt with no real reason"


def histogram_buckets() -> dict[str, set[str]]:
    """
    Every histogram's bucket bounds, formatted the way the client writes them.

    prometheus_client renders bounds through floatToGoString, so a bucket
    declared as the integer 4 is exposed as le="4.0". Using the client's own
    function rather than reimplementing it: the point is to match what actually
    appears on /metrics, not what looks right.
    """
    found: dict[str, set[str]] = {}
    for path in list((ROOT / "orchestrator").rglob("*.py")) + list(
        (ROOT / "agents").rglob("*.py")
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "Histogram" or not node.args:
                continue
            metric = getattr(node.args[0], "value", None)
            buckets = next(
                (kw.value for kw in node.keywords if kw.arg == "buckets"), None
            )
            if not isinstance(metric, str) or buckets is None:
                continue
            bounds = set()
            for element in getattr(buckets, "elts", []):
                try:
                    bounds.add(floatToGoString(ast.literal_eval(element)))
                except ValueError:
                    # float("inf") is a Call, not a literal; the client writes
                    # it as +Inf and every histogram has it.
                    bounds.add("+Inf")
            bounds.add("+Inf")
            found[metric] = bounds
    return found


def bucket_selectors() -> list[tuple[str, str, str]]:
    """(file, metric, le) for every le="..." selector on a _bucket series."""
    out = []
    for path in (DASHBOARD, ALERTS):
        text = path.read_text(encoding="utf-8")
        for metric, le in re.findall(
            r'([a-z_][a-z0-9_]*)_bucket\{[^}]*?le=\\?"([^"\\]+)\\?"', text
        ):
            out.append((path.name, metric, le))
    return out


def test_every_bucket_selector_names_a_real_bound():
    """
    le="4" and le="4.0" are not the same selector, and the wrong one matches no
    series at all - a query that returns nothing forever and draws exactly like
    a healthy zero. That is what the first version of the tools-discovered
    panel did, and only asking live Prometheus found it.

    Reading catches it in a millisecond now.
    """
    buckets = histogram_buckets()
    assert buckets, "no histograms found; the extractor is broken"

    wrong = []
    for filename, metric, le in bucket_selectors():
        known = buckets.get(metric)
        if known is None:
            wrong.append(f"{filename}: {metric}_bucket is not a histogram in this repo")
        elif le not in known:
            wrong.append(
                f'{filename}: {metric}_bucket{{le="{le}"}} - bounds are '
                f"{sorted(known)}"
            )

    assert not wrong, "bucket selectors that match nothing:\n  " + "\n  ".join(wrong)


def test_the_bucket_check_knows_the_client_formats_floats():
    """
    The guard needs a guard: if floatToGoString were bypassed and bounds
    compared as written in the source, le="4" would look correct and the real
    bug would pass.
    """
    buckets = histogram_buckets()
    assert "4.0" in buckets["orchestrator_tools_discovered"]
    assert "4" not in buckets["orchestrator_tools_discovered"]
    assert "+Inf" in buckets["orchestrator_tools_discovered"]
    # ...and the duration histogram the alert rules lean on.
    assert "300.0" in buckets["orchestrator_run_duration_seconds"]


# ---------------------------------------------------------------------------
# ...and the dashboard has to be a dashboard
# ---------------------------------------------------------------------------


def test_panels_do_not_overlap():
    """
    Grafana renders overlapping panels by shoving them around, so a layout
    mistake shows up as a dashboard that looks nothing like the file.
    """
    occupied: dict[tuple[int, int], str] = {}
    for panel in dashboard()["panels"]:
        box = panel["gridPos"]
        for x in range(box["x"], box["x"] + box["w"]):
            for y in range(box["y"], box["y"] + box["h"]):
                clash = occupied.get((x, y))
                assert clash is None, (
                    f"{panel['title']!r} overlaps {clash!r} at ({x}, {y})"
                )
                occupied[(x, y)] = panel["title"]


def test_panels_fit_the_grid():
    """Grafana's grid is 24 columns; anything wider is silently clipped."""
    for panel in dashboard()["panels"]:
        box = panel["gridPos"]
        assert box["x"] + box["w"] <= 24, f"{panel['title']!r} runs past the grid"


def test_panel_ids_are_unique():
    ids = [panel["id"] for panel in dashboard()["panels"]]
    assert len(ids) == len(set(ids)), f"duplicate panel ids: {sorted(ids)}"


@pytest.mark.parametrize(
    "panel", dashboard()["panels"], ids=lambda p: p["title"]
)
def test_every_panel_says_what_it_is_for(panel):
    """
    Whoever opens a panel at 2am is not whoever wrote the query. Several of
    these carry a caveat that changes how the line is read - that run duration
    cannot exceed 300, that the cache is per pod - and those belong on the panel
    rather than in a commit message.
    """
    assert panel.get("description", "").strip(), f"{panel['title']!r} has no description"
    assert panel.get("targets"), f"{panel['title']!r} queries nothing"
