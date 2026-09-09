"""
The alert rules have to reference metrics that exist.

WHY THIS EXISTS. An alert rule fails silently in a way almost nothing else
does. A typo in a metric name is valid PromQL - it is simply a selector that
matches no series - so Prometheus loads the rule, reports it healthy, and the
alert sits at "inactive" forever. There is no error anywhere. The rule looks
like coverage and provides none, and the only way to find out is for the
failure it was meant to catch to happen unnoticed.

Reading catches that in milliseconds, and names the offending rule. It cannot
catch a threshold that is wrong, or an expression that fires at the wrong time
- those need real evaluation, which is what tests/alerts_test.yml does through
`promtool test rules` in CI. The two are deliberately different checks: this
one asks "does this rule refer to things that exist", promtool asks "does it
fire when it should".

The same split as tests/test_agent_images.py: read locally, run for real in CI.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "k8s" / "prometheus.yaml"
PROMTOOL_TESTS = Path(__file__).resolve().parent / "alerts_test.yml"

# Prometheus synthesises these; no code declares them.
SYNTHETIC_METRICS = {"up"}

# PromQL functions, aggregators and keywords. Anything left after these are
# removed is being used as a metric name.
PROMQL_WORDS = {
    "sum", "rate", "increase", "absent", "histogram_quantile", "by", "without",
    "on", "ignoring", "group_left", "group_right", "and", "or", "unless",
    "avg", "min", "max", "count", "topk", "bottomk", "quantile",
    # The _over_time family and friends. Added after CorpusMostlyGone was
    # written with max_over_time and this check reported it as an undeclared
    # metric - which is the guard working, but on the wrong target. A function
    # missing from here fails loudly, so the list only needs to grow when a rule
    # reaches for something new; it can never silently pass a real typo.
    "max_over_time", "min_over_time", "avg_over_time", "sum_over_time",
    "count_over_time", "last_over_time", "stddev_over_time", "quantile_over_time",
    "delta", "idelta", "irate", "deriv", "predict_linear", "changes", "resets",
    "clamp_max", "clamp_min", "abs", "ceil", "floor", "round", "vector",
    "scalar", "time", "timestamp",
}

# Suffixes Prometheus derives from a histogram; the base name is what the code
# declares.
DERIVED_SUFFIXES = ("_bucket", "_count", "_sum")

API = ROOT / "orchestrator" / "api.py"

# Run outcomes that deliberately have no alert, and why. Being in this map is a
# decision; being absent from both this map and the rules is an oversight, and
# test_every_run_outcome_is_alerted_or_exempt cannot tell the difference unless
# the decision is written down.
EXEMPT_OUTCOMES = {
    "answered": "the run worked",
    "unanswered": (
        "the loop ended saying it could not answer. That is the honest path and "
        "the thing this project spent weeks building; alerting on it would raise "
        "an alarm every time the system correctly declined to invent something, "
        "and the obvious way to silence it would be to remove the guardrail"
    ),
}


def manifest_docs():
    return [d for d in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if d]


def named(kind: str, name: str):
    for d in manifest_docs():
        if d["kind"] == kind and d["metadata"]["name"] == name:
            return d
    raise AssertionError(f"{kind}/{name} is not in {MANIFEST.name}")


def alert_rules():
    """Every alert rule, as (group_name, rule_dict)."""
    rules_cm = named("ConfigMap", "prometheus-rules")
    parsed = yaml.safe_load(rules_cm["data"]["alerts.yml"])
    return [
        (group["name"], rule)
        for group in parsed["groups"]
        for rule in group["rules"]
        if "alert" in rule
    ]


def declared_metrics() -> set[str]:
    """
    Metric names the code actually creates.

    Read from the source rather than from a running Prometheus: a metric that
    has never been incremented does not appear in /metrics yet, so scraping
    would call a correct rule broken on a quiet system.
    """
    pattern = re.compile(
        r"(?:Counter|Histogram|Gauge|Summary)\(\s*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']"
    )
    found = set()
    for path in list((ROOT / "orchestrator").rglob("*.py")) + list(
        (ROOT / "agents").rglob("*.py")
    ):
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    return found


def metrics_in(expr: str) -> set[str]:
    """Bare metric names in a PromQL expression."""
    # Order matters: strip the things that legitimately contain identifiers
    # which are not metric names, so what survives is unambiguous.
    expr = re.sub(r"\"[^\"]*\"|'[^']*'", " ", expr)  # string literals
    expr = re.sub(r"\{[^}]*\}", " ", expr)  # label matchers
    expr = re.sub(r"\[[^\]]*\]", " ", expr)  # range selectors
    expr = re.sub(r"\b(?:by|without|on|ignoring)\s*\([^)]*\)", " ", expr)  # groupings

    words = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr))
    return {w for w in words if w not in PROMQL_WORDS}


def base_name(metric: str) -> str:
    for suffix in DERIVED_SUFFIXES:
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric


def write_rules_file(path: Path) -> Path:
    """
    Extract the rules from the ConfigMap into a plain file promtool can read.

    The rules live inside a Kubernetes ConfigMap so that `kubectl apply -f k8s/`
    deploys them with everything else, and promtool wants a bare rules file.
    Rather than keep a second copy in sync, CI calls this module as a script to
    bridge the two - see the alerts job in .github/workflows/tests.yml.
    """
    path.write_text(named("ConfigMap", "prometheus-rules")["data"]["alerts.yml"],
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The check that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group,rule", alert_rules(), ids=lambda x: x["alert"] if isinstance(x, dict) else x
)
def test_every_metric_an_alert_references_exists(group, rule):
    known = declared_metrics() | SYNTHETIC_METRICS
    used = {base_name(m) for m in metrics_in(rule["expr"])}
    missing = used - known

    assert not missing, (
        f"alert {rule['alert']} references {sorted(missing)}, which no Counter, "
        "Histogram or Gauge in this repo declares. The rule would load cleanly "
        "and never fire."
    )


def run_outcomes() -> set[str]:
    """
    Every value orchestrator_runs_total can be labelled with.

    Read from api.py rather than from the comment in metrics.py that lists
    them, because that comment was wrong - it omitted "unanswered" - and alert
    rules written against it missed three outcomes. The code is the only thing
    that cannot drift from itself.

    Catches both forms: the `run.outcome = "..."` assignments and RunState's
    `outcome: str = "failed"` default, which is what an escaping exception or a
    disconnected client is recorded as.
    """
    source = API.read_text(encoding="utf-8")
    return set(re.findall(r'\boutcome\b[^=\n]*=\s*"([a-z_]+)"', source))


def alerted_outcomes() -> set[str]:
    """Outcomes some alert rule actually selects on."""
    found = set()
    for _, rule in alert_rules():
        found.update(re.findall(r'outcome\s*=\s*"([a-z_]+)"', rule["expr"]))
    return found


def test_every_run_outcome_is_alerted_or_exempt():
    """
    The check that would have caught the hole this test was written for.

    Nine rules shipped covering exactly one of the six run outcomes. The one
    that bit us was `failed`: Ollama died mid-session on 2026-09-08, every run
    returned ConnectionError, the counter recorded it correctly - and no rule
    referenced it, so nothing fired. `truncated` and `no_tools` were missing
    too.

    Alerting cannot be complete by inspection, because a missing rule looks
    exactly like a rule that has not fired. This makes the omission a test
    failure instead.
    """
    outcomes = run_outcomes()
    assert outcomes, "found no outcomes in api.py; the extractor is broken"

    uncovered = outcomes - alerted_outcomes() - set(EXEMPT_OUTCOMES)

    assert not uncovered, (
        f"run outcome(s) {sorted(uncovered)} have no alert and are not listed in "
        "EXEMPT_OUTCOMES. Either write a rule, or record why not - an outcome "
        "nothing watches is invisible until somebody notices the system was "
        "broken all along."
    )


def test_exemptions_name_real_outcomes():
    """A renamed outcome leaves an exemption silently excusing nothing."""
    stale = set(EXEMPT_OUTCOMES) - run_outcomes()
    assert not stale, (
        f"EXEMPT_OUTCOMES lists {sorted(stale)}, which api.py never sets"
    )


def test_exemptions_are_justified():
    """An exemption with no reason is an oversight with better paperwork."""
    for outcome, reason in EXEMPT_OUTCOMES.items():
        assert len(reason.split()) >= 3, f"{outcome} is exempt with no real reason"


def test_the_outcome_extractor_finds_what_is_there():
    """
    The guard needs a guard. If this regex quietly matched nothing, the coverage
    check above would pass forever while covering nothing.
    """
    outcomes = run_outcomes()

    # The pessimistic default, which is only reachable as a dataclass field.
    assert "failed" in outcomes, "RunState's default outcome was not found"
    # ...and an ordinary assignment.
    assert "answered" in outcomes

    # The label call itself must not be mistaken for an assignment.
    assert "run" not in outcomes and "outcome" not in outcomes


# ---------------------------------------------------------------------------
# ...and the things that make a firing alert readable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group,rule", alert_rules(), ids=lambda x: x["alert"] if isinstance(x, dict) else x
)
def test_every_alert_says_what_is_wrong_and_how_bad(group, rule):
    """
    An alert with no annotations is a name and a timestamp. Whoever reads it is
    usually not whoever wrote it, and by then the expression is not on screen.
    """
    severity = rule.get("labels", {}).get("severity")
    assert severity in ("warning", "critical"), (
        f"alert {rule['alert']} has severity {severity!r}; expected warning or critical"
    )

    annotations = rule.get("annotations", {})
    for field in ("summary", "description"):
        assert annotations.get(field, "").strip(), (
            f"alert {rule['alert']} has no {field}"
        )


def test_alert_names_are_unique():
    """Two rules with one name are indistinguishable once firing."""
    names = [rule["alert"] for _, rule in alert_rules()]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate alert names: {sorted(duplicates)}"


# ---------------------------------------------------------------------------
# The wiring, which is easy to get right once and then break
# ---------------------------------------------------------------------------


def test_prometheus_is_told_to_load_the_rules():
    config = yaml.safe_load(named("ConfigMap", "prometheus-config")["data"]["prometheus.yml"])
    rule_files = config.get("rule_files") or []
    assert rule_files, "prometheus.yml has no rule_files; the rules would never load"

    deployment = named("Deployment", "prometheus")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    volumes = {
        v["name"]: v.get("configMap", {}).get("name")
        for v in deployment["spec"]["template"]["spec"]["volumes"]
    }

    assert volumes.get("rules") == "prometheus-rules", (
        "the prometheus-rules ConfigMap is not mounted; rule_files would point at "
        "an empty directory"
    )

    # The glob has to sit under the directory the ConfigMap is mounted at, or
    # it matches nothing - and an unmatched glob is not an error in Prometheus.
    mount_path = mounts["rules"]
    assert any(f.startswith(mount_path.rstrip("/") + "/") for f in rule_files), (
        f"rule_files {rule_files} does not point inside {mount_path}"
    )


def test_the_rules_configmap_is_not_also_the_config_configmap():
    """
    A ConfigMap mounts as a directory. Both were nearly put in one, which
    would have mounted over /etc/prometheus and hidden prometheus.yml itself.
    """
    deployment = named("Deployment", "prometheus")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    paths = [m["mountPath"].rstrip("/") for m in container["volumeMounts"]]
    assert len(paths) == len(set(paths)), f"two volumes mount at the same path: {paths}"


# ---------------------------------------------------------------------------
# Keeping the promtool fixture honest
# ---------------------------------------------------------------------------


def promtool_test_doc():
    return yaml.safe_load(PROMTOOL_TESTS.read_text(encoding="utf-8"))


def alerts_with_a_firing_case() -> set[str]:
    """Alerts that some case asserts actually fires."""
    tested = set()
    for case in promtool_test_doc()["tests"]:
        for assertion in case.get("alert_rule_test", []):
            if assertion.get("exp_alerts"):
                tested.add(assertion["alertname"])
    return tested


def test_every_alert_has_a_test_that_it_fires():
    """
    The rule this file cannot check by reading is whether an expression fires
    at the right moment, so at minimum every alert must be exercised by
    promtool somewhere. Without this, a new rule can be added with no test and
    nothing says so.
    """
    defined = {rule["alert"] for _, rule in alert_rules()}
    untested = defined - alerts_with_a_firing_case()

    assert not untested, (
        f"{sorted(untested)} have no case in {PROMTOOL_TESTS.name} asserting they "
        "fire. An alert nobody has seen fire is a guess."
    )


def test_the_promtool_fixture_only_names_real_alerts():
    """A renamed alert leaves its test asserting on a name that cannot fire."""
    defined = {rule["alert"] for _, rule in alert_rules()}
    named_in_tests = {
        assertion["alertname"]
        for case in promtool_test_doc()["tests"]
        for assertion in case.get("alert_rule_test", [])
    }
    unknown = named_in_tests - defined

    assert not unknown, (
        f"{PROMTOOL_TESTS.name} tests {sorted(unknown)}, which no rule defines. "
        "promtool passes such a case vacuously."
    )


def test_a_promql_function_is_not_mistaken_for_a_metric():
    """
    CorpusMostlyGone uses max_over_time, and this check first reported that as
    an undeclared metric. The fix was to name the function - which widens what
    the check excuses, so this pins that the widening did not swallow the real
    thing: the metric inside the call is still extracted.
    """
    expr = (
        "retrieval_documents_total < 0.5 * "
        "max_over_time(retrieval_documents_total[6h])"
    )
    assert metrics_in(expr) == {"retrieval_documents_total"}

    # A typo inside a function call is still caught.
    assert metrics_in("max_over_time(retrieval_documents_totl[6h])") == {
        "retrieval_documents_totl"
    }


def test_the_metric_check_would_notice_a_typo():
    """
    The guard needs a guard - the same reasoning as test_agent_images.py. This
    is all regex over two file formats, and a check that quietly matches
    nothing would report success forever.
    """
    known = declared_metrics()
    assert "orchestrator_runs_total" in known, "the source scan found nothing"

    typo = 'sum(rate(orchestrator_runs_totl{outcome="rejected"}[15m])) > 0'
    assert metrics_in(typo) == {"orchestrator_runs_totl"}
    assert metrics_in(typo) - known == {"orchestrator_runs_totl"}

    # ...and does not flag the real thing, its derived suffixes, or label names
    real = (
        'histogram_quantile(0.95, sum by (le) '
        '(rate(orchestrator_run_duration_seconds_bucket[30m]))) > 300'
    )
    assert {base_name(m) for m in metrics_in(real)} - known - SYNTHETIC_METRICS == set()


if __name__ == "__main__":
    # Not a test entry point: CI uses this to hand the rules to promtool.
    import sys

    print(write_rules_file(Path(sys.argv[1])))
