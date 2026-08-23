"""
Metrics for the orchestrator.

WHY THIS EXISTS. Every agent exports tool_calls_total and a latency histogram,
so the system could report which tools ran and how long each took - while the
component that DECIDES which tools to run exported nothing at all. The one part
of the system with judgement in it was the one part with no telemetry.

WHY THESE NUMBERS AND NOT MORE. The agents already count tool calls, and
counting them again here would double-report the same event under a second
name. What only the orchestrator knows is the shape of a RUN:

  - did it end with an answer, get cut short by the guardrail, or fail
  - how long the whole reason -> act -> reason cycle took
  - how many iterations it needed to get there

The iteration histogram is the interesting one. Runs clustering near
MAX_ITERATIONS mean the loop is regularly running out of road, which is a
prompt or model problem that no per-tool metric would reveal - each individual
call looks fine.

WHY A SEPARATE PORT RATHER THAN A ROUTE. This process owns its Starlette app,
so /metrics could have been a route on 8080 - and that was the first attempt.
Two things ruled it out:

  - Prometheus here keeps only targets whose CONTAINER PORT IS NAMED "metrics"
    (see the relabel rules in k8s/prometheus.yaml, and the RE2 backreference
    problem documented there). A route on the port named "http" is annotated
    and then silently dropped - scraped by nothing, with no error anywhere.
    Declaring 8080 twice under two names is accepted by the API server but
    warns: "duplicate port definition".
  - 8080 is the port published to the host by the NodePort mapping, so metrics
    on it are readable by anyone who can reach the UI. A separate port is not
    in the Service's NodePort and stays inside the cluster.

So this follows the agents exactly: a side port served by
prometheus_client.start_http_server(), which costs one daemon thread and keeps
the fleet uniform.
"""

import os

from prometheus_client import Counter, Histogram

# 9100/9101/9102 belong to research, retrieval and code-analysis respectively.
METRICS_PORT = int(os.environ.get("ORCHESTRATOR_METRICS_PORT", "9103"))

# Outcome is a label rather than three counters so a dashboard can show the
# split without hardcoding which outcomes exist. Values: "answered",
# "truncated" (hit the guardrail), "failed" (raised), "no_tools" (discovery
# came back empty, so the run never started).
RUNS = Counter(
    "orchestrator_runs_total",
    "Orchestrator runs, by how they ended",
    ["outcome"],
)

RUN_DURATION = Histogram(
    "orchestrator_run_duration_seconds",
    "Wall time of a complete run",
    # A run is one or more local LLM calls plus tool calls, so it lives in
    # seconds-to-minutes, not the sub-second range the default buckets assume.
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, float("inf")),
)

RUN_ITERATIONS = Histogram(
    "orchestrator_run_iterations",
    "reason/act cycles per run",
    # Explicit small-integer buckets: the default latency buckets are
    # meaningless for a count that is nearly always between 1 and 10.
    buckets=(1, 2, 3, 4, 5, 6, 8, 10, float("inf")),
)

# Discovery is best-effort by design - MCPToolRegistry logs an unreachable
# agent and carries on, so a partial system still serves requests. That
# tolerance is only safe if the degradation is visible somewhere.
DISCOVERY_FAILURES = Counter(
    "orchestrator_discovery_failures_total",
    "Discovery attempts that reached no agents at all",
)

TOOLS_DISCOVERED = Histogram(
    "orchestrator_tools_discovered",
    "Tools visible after a discovery pass",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10, float("inf")),
)
