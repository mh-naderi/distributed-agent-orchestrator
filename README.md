# Distributed Agent Orchestrator

[![tests](https://github.com/mh-naderi/distributed-agent-orchestrator/actions/workflows/tests.yml/badge.svg)](https://github.com/mh-naderi/distributed-agent-orchestrator/actions/workflows/tests.yml)

A LangGraph orchestrator that routes tasks across specialized agents, each
exposed as its own MCP (Model Context Protocol) server and deployed as an
independent Kubernetes service, with Prometheus/Grafana observability.

Runs entirely locally — no cloud account, no GPU rental, no API keys.

## Architecture

| Agent | Tools | State | Deployed as |
|---|---|---|---|
| **research** | `search_web` | none | Deployment |
| **retrieval** | `index_documents`, `retrieve` | vector index | StatefulSet + PVC |
| **code-analysis** | `analyze_code` | none | Deployment |

The orchestrator runs a reason → act → reason loop: ask the LLM what to do, call
the MCP tool it picks, feed the result back, repeat until it answers or hits the
max-iteration guardrail.

```mermaid
flowchart TB
    TASK(["User task"])

    subgraph HOST["Host machine"]
        direction TB
        ORCH["Orchestrator — LangGraph<br/>reason → act → reason loop"]
        OLLAMA["Ollama :11434<br/>qwen3:1.7b · nomic-embed-text"]
    end

    subgraph CLUSTER["Kubernetes cluster"]
        direction TB
        subgraph AGENTS["MCP agents — Streamable HTTP on /mcp"]
            direction TB
            RES["research-agent · Deployment<br/>search_web"]
            RET["retrieval-agent · StatefulSet<br/>index_documents · retrieve"]
            CODE["code-analysis-agent · Deployment<br/>analyze_code"]
        end
        PVC[("PersistentVolume<br/>sqlite-vec index")]
        PROM["Prometheus"]
        GRAF["Grafana"]
    end

    WEB(["DuckDuckGo"])

    TASK --> ORCH
    ORCH <-->|"messages + tool schemas"| OLLAMA
    ORCH ==>|"tool calls over MCP"| AGENTS

    RES --> WEB
    RET -->|"embed"| OLLAMA
    RET <--> PVC

    AGENTS -.->|"scrape /metrics"| PROM
    PROM --> GRAF

    classDef stateful fill:#fcd34d,stroke:#b45309,color:#111
    classDef stateless fill:#bfdbfe,stroke:#1d4ed8,color:#111
    classDef external fill:#e5e7eb,stroke:#6b7280,color:#111
    classDef core fill:#bbf7d0,stroke:#15803d,color:#111
    classDef obs fill:#e9d5ff,stroke:#7e22ce,color:#111

    class RET,PVC stateful
    class RES,CODE stateless
    class WEB,OLLAMA external
    class ORCH core
    class PROM,GRAF obs
```

Amber is the only stateful part of the system, which is why it's a StatefulSet
with a volume and why it alone can't be scaled horizontally. Dotted arrows are
metrics scraping.

**[docs/architecture.md](docs/architecture.md)** covers the design decisions —
why separate MCP servers, why the summarizer agent was cut, and the hardware
limits that shaped the model and deployment choices.

## Quickstart

**1. Models** (must support tool calling — check `ollama show <model>` for `tools`)

```bash
ollama pull qwen3:1.7b && ollama pull nomic-embed-text
```

**2. Dependencies**

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

**3. Agents** — one per terminal

```bash
MCP_PORT=18000 .venv/Scripts/python.exe agents/research_agent/server.py
```

```bash
MCP_PORT=18001 .venv/Scripts/python.exe agents/retrieval_agent/server.py
```

```bash
MCP_PORT=18002 .venv/Scripts/python.exe agents/code_analysis_agent/server.py
```

**4. Orchestrator**

```bash
.venv/Scripts/python.exe -m orchestrator.main
```

## Kubernetes

The cluster must be created with the config file — it publishes the node port
that serves the UI, and Docker can only publish a container's ports at creation
time, so this cannot be added to an existing cluster:

```bash
kind create cluster --name agent-orchestrator --config kind-cluster.yaml
```

Build and load each image — three agents plus the orchestrator (the manifests
use `imagePullPolicy: IfNotPresent`):

```bash
docker build -t agent-orchestrator/research-agent:latest agents/research_agent
```

```bash
kind load docker-image agent-orchestrator/research-agent:latest --name agent-orchestrator
```

```bash
kubectl apply -f k8s/
```

The orchestrator runs **inside** the cluster and reaches the agents by their
Service DNS names, so the UI needs no tunnel:

    http://localhost:18080

Everything else is `ClusterIP`. Port-forward only what you need — Grafana,
Prometheus, or an agent you want to drive from the host:

```bash
kubectl port-forward service/retrieval-agent-service 18001:8000
```

Ollama stays on the host; pods reach it via `host.docker.internal`. See
`docs/RUNBOOK.md` for the full sequence and the environment traps.

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

Loop and vector-store tests use fakes and need nothing running. Integration
tests exercise the real MCP protocol and skip when the agents are down, so the
suite is green on a fresh checkout.

## Observability

Prometheus and Grafana deploy with everything else.

```bash
kubectl port-forward service/grafana-service 13000:3000
```

Grafana at `localhost:13000`, datasource and dashboard already provisioned.
Discovery is annotation-driven — a new component opts in with
`prometheus.io/scrape`, no config edit. The annotated port must be **named**
`metrics`; Prometheus selects targets by port name, so a metrics endpoint on a
port named anything else is annotated and then silently never scraped.

From the agents, at the tool boundary:

- `tool_calls_total{tool_name,status,agent}`
- `tool_call_duration_seconds` (histogram, so p95 is computable)
- `retrieval_documents_total` — corpus size, the one number that should
  survive a pod restart

From the orchestrator, about the loop rather than individual tools:

- `orchestrator_runs_total{outcome}` — `answered`, `truncated`, `failed`,
  `no_tools`
- `orchestrator_run_duration_seconds`
- `orchestrator_run_iterations` — runs clustering near the max-iteration
  guardrail mean the loop is regularly running out of road, which no per-tool
  metric would reveal because each call looks fine
- `orchestrator_tools_discovered`, `orchestrator_discovery_failures_total`
- `orchestrator_runs_queued_total` — runs that waited for a slot. Only one
  run executes at a time, because inference shares a single 4GB GPU; a second
  request is queued and told so, and refused past a cap. Raise
  `MAX_CONCURRENT_RUNS` where there is headroom.

## Escalating to Claude

Local inference is the default because it is free. `docs/architecture.md`
concludes that a 4GB laptop GPU is under-specified for this workload, and the
Claude API is the escalation path for runs where the small model is not good
enough — its measured cost is tool-selection accuracy, not fluency.

Escalation is **manual, per request**: tick *escalate to Claude* in the UI, or
add `&escalate=1` to `/stream`. An automatic rule (after N iterations, or on a
failed tool selection) was left for later — a three-case eval cannot tell
whether such a heuristic helps, and spending money on a guess is worse than a
switch somebody chose to flip.

```bash
export ANTHROPIC_API_KEY=...   # read from the environment, never stored here
```

Without the key, an escalated request **fails** with a message naming the
missing variable. It does not quietly answer with the local model — a request
that asked for the better model and silently got the weaker one is the failure
this project keeps having to correct.

Set `CLAUDE_MODEL` to change the model and `CLAUDE_FALLBACKS=off` to drop the
server-side refusal fallbacks, which ride on a beta not every organisation has
enabled.

## Evaluation

```bash
.venv/Scripts/python.exe -m eval.run_eval
```

Runs `eval/test_cases.json` through the full system and scores each result on
automated signals (required tools called, keyword match) plus an LLM judge that
grades the answer against the tool output it was actually given.

Latest run — `qwen3:1.7b`, three cases, ~38 s:

| case | required tool called | grounding | completeness | relevance |
|---|---|---|---|---|
| mcp-adoption-summary | yes | 5 | 5 | 5 |
| cached-retrieval | yes | 5 | 5 | 5 |
| code-review-basic | yes | 5 | 5 | 5 |

Straight fives are not a brag — two of these numbers moved because the harness
was fixed, not because the system got smarter. Both fixes are worth recording,
since each was a case of a measurement quietly measuring the wrong thing.

**`cached-retrieval` was testing a side effect.** It carried a note reading
"Run after mcp-adoption-summary", which nothing enforced. It could only find
MCP content in the corpus if that earlier case had chosen to call
`index_documents` — and the small default model skips indexing, correctly from
its point of view, because indexing helps the *next* run and does nothing for
the answer in progress. So the case scored completeness 1 against a corpus that
genuinely held nothing about MCP, and looked like a retrieval failure. Cases now
declare `seed_documents`, which the harness indexes through the MCP tool before
the case runs. Completeness 1 → 5, with the answer drawn from the seeded text.

**A previously documented failure did not reproduce.** Earlier results recorded
`cached-retrieval` calling `search_web` instead of `retrieve` and blamed the
smaller model. Re-measured, it calls `retrieve` first in 4 of 4 runs. What
changed is not established, and this says so rather than inventing a cause.

**Grounding is not truth.** Worth stating plainly, because this harness was
fooled by it. The judge scores whether every claim in the answer is supported by
the tool output. It cannot tell that the tool output was itself false. While
`analyze_code` was a stub returning "no issues found", this table published
`code-review-basic` at grounding 5/5 — a perfectly grounded answer built on a
fabrication. The score was right by its own definition; the conclusion drawn
from it was wrong. That row now reflects real static analysis.

Keeping the two signals separate is what makes any of this visible. A single
blended score would have hidden both the routing question and the empty corpus.

## Status

Orchestration loop, all three agents, Kubernetes deployment, observability and
the evaluation harness are working and verified.

The streaming UI runs in the cluster too, verified end to end: reachable on
`localhost:18080` with no port-forward, resolving the agents by Service DNS,
and answering from the retrieval corpus with the full `tools` -> `tool_call`
-> `tool_result` -> `answer` -> `done` sequence.
