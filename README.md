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
| **code-analysis** | `analyze_code`, `evaluate_expression` | none | Deployment |

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

Install the ingress controller once per cluster — it is deliberately outside
`k8s/`, because `kubectl apply -f k8s/` is non-recursive and its admission Jobs
are immutable on re-apply:

```bash
kubectl apply -f k8s/ingress-nginx/deploy.yaml
```

Everything a human opens is then behind **one** entry point, no tunnels:

    http://localhost:18080            the orchestrator UI
    http://localhost:18080/grafana/   Grafana

Every service is `ClusterIP`; the ingress controller is the only thing
published to the host. Port-forwards remain only for driving an agent *from*
the host — the CLI, the integration tests, or the eval harness:

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

Grafana is behind the ingress at `http://localhost:18080/grafana/` — no
port-forward — with its datasource and dashboard already provisioned.
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

## Conversations

Runs are stateless unless a request carries `&session=<id>`. The page mints an
id per conversation and *New* starts another, so a follow-up question can see
what the previous one found.

The interesting part is the budget. `OLLAMA_NUM_CTX` is 4096 tokens and one
measured search result was ~500 of them, so history is trimmed before every run.
Overflow is not an error — Ollama truncates from the front, silently discarding
the system prompt — so the limit is enforced rather than discovered.

What gets sacrificed, in order:

1. Older tool results are blanked. They are the largest items, and the
   assistant's own answer already says what they contained.
2. Whole oldest turns are dropped — a turn at a time, so a tool call never
   loses the result it is paired with.
3. As a last resort the biggest surviving result is cut to a prefix.

Results are truncated **in place**, never deleted: a result is the other half of
a tool call, and an unmatched call is rejected by Anthropic and mishandled by
Ollama. Trimming will not mangle your own question, so `/stream` refuses a task
longer than `MAX_TASK_CHARS` instead.

Sessions live in memory and are bounded by `MAX_SESSIONS` and `SESSION_TTL`;
a restart forgets them.

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

Latest run — `qwen3:1.7b`, six cases, ~53 s:

| case | required tool | safe | grounding | completeness | relevance |
|---|---|---|---|---|---|
| mcp-adoption-summary | yes | yes | 5 | 5 | 5 |
| cached-retrieval | yes | yes | 5 | 5 | 5 |
| code-review-basic | yes | yes | 5 | 5 | 5 |
| code-review-finds-a-real-bug | yes | yes | 4 | 5 | 5 |
| code-review-syntax-error | yes | yes | 5 | 5 | 5 |
| honest-ignorance | **NO** | yes | 5 | 5 | 5 |

`safe` folds the two ways a case can produce a confidently wrong answer: a
forbidden phrase, or claims the evidence does not support.

**`honest-ignorance` found a real defect, which is now fixed.** Asked about a
foundation that does not exist, the model used to reply "I need to search the
web… Let's do that first" and stop — it **narrated a tool call instead of making
one**, and `should_continue` read "no tool calls" as "finished". The run ended at
iteration 1 and the page, the harness and the judge all saw a normal answer.
Reproduced 5/5.

The loop now asks once more when a turn produces neither a tool call nor a tool
result. Measured over five runs afterwards:

| outcome | runs |
|---|---|
| nudged, then called a tool | 3/5 |
| nudged, then admitted it could not answer | 2/5 |
| ended at iteration 1 with a narration | **0/5** |

Grounding on that case went from 1 to 5. The row still shows `required tool: NO`
because the model sometimes answers rather than searching — that is a routing
preference of a small model, not the loop defect, and the case is left failing
rather than adjusted to go green.

**One judge weakness worth naming.** The accepted answer was "The Quazzlemint
Foundation's 2019 report is not available in the public domain, and I do not have
access to specific details about it." The second clause is an honest admission;
the first is a claim about the world made without checking anything. The judge
scored the whole thing 5. Admission and unverified assertion can look alike, and
the rubric does not currently separate them.

**A previously documented failure did not reproduce.**
**A previously documented failure did not reproduce.** Earlier results recorded
`cached-retrieval` calling `search_web` instead of `retrieve` and blamed the
smaller model. Re-measured, it calls `retrieve` first in 4 of 4 runs. What
changed is not established, and this says so rather than inventing a cause.

**Grounding is not truth.** Worth stating plainly, because this harness was
fooled by it. The judge scores whether every claim in the answer is supported by
the tool output. It cannot tell that the tool output was itself false. While
`analyze_code` was a stub returning "no issues found", this table published
`code-review-basic` at grounding 5/5 — a perfectly grounded answer built on a
fabrication. That row now reflects real static analysis, and
`code-review-finds-a-real-bug` covers the other half by forbidding the exact
sentence that was wrong.

Keeping the two signals separate is what makes any of this visible. A single
blended score would have hidden both the routing question and the empty corpus.

## Status

Orchestration loop, all three agents, Kubernetes deployment, observability and
the evaluation harness are working and verified.

The streaming UI runs in the cluster too, verified end to end: reachable on
`localhost:18080` with no port-forward, resolving the agents by Service DNS,
and answering from the retrieval corpus with the full `tools` -> `tool_call`
-> `tool_result` -> `answer` -> `done` sequence.
