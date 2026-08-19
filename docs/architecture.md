# Architecture Notes

## Why MCP, and why multiple servers

MCP standardizes how an agent discovers and calls tools, independent of
which LLM is driving it (like a USB port for AI tools: any client, any
compatible server). A single MCP server with several `@mcp.tool()`
functions would be simpler and still valid MCP usage.

This project splits tools into separate MCP servers deliberately,
to practice and demonstrate:
- Independent deployment and scaling per agent
- Isolated failure handling (one agent crashing doesn't take down the others)
- Per-agent observability (Prometheus metrics scoped per service)
- Different resource needs per agent

The last two points used to be aspirational. They aren't anymore - see below.

## The agents

| Agent | Tools | State | Deployed as |
|---|---|---|---|
| research | `search_web` | none | Deployment |
| retrieval | `index_documents`, `retrieve` | **persistent index** | **StatefulSet + PVC** |
| code-analysis | `analyze_code` | none | Deployment |

## Stateful vs stateless, and why it matters here

An earlier version of this project had three stateless agents. That made the
multi-server architecture **unfalsifiable**: every agent held nothing, restarted
instantly, and could be scaled to any number of replicas. Merging all three into
one MCP server would have broken nothing. The distribution was asserted in this
document but no code depended on it.

The retrieval agent changes that, because it owns a sqlite-vec index that has to
outlive the process. That single property is what forces the rest:

- **StatefulSet, not Deployment.** A Deployment treats pods as interchangeable,
  which is correct when they hold nothing. A StatefulSet gives stable pod
  identity and `volumeClaimTemplates`, so each replica gets its own
  PersistentVolumeClaim reattached to it by name after a restart.
- **A headless Service.** Stable identity is meaningless if you can't address a
  specific pod, so a StatefulSet requires one alongside the normal Service.
- **Readiness matters.** The agent must load the sqlite-vec extension and open
  the index before it can serve. Routing traffic to a pod mid-warm-up returns
  confusing empty results rather than an honest failure.
- **It does not scale horizontally the way the others do.** Three replicas would
  not be three workers sharing an index - they would be three pods with three
  separate volumes and three divergent corpora, so a document indexed via one
  pod would be invisible to queries hitting another. Sharing one index across
  replicas needs a different design (a real vector database, or read replicas
  over shared storage). This constraint is the point: it is what makes the
  stateless agents' freedom to scale meaningful by contrast.

## Decision: why the summarizer agent was removed

The original lineup had a summarizer exposing `summarize(text: str)`. It was cut,
and retrieval took its place.

The problem was **pass by value**. To delegate summarization, the orchestrator had
to pass the document as a tool-call argument - meaning it already held the text,
in a model that can summarize. Nothing was saved: the document still transited the
orchestrator's context, the model burned output tokens re-emitting text it already
had, and input size stayed bounded by the orchestrator's own context window. If the
agent were down, the orchestrator would simply do the work itself, so the service
was never load-bearing.

This was confirmed empirically before it was removed: given a task worded
"research ... and summarize it", the model called `search_web` and then summarized
in its own final answer, declining to call `summarize` at all. That was the model
being right.

Retrieval is the opposite shape. It takes a *query* and returns text the
orchestrator has never seen, from a corpus that outlives any single run. That is a
capability the orchestrator genuinely lacks, and the reason the tool earns a
network round trip.

The general rule extracted from this: **a tool earns its place when it does
something the calling model structurally cannot** - reach external information,
run deterministic computation, or read state the model has no access to. A tool
that merely relocates work the model could do in-context is a network hop for
nothing.

## Transport: HTTP+SSE to Streamable HTTP

MCP is JSON-RPC, which needs a transport. stdio suits local subprocesses; this
project needs a network transport, since the whole premise is that agents are
independently deployed services.

The original HTTP transport was HTTP+SSE: two endpoints, one to POST requests and
a separate Server-Sent Events stream for responses. It was deprecated in MCP spec
revision 2025-03-26 and reached end-of-life on 2026-04-01, replaced by **Streamable
HTTP** - a single `/mcp` endpoint that upgrades to a stream only when the server
needs to push. The Python SDK still ships SSE for backwards compatibility only.

All agents use `transport="streamable-http"`. The client uses
`streamable_http_client`; the older `streamablehttp_client` spelling still exists
but is deprecated.

## Sync vs. async tool execution

MCP's core request/response model is synchronous by default: a tool
call blocks until the server finishes executing it and returns a
result. There's no built-in message queue.

All agents stay synchronous. The fast/simple option is the right one when a tool
completes in a few seconds and the added complexity of a queue wouldn't teach
anything new.

The stretch-goal candidate for a Redis-backed async worker pattern is a **code
execution sandbox** - running untrusted code is both genuinely slow and the case
where decoupling the listener from execution actually earns its complexity. (The
code-analysis agent previously held this note; a sandbox is the better fit, since
static analysis is fast.)

## Grounding, and why the stubs had to go

`search_web` initially returned a canned string. That turned out to be actively
harmful rather than merely incomplete: the tool call still *looked* successful, so
the model received a well-formed result containing no information and confabulated
around it. Two runs produced two different fictional expansions of "MCP" - one
invented a cable-modem protocol, the other a multi-cloud platform complete with a
fabricated adoption statistic.

**An empty result shaped like a good one is worse than an error**, because nothing
downstream can detect it. Both retrieval and search now say plainly when they have
nothing, and the system prompt instructs the model to report that rather than fill
the gap.

Real search brought a second, subtler problem: DuckDuckGo returns sponsored results
that look identical to organic ones, distinguishable only by ad-network redirect
URLs. Left in, they were indexed and cited as evidence - one run reported a security
vendor's ebook marketing as a finding about MCP adoption. A model cannot tell an
advertisement from a source, so the filtering happens in the research agent.

## Retrieval design notes

- **sqlite-vec over Chroma.** One file, no server process, and deploying an index
  is copying that file - which makes the PersistentVolume story concrete. Its exact
  brute-force search is linear in corpus size and entirely fine at this scale;
  approximate indexing solves a problem this project doesn't have yet.
- **Two tables joined on rowid.** A `vec0` virtual table holds only vectors; text
  lives in an ordinary table. Newer sqlite-vec supports auxiliary columns inside
  `vec0`, but the split works across versions and separates "which rows are
  nearest" from "what were they".
- **Chunking on blank lines.** One embedding is a single point in vector space, so
  it represents one coherent idea well and five unrelated ones badly. Indexing a
  whole multi-result search blob scored a correct match at distance 0.915;
  chunking per result brought it to 0.787.
- **`num_ctx: 8192` set explicitly.** Ollama serves nomic-embed-text with a
  2048-token window by default and truncates silently, so a long document would be
  embedded from its opening fragment with no error raised.
- **Retrieve before searching.** The orchestrator checks the persistent corpus
  first and only searches on a miss, then indexes what it found. Searching and
  then immediately retrieving the same results would be circular - the results are
  already in context. The index step pays off on *later* runs, which is exactly
  why the store has to be persistent.

## Observability

Prometheus runs in-cluster so it can use the Kubernetes API for service
discovery rather than a hardcoded target list. Three things about that turned
out to matter more than expected, and all three fail quietly:

- **RBAC is not optional.** `kubernetes_sd_configs` works by calling the
  Kubernetes API. A pod cannot do that without a ServiceAccount bound to a role
  granting list/watch on pods; without it discovery returns 403 and Prometheus
  sits with zero targets and nothing obviously wrong on screen.
- **`role: pod` creates one target per declared container port.** Each agent
  declares two (MCP and metrics), so a naive config produces six targets, three
  permanently failing, because `/metrics` on the MCP port does not exist. The
  fix is to name the ports and keep only `metrics`.
- **Prometheus relabel regexes are RE2, which has no backreferences.** An
  earlier attempt matched "container port equals annotated port" with a ``
  backreference; Prometheus rejected the config outright at load. Selecting by
  port *name* is both valid and clearer.

Discovery is annotation-driven (`prometheus.io/scrape`), not a regex over agent
names. The earlier config filtered on a hardcoded list of the three agent names
while the comment above it claimed new agents would appear automatically - which
was false. Annotations make the claim true.

Grafana's datasource and dashboard are provisioned from ConfigMaps. A dashboard
built by clicking through the UI lives only in that container's database and
dies with the pod; provisioned, it is reproducible from `kubectl apply` and
reviewable in version control.

Prometheus stores metrics on an `emptyDir` with short retention, deliberately in
contrast to the retrieval agent's PersistentVolume: metrics here are disposable,
the index is not. Both workloads carry explicit memory limits, because the
cluster shares a 16GB laptop with Docker's VM and local inference.

## Known gaps

- Schema-invalid tool input is rejected by FastMCP *before* the instrumented
  function runs, so validation failures never appear as
  `tool_calls_total{status="error"}`. Worth knowing before building Grafana panels
  on that label.
- `ddgs` scrapes HTML rather than calling a supported API. It rate-limits under
  rapid use; swapping in a keyed search API means changing `SearchService` only.
- A small local model will skip `index_documents` unless told emphatically to call
  it, because indexing benefits future runs and does nothing for the answer in
  progress. Currently handled in the system prompt; making it deterministic would
  mean the orchestrator hardcoding the search/index pairing, which trades away the
  routing generality the tool-ownership map provides.

## Build plan

- Week 1: MCP servers -> orchestrator graph -> end-to-end local run **(done)**
- Week 2: Containerize + deploy to local K8s -> observability -> evaluation
  suite -> documentation

## Local-only, no cloud budget

Entire system runs on a local Kubernetes cluster (kind/minikube) and
local LLM inference (Ollama), with a thin Claude API fallback for harder
reasoning steps planned. No cloud GPU rental required, in contrast to an
inference-serving-style project, which was considered and set aside
specifically because of the budget constraint.

Verified working on 4GB VRAM (RTX 3050 Ti laptop): `qwen3:4b` for orchestration,
`nomic-embed-text` for embeddings. Web search via `ddgs` needs no API key.
