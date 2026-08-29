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
instantly, and could be scaled to any number of replicas (with a caveat about the
transport - see below). Merging all three into
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

## A stateless agent is not a stateless protocol

The section above says the stateless agents can be scaled to any number of
replicas. That was asserted for a long time while every `replicas:` in the repo
said `1`, and when it was finally tested it turned out to be false as written.

Two replicas of the research agent behind its Service, and the client failed
three times out of three with `McpError: Session terminated`. At one replica the
identical code succeeded three out of three.

The cause is the transport, not the agent. MCP Streamable HTTP issues an
`Mcp-Session-Id`; the handshake established a session on one pod, kube-proxy
round-robined the next request to the other pod, and that pod had never heard of
the id - so it answered 404 and the client concluded the session was gone.

Holding no state does not make a service horizontally scalable. The protocol in
front of it has to be stateless too, and these two properties were being treated
as one.

The fix is `FastMCP(..., stateless_http=True)` on all three agents: no session
ids, every request self-contained, any replica able to serve any of them. The
test that failed 3/3 then passed 10/10, and eight tool calls through the Service
split 3/5 across the two pods. Nothing was given up, because the tools are pure
functions and the retrieval agent keeps its state in sqlite rather than in a
session.

Two things follow that are worth keeping in mind:

- **The retrieval agent still cannot be scaled**, and now for a cleaner reason.
  It is not the transport - it is the volume. That is the distinction this
  document was trying to draw all along, and it is sharper once the protocol
  stops being a confounding factor.
- **Session affinity would have hidden this.** `sessionAffinity: ClientIP` on the
  Service pins a client to one backend, which would make the errors disappear
  while sending every request to a single replica. The second pod would sit idle
  and the scaling claim would look proven when nothing had changed.

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

That was written before anyone checked what containment is actually available on
this machine. It is designed out in full below - "Design: the code execution
sandbox, and why it is not built" - and the conclusion is that the async worker
is justified by slow execution, and slow execution is the part this hardware
cannot safely host.

## Design: the code execution sandbox, and why it is not built

The section above names a Redis-backed async worker running a code execution
sandbox as the stretch goal. This is the design for it, written before any code,
because the containment argument IS the feature. An execution path that runs and
is not safe would be worse than no execution path at all, and every other agent
here is safe by accident - they are pure functions over text and there is nothing
to contain.

### The threat model, stated plainly

The code would not come from the person using this. It would come from the
orchestrating model, which is frequently repeating something it read in a
DuckDuckGo result. That is an untrusted input path that already exists in this
system: `search_web` scrapes arbitrary HTML, `index_documents` stores it, and
`retrieve` feeds it back to the model as evidence. Adding execution turns a
prompt-injection in a search result into code that runs on the machine.

What such code would be trying to do, in rough order of how much it would matter:

- **Read the filesystem.** The retrieval agent's volume holds the corpus; the
  node holds a kubeconfig with cluster-admin.
- **Reach the network.** Pods can reach every Service, and `host.docker.internal`
  reaches the host's loopback - which is exactly how the agents reach Ollama.
  Egress is not blocked anywhere in this cluster.
- **Exhaust the machine.** A fork bomb or an allocation loop on a laptop that is
  already the binding constraint, and which has twice had the GPU driver fall
  over under memory pressure.
- **Escape the container.** The least likely and the most severe.

### Containment options, ranked against THIS hardware

Measured on this machine rather than assumed, because most of the standard
answers turn out not to be available here.

**1. A restricted AST subset - no arbitrary execution at all.** Walk the parsed
tree and refuse anything not on a whitelist: literals, arithmetic, comparisons,
a handful of builtins. No imports, no attribute access, no calls to anything not
explicitly allowed, with step and wall-clock ceilings. This is not a sandbox
around dangerous operations; it is the absence of dangerous operations. The
code-analysis agent already parses with `ast` for exactly this kind of walk, so
the machinery exists. Honest name: an *evaluator*, not an executor.

**2. Unix resource limits in a subprocess.** `RLIMIT_AS` and `RLIMIT_CPU` were
verified present inside the agent containers (Linux). They are NOT present on the
Windows host - `import resource` raises `ModuleNotFoundError` there - and
`docs/RUNBOOK.md` documents running the agents as host processes as the normal
way to develop. So a containment scheme built on rlimits would work in the
cluster and do NOTHING in host-process mode, while looking identical in the code.
That asymmetry is disqualifying on its own: this project has repeatedly been bitten
by protections that were silently weaker than they appeared, and one that depends
on which way you happen to be running is the same failure wearing a new hat.

**3. A container per execution.** The standard answer, and unavailable without
making things worse. `/var/run/docker.sock` is not present inside a pod - checked -
so it would have to be deliberately mounted from the host. Mounting the host's
Docker socket into a pod is equivalent to handing that pod root on the host, which
means the containment mechanism would be a larger hole than the thing being
contained.

**4. gVisor or Kata.** The right tool. Neither runs under kind on Docker Desktop
on Windows, so this is not an option on this machine at all.

### What the async worker buys, and what it costs

It buys a real thing. MCP's request/response model is synchronous, and this
client sets explicit timeouts of 30s at the HTTP layer and 120s for a JSON-RPC
response. Anything slower than that has to be decoupled from the call, and
"run this code" is the first tool here that could legitimately take minutes. That
is a genuine architectural reason, not a resume line.

The costs are concrete:

- **A queue is a second stateful service.** This document argues that the
  retrieval agent owning a durable index is what makes the multi-server split
  load-bearing rather than decorative. Adding Redis adds a second component with
  state, and the argument would need rewriting rather than merely extending.
- **Memory.** The node currently declares 3078Mi of limits against a 16GB machine
  that also runs Docker's VM and local inference. Redis plus a worker is another
  ~300-500Mi, on the constraint that has already caused two driver crashes.
- **The result protocol changes shape.** A tool that returns a job id and is
  polled is a different contract from one that returns an answer, and every
  consumer - the graph, the SSE layer, the eval harness - assumes the latter.

### Recommendation

Build option 1, the restricted evaluator, and call it what it is. It is
genuinely safe because nothing dangerous is reachable, it needs no new
infrastructure, it runs identically on Windows and in the cluster, and it can
state its own limits the way `analyze_code` already does - which is the pattern
this project settled on after the stub taught it that a tool implying more than
it knows is worse than a tool that fails.

Do NOT build options 2 or 3 on this hardware. Option 2 is a protection that
evaporates in the documented development mode; option 3 trades a small risk for
a root-equivalent one.

Leave the async worker unbuilt until there is something that actually needs it.
A restricted evaluator returns in milliseconds, so wrapping it in a queue would
be adding a second stateful service and a new tool contract to solve a latency
problem that does not exist. The honest version of the stretch goal is: the
async pattern is justified by slow execution, and slow execution is the part
this hardware cannot safely host.

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

## Hardware constraints, and what they forced

The "everything runs locally, no cloud budget" constraint collided with a real
ceiling: a thin laptop with 16GB RAM, a 4GB laptop GPU, and an H-series CPU.
Local LLM inference is a sustained 100%-utilisation workload by nature, and this
chassis cannot hold that without thermal throttling - which slows everything
else on the machine, including the cluster it is hosting.

What the measurements showed:

- **The desktop takes the VRAM first.** Windows compositing, Explorer and a
  browser occupy 1.3-1.6GB of the 4GB card before any model loads.
- **Two models do not fit.** The orchestrator model and the embedding model were
  both held resident by Ollama, peaking at 3806MiB used and **157MiB free**. Any
  transient allocation on top of that fails.
- **That is what crashed it.** Two `VIDEO_TDR_FAILURE` bugchecks in
  `nvlddmkm.sys`, the second carrying `STATUS_INSUFFICIENT_RESOURCES` - the
  driver stopped responding under memory pressure and Windows could not reset it.
- **Model size is a thermal decision, not just a quality one.** One short call on
  `qwen3:4b` took the GPU from 55C to 84C; `qwen3:1.7b` reached 64C and finished
  four times faster in wall-clock. Both emit well-formed tool calls, but the
  smaller model is less reliable at *deciding* to call one - the evaluation
  harness later showed it skipping `retrieve` where the 4B does not. Choosing
  the small model bought thermal headroom and cost tool-selection accuracy;
  both halves of that trade are measured rather than assumed.
- **Partial GPU offload bought nothing.** Capping layers at 12 measured the same
  throughput as running entirely on CPU while consuming ~800MiB more VRAM.

What changed as a result: `qwen3:1.7b` is the default, context is capped, models
unload promptly, and the MCP client has explicit timeouts (a wedged dependency
used to hang the orchestrator forever, since the max-iteration guardrail bounds
loop count and not call duration).

The honest conclusion is that a 4GB laptop GPU is under-specified for this
workload, and the documented Claude API fallback - already the plan for harder
reasoning - is the real answer for anything sustained. Running the Kubernetes
cluster and local inference simultaneously is also avoidable: the agents run
fine as host processes while iterating, and the cluster is for demonstrating the
Kubernetes story.

## Known gaps

- Schema-invalid tool input is rejected by FastMCP *before* the instrumented
  function runs, so validation failures never appear as
  `tool_calls_total{status="error"}`. Worth knowing before building Grafana panels
  on that label.
- `ddgs` scrapes HTML rather than calling a supported API. It rate-limits under
  rapid use; swapping in a keyed search API means changing `SearchService` only.
- ~~A small local model will skip `index_documents`~~ - **resolved**, see
  "Decision: the producer indexes its own output" below.

## Decision: the producer indexes its own output

`index_documents` was never being called. The system prompt asks the model to
index after searching; the small local model reliably declines, and it is right
to - indexing pays off on the NEXT run and only costs tokens on this one. The
result was a durable index that nothing ever wrote to, which made the retrieval
agent's persistence story hollow.

The Known gaps entry above framed the only alternative as the orchestrator
calling `index_documents` itself after every `search_web`, and rejected it,
because hardcoding two specific tools into the router trades away the routing
generality that the tool-ownership map exists to provide. That reasoning still
holds. What it missed is a third option: the side effect can live with the agent
that PRODUCED the data. The research agent now indexes its own results, and the
loop is untouched - a tool simply does its own housekeeping.

This is real coupling. The research agent has an opinion about another agent now,
where before it had none, and that is a cost rather than a free win. Two
properties are what make it acceptable:

- **Best effort.** If the retrieval agent is unreachable, the search still
  succeeds. Verified by pointing the research agent at a dead address in the
  cluster: the search returned normally in 2.5s. The isolation this document
  claims for the multi-server split is only true if clients are written to
  uphold it, and this is the first place in the project where that was actually
  at stake.
- **Counted.** `search_results_indexed_total{status="stored"|"skipped"}`.
  Best-effort work that fails silently is precisely how a corpus stays empty
  while every dashboard looks healthy - the same shape of failure as the
  analyze_code stub and the eval case that measured a side effect.

That counter proved its worth immediately. The first deployment indexed nothing:
the client called `asyncio.run` from a FastMCP sync tool, which runs on the event
loop thread rather than a worker, so it raised and the best-effort handler
swallowed it. Searches kept succeeding. Without the skipped counter and a log
line, an empty corpus would have looked exactly like a working one.

Measured: one search took the corpus from 14 documents to 19. Repeats do not
accumulate, because the store now skips text it already holds.

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

Verified working on 4GB VRAM (RTX 3050 Ti laptop): `qwen3:1.7b` for orchestration,
`nomic-embed-text` for embeddings. Web search via `ddgs` needs no API key.
