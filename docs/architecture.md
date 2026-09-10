# Architecture Notes

## What is in this document

Thirty-two sections, most of them short. They are grouped here rather than
listed in order, because the order is chronological - the document grew as the
project did - and chronology is rarely what a reader wants.

**How the system is put together.** The decisions taken before anything was
measured - what the pieces are and why they are separate.

- [Why MCP, and why multiple servers](#why-mcp-and-why-multiple-servers)
- [The agents](#the-agents)
- [Stateful vs stateless, and why it matters here](#stateful-vs-stateless-and-why-it-matters-here)
- [A stateless agent is not a stateless protocol](#a-stateless-agent-is-not-a-stateless-protocol)
- [Transport: HTTP+SSE to Streamable HTTP](#transport-httpsse-to-streamable-http)
- [Sync vs. async tool execution](#sync-vs-async-tool-execution)
- [Retrieval design notes](#retrieval-design-notes)
- [Observability](#observability)
- [Decision: the producer indexes its own output](#decision-the-producer-indexes-its-own-output)
- [Decision: tool metrics are recorded at the MCP boundary](#decision-tool-metrics-are-recorded-at-the-mcp-boundary)

**What was deliberately not built.** Absences are decisions too, and the
reasoning for them is easier to lose than the reasoning for code.

- [Decision: why the summarizer agent was removed](#decision-why-the-summarizer-agent-was-removed)
- [Design: the code execution sandbox, and why it is not built](#design-the-code-execution-sandbox-and-why-it-is-not-built)
- [Hardware constraints, and what they forced](#hardware-constraints-and-what-they-forced)
- [Local-only, no cloud budget](#local-only-no-cloud-budget)

**What measurement changed.** Each of these began as an assumption that a run
contradicted. They are the ones worth reading if you want to know how the
system actually behaves rather than how it was meant to.

- [Grounding, and why the stubs had to go](#grounding-and-why-the-stubs-had-to-go)
- [Adding a tool is mechanically free and behaviourally not](#adding-a-tool-is-mechanically-free-and-behaviourally-not)
- [The corpus learned to vouch for a fiction](#the-corpus-learned-to-vouch-for-a-fiction)
- [Decision: a failed search is not an absence](#decision-a-failed-search-is-not-an-absence)
- [Decision: the cache remembers evidence and nothing else](#decision-the-cache-remembers-evidence-and-nothing-else)
- [Saying which results are not about what was asked](#saying-which-results-are-not-about-what-was-asked)
- [The guardrail for answering from nothing](#the-guardrail-for-answering-from-nothing)
- [When asking again does not work](#when-asking-again-does-not-work)
- [A signal for claims about a subject the evidence never mentioned](#a-signal-for-claims-about-a-subject-the-evidence-never-mentioned)
- [One case that checks whether the answer is true](#one-case-that-checks-whether-the-answer-is-true)

**How the claims here are kept honest.** A document full of numbers is only
worth as much as the ability to re-take them.

- [The measurements are code now](#the-measurements-are-code-now)
- [The instrument had the bug it was built to find](#the-instrument-had-the-bug-it-was-built-to-find)
- [Decision: what the alert rules are allowed to assume](#decision-what-the-alert-rules-are-allowed-to-assume)
- [A recovery procedure that referred to itself](#a-recovery-procedure-that-referred-to-itself)
- [The images are checked by reading, not by building](#the-images-are-checked-by-reading-not-by-building)
- [One id per run, across four services](#one-id-per-run-across-four-services)
- [Known gaps](#known-gaps)
- [Build plan](#build-plan)
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
| code-analysis | `analyze_code`, `evaluate_expression` | none | Deployment |

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

### Recommendation, and what was built

Option 1 was built - `agents/code_analysis_agent/evaluator.py`, exposed as
`evaluate_expression`. It is an interpreter rather than `compile()` plus
`eval()`, because whitelisting node types and then evaluating checks the shape
of an expression and nothing about what it does.

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

## Adding a tool is mechanically free and behaviourally not

This document says more than once that adding a fourth agent needs no changes -
the tool-ownership map is built at discovery, so dispatch is a lookup and no
router has to learn about it. That is true, and it is only half the story.

Adding `evaluate_expression` to the code-analysis agent changed how the model
routes a question that has nothing to do with arithmetic.

`cached-retrieval` asks "What is the Model Context Protocol?" and the corpus is
seeded with the answer. Before the evaluator existed it called `retrieve` in 4
of 4 runs. Afterwards it called `search_web` in 6 of 6 - the eval run plus five
repeats.

That could have been variance, so it was tested directly. The identical task was
run five more times with one change: `evaluate_expression` filtered out of the
tool list handed to the model. Same prompt, same model, same corpus, same
everything else.

| tool list | routing |
|---|---|
| five tools (evaluator visible) | 6/6 `search_web` |
| four tools (evaluator hidden) | 5/5 `retrieve` |

The presence of an unrelated tool is what moved it. Nothing else differed.

Three things follow that are worth holding onto:

- **A routing measurement is only valid for the tool set it was taken with.**
  The earlier "4 of 4 `retrieve`" was a true measurement that a later change
  invalidated without touching the case, the prompt, or the model.
- **Tool descriptions compete for attention.** The model sees one list and picks
  from it; a fifth entry changes the shape of that decision even when it is
  irrelevant to the question. This is a property of tool-calling models, not a
  bug in this system, but a system that adds agents freely has to expect it.
- **The tool-ownership map removes the mechanical cost of adding an agent, not
  the behavioural one.** The dispatch claim stands. The implication some readers
  would draw from it - that a new agent cannot affect existing behaviour - does
  not.

The eval suite is what makes this visible at all. Without a case pinning
`cached-retrieval`'s expected tool, adding the evaluator would have silently
changed retrieval routing across the system and nothing would have said so.

### The fix: guidance belongs next to the tool

The regression was closed by changing one thing - `retrieve`'s own description.

The instruction "try retrieve before search_web" existed only in the system
prompt, where it competed with every tool description at once and lost ground
as the list grew. Meanwhile `retrieve` had the second-shortest description in
the system, 135 characters that said what it did and nothing about when to
reach for it; `search_web`, at 90 characters, was winning on fit rather than
merit. Moving the rule into the description puts it where the choice is
actually made.

| retrieve description | routing on the same task |
|---|---|
| 135 chars, "what it does" | 1/5 `retrieve`, 4/5 `search_web` |
| 777 chars, "try me before search_web, and why" | **5/5 `retrieve`** |

Nothing else changed - same prompt, same model, same corpus, same five tools.
The eval agrees: `cached-retrieval` passes its required-tool check again.

The general lesson is the one worth keeping. **A tool description is not
documentation; it is the argument for choosing that tool over its neighbours**,
and it has to keep holding as neighbours are added. A system prompt does not
scale that way, because every new tool dilutes it while each description stays
attached to the choice it informs.

## Known gaps

- ~~Schema-invalid tool input never appears as
  `tool_calls_total{status="error"}`~~ - **resolved**, see "Decision: tool
  metrics are recorded at the MCP boundary" below.
- `ddgs` scrapes HTML rather than calling a supported API, and rate-limits under
  rapid use; swapping in a keyed search API means changing `SearchService` only.
  The throttling itself is no longer silent - see "Decision: a failed search is
  not an absence" below - and repeated queries no longer re-scrape, which is
  where the throttling was worst: see "Decision: the cache remembers evidence and
  nothing else". **Still open**, because the cache narrows the exposure rather
  than removing it. A first search of anything is still a scrape, the cache is
  per-pod across two replicas, and nothing here makes DuckDuckGo a supported
  interface.
- ~~A small local model will skip `index_documents`~~ - **resolved**, see
  "Decision: the producer indexes its own output" below.

## Decision: what the alert rules are allowed to assume

Prometheus scraped every service and Grafana drew the result, and for a while
that was described as observability. It was not. Nothing in the system said
what counted as wrong. `/api/v1/rules` returned an empty list, so the whole
arrangement worked exactly as long as a human happened to be looking at a
dashboard, and no longer.

Fourteen rules now sit in the `prometheus-rules` ConfigMap, in four groups: is
the system there, can it do the work, is the one irreplaceable thing still
there, is what it produces trustworthy. Nine shipped first and covered one run
outcome in six; see below.

**There is no Alertmanager, and that is the honest limit.** Alertmanager routes
alerts - to email, to Slack, to PagerDuty - and this project has none of those.
Prometheus evaluates rules and shows what is firing at `/alerts` on its own.
So the system can now say it is unwell; nobody is told. Those are different
claims and only the first one is being made.

### Where the thresholds came from

This was the part that needed care. Prometheus keeps six hours of data in an
`emptyDir`, deliberately - the contrast with the retrieval agent's volume is
the point - so there is no history to fit thresholds against. It would have
been easy to write `> 0.05` everywhere and produce nine rules that fire on
nothing in particular, which is worse than no rules at all, because a rule that
has never fired looks like coverage.

Every threshold instead derives from something already fixed:

- **Rejections** alert at more than zero, because `orchestrator/metrics.py`
  already says what that counter means: a rising rejection count means the
  queue cap is being hit and people are being turned away. With
  `MAX_CONCURRENT_RUNS` at 1 and `MAX_QUEUED_RUNS` at 4, waiting is
  backpressure working, and a rejection is a request refused outright.
- **Discovery reaching no agents** also alerts at more than zero, for a
  stronger reason. Discovery is best-effort by design: an unreachable agent is
  logged and the run proceeds with a partial toolset. Reaching *no* agent is
  the one case that tolerance cannot absorb, because the run then has zero
  tools - which is the answering-from-nothing failure the reground guardrail
  exists to catch.
- **Run duration** uses the histogram's own top bucket, 300s, chosen when that
  histogram was defined as the point past which a run is off the anticipated
  scale.
- Where no such anchor existed, the rule asks whether a failure is the
  **majority** of what happened. "Most runs are doing this" needs no
  calibration to be alarming, and both grounding ratios are bounded at 1
  because `MAX_REGROUNDS` and `MAX_NUDGES` are 1.

### Regrounds and nudges are not alerted on individually

A reground firing means the guardrail worked. Alerting on each one would raise
an alarm every time the system successfully refused to fabricate, and the
obvious way to silence it would be to remove the guardrail. Only the proportion
is abnormal: over half of recent runs having answered from empty evidence
before being sent back says the corpus or the search path has degraded, not
that the guardrail is doing its job.

The tests encode that distinction directly. One reground in ten runs must
produce no alert; nine in ten must.

### An alert that could never fire

`RunsSlowerThanDesigned` was written as `> 300` and would never have fired
under any circumstance. `histogram_quantile` cannot return a value above the
highest finite bucket bound: once the quantile falls in the `+Inf` bucket it
reports that bound and stops. With every single run taking an hour, the
expression still returns exactly 300.

This was not reasoned out. The rule was written, a test was added asserting it
fires, and promtool returned `3E+02` for a series whose mass was entirely above
300. The rule is now `>= 300` and that case is pinned in
`tests/alerts_test.yml`.

It is worth being clear about how close this came to shipping. The rule loaded
cleanly, `promtool check rules` passed it, Prometheus reported its health as
`ok`, and it sat at `inactive` - which is exactly what a correct rule looks
like on a healthy system. Nothing distinguishes a rule that is not firing from
one that cannot fire, except evaluating it against data that should trigger it.

### The rules covered one outcome in six

Shipped, the nine rules watched exactly one of the six values
`orchestrator_runs_total` can carry. `failed`, `truncated` and `unanswered` had
nothing at all, and `no_tools` was covered only by accident.

It was found the honest way rather than by review. Ollama died mid-session the
following day, every run came back `ConnectionError`, the counter recorded each
one correctly - and nothing fired. The instrumentation was right; the thing
reading it was not looking.

This is the third instance of the same shape in a week, and the pattern is now
explicit enough to state: **a rule that has never fired and a rule that cannot
fire are indistinguishable from the outside.** An alert that could never fire,
a recovery procedure that referred to itself, and now a set of rules that simply
did not mention most of the ways a run can end. None was visible by reading; all
three were found by asking the running system.

So the fix is not only three more rules. `test_every_run_outcome_is_alerted_or_exempt`
reads the outcomes out of `api.py` and requires each one to be either selected
by a rule or listed as deliberately exempt with a reason. A missing rule is now
a test failure rather than a silence.

It reads the code and not the comment in `metrics.py` that lists the outcomes,
because that comment was itself wrong - it omitted `unanswered` - and the rules
written against it inherited the omission. A list maintained by hand next to the
thing it describes will drift from it; the only question is whether anything
notices.

### Two outcomes are deliberately not alerted

`answered` is the run working. `unanswered` is the loop ending by saying it
could not answer, which is the honest path this project spent weeks building.
Alerting on it would raise an alarm every time the system correctly declined to
invent something, and the obvious way to silence that alarm would be to remove
the guardrail. Both are recorded in `EXEMPT_OUTCOMES` with the reasoning, so
that being unalerted is a decision rather than an oversight - which is the whole
distinction the coverage test exists to enforce.

### Why the failure alert is a proportion

`MostRunsFailing` looks like it should be a per-occurrence rule. A run that
raised is never good, and `RunsBeingTurnedAway` fires on a single rejection for
exactly that reasoning.

The difference is what `failed` actually counts. It is `RunState`'s pessimistic
default, so anything that escapes without setting an outcome lands there - and
that includes a client disconnecting mid-run, because a generator's `finally`
runs when it is closed. Closing the browser tab during a slow answer records a
failure.

That was measured rather than reasoned about: killing a `curl` three seconds
into a run incremented `orchestrator_runs_total{outcome="failed"}`. A rule
firing on any occurrence would therefore alert on somebody navigating away,
which is ordinary use. A majority cannot be explained that way, and a dead model
backend produces precisely that.

The same measurement argues against "fixing" the conflation by splitting the
label. A disconnect and an exception are both runs that ended without an answer,
and the run that matters - did the user get one - is the same question in both
cases.

### The failure that hides behind a working system

Alerting on the corpus came last, and it is the one gap none of the earlier
reasoning would have found, because the corpus failing does not look like a
failure.

Every other rule here watches something that stops working. An emptied index
does not. `retrieve` reports no evidence - correctly, there is none - the model
falls back to `search_web`, search succeeds, and answers keep arriving at the
page. The reground guardrail does not fire, because it requires *every* tool
that ran to have come back empty, and search did not. Run outcomes stay
`answered`. Latency does not move. Nothing in the first twelve rules is
sensitive to it at all.

What makes that worse than an outage is that the corpus is the only state here
with no source to rebuild it from. The agents are stateless and redeploy from
their images; Prometheus keeps six disposable hours in an `emptyDir`. The index
is a PersistentVolume holding documents accumulated since the first week, and
losing it silently means losing it permanently.

`CorpusIsEmpty` can be a bare `== 0` only because the gauge is not incremental.
`server.py` sets it from `store.count()` at module load and again after each
index, so it carries the true size from the moment the process starts - checked
against the running agent, where the gauge, `SELECT COUNT(*)` and Prometheus all
said 401. Had it only been set on indexing, zero would have meant "nothing
indexed since this pod started" and the rule would have fired on every restart.

`CorpusMostlyGone` leans on a property of the design rather than a tuned number:
the corpus only grows in normal operation, because `index_documents` adds and
nothing in the serving path deletes. Any sustained decrease is a maintenance
script, a restore, or data loss.

### An alert that fires when you meant it

Both corpus rules will fire on deliberate acts - a fresh deploy, a restore to an
older snapshot, a purge that removed enough. The first instinct is to suppress
that, and it is wrong.

Neither is a false alarm. The rule is describing the state accurately; what it
cannot know is intent, and intent is the one thing the operator does know. The
asymmetry decides it: somebody who has just run a restore can dismiss the alert
in a second, while nobody can reconstruct a silent loss after the fact. On the
live corpus the threshold computes to 200.5 documents, so the restore down to
170 performed the day before would have fired it - which is precisely the event
worth surfacing if it had been an accident rather than a test.

The cost of getting this wrong in the other direction is a muted rule. An alert
that fires on ordinary operations gets ignored, then silenced, and is then
absent on the day it mattered. That is why the promtool cases spend most of
their effort on the negative side: a corpus that was always empty has lost
nothing, a growing corpus is fine, and the real purge this project ran - 34
documents out of several hundred - must not read as data loss.

### How they are checked

Two layers, the same split as the image checks:

- `tests/test_alert_rules.py` reads. Every metric an alert references must
  resolve to a `Counter` or `Histogram` declared in this repo; every alert must
  carry a severity and both annotations; the rules ConfigMap must actually be
  mounted where `rule_files` points. A typo in a metric name is valid PromQL -
  a selector matching no series - so without this a misspelled rule loads,
  reports healthy, and never fires.
- `tests/alerts_test.yml` evaluates, through `promtool test rules` in CI,
  against the same Prometheus image the cluster runs.

`test_every_alert_has_a_test_that_it_fires` ties the two together and earned
its place immediately, by failing: two of the nine rules had no case asserting
they fire, and writing one of them is what exposed the `>= 300` bug.

The rules were then mutation-tested - a raised threshold, a comparison against
the wrong value, a dropped grace period, a lost `by (tool_name)` - and all five
breaks were caught. That check mattered more than it sounds: the first attempt
at it reported success against a fixture that had transferred as zero bytes,
because **`promtool test rules` reports SUCCESS on an empty test file**. CI now
refuses to run if either file is empty.

Finally, one rule was temporarily inverted against the live cluster to confirm
the path end to end: it moved from `inactive` to `pending` to `firing`, with
per-pod labels and annotations correctly templated, and returned to silence
when restored. promtool proves the rules; only that proves the running server
surfaces them.

## A recovery procedure that referred to itself

The corpus is the only state here that redeploying cannot rebuild, and the docs
described restoring it in three places: the backfill note, the note saying
`backfill_claims.py` can be re-run whenever a corpus is restored, and the
disaster-recovery note explaining that a volume rescued with `docker cp`
"restores into a fresh cluster with the same `kubectl exec` pipe used for any
other backup".

There was no such pipe. Every `kubectl exec -i ... python -` in the repo streams
a *script* into the pod; none of them writes a database. Three recovery paths
terminated in a step that had never been written or run, and the two snapshots
sitting on the volume could not have been restored by any documented means.

This is the same shape as the alert rule that could never fire, and it is worth
naming as a category: **a procedure nobody has executed is a hypothesis.** Both
were plausible, both were written down in good faith, and both were false in a
way that reading could not reveal - one because a rule that cannot fire looks
exactly like a rule that has not fired, the other because a cross-reference to a
procedure looks exactly like a procedure.

### What verification changed

Neither of the two substantive design decisions in `restore_corpus.py` survived
contact with the real system in its original form.

**It refused a corpus predating the provenance split.** A backup from before
`claimed_source` existed was rejected as malformed - which would have blocked
precisely the recovery the runbook describes, since `store.py` adds that column
on startup and `backfill_claims.py` exists to move the old labels across
afterwards. Missing *migrated* columns now warn; only a corpus that cannot be
read, or holds nothing, blocks. Found by restoring the real 170-document backup
from 2026-09-03 rather than by thinking about it.

**It described the wrong reason for restarting the pod.** The script said the
agent holds the database open, keeps the old inode after the swap, and serves
the stale corpus until restarted. `store.py` opens a connection per operation
and closes it - deliberately, since FastMCP runs sync tools in a thread pool and
a shared sqlite3 connection would eventually be used from the wrong thread.
There is no long-lived handle and the data is live immediately, measured by
swapping a 391-document corpus for a 170-document one and watching the running
agent report 170.

The restart is for the schema. `_create_schema` runs once, in `__init__`, so
restoring a pre-split corpus leaves the agent running against a database with no
`claimed_source` column: reads work, while `index_documents` and the corpus
audit fail with `no such column`. Confirmed by running that query and watching
it raise, then deleting the pod and watching the column appear.

Both corrections make the guidance narrower and more useful. The first draft
would have told an operator to restart every time, which is harmless but trains
them to ignore the instruction on the one occasion it matters.

### Why it verifies before it moves anything

The failure to design against is not a missing backup. That announces itself.
It is a restore that completes and replaces a working index with a broken one,
because a transfer that stops partway still produces a file and SQLite opens the
intact prefix without complaint.

Cutting a 400-document corpus at 99% settled what the check had to be: it opens,
reports 400 documents and 400 matching vectors, and passes every count-based
test. Only `PRAGMA integrity_check` notices, so the check reads every page
rather than trusting counts. Below about 90% the file fails to open at all, but
the narrow band near the top is exactly where a real interrupted transfer lands.

The whole path was then run end to end against the live corpus: a truncated file
refused in `apply` mode with the corpus untouched at 391, a real restore down to
170 and back, and a query afterwards that retrieved and then indexed - the
second being the real check, since indexing writes `claimed_source` and fails
loudly if the schema is wrong.

## The instrument had the bug it was built to find

`eval/experiment.py` prints a line like `fabricated 0/8`, and those fractions
are what README.md quotes as the durable record - "0 fabrications in 8", "9 runs
in 10 calling `retrieve`". The denominator was whatever survived.

Failed runs were printed as `ERR` and then dropped. Eight runs where six could
not reach the model reported `fabricated 0/2`: a clean result on a small sample,
indistinguishable from a deliberate two-run measurement. Nothing in the summary
said six were missing.

It was found by hitting it rather than by reading. A repeat run against a host
where Ollama had moved ports printed:

```
  ERR run 1: ConnectionError: Failed to connect to Ollama...
  ERR run 2: ConnectionError: Failed to connect to Ollama...
  ERR run 3: ConnectionError: Failed to connect to Ollama...
  ERR run 4: ConnectionError: Failed to connect to Ollama...

  -> arithmetic-uses-the-evaluator: fabricated 0/4
```

Four total failures and a bottom line that reads like success. The `ERR` rows
are right there, so nothing was hidden - but the summary is the part that gets
copied into a table, and it travelled without them.

This is the fourth instance of one pattern in a week, and the first inside the
measuring equipment. An alert rule that could never fire. A recovery procedure
that referred to itself. A set of rules that ignored most ways a run can end.
And now a harness that reports a confident number with no evidence behind it -
which is precisely the failure the harness exists to detect in the system it
measures.

The common thread is worth naming: **absence and a clean result look identical
unless something is counting.** A rule at `inactive`, a procedure nobody ran, an
outcome no rule selects, a denominator with the failures removed. In each case
the artefact was well-formed and said nothing false; it simply did not say the
one thing that would have revealed it.

### What changed

`_report` now takes the number of runs asked for. When some are lost it names
them in the same line as the fraction, so a quoted number carries its own
caveat:

```
  -> case: fabricated 0/2   (6 of 8 runs failed and are NOT counted below)
```

When every run is lost it refuses to print a fraction at all, because `0/0` is
not a result and there must be nothing there to copy:

```
  -> case: NOTHING MEASURED - all 4 run(s) failed.
     There is no result here to quote. See the ERR lines above.
```

`eval/run_eval.py` had a milder version of the same thing. Errored cases print
an `ERROR` row and are filtered out of the scoring, and the means below were
computed over the survivors with no denominator - five failures out of nine
still produced a confident `mean grounding: 5.0 / 5`. Each mean now carries
`(over N of M cases)`, the excluded cases are tallied, and a suite where nothing
scored says so rather than averaging an empty set.

`ab()` was left alone deliberately. It does not catch errors, so a failure ends
the run rather than shrinking it, and every run it reports on completed by
definition. Passing it a count would have implied a check it does not need.

Both fixes were mutation-tested by reverting each to its previous behaviour and
confirming the new tests fail - which is the only way to know a test written
after the fact would have caught the thing it describes.

## Decision: the cache remembers evidence and nothing else

`ddgs` scrapes HTML rather than calling a supported API, and throttles under
rapid use. That is the open gap in "Known gaps", and it is felt hardest by the
measurement tools: an eval run puts nine cases through the loop, and
`eval/experiment.py repeat` puts one case through it eight times. The same few
queries are asked over and over within a few minutes, and every repetition was a
fresh scrape.

Measured in the pod against real DuckDuckGo: **2.78s to fetch, 0.00s to serve
the same query again**, byte-identical text.

### The part that needed deciding

Caching the successes is obvious. What to do with everything else is not, and it
is where a search cache stops being a performance question.

This agent produces five other outcomes: rate-limited, failed, nothing matched,
every result was an advertisement, and results-with-a-coverage-note. Four of
those are statements about **a moment**, not about the query. Remembering one
would take a passing throttle and serve it to every later run for the whole TTL,
each of them correctly told that the lookup could not happen - long after it
could.

That is not a hypothetical failure for this project. The corpus once held a
message about a failed search, indexed as though it were a document, and a later
`retrieve` returned it as evidence. Same mistake, different store.

So: **cache evidence, never absence.** `SearchOutcome.indexable` already draws
exactly that line for the corpus - it is what decides whether an outcome is
worth storing - so the cache reuses it rather than growing a second predicate
that could drift from the first. The results-with-a-note case is cached, because
a note saying the results are not about the subject is commentary on real
results; the outcome is still evidence. The two hard failures never reach the
cache at all, because they raise.

### It replaces the network call and nothing else

The cache sits inside `SearchService.run`, not in the tool adapter. A hit
returns the same `SearchOutcome` the fetch produced, and the tool goes on to
index and count it exactly as it would have, so every behaviour downstream is
identical to an uncached run.

Indexing on a hit costs nothing worth avoiding: `store.index` filters texts
against what the corpus already holds *before* embedding anything, so a repeat
is a lookup and an early return rather than a round trip to the embedder. Making
a hit take a different path through the rest of the tool would have bought
nothing and given the cache a second way to change behaviour.

### The TTL is sized to the workload, not to the web

Fifteen minutes, and that is a judgement rather than a measurement - nothing
here measures how fast search results change. It is sized to what the cache is
*for*: one eval run and a set of repeats, which is minutes. Short enough that no
answer is built on a result from a previous working session.

`SEARCH_CACHE_TTL=0` turns it off. An experiment that wants live results every
time has to be able to say so, or the cache silently changes what is being
measured - which would be a strange way to repay a project organised around
trustworthy measurement.

### Two replicas, two caches

`research-agent` runs two replicas, so a repeated query is fetched once per pod
rather than once. Eight repeats cost two fetches instead of eight, not the one a
shared cache would give.

A shared cache needs somewhere to share it, and this project deliberately has no
such place - the only stateful service is the retrieval agent, and putting a
search cache in the vector store would confuse two different kinds of memory.
The hit/miss counter is on the dashboard so the real rate is observable rather
than assumed, which matters more than the missing third of it.

Nothing alerts on the hit rate. A cold cache is not a fault, and the rate is a
property of the workload rather than of the system's health - unlike every
metric in the alerting rules, there is no value of it that means something is
wrong.

## The images are checked by reading, not by building

Each agent builds from its own directory and its Dockerfile names the files to
copy one by one. That is the price of the duplication the split buys - each
image stays small and independent - and it has a failure mode with no local
signal: adding `instrumentation.py` needed a matching COPY line, and so did
`coverage.py`. Both were remembered by hand.

The third time would not fail in the build. An image builds perfectly well
without a file nothing in the build references; it fails on deploy as a crash
loop on import, or later still if the missing module is only reached when a
particular tool runs. The tests would not see it either, because they import
from the working tree rather than from the image.

`tests/test_agent_images.py` reads each agent's imports, follows them
transitively, and asserts the Dockerfile copies every local module reachable
from `server.py`. Reading rather than building is the point: the question is not
"does this image build" but "does it contain what the code needs", which needs no
Docker, runs in a fifth of a second, and names the missing file and the
Dockerfile instead of leaving a traceback in a pod log.

It checks the other direction too - a COPY naming a file that no longer exists,
and a maintenance script shipped into an image it does not belong in, since
`backfill_claims.py` is piped through `kubectl exec` and reads the volume rather
than the image.

The same reading catches the other half of "would this image run": a third-party
import with no matching line in that agent's `requirements.txt`. Each agent
carries its own, deliberately, and the tests import from the working tree where
the dev virtualenv has every package any agent might want - so an undeclared
import passes every test and fails at `pip install` in the build, or at import
time in the pod if the package happens to arrive as somebody else's transitive
dependency. Requirement names are normalised the way PyPI does, since
`mcp[cli]>=1.27,<2` declares `mcp` and the import `sqlite_vec` is the
distribution `sqlite-vec`.

That is most of what building the images in CI would have caught, in a fifth of a
second rather than minutes, and without a Docker daemon on the runner. What it
still cannot see is a Dockerfile that no longer builds for some other reason - a
base image that moved, a pip resolution failure - which remains an argument for
building them in CI eventually, just not an urgent one.

The guard has its own guard. Two tests build a fake agent directory with a
missing COPY and assert the check notices, because a test that passes by looking
at the wrong thing is worse than no test, and this one is entirely path
handling. Verified once by hand as well: removing `coverage.py` from the
retrieval agent's Dockerfile fails with "retrieval_agent/Dockerfile does not COPY
['coverage']".

## One id per run, across four services

The metrics say how many tool calls failed and how long they took. They cannot
say which run a particular failure belonged to, and with four services logging
independently, reconstructing one request meant reading three pod logs side by
side and matching on timestamps. That has already produced wrong conclusions
here: runs that looked broken turned out to be a dead port-forward and a partial
tool set rather than anything in the code, and an invalidated measurement was
only caught because one answer mentioned a tool that should not have been
reachable.

Each run now carries an eight-character id, logged at its start and end, on every
MCP call the orchestrator makes, and on every tool call each agent serves.

### It travels as protocol metadata, not as an argument

MCP requests carry a `_meta` field, and `RequestParams.Meta` is declared
`extra="allow"`, so an unknown key rides along untouched. The client passes
`meta={"traceId": ...}` and the agent reads it back off the request context.

The alternative - a `trace_id` argument on every tool - would put it in the JSON
Schema the model is shown. That is one more field for a small model to get wrong,
on every tool, in exchange for nothing it can use. Tool arguments are the model's
business; this is the transport's.

### A contextvar, not a parameter

Threading the id through `build_graph`, every node, and the registry would give
each of them a reason to know about tracing, and the graph is deliberately
ignorant of what is watching it. A contextvar is set at the edge of a run and
read at the edge of an MCP call, with nothing in between aware of it.

It also has to be a contextvar rather than a module global: two runs in flight
would otherwise share whichever id was set last. A test runs two concurrently
and asserts each keeps its own.

### The absence of an id is never an error

Every way the id can be missing - a tool called by hand, an older orchestrator,
no request context at all - reads as `-`. A tool must not stop working because
nobody was watching it, which is the same reasoning as the indexing side effect
that is allowed to fail without failing the search.

### Deploying it revealed that the orchestrator was logging nothing

The first deployment produced no trace lines at all. `basicConfig` lived in
`main.py`'s `__main__` block, and in the container there is no `__main__` of
ours - uvicorn is the entrypoint and imports the module, so the root logger sat
at WARNING and uvicorn's access log was the only thing reaching `kubectl logs`.
Every `logger.info` in the orchestrator had been discarded in the cluster since
it was containerised. Logging is now configured at import, with the level from
`LOG_LEVEL`.

Worth recording because the feature looked finished at that point: the code was
correct, the tests passed, and the thing it exists to produce did not exist.

## The measurements are code now

Every strong claim in this document is a number: routing went 1 in 5 to 5 in 5
when a tool description changed, fabrication went 8 in 8 to 0 in 8 when search
results carried a coverage note, a floor of 0.90 produced 7 fabrications in 8
where 0.70 produced none. Each of those came from a script that was written, run
once and thrown away. The conclusions outlived the evidence for them, which is
the same shape of problem this project keeps finding in its own agents - an
assertion that reads as established because nothing is left to check it against.

`eval/experiment.py` has the two shapes that kept recurring. `repeat` runs a case
through the whole loop N times and reports the fabrication count *and the route
taken*, because the route has twice been the finding: "6 runs in 8 never reached
the web" was how the last fabrication path got narrowed down. `ab` holds the
evidence fixed and varies one thing, which the whole loop cannot do - it reaches
`search_web` about once in eight runs, so a change to search results would need
thirty runs to yield a handful of samples.

`eval/distance_study.py` is the third shape and runs inside the pod, since it
needs the store's embedder and the MCP tool applies the floor under test. It
prints the top hit for every query, because the labels are the one part no code
can verify: a query filed under "the corpus cannot answer this" that the corpus
actually covers would quietly corrupt the threshold chosen from it.

It also carries the trap that caught this project out. The output now counts, for
each candidate floor, how many *wrong-subject* documents it would admit - 0 out
of 2 at 0.70, 2 out of 2 at 0.90 - so the next reader sees the reason 0.90 failed
in the same table that makes 0.90 look attractive.

## Saying which results are not about what was asked

With the corpus cleaned up and the empty-evidence guardrail in place, one
fabrication path was left, and measuring it showed how narrow it had become:
over eight runs of `honest-ignorance`, six never reached the web at all -
`retrieve` came back empty, the guardrail fired, the answer was honest. Two ran a
search, and one of those two fabricated. All remaining invention lived there,
where the search *succeeds* and returns real documents about something else.

The signal is a fact the agent can compute. It knows the query and it knows what
came back, so it can say which capitalised words from the query appear in none of
the results:

> Note: none of these results mention Quazzlemint. They were the closest matches,
> not necessarily results about it.

No model, no inference, nothing to hallucinate. Only capitalised words, and never
the first - that one is capitalised because it starts a sentence. Proper nouns are
where misattribution happens, and a note that fired on ordinary words would be
noise, which is what a model learns to skip.

### The measurement, and what it took to trust it

The full loop now reaches `search_web` about once in eight runs, so measuring the
note through it would need thirty runs for a handful of samples. Instead the step
the note is meant to change was isolated: one real result set, the same question,
the note as the only difference.

| | fabricated |
|---|---|
| without the note | **8 of 8** |
| with the note | **0 of 8** |

Two honest qualifications. An earlier attempt at the same A/B produced 0 of 8
both ways, because the live search had returned a different, less answerable-
looking set of documents - so the note matters when the results invite an answer
and is neutral when they do not. And the **LLM judge flagged none of the eight
fabrications**: it scored them grounded, several at 5 out of 5, because every
claim did trace back to the documents. Only `check_subject_grounding` caught
them, which is the clearest vindication yet of building that signal.

### The instrument was wrong before the result was right

The first run of this A/B reported 2 of 8 without the note and 3 of 8 with it,
which would have said the note made things slightly worse. Reading the flagged
sentences rather than the counts showed three of them were honest denials:
"None of the results mention the Quazzlemint Foundation" and "did not return any
relevant information" were not matched, because the pattern knew `not` and
`never` but not `none`, and its verb list had no `return`.

That is the third time this lexical detector has been widened by observation, and
the failure mode is worth stating plainly: a denial detector that miscounts does
not merely miss things, it produces a number that still looks like a number.
Both directions are now pinned in tests, so widening it cannot quietly let an
invention through.

## One case that checks whether the answer is true

Every signal in this harness asks whether an answer is *supported*. None asked
whether it is *right*, and the gap is not academic. Asked who won the 2018 World
Cup final, the system searched, received real results, and answered:

> The 2018 FIFA World Cup final was won by Argentina.

France won. The required tool was called, the judge scored it grounded because
the claim did trace back to documents about the 2018 final, and no forbidden
phrase existed to catch it. Every signal passed a confidently wrong answer, which
is the failure this project is organised around wearing yet another shape.

`a-checkable-fact` pins both halves: `must_contain` is "France", and
`must_not_contain` carries the phrasings actually observed. Both are needed -
`must_contain` alone passes an answer naming France and Argentina both, and
`must_not_contain` alone passes "I could not find it".

The forbidden phrases are "won by Argentina" and "Argentina won" rather than the
bare word, because a correct answer may legitimately name a losing side. A check
that fires on a right answer is worse than one that misses a wrong answer: it
makes the harness untrustworthy about everything else it reports.

No tool is required. The source does not matter for this question, and pinning
one would fail the case whenever an earlier run had already indexed the answer -
which is exactly what happened while testing it, with all five runs answering
correctly from the corpus rather than from a fresh search.

### It is narrower than it first looks

Other cases already assert correctness: `code-review-finds-a-real-bug` requires
"zero" and the arithmetic case requires "367303". A first version of the test for
this claimed to be the only case checking a fact, and that was wrong - it failed
immediately against the existing cases. The real distinction is that those facts
are derivable from what was handed to the system, while this one must be fetched
from the world. That also makes it the one case that can fail without a
regression, when a rate-limited search leaves nothing to fetch and the corpus
does not already hold the answer.

## When asking again does not work

The nudge asks once when the model writes out a tool call instead of making one.
Measured today: sometimes the model does it again. The nudge budget is spent,
`should_continue` returns "end", and the run finishes with

```json
{"name": "retrieve", "arguments": {"query": "Quazzlemint Foundation 2019 report"}}
```

as its answer. The API holds the first narration back precisely so it never
reaches the page - and then publishes the second one, because by then it is the
final message and looks like a result.

A terminal node replaces it with an explicit non-answer, the same shape as the
max-iteration stop notice, and the run is counted as `unanswered` rather than
`answered`. A dashboard that cannot tell those apart reports a healthy system
that is publishing tool calls as results.

### Only the unambiguous case

`looks_like_a_raw_tool_call` catches a payload that parses as JSON and carries a
tool name. It does not attempt prose - "I will now search for that" - even though
that is the more common narration, because the nudge already covers it and
catching prose reliably means the lexical guessing this project has got wrong
twice already.

The asymmetry decides it. A missed narration ends one run badly. A false positive
replaces a real answer with a failure notice, which is worse than the bug being
fixed, so the detector only fires on something no answer to a human question
could ever look like.

Verified with fakes at both the graph and the stream level rather than in the
cluster: provoking it needs the model to narrate twice in a row, which is not
reliably reproducible on demand. The happy path was checked against the deployed
service.

## The guardrail for answering from nothing

Every fix so far made the corpus honest: a relevance floor so a far neighbour is
not returned as a match, provenance taken from the document rather than the
caller, a label marked unverified when it was only asserted. None of them moved
the fabrication rate, which sat at 1 to 2 runs in 6. That is the measurement that
mattered, because it said the cause was no longer the corpus lying. It was that
an answer was possible at all when nothing supported one.

The `reground` node closes that. If tools ran this turn and **every one of them
reported having nothing**, the model does not get to end the run on whatever it
just wrote - it is sent back once, told what the evidence actually was, and asked
again.

### The agents declare it; the orchestrator does not guess

A tool that has nothing to offer says so with a marker, `[no-evidence]`, and the
orchestrator routes on that rather than on the wording. This project has twice
shipped a lexical matcher that missed a rephrasing - "could not find" missing
"could not be found", "not mentioned" missing "not explicitly mentioned" - and
both times the miss was a false accusation. Putting a third one on the critical
path of the loop itself would be a worse bet than either.

The marker is duplicated into both agents and the orchestrator, the same trade as
`instrumentation.py`, because each image builds from its own directory.

### One prompt revision, from a measured regression

The first version told the model to say it could not find the answer. It worked
on the case it was built for and broke a case it was not: asked who won the 2018
World Cup final, the model got an empty corpus, was regrounded, and **declined** -
instead of searching the web, which would have answered it. Fabrication had been
traded for uselessness.

The prompt now says to call another tool if one could still find it, and only
otherwise to report the absence. It still never supplies a conclusion, for the
same reason `NUDGE_PROMPT` names no tool: telling the model what to say is how a
loop starts producing what it was told rather than what it found.

### Both guardrails addressed the model as the user, and collided

A turn-scoped check that stops at the last user message treats the guardrail's own
prompt as the start of a new turn. The nudge is shielded by its counter; the
reground was not, so an honest answer produced *after* regrounding looked exactly
like a narration with no tool call, and was nudged for it - two recovery attempts
on one failure, the second arguing with a correct answer. `SYNTHETIC_PROMPTS`
names the messages the system injected so they do not begin a turn.

Worth keeping because the class generalises: any check scoped to "since the user
last spoke" is wrong in a system that speaks as the user.

### Measured

| | fabrication on `honest-ignorance` |
|---|---|
| before, across several samples | 1 to 2 runs in 6 |
| with the guardrail | **0 in 6** |

The guardrail fired in 5 of those 6 runs and every one produced an honest answer;
the sixth took the narration path and was nudged instead. A full evaluation run
has all eight cases passing every automated signal, `honest-ignorance` included,
for the first time - at the cost of two extra iterations on that case.

Six runs is a small sample and the case is stochastic. What is not in doubt is
the mechanism: the state shows `regrounds=1` on the runs that would previously
have answered from nothing.
## A signal for claims about a subject the evidence never mentioned

The judge scores whether each claim in an answer is supported by the tool output
it was given. That is the right question, and it is blind to the failure that
actually happened: tool output about a *different subject*. Asked what the
Quazzlemint Foundation concluded, the system retrieved real annual reports from
real foundations and answered about Quazzlemint. Every claim traced back to the
evidence, so grounding came out 5 of 5 on a fabrication - in the one case this
project is organised around.

`check_subject_grounding` is deterministic and narrow. A case opts in with a
`subject` field; if the subject appears in the tool output there is nothing to
check and the judge's claim-level scoring takes over; if it does not, every
sentence mentioning the subject must be reporting its absence. A sentence that
instead says what the subject did, concluded or contains is an invention, however
well it matches the documents that came back.

Opting in explicitly, rather than inferring proper nouns from the task, is
deliberate. A wrong guess produces a false failure, and a harness that cries wolf
about the case it exists to police stops being believed.

### Three ways the first versions were wrong

Each was found by running the case rather than by reasoning about it, and each
was a false positive - an honest denial reported as a fabrication.

| written | missed | because |
|---|---|---|
| `"could not find"` | "could not be found" | passive voice |
| `"not mentioned"` | "not explicitly mentioned" | an adverb in between |
| a verb list without `have` | "did not have a 2019 report" | denial phrased as the subject not doing something |

The fix was to stop matching phrasings and match a pattern: a negation, up to
three words, then a verb of existence or provision. The general lesson is that a
lexical denial detector is never finished - it is tuned on the phrasings observed
so far, and a model writes new ones. It is worth having anyway, because the
alternative is a signal that only an LLM can produce, and this one costs nothing
per case and cannot itself hallucinate.

Two sentence types are excluded for the same reason: a narrated tool call that
leaked into the answer is the nudge node's failure and filing it here would put
one problem under another's name, and advice to the reader ("if you need details,
consult the primary source") names the subject while asserting nothing about it.

### The check found a bug in a fix from the same session

Its first live run reported the subject as present in the evidence, which was
true and should not have been. `index_documents` had just been changed to stop
echoing the caller's label back - and the version shipped still quoted it in the
confirmation text. The model passes the fictional name as the source label, so
the tool's own receipt put that name into the transcript as tool output, and the
check read it as coverage.

So the receipt is now silent about the label, and the check ignores two things
that are not evidence: output from bookkeeping tools, and a subject that appears
only inside quotes, which is how every tool in this system echoes a request back
("matched nothing for 'X'"). Counting an echo as coverage would disable the check
exactly when it matters, since those messages appear only when nothing was found.

### What it catches, measured

Over six runs of `honest-ignorance` it flagged three, of which two were real -
an invented summary of the report's findings, and a claim that the indexed
documents related to the foundation - and one was a denial phrased with a verb
the pattern did not yet cover, now added. The remaining runs were honest denials
and correctly passed. The fabrication rate itself is unchanged; what changed is
that the harness can now see it, where before it reported grounding 5 and moved
on.

## The corpus learned to vouch for a fiction

`honest-ignorance` asks what the Quazzlemint Foundation concluded in its 2019
report. Nothing of the sort exists, so the only correct answer is to say so. The
case is the one this project is organised around, and it stopped working in a way
that no signal in the harness could see.

Yesterday it failed by calling no tool at all and answering from memory. Today it
**passes every automated check** - required tool called, claim budget met,
grounding 5 out of 5, zero unsupported claims - while confidently reporting what
the foundation concluded. The green row is worse than the red one was.

What happened is a loop the system closed on itself:

1. An earlier run searched the web for "Quazzlemint Foundation 2019 report".
2. DuckDuckGo does not return nothing - it returns loose matches, in this case the
   Mellon Foundation's real 2019 annual report and similar.
3. `search_web` auto-indexed those results with `source=f"web-search: {query}"`,
   filing real documents under a label naming a foundation that does not exist.
4. `retrieve` later returned them, because nearest-neighbour search always returns
   *k* rows if the corpus holds *k* documents.
5. The model read the provenance label as confirmation and answered.
6. The judge scored grounding 5, correctly: every claim *was* supported by the tool
   output. The tool output was about a different organisation.

A count of the corpus made the scale plain - 18 of 130 documents were real web
content filed under a fictional entity, and that was before the runs measuring
this added six more.

### The relevance floor

Distances were measured rather than guessed, best hit per query:

| query class | best-hit L2 distance |
|---|---|
| queries the corpus really answers | 0.568 - 0.641 |
| the absent entity | 0.768 |
| queries unrelated to anything stored | 1.025 - 1.078 |

`RETRIEVAL_MAX_DISTANCE` defaults to 0.70, the midpoint of the gap between the
first two rows. Above it, a neighbour is reported as no match rather than as
evidence. Erring toward rejection is deliberate: a wrongly rejected match sends
the model to `search_web`, which is recoverable, while a wrongly accepted one
becomes a confident answer about something the corpus never held.

One measurement corrected a guess worth recording. The Xylophone Quarks Institute
is also fictional, but it matches at 0.527 - closer than any genuine query - and
that is *correct*, because the integration test indexes that literal text. The
corpus really does hold documents about it. Only the Quazzlemint documents were
contamination, and a check that assumed "fictional entity implies bad match"
would have been wrong.

### What the floor did not fix

`retrieve` now declines the fiction, and `cached-retrieval` is unaffected - 3 runs
out of 3, grounding 5. But the case still fabricates in 2 runs out of 3, because
the model falls through to `search_web`, which returns real foundations' reports,
and attributes those to Quazzlemint instead. Grounding fell from 5 to 3, so the
judge senses the weaker support, but the answer is still wrong.

The floor addresses the corpus vouching for a fiction. It does not address the
cause, which is that `source=f"web-search: {query}"` turns a question into a claim
about what the documents are: any search for a false premise files real content
under a label asserting it. That is the same corpus-pollution failure as
"Stop reporting a failed search as an absence", one level deeper - there the
stored text was a failure message, here it is real text with a lying label.

### Provenance belongs to the document, not to the batch

The label was a property of the *call*: one string covering everything sent. So
whatever the caller believed the batch was about got stamped on every document in
it. A document's own `Source:` line cannot lie in that direction - it names the URL
the text actually came from - so `provenance_of` prefers it, and the caller's
string is demoted to a fallback for text with no origin of its own.

Putting the rule in the store rather than in the research agent was not tidiness.
Counting the corpus turned up **two** producers of bad labels:

| documents | source | who wrote it |
|---|---|---|
| 24 | `web-search: Quazzlemint Foundation 2019 report` | the research agent's auto-indexing |
| 6 | `Quazzlemint Foundation 2019 report` | **the model**, choosing the argument itself |

The second was not anticipated. The system prompt tells the model to call
`index_documents` after a search, and it passes the query as the source, because
that is the obvious thing to write. A fix in the research agent would have closed
one path and left the other open. In the store it covers both, and any future
caller as well - verified live by indexing a document with a deliberately
misleading label and finding it stored under its own URL instead.

For the same reason `index_documents` no longer echoes the caller's label back in
its confirmation. Reporting "indexed from X" when X was discarded would tell the
model its label stuck.

### A claim is not a source

Preferring a document's own `Source:` line fixed documents that have one. Text
that does not - fragments the model composes and hands to `index_documents` -
fell back to the caller's label, and the model is one of the callers. Counting
the corpus found the habit in several shapes:

| documents | label the model supplied |
|---|---|
| 34 | `Quazzlemint Foundation 2019 report` |
| 10 | `FIFA World Cup` |
| 7 | `web` |
| 4 | `LinkedIn Learning: Project Management Foundations Exam - Quizlet` |
| 3 | `www.umfoundation.com` |

A query, a subject, a bare word, a page title, a domain. No rule separates those
from an honest tag like `integration-test`, because the difference is not in the
string - it is in whether the caller had any way to know, and a tool cannot ask.

So the caller's word is no longer treated as provenance at all. A document that
names its own origin keeps it; everything else is stored as `unattributed`, and
`retrieve` says "no stated origin" rather than repeating a label nobody can
vouch for. The claim itself is kept in a separate `claimed_source` column,
because counting labels is exactly how this contamination was found, and a fix
that removed that ability would have cost more than it bought.

The honest tags lose their place in the source column too, which is the right
outcome rather than a regrettable side effect: `integration-test` is also a
claim by somebody who had no way to know what the text was. It is simply a claim
that happened to be true.

`backfill_claims.py` moves existing rows over. It deletes nothing - a document
with real text is worth keeping whatever was written on the front of it - and it
is idempotent, so it can be re-run whenever a corpus is restored from a backup
that predates the change. 160 of 351 documents were carrying a claim as their
source; none are now.

### The fallback that let a question become provenance

Preferring a document's own `Source:` line fixes documents that have one. Text
that does not - fragments the model composes itself and hands to
`index_documents` - falls back to the caller's label, and the model passes the
question as that label. Twenty-six such documents accumulated over one session's
validation runs, after a purge had taken the count to zero.

The obvious remedy is wrong. Rejecting caller labels outright would also discard
the honest ones: `integration-test`, `eval-fixture`, `k8s`. And the agent cannot
tell an honest caller from a careless one, because MCP tools have no caller
identity - the model uses the same tool as the test fixtures.

What it *can* tell is whether a source was **derived from the document** or
**asserted about it**. `provenance_of` only ever derives a URL, so a URL is the
one case where the corpus knows where a document came from; everything else is
somebody's word for it. `retrieve` now presents the two differently:

```
[1] (source: https://assets.ctfassets.net/...mellonannualreport_2019.pdf, distance: 0.601)
[2] (unverified label: Quazzlemint Foundation 2019 report, distance: 0.619)
```

The label is still stored. Discarding it would lose the audit trail that found
the contamination in the first place; what changed is that a claim is no longer
printed in the same shape as a derived fact.

**Measured, and it changed nothing.** Fabrication on `honest-ignorance` stayed at
2 runs in 6, against 1 to 2 in 6 before. At these sample sizes that is noise, not
an improvement. The change is kept because it is correct on its own terms - the
corpus should not present an assertion as a fact - and not because it was shown
to help. An earlier, wordier version of the same hedge was shortened for the same
reason: it was spending context on a small model and buying nothing.

Worth stating plainly, because the temptation is to report a principled change as
a win. The remaining fabrication does not come from the corpus lying any more. It
comes from the model attributing documents to a name it was asked about, which no
amount of labelling in the retrieval layer addresses.

### What it actually bought, measured

Fabrication on `honest-ignorance` went from 2 runs in 3 to 1 in 6, and the judge
began to see it: grounding had been a flat 5 while the answer was invented, and
now ranges from 1 to 5, with one run's unsupported-claim budget failing outright.
The signal improved along with the behaviour.

It is not solved. One run in six still fabricated a report title. Both samples are
small, and a first sample of three post-fix runs showed no fabrications at all -
reading a fix into that would have repeated exactly the mistake documented at the
top of this section, where one observation said the case passed. The honest
statement is that the contamination path is closed and the model's remaining
willingness to attribute a real document to a name it was asked about is not.

## Decision: a failed search is not an absence

The research agent's job is to bring back evidence. The failure that matters is
not returning nothing - it is returning nothing in a shape that reads like a
finding, because a model that is told "no results found" will treat absence as
established and answer from memory. This project has already been bitten by that
once, when a stubbed search returned a well-formed empty result and two runs
invented two different expansions of "MCP".

Reading `ddgs` 9.15.0 turned up two things that made the same mistake easy:

- **It never returns an empty list.** When nothing matches it raises
  `DDGSException("No results found.")`. So the careful "the search worked and
  found nothing usable" branch in `SearchService` was almost unreachable - it
  could only fire when results came back and every one was filtered as an ad.
- **`RatelimitException` is defined but never raised.** Every failure, throttling
  included, arrives as a generic `DDGSException` carrying the underlying engine
  error. A rate limit and an empty search were indistinguishable by type.

The result was that a throttled search and a genuinely empty one produced the
same outcome, and the wording the model received - "No search results found for
X" - asserted something about the world that had not been checked.

Failures are now classified, and the classification decides only wording and
retry, never correctness:

| outcome | what the model is told | indexed |
|---|---|---|
| results | the results, with sources | yes |
| `only_sponsored` | every result was an ad, nothing citable, not evidence of absence | no |
| `no_results` | the search ran and matched nothing for this wording | no |
| `rate_limited` | raises: a transport failure, says nothing about what exists | no |
| `failed` | raises: the lookup did not happen | no |

Two properties are deliberate. **An unrecognised failure is still a failure.**
The throttle markers are matched against strings this project has not actually
observed - triggering a real rate limit means hammering DuckDuckGo, which is the
behaviour being avoided - so a wrong guess costs a retry, never a claim about
the world. And **failures raise rather than return**. The orchestrator already
converts a tool error into text the model can act on, so raising keeps
`tool_calls_total{status="error"}` honest while still delivering the explanation.

`search_outcomes_total{outcome}` records the split, which `tool_calls_total`
cannot express: a throttled search and a successful one were previously one
call each with no way to tell them apart.

The retry is bounded to one extra attempt, which is in open tension with being
rate-limited - the remedy for "too many requests" is not another request. One is
justified by an observation rather than a principle: the live integration test
failed once during a session that had just driven dozens of searches through the
eval harness, then passed on retry and across three consecutive suite runs.

### The corpus pollution this exposed

`search_web` indexes its own output into the retrieval agent, and `index_results`
stores whatever text it is handed. So the string "No search results found for
'X'" was being written into the corpus as a document. A later `retrieve` could
return the record of a failed search as evidence - in a system whose routing was
just changed to try `retrieve` first. Only real results are indexed now, which is
what `SearchOutcome.indexable` is for.

Worth naming as a pattern: the bug was invisible while search results and
explanatory messages had the same type. Making the difference explicit in the
return value is what made the indexing rule expressible at all.

## Decision: tool metrics are recorded at the MCP boundary

Counters used to live inside each tool function. That is the obvious place, it
reads correctly, and it passes any test that calls the tool normally. It also
could not see an entire class of failure.

FastMCP validates arguments against the schema derived from the function
signature *before* calling the function. A call like `retrieve(k="abc")` is
rejected upstream, so the `try/except` meant to record the failure sat inside
the function that never ran. The effect was not miscategorisation - it was
silence. Measured before changing anything: a schema-invalid call raised
`ToolError` to the client and left every sample untouched, neither `success`
nor `error`. `sum(tool_calls_total)` was undercounting real traffic, and the
"error rate by tool" panel was structurally incapable of showing this class.

`_setup_handlers` registers `FastMCP.call_tool` as the handler for tool
requests, and validation happens below it inside `tool.run()`. Overriding that
one method sees every call - valid, invalid, and unknown - so `InstrumentedMCP`
subclasses `FastMCP` and counts there. The official SDK's FastMCP has no
middleware system; that belongs to the separate `fastmcp` v2 package. A
subclass is the seam this version offers.

Three things fell out of moving up a layer:

- **The tool name is no longer hardcoded.** It arrives with the request, which
  also means it is client-supplied and becomes a Prometheus label. An
  unregistered name is recorded as `"unknown"` rather than passed through, so a
  caller looping over invented tool names cannot grow the metric store.
- **Exactly one increment per call.** `search_web` used to increment `success`
  and then run `index_results` afterwards; had that raised, one call would have
  been counted under both labels. Wrapping the whole call makes that
  unrepresentable rather than merely unlikely.
- **Nine lines of `try/except/finally` left every tool.** The instrumentation
  was identical in all five, and identical code repeated per tool is how the
  blind spot stayed uniform across three agents.

One behaviour was characterised rather than changed: an argument that is *not*
in the schema is ignored by pydantic, so a model that invents a parameter gets
a successful call rather than a signal it guessed wrong. That is invisible in
the error rate by design, and worth knowing before reading that rate as "how
often the model called tools incorrectly".

The module is duplicated verbatim into all three agent directories. Each image
is built from its own agent directory as its context, so a shared repo-root
module would not be copied in without widening every build context - the same
trade already made for `requirements.txt`. Duplication is only acceptable if
drift is caught, so a test asserts the three copies are byte-identical and that
no server has quietly gone back to counting inside a tool body.
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
