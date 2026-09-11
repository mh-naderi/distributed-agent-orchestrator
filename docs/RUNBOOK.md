# Runbook

How to start, stop and troubleshoot this project locally. Written because the
environment has several non-obvious traps that cost real time to rediscover.

## Two ways to run

**Host processes** — lighter, no Docker. Use this for developing the loop.
**Kubernetes (kind)** — use when demonstrating or verifying the k8s story.

Do not run both at once. On a 16GB machine the cluster plus local inference
leaves no headroom, and the laptop throttles.

## Host processes

Start Ollama first. Its default port is usually fine; when it is not, how
loudly it says so depends on how it was started - see the port trap below:

```bash
ollama serve
OLLAMA_HOST=http://localhost:11434 .venv/Scripts/python.exe -m orchestrator.main
```

If it will not bind, pick a free port and pass it to both sides. This runbook
used to name 11500 as the fallback; on 2026-09-05 that was reserved too.

Each agent in its own terminal:

```bash
MCP_PORT=18000 .venv/Scripts/python.exe agents/research_agent/server.py
MCP_PORT=18001 .venv/Scripts/python.exe agents/retrieval_agent/server.py
MCP_PORT=18002 .venv/Scripts/python.exe agents/code_analysis_agent/server.py
```

Then either the CLI:

```bash
.venv/Scripts/python.exe -m orchestrator.main
```

or the streaming UI at http://localhost:18080 :

```bash
.venv/Scripts/python.exe -m uvicorn orchestrator.api:app --port 18080
```

## Kubernetes

If the cluster already exists but is stopped - which is the normal daily
case, see "Stopping and starting again" below - start it rather than creating
it:

```bash
docker start agent-orchestrator-control-plane
```

Everything comes back: the workloads, the ingress, and the corpus on the
PersistentVolume. No rebuild, no re-apply. The API server takes about ten
seconds and the pods another thirty.

`kind create cluster` is for the FIRST time only. Run against an existing
cluster it stops with `ERROR: failed to create cluster: node(s) already exist
for a cluster with the name "agent-orchestrator"` and offers nothing further,
which is a dead end worth knowing before meeting it.

Create the cluster **with the config file**:

```bash
kind create cluster --name agent-orchestrator --config kind-cluster.yaml
```

The config publishes node port 30080 to host port 18080, which is what makes
the orchestrator UI reachable without a tunnel. Docker can only publish a
container's ports when the container is created, so kind can only honour this
at cluster-creation time - a cluster made without `--config` cannot be fixed
afterwards and has to be deleted and recreated.

Nothing else on the host may be holding 18080 when the cluster is created, or
the bind fails. The most likely culprit is the host-process UI from the section
above, which serves on the same port on purpose - one more reason the two ways
of running are not meant to overlap.

Build and load each image, then apply. Four images now, one per agent plus the
orchestrator:

```bash
docker build -t agent-orchestrator/research-agent:latest agents/research_agent
docker build -t agent-orchestrator/retrieval-agent:latest agents/retrieval_agent
docker build -t agent-orchestrator/code-analysis-agent:latest agents/code_analysis_agent
docker build -t agent-orchestrator/orchestrator:latest orchestrator
```

```bash
kind load docker-image agent-orchestrator/research-agent:latest --name agent-orchestrator
kind load docker-image agent-orchestrator/retrieval-agent:latest --name agent-orchestrator
kind load docker-image agent-orchestrator/code-analysis-agent:latest --name agent-orchestrator
kind load docker-image agent-orchestrator/orchestrator:latest --name agent-orchestrator
kubectl apply -f k8s/
```

`kind load` is not optional and is easy to forget on a rebuild: the node has its
own image store, so a freshly built image on the host is invisible to it. With
`imagePullPolicy: IfNotPresent` a missing image becomes `ErrImagePull` against
Docker Hub, which reads like a network problem rather than a missing load. After
rebuilding an image you must also `kubectl rollout restart` its workload -
reloading the same `:latest` tag does not restart anything.

Ollama still runs on the host and must be up before the pods need it, on the
port the manifests expect - 11434 (see the port trap below):

```bash
ollama serve
```

### Reaching things

Install the ingress controller once per cluster. It lives in a subdirectory
and `kubectl apply -f k8s/` is non-recursive, so it is not applied by that
command and does not need to be - it survives a cluster stop and start like
everything else. (An earlier note here said a re-apply would fail because the
admission Jobs are immutable. Re-applying `deploy.yaml` against an existing
install was tested on 2026-09-07 and reported only `unchanged` and `configured`,
so that is not a reason to avoid it; keeping it out of the main apply is simply
tidier.)

```bash
kubectl apply -f k8s/ingress-nginx/deploy.yaml
```

Everything a human opens is then behind one entry point, with no tunnels:

    http://localhost:18080            the orchestrator UI
    http://localhost:18080/grafana/   Grafana

Port-forwards are now only needed to drive the agents *from the host* - the CLI,
the test suite's integration cases, or the eval harness against pods:

```bash
kubectl port-forward service/research-agent-service      18000:8000
kubectl port-forward service/retrieval-agent-service     18001:8000
kubectl port-forward service/code-analysis-agent-service 18002:8000
```

Each tunnel holds a terminal and dies when the pod is replaced, the terminal
closes, or the machine reboots. A rolling update silently breaks them: the
local port keeps accepting TCP while the tunnel behind it is dead, so a plain
port check reports "up" misleadingly.

## Auditing what the corpus says about itself

A document's `source` is either a URL it names in its own text or
`unattributed`. What the indexing caller claimed is kept separately, so the two
questions - where did this come from, and what did somebody think they were
indexing - stay apart:

```bash
kubectl exec retrieval-agent-0 -- python -c "
import sqlite3, os
con = sqlite3.connect(os.environ.get('RETRIEVAL_DB_PATH','/data/retrieval.db'))
for row in con.execute('select claimed_source, count(*) from documents group by 1 order by 2 desc limit 10'):
    print(row)"
```

After restoring a corpus from a backup that predates the split, move the old
labels across. It deletes nothing and is idempotent:

```bash
kubectl exec -i retrieval-agent-0 -- python - < agents/retrieval_agent/backfill_claims.py
kubectl exec -i retrieval-agent-0 -- python - < agents/retrieval_agent/backfill_claims.py apply
```

Back the volume up first - `VACUUM INTO` works while the agent is running:

```bash
kubectl exec retrieval-agent-0 -- python -c "
import sqlite3, os, sqlite_vec, datetime
db = os.environ.get('RETRIEVAL_DB_PATH','/data/retrieval.db')
con = sqlite3.connect(db); con.enable_load_extension(True); sqlite_vec.load(con)
con.execute('VACUUM INTO ?', (f'{db}.bak-{datetime.datetime.now():%Y%m%d-%H%M%S}',))"
```

## Restoring a corpus

The retrieval index is the only state in the system that cannot be rebuilt by
redeploying. `docker stop` keeps it; `kind delete cluster` does not.

Take a snapshot first - `VACUUM INTO` works while the agent is running, and
writes a consistent image rather than whatever the page cache had flushed:

```bash
kubectl exec retrieval-agent-0 -- python -c "
import sqlite3, os, sqlite_vec, datetime
db = os.environ.get('RETRIEVAL_DB_PATH','/data/retrieval.db')
con = sqlite3.connect(db); con.enable_load_extension(True); sqlite_vec.load(con)
con.execute('VACUUM INTO ?', (f'{db}.bak-{datetime.datetime.now():%Y%m%d-%H%M%S}',))"
```

That snapshot lives on the same volume, which protects against a bad migration
and not against losing the volume. To get a copy onto the host:

```bash
kubectl exec retrieval-agent-0 -- sh -c 'base64 -w0 /data/retrieval.db.bak-YYYYMMDD-HHMMSS' \
  | base64 -d > corpus.db
```

### Putting one back

```bash
# 1. stream it in. base64 rather than a raw pipe: stdin crosses a Windows
#    shell here, and one mangled byte gives a database that opens and is
#    subtly wrong.
base64 -w0 corpus.db | kubectl exec -i retrieval-agent-0 -- \
    sh -c 'base64 -d > /data/retrieval.db.incoming'

# 2. check it. Touches nothing, prints both sides so you can see what you
#    are about to replace and with what.
kubectl exec -i retrieval-agent-0 -- python - < agents/retrieval_agent/restore_corpus.py

# 3. commit. Snapshots the current corpus first, then swaps.
kubectl exec -i retrieval-agent-0 -- python - apply < agents/retrieval_agent/restore_corpus.py
```

Step 2 refuses anything it cannot vouch for - a truncated transfer, a file that
is not a database, a corpus with no documents, or one whose vector count does
not match its document count - and says so without touching the live corpus.
A rejected file is left in place to look at.

### Whether to restart the pod

Usually not. The agent opens a SQLite connection per operation and closes it,
so a restored corpus is live immediately. An earlier version of this runbook
would have told you otherwise.

The exception is a corpus old enough to need a schema migration, which step 2
reports as a `NOTE`. `_create_schema` runs once when the agent starts, so until
it does, the agent is running against a database missing `claimed_source`:
reads work, and `index_documents` and the corpus audit fail with `no such
column`. In that case only:

```bash
kubectl delete pod retrieval-agent-0
kubectl exec -i retrieval-agent-0 -- python - < agents/retrieval_agent/backfill_claims.py
kubectl exec -i retrieval-agent-0 -- python - < agents/retrieval_agent/backfill_claims.py apply
```

Restarting the pod kills any `kubectl port-forward` to it, so restart the
retrieval forward afterwards.

### Checking it worked

Row counts are not proof. Ask the system to use the corpus:

```bash
curl -s -N "http://127.0.0.1:18080/stream?task=Search+the+indexed+documents+for+X"
```

A restore that produced a working index will `retrieve`, and - if the query
leads to indexing - `index_documents` will succeed. That second one is the real
check, because it writes `claimed_source` and fails loudly if the schema is
wrong.

## The search cache

Repeated searches are served from memory for fifteen minutes, which is most of
why an eval run is no longer a pile of fresh scrapes. Only real results are
kept - a rate limit, a failure, an empty result and an all-sponsored page are
all re-fetched every time, deliberately, so a passing failure never becomes a
sticky one.

Two replicas means two caches, so a repeated query is fetched once per pod.

Check the hit rate:

```bash
kubectl exec deploy/research-agent -- python -c "
import urllib.request
for line in urllib.request.urlopen('http://localhost:9100/metrics').read().decode().splitlines():
    if line.startswith('search_cache_total'): print(line)"
```

To take a measurement with no caching at all - which is what you want when the
question is about search itself rather than about the loop:

```bash
kubectl set env deployment/research-agent SEARCH_CACHE_TTL=0
```

That restarts the pods, so restart the research port-forward afterwards. Put it
back with `SEARCH_CACHE_TTL=900`, and remember that `kubectl apply -f k8s/`
reverts it along with everything else set this way - the same trap as
`OLLAMA_HOST`.

## Following one run across the services

Every run gets an eight-character id, logged by the orchestrator and by every
agent it calls, and shown at the foot of the page so somebody reporting a wrong
answer can quote it without reading any logs. Find it, then grep for it:

```bash
kubectl logs deployment/orchestrator --tail=50 | grep "run start"
kubectl logs deployment/orchestrator --tail=200 | grep <id>
kubectl logs statefulset/retrieval-agent --tail=200 | grep <id>
kubectl logs deployment/research-agent --tail=200 | grep <id>
```

The orchestrator prints the task, each tool it calls, and the outcome; each
agent prints one line per call with the tool, status and duration.

An evaluation case that fails can be taken back to the logs the same way. The
results table has a `trace` column and every row of `eval/results.json` carries
`trace_id`, so a verdict leads to what the services actually did rather than
stopping at the verdict:

```bash
OLLAMA_HOST=http://localhost:11434 .venv/Scripts/python.exe -m eval.run_eval
kubectl logs statefulset/retrieval-agent --tail=200 | grep <trace from the failing row>
```

These runs go through `arun_traced` rather than the API, which is the point:
they are the ones behind every measurement in the docs. An agent that
was not involved simply returns nothing, which is an answer too - a run that
should have searched and did not shows up as silence in the research agent.

The agents log through FastMCP's rich handler, which wraps at 80 columns without
a TTY, so the id leads the message and the remaining fields land on the
continuation line. `kubectl logs ... | tr -s ' ' | grep -A1 <id>` gets the whole
thing.

Nothing is traced when the id is absent - a tool called by hand, or an older
orchestrator - and calls still work; the field simply reads `-`.

## Reproducing a measurement

Claims in `docs/architecture.md` come with numbers. These are how the numbers
were produced, so they can be checked rather than believed. All of them need the
cluster up, the agents reachable and Ollama running.

How often does something happen, through the whole loop:

```bash
OLLAMA_HOST=http://localhost:11434 .venv/Scripts/python.exe -m eval.experiment repeat --case honest-ignorance --runs 8
```

**Check the denominator before quoting the number.** Failed runs are not
counted, so the summary says how many were lost:

```
  -> honest-ignorance: fabricated 0/2   (6 of 8 runs failed and are NOT counted below)
```

and refuses to print a fraction when none completed:

```
  -> honest-ignorance: NOTHING MEASURED - all 8 run(s) failed.
```

The usual cause of a whole run failing is the model backend, not the case: the
harness runs on the host, so it needs `OLLAMA_HOST` pointing at the port Ollama
actually bound, which is not always 11434 - see the port trap below.

It reports which route the loop took as well as the count, and that matters: a
change which stops the model calling `search_web` at all looks like a fabrication
fix if only the totals are read.

Did one change cause it - same evidence, one difference:

```bash
OLLAMA_HOST=http://localhost:11434 .venv/Scripts/python.exe -m eval.experiment ab --case honest-ignorance --tool search_web --strip "Note: none of these" --runs 8
```

The loop reaches `search_web` about once in eight runs, so measuring a change to
search results through the whole loop would need thirty runs to collect a
handful. `ab` fetches the evidence once and varies only the text after `--strip`.
Live results differ between fetches, so a `--strip` that was present yesterday
may not be today; the command says so rather than silently measuring nothing.

What distances the relevance floor is separating - runs inside the pod, because
it needs that agent's embedder and the MCP tool applies the floor being tested:

```bash
kubectl exec -i retrieval-agent-0 -- python - < eval/distance_study.py
```

## Regenerating the dashboard screenshot

The README image is a real capture, so it goes stale as the dashboard changes.
With the cluster up and Grafana reachable through the ingress:

```bash
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --hide-scrollbars --window-size=1600,2680 --virtual-time-budget=45000 --screenshot="D:\Projects\agent-orchestrator\docs\images\grafana-dashboard.png" "http://localhost:18080/grafana/d/agent-orchestrator/agent-orchestrator?kiosk&from=now-3h&to=now"
```

No credentials are needed because the manifest enables anonymous viewing, and
`kiosk` drops Grafana's own navigation. The window must be tall enough to hold
every panel: Grafana renders lazily, so a panel below the fold comes out blank
rather than missing, which is easy to miss when checking the file. **Open the
resulting PNG and look at it** - the height here is not a constant, it tracks
the dashboard, and it went from 1850 to 2680 when the dashboard grew from
thirteen panels to twenty. A capture that is too short loses the bottom row
silently; one that is too tall leaves a band of empty background.

Drive a few runs through the deployed orchestrator first, or the three
orchestrator panels will read "No data" - the evaluation harness runs the graph
in-process and never touches the deployed service, and the pod's counters reset
when it restarts:

```bash
curl -s -N --get --data-urlencode "task=What is Kubernetes?" http://localhost:18080/stream
```

## Reading the alerts

Fourteen alert rules ship in the `prometheus-rules` ConfigMap. **Nothing pages
anyone** - there is no Alertmanager, so alerts exist in Prometheus' own UI and
nowhere else. Port-forward it and open `/alerts`:

```bash
kubectl port-forward service/prometheus-service 19090:9090
```

Or ask without a browser, which is faster when checking whether anything is
wrong at all:

```bash
kubectl exec deploy/prometheus -- wget -qO- http://localhost:9090/api/v1/alerts
```

An empty `alerts` array is the healthy answer. A firing alert carries the pod
and agent it concerns, plus a description saying what to look at.

To confirm the rules loaded at all - worth doing after any change to the
ConfigMap, because a rules file that fails to parse stops Prometheus from
starting, and one that is simply absent does not:

```bash
kubectl exec deploy/prometheus -- wget -qO- http://localhost:9090/api/v1/rules
```

Nine rules, each with `"health":"ok"`. `"health":"unknown"` immediately after a
restart is normal - groups are evaluated on a stagger, so a group can sit
unevaluated for up to one `evaluation_interval` (30s). If it is still unknown
after a minute, something is wrong with that group.

### What each one means when it fires

| Alert | What it is telling you |
| --- | --- |
| `ScrapeTargetDown` | A pod is scheduled but its metrics port stopped answering. The service may still be serving MCP traffic fine. |
| `AllTargetsVanished` | Discovery itself broke - RBAC, or the `metrics` port naming - rather than one agent being down. |
| `RunsBeingTurnedAway` | The queue cap is being hit and requests are refused. Concurrency is 1 with 4 queued by design. |
| `DiscoveryReachedNoAgents` | Runs are starting with no tools at all. This is the answering-from-nothing case; treat it as urgent. |
| `RunsSlowerThanDesigned` | p95 has reached the 300s top bucket. Usually the model being evicted from GPU memory - `OLLAMA_KEEP_ALIVE` is 2m. |
| `MostRunsAnsweredFromNothing` | Over half of runs answered from empty evidence before being sent back. Check the corpus and the search path, not the prompt. |
| `MostRunsNarratedToolCalls` | Over half of runs described a tool call instead of making one. If a prompt or model changed recently, that is the regression. |
| `ToolFailingOnMostCalls` | One named tool is erroring on most calls - broken rather than flaky. |
| `SearchMostlyRateLimited` | DuckDuckGo is throttling past the built-in retry. Answers will lean on the existing corpus. |
| `RunsStartedWithNoTools` | A run began with an empty tool list. Either discovery failed, or every agent is up and advertising nothing. Urgent. |
| `MostRunsFailing` | Over half of runs ended in an unhandled failure. Usually the model backend: check `OLLAMA_HOST` against the port Ollama actually bound. |
| `MostRunsTruncated` | Over half of runs ran out of iterations instead of answering. The loop is going round without converging. |
| `CorpusIsEmpty` | The index holds nothing. Expected on a fresh cluster until seeded; otherwise a volume with no other copy has been emptied. |
| `CorpusMostlyGone` | The index is under half its recent size. Expected right after a restore or a purge; otherwise data loss. |

The `MostRuns...` alerts need one caveat, and it is the reason they are all
proportions. Regrounds, nudges and truncations firing are the guardrails
**working**; only the proportion is abnormal. A handful in a normal day is the
system refusing to fabricate or to spin forever, and is not something to fix.

`MostRunsFailing` is a proportion for a different reason. `failed` is the
pessimistic default for any run that ends without setting an outcome, which
includes **a client disconnecting mid-run** - closing the browser tab during a
slow answer records a failure. Measured, not assumed: killing a `curl` three
seconds into a run ticks the counter. So a handful of failures usually means
somebody navigated away, and only a majority means the system is broken.

If it does fire, the first thing to check is the model backend:

```bash
kubectl get deployment orchestrator -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="OLLAMA_HOST")].value}'
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18434/api/tags
```

Ollama going down mid-session is what prompted these three rules - every run
returned `ConnectionError` and nothing fired, because no rule referenced that
outcome.

### The two corpus alerts will fire when you meant it

Both `Corpus...` rules fire on things you might have done on purpose, and that
is deliberate. A fresh cluster's corpus really is empty. A restore to an older
snapshot really did lose documents. Neither is a false alarm - the rule is
describing the state correctly, and the operator who caused it is the one person
who can say it was intended.

They are not suppressed because the alternative is worse. This is the only state
in the system with no source to rebuild it from, and it fails silently: an
emptied index makes `retrieve` report no evidence, the model falls back to
`search_web`, search succeeds, and answers keep arriving. No reground fires,
because a reground needs *every* tool to have come back empty. Nothing else in
these fourteen rules would notice.

So if you have just restored or purged, expect them and move on. If you have
not, check the volume before doing anything else:

```bash
kubectl exec retrieval-agent-0 -- python -c "
import sqlite3
con = sqlite3.connect('/data/retrieval.db')
print('documents:', con.execute('select count(*) from documents').fetchone()[0])"
kubectl exec retrieval-agent-0 -- sh -c 'ls -la /data/'
```

The `.bak-*` files on the volume are the nearest snapshots; "Restoring a corpus"
above is how to put one back. `CorpusMostlyGone` compares against the highest
value Prometheus has seen in six hours, which is all it retains, so after a
Prometheus restart the baseline is whatever the corpus was at that point - the
corpus-size panel on the dashboard is the longer memory.

### Changing a rule

The rules are embedded in `k8s/prometheus.yaml`, so they deploy with everything
else. After editing:

```bash
kubectl apply -f k8s/prometheus.yaml
kubectl rollout restart deployment/prometheus
```

The restart is required. A mounted ConfigMap updates eventually, but Prometheus
does not reload rules on its own.

Then check the change the way it needs checking, which is not by reading it:

```bash
python tests/test_alert_rules.py alerts.yml
cp tests/alerts_test.yml .
docker run --rm -v "$PWD:/w" -w /w --entrypoint promtool prom/prometheus:v3.1.0 test rules alerts_test.yml
```

**A rule that loads healthy and sits at `inactive` looks identical to a correct
rule on a healthy system.** One of these was written as `> 300` when
`histogram_quantile` can never return more than 300, so it could not have fired
under any circumstance - and it passed `promtool check rules`, reported
`"health":"ok"`, and looked entirely normal. Only evaluating it against data
that should trigger it found the problem. `pytest tests/test_alert_rules.py`
enforces that every rule has such a case.

One more trap, because it wasted a verification round here: **`promtool test
rules` reports SUCCESS on an empty test file.** If a copy silently produced
nothing, the result is a green tick that means "there was nothing to check".
Check the byte count before believing a pass.

## Stopping and starting again

Stop the cluster; do not delete it. `docker stop` keeps the PersistentVolume and
everything on it, and `docker start` brings the whole cluster back:

```bash
docker stop agent-orchestrator-control-plane
```

**Order matters for the rest, and getting it wrong looks like it worked.** Quit
Docker Desktop BEFORE `wsl --shutdown`: otherwise its still-running UI restarts
the WSL backend, which restarts the kind node - which carries
`--restart=on-failure:1`, and every stop exits 137 because the node does not
handle SIGTERM in time, so Docker treats a clean stop as a failure and starts it
again. `vmmemWSL` comes back with it.

```powershell
Get-Process | Where-Object { $_.ProcessName -match '^(kubectl|ollama)$' } | Stop-Process -Force
Get-Process | Where-Object { $_.ProcessName -match 'Docker Desktop|com\.docker' } | Stop-Process -Force
wsl --shutdown
```

Verify with `Get-Process`, not with `tasklist` piped through grep in Git Bash:
the Bash tool's PATH sometimes arrives with Windows-style separators, and a
missing `tasklist` makes `grep -c` print 0, which reads as "all stopped" and is
not. `wsl --shutdown` sometimes needs running twice; check that `vmmemWSL` is
gone.

To destroy the cluster and its data deliberately - a genuine reset, not a daily
stop:

```bash
kind delete cluster --name agent-orchestrator   # destroys the PVC and its data
```

## Traps

### Windows reserves TCP ports, and the ranges move

Hyper-V, WSL and Docker reserve large blocks of ports. A port inside one cannot
be bound by anything - `WinError 10013`, "forbidden by its access permissions" -
even as Administrator with nothing listening. The blocks are redrawn on boot.

This has hit the project twice with different ports:

- **8000**, the agents' default, fell inside 7976-8075. Agents default to
  18000-18002 locally. Kubernetes is unaffected: each pod has its own network
  namespace, so the manifests keep 8000.
- **11434, Ollama's own default**, fell inside 11359-11458 after a reboot.
  Ollama could not bind its own port. (Later freed again - the ranges move
  both ways, so check rather than assume.)
- **11434 AND 11500 together**, on 2026-09-05: 11434 inside 11375-11474, 11500
  inside 11475-11574. The default and the fallback this runbook used to name
  were gone at once. Nothing said so: Ollama's server never bound, `ollama list`
  printed its startup log as though all were well, and the first real symptom
  was an integration test reporting "Failed to connect to Ollama". Treat no port
  as the fallback - pick one from the current ranges. 18434 was free that day
  and echoes the 18000-18002 the agents use locally.
- **4863, the kind API server port**, fell inside 4856-4955 overnight. This one
  was the worst of the three, because nothing chose that port: kind asked the
  OS for an ephemeral one at creation and baked it into the container. A
  container's port mappings cannot be changed afterwards, so the cluster simply
  refused to start:

  ```
  bind: An attempt was made to access a socket in a way forbidden by its
  access permissions
  ```

  `kind-cluster.yaml` now pins `networking.apiServerPort`, which does not make
  the port immune but does make it a known value you can check before creating
  the cluster and change in one place.

  **Recovering data from a cluster that will not start:** `docker cp` works on
  a stopped container, so the PVC can be rescued without starting anything:

  ```bash
  docker cp agent-orchestrator-control-plane:/var/local-path-provisioner data/rescue/
  ```

  The retrieval index is at
  `data/rescue/local-path-provisioner/pvc-*_default_index-retrieval-agent-0/retrieval.db`
  and goes back into a fresh cluster with the procedure in "Restoring a corpus"
  above. That procedure did not exist when this note was first written - it
  referred to "the same `kubectl exec` pipe used for any other backup", and
  there was no such pipe. Every `kubectl exec -i ... python -` here streams a
  script in; none of them wrote a database.

Check the current ranges:

```bash
netsh interface ipv4 show excludedportrange protocol=tcp
```

That prints thirty-odd ranges, which is tedious to scan when the first two
candidates are already blocked. This answers the question for specific ports:

```powershell
$out = netsh interface ipv4 show excludedportrange protocol=tcp
$ranges = @(); foreach ($l in $out) { if ($l -match '^\s*(\d+)\s+(\d+)\s*$') { $ranges += ,@([int]$matches[1], [int]$matches[2]) } }
foreach ($p in 11434, 11500, 18434) { $hit = $false; foreach ($r in $ranges) { if ($p -ge $r[0] -and $p -le $r[1]) { $hit = $true } }; "$p : $(if ($hit) {'reserved'} else {'FREE'})" }
```

To move Ollama, every side must change. Ollama overloads `OLLAMA_HOST` to mean
the server's bind address *and* a client's connect URL - different processes, so
the values differ:

```bash
OLLAMA_HOST=127.0.0.1:18434 ollama serve
OLLAMA_HOST=http://localhost:18434 .venv/Scripts/python.exe -m orchestrator.main
```

A third side moves with them when the cluster is up. The manifests point the
retrieval agent and the orchestrator at `host.docker.internal:11434`, and a pod
that cannot embed fails only when something asks it to - the MCP port opens and
both probes pass either way. Patch the running workloads rather than editing the
manifests, so a port that is free again next week does not become a committed
default:

```bash
kubectl set env statefulset/retrieval-agent OLLAMA_HOST=http://host.docker.internal:18434
kubectl set env deployment/orchestrator OLLAMA_HOST=http://host.docker.internal:18434
```

Both restart their pods, so the port-forwards need restarting afterwards.

**`kubectl apply -f k8s/` silently undoes this.** The manifests name 11434, so a
re-apply puts it back - reported as `statefulset.apps/retrieval-agent
configured` among a screen of `unchanged` lines, which is easy to read past. The
pods then point at a port Ollama could not bind, both probes still pass, and the
failure arrives later as "Failed to connect to Ollama" from whatever first
needed an embedding. After any `kubectl apply -f k8s/`, check:

```bash
kubectl get statefulset retrieval-agent -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="OLLAMA_HOST")].value}'
```

### The GPU is the binding constraint

4GB VRAM, of which the Windows desktop takes 1.3-1.6GB. Sustained inference has
twice ended in a `VIDEO_TDR_FAILURE` bugcheck in `nvlddmkm.sys`, once carrying
`STATUS_INSUFFICIENT_RESOURCES` with both models resident and 157MiB free.

- `qwen3:1.7b` is the default: ~64C, ~3s per call.
- `qwen3:4b` reached 84C and 14.5s for the same call, and selects tools better.
- Partial GPU offload measured no faster than pure CPU while costing ~800MiB.
  `OLLAMA_NUM_GPU=0` runs entirely on CPU at roughly the same tokens/sec.

If the laptop gets hot and slow, stop the cluster first - it is the cheapest
thing to give up.

### host.docker.internal works here, but not everywhere

Pods reach Ollama on the host through `host.docker.internal`. This works on
Docker Desktop, whose DNS resolver knows the name and whose proxy can reach the
host's *loopback* interface - which matters because Ollama binds 127.0.0.1. It
does not work on kind over native Linux Docker, and the raw bridge gateway
(172.18.0.1) is refused because nothing is listening there.

### Verifying a port is "up" is not enough

A TCP connect succeeding proves only that something accepted the socket. After
a rolling update, `kubectl port-forward` tunnels keep accepting locally while
being dead behind. Verify by making a real request, not by checking the port.
