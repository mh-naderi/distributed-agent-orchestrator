# Runbook

How to start, stop and troubleshoot this project locally. Written because the
environment has several non-obvious traps that cost real time to rediscover.

## Two ways to run

**Host processes** — lighter, no Docker. Use this for developing the loop.
**Kubernetes (kind)** — use when demonstrating or verifying the k8s story.

Do not run both at once. On a 16GB machine the cluster plus local inference
leaves no headroom, and the laptop throttles.

## Host processes

Start Ollama first. Its default port is usually fine, and when it is not,
nothing says so clearly - see the port trap below:

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

Install the ingress controller once per cluster (it is not applied by
`kubectl apply -f k8s/`, which is non-recursive on purpose - its admission Jobs
are immutable and would fail on every re-apply):

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

## Following one run across the services

Every run gets an eight-character id, logged by the orchestrator and by every
agent it calls. Find it, then grep for it:

```bash
kubectl logs deployment/orchestrator --tail=50 | grep "run start"
kubectl logs deployment/orchestrator --tail=200 | grep <id>
kubectl logs statefulset/retrieval-agent --tail=200 | grep <id>
kubectl logs deployment/research-agent --tail=200 | grep <id>
```

The orchestrator prints the task, each tool it calls, and the outcome; each
agent prints one line per call with the tool, status and duration. An agent that
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
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --hide-scrollbars --window-size=1600,1850 --virtual-time-budget=30000 --screenshot="D:\Projects\agent-orchestrator\docs\images\grafana-dashboard.png" "http://localhost:18080/grafana/d/agent-orchestrator/agent-orchestrator?kiosk&from=now-3h&to=now"
```

No credentials are needed because the manifest enables anonymous viewing, and
`kiosk` drops Grafana's own navigation. The window must be tall enough to hold
every panel: Grafana renders lazily, so a panel below the fold comes out blank
rather than missing, which is easy to miss when checking the file.

Drive a few runs through the deployed orchestrator first, or the three
orchestrator panels will read "No data" - the evaluation harness runs the graph
in-process and never touches the deployed service, and the pod's counters reset
when it restarts:

```bash
curl -s -N --get --data-urlencode "task=What is Kubernetes?" http://localhost:18080/stream
```

## Stopping everything

```bash
kind delete cluster --name agent-orchestrator   # destroys the PVC and its data
wsl --shutdown                                  # reclaims Docker's VM memory
```

`wsl --shutdown` often needs running twice - the VM does not always release
memory on the first call. Verify with Task Manager that `vmmemWSL` is gone.

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
  and restores into a fresh cluster with the same `kubectl exec` pipe used for
  any other backup.

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
