# Distributed Agent Orchestrator

A multi-agent system where a LangGraph orchestrator routes tasks across
specialized agents, each exposed as its own MCP (Model Context Protocol)
server and deployed as an independent service on Kubernetes, with
Prometheus/Grafana observability across the whole system.

Built to extend prior work on adaptive distributed systems ([AdapTrain](#))
into the current agentic-AI tooling landscape.

## Architecture

Three specialized MCP servers, one orchestrator:

- **research-agent** - web search tool
- **summarizer-agent** - text summarization tool
- **code-analysis-agent** - code review tool (candidate for the async
  worker-queue pattern, documented as a stretch goal below)

The orchestrator (LangGraph) runs a reason -> act -> reason loop: it asks
an LLM what to do, calls the relevant MCP tool, feeds the result back, and
repeats until the LLM produces a final answer or a max-iteration guardrail
is hit.

Each agent runs as its own Kubernetes Deployment + Service, so it can be
restarted, scaled, and monitored independently. Prometheus scrapes
per-agent metrics (latency, call counts, error rates) via Kubernetes
service discovery; Grafana dashboards visualize them.

_(Architecture diagram: TODO, add after week 1 build)_

## Why multiple small MCP servers instead of one

A single MCP server with all tools would work and is simpler. This
project deliberately splits into separate services to practice and
demonstrate independent deployment, independent scaling, and isolated
failure handling, the same reasoning behind microservices generally.
See `docs/architecture.md` for the full writeup.

## Design decision: sync vs. async tool execution

MVP: all three agents execute tool calls synchronously (request in,
result out, no queue). The code-analysis agent is flagged as the
candidate for an async worker-queue pattern (Redis-backed) as a
documented stretch goal, since it's the one most likely to be genuinely
slow, rather than applying that complexity everywhere by default.

## Local setup (no cloud required)

This project runs entirely on your own machine. No cloud account or
GPU rental needed.

1. Install [kind](https://kind.sigs.k8s.io/) or minikube, and Docker.
2. Install [Ollama](https://ollama.com/) for local LLM inference (free).
3. `pip install -r orchestrator/requirements.txt`
4. Build agent images: `docker build -t agent-orchestrator/research-agent:latest agents/research_agent` (repeat per agent)
5. Create the local cluster: `kind create cluster`
6. Load images into the cluster: `kind load docker-image agent-orchestrator/research-agent:latest` (repeat per agent)
7. Deploy: `kubectl apply -f k8s/`
8. Run the orchestrator: `python -m orchestrator.main`

_(Detailed step-by-step instructions: TODO, finalize after week 2 testing)_

## Evaluation

See `eval/`. A fixed set of representative tasks run through the full
system, checked against automated signals (tool-call correctness,
keyword matches) and an LLM-judge quality score.

Results: TODO, populate after week 2 day 12.

## Observability

Grafana dashboard screenshots: TODO, add after week 2 day 10-11.

## Status

Work in progress. Build plan and progress tracked in `docs/architecture.md`.
