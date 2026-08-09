# Architecture Notes

## Why MCP, and why multiple servers

MCP standardizes how an agent discovers and calls tools, independent of
which LLM is driving it (like a USB port for AI tools: any client, any
compatible server). A single MCP server with several `@mcp.tool()`
functions would be simpler and still valid MCP usage.

This project splits tools into three separate MCP servers deliberately,
to practice and demonstrate:
- Independent deployment and scaling per agent
- Isolated failure handling (one agent crashing doesn't take down the others)
- Per-agent observability (Prometheus metrics scoped per service)
- Different resource needs per agent (a summarizer using a local LLM
  needs more memory than a lightweight search tool)

## Sync vs. async tool execution

MCP's core request/response model is synchronous by default: a tool
call blocks until the server finishes executing it and returns a
result. There's no built-in message queue.

For the MVP, all three agents stay synchronous, since the fast/simple
option is the right one when a tool completes in a few seconds and the
added complexity of a queue wouldn't teach anything new. The
code-analysis agent is flagged as the stretch-goal candidate for a
Redis-backed async worker pattern, since it's the one most plausibly
slow enough to justify decoupling the listener from execution.

## Build plan

- Week 1: MCP servers (working stubs) -> orchestrator graph -> end-to-end
  local run
- Week 2: Containerize + deploy to local K8s -> observability -> evaluation
  suite -> documentation

## Local-only, no cloud budget

Entire system runs on a local Kubernetes cluster (kind/minikube) and
local LLM inference (Ollama), with a thin Claude API fallback for harder
reasoning steps. No cloud GPU rental required, in contrast to an
inference-serving-style project, which was considered and set aside
specifically because of the budget constraint.
