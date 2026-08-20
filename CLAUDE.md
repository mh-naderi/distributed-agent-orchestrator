PROJECT CONTEXT

"distributed-agent-orchestrator": a multi-agent system where a LangGraph
orchestrator routes tasks across specialized agents. Each agent is exposed as
its own MCP (Model Context Protocol) server and deployed as an independent
Kubernetes service, with Prometheus/Grafana observability across the system.

Architecture:
- agents/research_agent - MCP server, web search via DuckDuckGo (ddgs, no API
  key). Filters sponsored results, which a model cannot distinguish from
  sources.
- agents/retrieval_agent - MCP server over a sqlite-vec index. The only
  stateful service: deployed as a StatefulSet with a PersistentVolume, and the
  one agent that cannot be scaled horizontally without a redesign.
- agents/code_analysis_agent - MCP server, code review tool (still stubbed)
- orchestrator/ - LangGraph reason-act loop that calls the agents over MCP,
  with a max-iteration guardrail and explicit MCP timeouts
- k8s/ - manifests per agent, plus the observability stack: Prometheus with
  RBAC for Kubernetes service discovery, Grafana with its datasource and
  dashboard provisioned from ConfigMaps
- eval/ - test cases and evaluation harness: automated signals (required tools
  called, keyword match) plus an LLM judge that scores answers against the tool
  output the agent actually received

Key constraints:
- No cloud budget. Everything runs locally: kind for Kubernetes, Ollama for
  inference, with the Claude API as a thin paid fallback for harder reasoning.
- A 4GB laptop GPU is the ceiling. Model choice, context size, keep-alive and
  what runs concurrently are all constrained by it, and the limits are measured
  rather than assumed - see the hardware section of docs/architecture.md.

Design decisions are recorded in docs/architecture.md: why separate MCP servers
rather than one, why an earlier summarizer agent was removed, why the transport
moved from SSE to Streamable HTTP, and what the hardware ceiling forced. Read it
before changing architecture.

WORKING CONVENTIONS

- Explain the concept or pattern behind a change before showing the
  implementation, rather than handing over finished code.
- Ask before any change that touches more than one file, or that revises a
  decision already documented in docs/architecture.md.
- Commit each semantically independent change on its own rather than batching.
  Commit directly to main; this repo does not use feature branches. Push only
  when asked.
- Verify claims by running things. Several conclusions in this project came
  from a single observation and turned out to be wrong under a second test.
