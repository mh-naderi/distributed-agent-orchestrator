"""
Configuration for the orchestrator.

Everything the orchestrator needs to *reach* something else lives here, read
from the environment with local-development defaults. The point is that the
same code runs unchanged in two very different places:

  - locally (now): three agent processes on localhost, on three ports, because
    one machine can't give them all port 8000
  - in Kubernetes (week 2): three pods behind Services, each reachable at a
    stable DNS name on port 8000, e.g. http://research-agent-service:8000/mcp

Deploying then means setting env vars on the Deployment, not editing code.
"""

import os

# ---------------------------------------------------------------------------
# Agent MCP endpoints
# ---------------------------------------------------------------------------
# Note the /mcp path: with Streamable HTTP, an MCP server exposes a single
# endpoint that handles both request POSTs and streamed responses. That path is
# part of the address, so it belongs in the URL rather than the client.
#
# The local defaults are 18000-18002, not 8000-8002, because Windows reserves
# large TCP ranges for Hyper-V/WSL/Docker and 8000 commonly falls inside one -
# binding it fails with WinError 10013 ("forbidden by its access permissions")
# even as Administrator. Check yours with:
#     netsh interface ipv4 show excludedportrange protocol=tcp
# This is a host-only problem: in Kubernetes each pod has its own network
# namespace, so the agents keep port 8000 there and the manifests are unchanged.
AGENT_URLS = {
    "research": os.environ.get("RESEARCH_AGENT_URL", "http://localhost:18000/mcp"),
    "retrieval": os.environ.get("RETRIEVAL_AGENT_URL", "http://localhost:18001/mcp"),
    "code_analysis": os.environ.get("CODE_ANALYSIS_AGENT_URL", "http://localhost:18002/mcp"),
}

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Ollama runs on the host and serves an HTTP API on 11434 by default.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# The model must support tool calling - not all local models do, and one that
# doesn't will simply answer in prose and never emit a tool call, which looks
# like a broken graph but isn't. See the README for verified-working options.
#
# qwen3:4b is the verified default: it fits entirely in 4GB of VRAM and handles
# this tool surface correctly. The bare "qwen3" tag resolves to the 8B build,
# which spills to CPU on a 4GB card.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

# Qwen3 and similar models generate an extended reasoning block before
# answering. That is useful for hard problems and pure overhead here: the
# orchestrator's job is picking a tool, and thinking made the first call take
# over ten minutes against the cluster. Off by default; set to "low", "medium",
# "high" or "true" to re-enable.
_think = os.environ.get("OLLAMA_THINK", "off").strip().lower()
OLLAMA_THINK = False if _think in ("off", "false", "0", "no", "") else (
    True if _think == "true" else _think
)

# Used by the retrieval agent, not the orchestrator - kept here so every model
# choice in the project is visible in one place. 768 dimensions, ~274MB.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
