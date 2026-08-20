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
# Timeouts
# ---------------------------------------------------------------------------
# Without these, a hung agent hangs the orchestrator indefinitely. The
# max-iteration guardrail in graph.py bounds how many times the loop runs; it
# does nothing about a single call that never returns. Both were observed
# during development: when the GPU driver failed, calls simply never came back.
#
# Two layers, because they catch different failures:
#   MCP_HTTP_TIMEOUT   - connect/read at the HTTP level (agent unreachable)
#   MCP_READ_TIMEOUT   - how long to wait for a JSON-RPC response after the
#                        connection is established (agent accepted the request
#                        then wedged)
MCP_HTTP_TIMEOUT = float(os.environ.get("MCP_HTTP_TIMEOUT", "30"))
MCP_READ_TIMEOUT = float(os.environ.get("MCP_READ_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Ollama runs on the host and serves an HTTP API on 11434 by default.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# The model must support tool calling - not all local models do, and one that
# doesn't will simply answer in prose and never emit a tool call, which looks
# like a broken graph but isn't. See the README for verified-working options.
#
# qwen3:1.7b is the default because it was MEASURED to be the right trade on
# 4GB laptop hardware, not because it is the most capable model available:
#
#   model        VRAM      GPU temp   wall time   tool call
#   qwen3:1.7b   3157MiB   64C        3.3s        correct
#   qwen3:4b     3690MiB   84C        14.5s       correct
#
# Both emit valid, correctly-schemad tool calls - which is the capability that
# actually matters for a four-tool surface with single string arguments. The 4B
# follows ordering instructions slightly better, and costs 20C and 4x the wall
# clock for it. On a thin chassis that heat becomes thermal throttling, which
# slows the whole machine, and sustained load on this GPU has twice ended in a
# driver reset (VIDEO_TDR_FAILURE).
#
# Set OLLAMA_MODEL=qwen3:4b when output quality matters more than heat.
# The bare "qwen3" tag resolves to the 8B build, which does not fit at all.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")

# Qwen3 and similar models generate an extended reasoning block before
# answering. That is useful for hard problems and pure overhead here: the
# orchestrator's job is picking a tool, and thinking made the first call take
# over ten minutes against the cluster. Off by default; set to "low", "medium",
# "high" or "true" to re-enable.
_think = os.environ.get("OLLAMA_THINK", "off").strip().lower()
OLLAMA_THINK = False if _think in ("off", "false", "0", "no", "") else (
    True if _think == "true" else _think
)

# ---------------------------------------------------------------------------
# GPU pressure controls
# ---------------------------------------------------------------------------
# This project ran on a 4GB laptop GPU where the Windows desktop alone occupies
# ~1.3GB, leaving ~2.6GB for a model that wants 3.5GB. Two models are in play -
# one for reasoning, one for embeddings - and Ollama will happily hold both
# resident at once. That combination produced a VIDEO_TDR_FAILURE bugcheck with
# STATUS_INSUFFICIENT_RESOURCES: the driver stopped responding under memory
# pressure and Windows could not reset it.
#
# These knobs exist to keep the footprint bounded. They cost throughput and buy
# stability, which is the right trade on constrained hardware.

# Context window. The KV cache scales with this, so it is a direct VRAM cost -
# and a tool-calling orchestrator does not need a large context.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))

# How many model layers to offload to the GPU. Unset lets Ollama decide, which
# means "as many as fit" - exactly the greedy behaviour that caused trouble.
# Set a number to cap it, or 0 to run entirely on CPU (slow but rock solid).
_num_gpu = os.environ.get("OLLAMA_NUM_GPU", "").strip()
OLLAMA_NUM_GPU = int(_num_gpu) if _num_gpu else None

# How long a model stays resident after its last use. Shorter frees VRAM sooner
# so the other model can load; too short causes reload churn, and a cold load is
# 10-15 seconds.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "2m")


def ollama_options() -> dict:
    """Runtime options shared by every Ollama call in this project."""
    options = {"num_ctx": OLLAMA_NUM_CTX}
    if OLLAMA_NUM_GPU is not None:
        options["num_gpu"] = OLLAMA_NUM_GPU
    return options


# Used by the retrieval agent, not the orchestrator - kept here so every model
# choice in the project is visible in one place. 768 dimensions, ~274MB.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
