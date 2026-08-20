"""
MCP client - the orchestrator's side of the protocol.

Same discipline as the agent servers: this module knows about MCP and nothing
else. It does not know what an LLM is, and it does not decide which tool to
call. It answers two questions for whoever does:

  - "what tools exist across all the agents?"  -> discover()
  - "run this tool with these arguments"       -> call()

Three concepts worth understanding before reading the code:

1. THE INITIALIZE HANDSHAKE. An MCP connection is stateful. Before you can
   list or call anything, client and server exchange protocol versions and
   capabilities via initialize(). That handshake is what lets a client talk to
   a server it knows nothing about in advance - capabilities are discovered at
   runtime rather than compiled in. It's the whole reason MCP is a protocol
   and not just "some HTTP endpoints".

2. TOOL SCHEMAS ARE THE BRIDGE. list_tools() returns each tool's name,
   description, and a JSON Schema describing its arguments. That is exactly
   the information an LLM's tool-calling API needs. So discovery isn't
   bookkeeping - the schemas fetched here are what get handed to the model so
   it knows what it's allowed to call. This is the actual join between the MCP
   half of the system and the LLM half.

3. SESSIONS ARE PER-CONNECTION. A ClientSession wraps one live connection, so
   it has a lifetime and is used as a context manager. We open a fresh session
   per operation rather than holding three connections open for the life of
   the process. That costs a connection setup per call, but it means no
   reconnect logic, no stale-connection bugs, and no shared state between
   graph nodes. (langchain-mcp-adapters defaults to the same tradeoff.)
"""

import logging
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import (
    create_mcp_http_client,
    streamable_http_client,
)

from orchestrator.config import AGENT_URLS, MCP_HTTP_TIMEOUT, MCP_READ_TIMEOUT

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _session(url: str):
    """
    Open a connection to one MCP server and complete the handshake.

    streamable_http_client yields three things: a read stream, a write stream,
    and a callable that returns the session id. ClientSession is the protocol
    layer on top of those raw streams - it turns method calls like list_tools()
    into JSON-RPC messages and matches responses back up to requests.

    (Note the underscores: the older streamablehttp_client spelling still
    exists in the SDK but is deprecated.)
    """
    # Two independent timeouts. Without them a wedged agent hangs the whole
    # orchestrator: the max-iteration guardrail limits how many times the loop
    # runs, not how long a single call may block. This is not hypothetical -
    # when the GPU driver failed mid-development, calls stopped returning and
    # the run hung until killed by hand.
    http_client = create_mcp_http_client(timeout=MCP_HTTP_TIMEOUT)

    async with streamable_http_client(url, http_client=http_client) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=MCP_READ_TIMEOUT),
        ) as session:
            await session.initialize()
            yield session


class MCPToolRegistry:
    """
    Knows every tool across every agent, and which agent owns each one.

    That ownership map is what removes the need for a hardcoded router. The
    LLM says "call summarize"; this class already knows summarize lives on the
    summarizer agent, because it asked at startup. Adding a fourth agent later
    means adding a URL to config - no dispatch code changes.
    """

    def __init__(self, agent_urls: dict[str, str] | None = None):
        self._agent_urls = agent_urls if agent_urls is not None else AGENT_URLS
        # tool name -> the URL of the agent that owns it
        self._owner: dict[str, str] = {}
        # neutral tool descriptions, in MCP's terms. Translating these into a
        # particular LLM's tool format is the LLM provider's job, not ours.
        self.tools: list[dict] = []

    async def discover(self) -> list[dict]:
        """
        Ask every configured agent what tools it has.

        An agent that's down is logged and skipped rather than fatal: one
        agent crashing shouldn't take the orchestrator with it. That's the
        failure isolation the multi-server design exists to demonstrate - and
        it only actually holds if the client is written to tolerate it.
        """
        self._owner.clear()
        self.tools.clear()

        for agent_name, url in self._agent_urls.items():
            try:
                async with _session(url) as session:
                    result = await session.list_tools()
            except Exception as exc:
                logger.warning("agent %s unreachable at %s: %s", agent_name, url, exc)
                continue

            for tool in result.tools:
                if tool.name in self._owner:
                    logger.warning(
                        "tool name %r exported by more than one agent; keeping the first",
                        tool.name,
                    )
                    continue
                self._owner[tool.name] = url
                self.tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                    }
                )

        logger.info("discovered %d tool(s): %s", len(self.tools), sorted(self._owner))
        return self.tools

    async def call(self, name: str, arguments: dict) -> str:
        """
        Execute one tool on whichever agent owns it, and return its text output.

        Errors are returned as text rather than raised. That's deliberate: in
        an agent loop a failed tool call is information the model can act on
        ("that tool errored, try a different approach"), whereas an exception
        just kills the run. Feeding the error back keeps the loop alive.
        """
        url = self._owner.get(name)
        if url is None:
            return f"Error: no agent exposes a tool named {name!r}."

        try:
            async with _session(url) as session:
                result = await session.call_tool(name, arguments)
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return f"Error calling {name}: {exc}"

        # An MCP tool returns a list of content blocks rather than a bare
        # string, because a tool can return text, images, or embedded
        # resources. These agents only ever return text, so pull the text out.
        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )

        if getattr(result, "isError", False):
            return f"Error from {name}: {text}"

        return text or f"{name} returned no text content."
