"""
One id per run, carried to every agent it touches.

WHY THIS EXISTS. The metrics say how many tool calls failed and how long they
took; they cannot say which run a particular failure belonged to. Four services
log independently, so reconstructing one request meant reading three pod logs
side by side and matching on timestamps - which has already produced wrong
conclusions in this project, when runs that looked broken turned out to be a
dead port-forward and a partial tool set rather than anything in the code.

WHY A CONTEXTVAR rather than an argument. The id would otherwise have to be
threaded through build_graph, every node, and the registry, none of which have
any business knowing about it - and the graph is deliberately ignorant of what
is watching it. A contextvar is set once at the edge of a run and read at the
edge of an MCP call, with nothing in between aware. It also follows asyncio
tasks correctly, which a module global would not: two runs in flight would
otherwise share whichever id was set last.

WHY IT TRAVELS AS MCP `_meta` rather than a tool argument. Arguments are the
model's business and appear in the schema it is shown; a trace id there would be
one more thing for it to get wrong, and would change every tool's signature. The
protocol already has a metadata channel for exactly this, so the tools never see
it and the agents read it without a parameter.
"""

import contextvars
import uuid
from contextlib import contextmanager

# Short on purpose. This appears on every log line in four services, and its job
# is to be greppable and to distinguish concurrent runs - not to be unique
# across the internet. Eight hex characters is one collision in ~4 billion,
# against a system that serialises runs and rarely exceeds one at a time.
TRACE_ID_LENGTH = 8

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "orchestrator_trace_id", default=None
)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:TRACE_ID_LENGTH]


def current_trace_id() -> str | None:
    """The id of the run in flight, or None outside one."""
    return _current.get()


def begin(trace_id: str | None = None):
    """
    Start a trace and return the token that ends it.

    The paired form for code that cannot wrap its body in a `with` - a long
    generator, say, whose whole point is that it yields from the middle. It is
    only correct if the token reaches reset(), so use it where a `finally`
    already exists rather than adding one.
    """
    trace_id = trace_id or new_trace_id()
    return trace_id, _current.set(trace_id)


def reset(token) -> None:
    _current.reset(token)


@contextmanager
def use_trace(trace_id: str | None = None):
    """
    Mark everything inside as belonging to one run.

    Yields the id so a caller can log it, and restores the previous value on
    the way out, which matters because runs nest in the tests and could
    otherwise leak an id into whatever ran next.
    """
    trace_id = trace_id or new_trace_id()
    token = _current.set(trace_id)
    try:
        yield trace_id
    finally:
        _current.reset(token)
