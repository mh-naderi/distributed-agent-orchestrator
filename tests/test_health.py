"""
The health endpoint has to be honest, and it has to be fast.

WHY THIS EXISTS. /health reported `"status": "ok"` while Ollama was down and
every run was returning ConnectionError. It listed the model NAME and the agent
URLS - configuration, not reachability - so nothing in it was a check. Caught by
looking at it during a real outage rather than by reading it.

It also had no tests at all, while being the target of both the liveness and
the readiness probe in k8s/orchestrator.yaml.

THE TWO PROPERTIES THAT MATTER pull in opposite directions, which is what makes
this worth testing rather than eyeballing:

1. It must tell the truth about dependencies, or it is decoration.
2. It must always answer 200, quickly. Both probes use the default
   timeoutSeconds of 1. A non-200 restarts the orchestrator on liveness and
   withdraws the page on readiness - neither of which repairs a dependency, and
   the second of which hides the error message from the person who needs it.

So the bad news goes in the body and never in the status code, and the check
that produces it runs on a background task so a HANGING backend cannot make the
endpoint slow. The tests below hold both halves down.
"""

import asyncio
import json
import time

import pytest

from orchestrator import api

# Long enough to blow every timing assertion below, short enough that a broken
# build fails in seconds instead of wedging. Never reached when the code is
# correct: nothing waits on the probe, and the probe's own timeout bounds it.
WEDGE_SECONDS = 2.0


@pytest.fixture(autouse=True)
def fresh_probe(monkeypatch):
    monkeypatch.setattr(api, "_backend_probe", api._BackendProbe())


def call_health():
    """
    The handler, called directly, as test_api.py calls the rest of them.

    No TestClient: nothing else in this suite uses one, it would pull httpx in
    as an undeclared test dependency, and the handler ignores its request
    argument anyway. Calling it directly also skips the lifespan, which is the
    point - importing this module must not start the background poller.
    """
    response = asyncio.run(api.health(None))
    return response.status_code, json.loads(response.body)


def body() -> dict:
    status_code, payload = call_health()
    assert status_code == 200
    return payload


# ---------------------------------------------------------------------------
# It says what is actually true
# ---------------------------------------------------------------------------


def test_a_reachable_backend_reads_as_ok():
    api._backend_probe.reachable = True
    api._backend_probe.detail = "answered"
    api._backend_probe.checked_at = time.monotonic()

    report = body()

    assert report["status"] == "ok"
    assert report["backend"]["reachable"] is True


def test_an_unreachable_backend_reads_as_degraded():
    """
    The case this was written for. Before, this said "ok" while no run could
    produce an answer at all.
    """
    api._backend_probe.reachable = False
    api._backend_probe.detail = "ConnectError: [Errno 111] Connection refused"
    api._backend_probe.checked_at = time.monotonic()

    report = body()

    assert report["status"] == "degraded"
    assert report["backend"]["reachable"] is False
    # The reason travels with the verdict; otherwise the next step is guesswork.
    assert "ConnectError" in report["backend"]["detail"]
    assert report["backend"]["host"]


def test_an_unchecked_backend_is_not_reported_as_broken():
    """
    None and False have to stay different. Reporting "not looked yet" as
    unreachable would make every start read as an outage.
    """
    report = body()

    assert report["status"] == "starting"
    assert report["backend"]["reachable"] is None
    assert report["backend"]["checked_seconds_ago"] is None


# ---------------------------------------------------------------------------
# ...without ever giving Kubernetes a reason to act
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reachable", [True, False, None])
def test_the_status_code_is_always_200(reachable):
    """
    A non-200 here restarts the orchestrator and withdraws the UI. Neither
    repairs a backend that is down, and the second hides the error from the
    person trying to read it.
    """
    api._backend_probe.reachable = reachable
    api._backend_probe.checked_at = time.monotonic()

    status_code, _ = call_health()
    assert status_code == 200


def test_health_does_not_wait_on_the_backend():
    """
    The property the whole design turns on. A refused connection is instant;
    the failure that actually happens is a HANG, and an inline check would put
    it on the critical path of a probe with timeoutSeconds: 1.

    Here the probe is slow and the endpoint still answers at once.

    The fake sleeps for a bounded WEDGE rather than forever, and that detail is
    the difference between a useful test and a wedged CI run. With sleep(3600)
    this test does not fail when somebody moves the check inline - it hangs,
    and a suite that hangs tells you less than one that fails. Verified by
    doing it: the inline mutation produced no output and had to be killed.
    """
    async def slow():
        await asyncio.sleep(WEDGE_SECONDS)

    api._backend_probe.check_once = slow

    started = time.monotonic()
    status_code, _ = call_health()
    elapsed = time.monotonic() - started

    assert status_code == 200
    assert elapsed < 0.5, f"/health took {elapsed:.2f}s with a hanging backend"


def test_the_agents_are_still_not_called(monkeypatch):
    """
    A decision that predates this one and is unchanged: a wedged agent must not
    be able to affect the orchestrator. The discovery block is the cheap
    substitute - what the last discovery found, not a fresh round trip.
    """
    called = []
    monkeypatch.setattr(
        api._registry_cache, "get",
        lambda *a, **k: called.append(1),
    )

    report = body()

    assert not called, "/health performed discovery"
    assert "discovery" in report and "tools" in report["discovery"]


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


def test_a_failing_check_records_why_rather_than_raising():
    probe = api._BackendProbe(interval=0.01, timeout=0.01)

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def list(self):
            raise ConnectionError("no route to host")

    original = api.ollama.AsyncClient
    api.ollama.AsyncClient = Boom
    try:
        asyncio.run(probe.check_once())
    finally:
        api.ollama.AsyncClient = original

    assert probe.reachable is False
    assert "no route to host" in probe.detail
    assert probe.checked_at is not None


def test_a_successful_check_clears_a_previous_failure():
    """Recovery has to be visible, or the endpoint is only useful once."""
    probe = api._BackendProbe(interval=0.01, timeout=0.5)
    probe.reachable = False
    probe.detail = "ConnectionError: earlier"

    class Fine:
        def __init__(self, *a, **k):
            pass

        async def list(self):
            return {"models": []}

    original = api.ollama.AsyncClient
    api.ollama.AsyncClient = Fine
    try:
        asyncio.run(probe.check_once())
    finally:
        api.ollama.AsyncClient = original

    assert probe.reachable is True
    assert probe.detail == "answered"


def test_a_hanging_backend_is_bounded_by_the_timeout():
    """
    Without the timeout the background task would stop polling entirely on the
    first hang, and the endpoint would report a stale verdict forever.
    """
    probe = api._BackendProbe(interval=0.01, timeout=0.05)

    class Hangs:
        def __init__(self, *a, **k):
            pass

        async def list(self):
            # Bounded, for the reason test_health_does_not_wait_on_the_backend
            # gives: removing the timeout must make this FAIL, not hang.
            await asyncio.sleep(WEDGE_SECONDS)

    original = api.ollama.AsyncClient
    api.ollama.AsyncClient = Hangs
    try:
        started = time.monotonic()
        asyncio.run(probe.check_once())
        elapsed = time.monotonic() - started
    finally:
        api.ollama.AsyncClient = original

    assert probe.reachable is False
    assert elapsed < 1.0, f"check_once took {elapsed:.2f}s against a hang"
