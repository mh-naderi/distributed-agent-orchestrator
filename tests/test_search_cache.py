"""
The cache must not remember the wrong things.

WHY THIS EXISTS. A search cache is easy to make fast and hard to make honest.
The dangerous entry is not a stale result - the TTL bounds that - but a cached
FAILURE: `ddgs` throttles, the agent correctly reports that it could not look
something up, and if that answer were remembered then a passing rate limit
would become a sticky one for the whole TTL, with every later run told the web
had nothing to say.

That is the same mistake this project already made once in the corpus, where a
message about a failed search was indexed as though it were a document. So the
rule is "cache evidence, never absence", and most of what is tested here is the
never half.
"""

import importlib.util
import sys
import time
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1] / "agents" / "research_agent"
sys.path.insert(0, str(AGENT_DIR))

# Loaded under an explicit name, for the reason test_search_service.py gives:
# Python caches modules by name, and "cache" is generic enough that a plain
# import could hand back somebody else's.
_spec = importlib.util.spec_from_file_location("research_cache", AGENT_DIR / "cache.py")
research_cache = importlib.util.module_from_spec(_spec)
sys.modules["research_cache"] = research_cache
_spec.loader.exec_module(research_cache)

SearchCache = research_cache.SearchCache


def test_a_stored_value_comes_back():
    cache = SearchCache(ttl=60, max_entries=8)
    cache.put("mcp adoption", "results")

    assert cache.get("mcp adoption") == "results"


def test_an_unknown_query_is_a_miss():
    cache = SearchCache(ttl=60, max_entries=8)
    assert cache.get("never asked") is None


def test_wording_that_only_differs_in_case_or_spacing_is_the_same_query():
    """
    A model rephrases slightly between runs. Treating "MCP  Adoption" as a
    different question would halve the hit rate for no reason - it is the same
    request to a search engine.
    """
    cache = SearchCache(ttl=60, max_entries=8)
    cache.put("MCP  Adoption", "results")

    assert cache.get("mcp adoption") == "results"
    assert cache.get("  mcp   adoption  ") == "results"


def test_a_different_query_is_not_a_hit():
    cache = SearchCache(ttl=60, max_entries=8)
    cache.put("mcp adoption", "results")

    assert cache.get("mcp criticism") is None


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------


def test_an_expired_entry_is_a_miss():
    cache = SearchCache(ttl=0.05, max_entries=8)
    cache.put("q", "results")
    assert cache.get("q") == "results"

    time.sleep(0.08)

    assert cache.get("q") is None


def test_an_expired_entry_is_dropped_rather_than_kept():
    """Otherwise a long-running pod accumulates entries nothing will ever use."""
    cache = SearchCache(ttl=0.05, max_entries=8)
    cache.put("q", "results")
    time.sleep(0.08)

    cache.get("q")

    assert len(cache) == 0


# ---------------------------------------------------------------------------
# Bounds - this lives in a pod that runs for days
# ---------------------------------------------------------------------------


def test_the_cache_does_not_grow_without_limit():
    cache = SearchCache(ttl=60, max_entries=3)
    for i in range(10):
        cache.put(f"query {i}", f"results {i}")

    assert len(cache) == 3


def test_eviction_drops_the_oldest_entry_first():
    cache = SearchCache(ttl=60, max_entries=3)
    for i in range(4):
        cache.put(f"query {i}", f"results {i}")

    assert cache.get("query 0") is None
    assert cache.get("query 3") == "results 3"


def test_restoring_a_query_makes_it_recent_again():
    cache = SearchCache(ttl=60, max_entries=2)
    cache.put("a", "1")
    cache.put("b", "2")
    cache.put("a", "1 again")  # a is now the newer of the two
    cache.put("c", "3")  # evicts b

    assert cache.get("a") == "1 again"
    assert cache.get("b") is None


# ---------------------------------------------------------------------------
# Turning it off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [{"ttl": 0}, {"max_entries": 0}], ids=["ttl-zero", "no-entries"]
)
def test_the_cache_can_be_disabled(kwargs):
    """
    A measurement may want live results every time, and a cache that cannot be
    turned off would silently change what an experiment measures.
    """
    cache = SearchCache(**{"ttl": 60, "max_entries": 8, **kwargs})
    cache.put("q", "results")

    assert not cache.enabled
    assert cache.get("q") is None
    assert len(cache) == 0


def test_clearing_empties_it():
    cache = SearchCache(ttl=60, max_entries=8)
    cache.put("q", "results")
    cache.clear()

    assert cache.get("q") is None
