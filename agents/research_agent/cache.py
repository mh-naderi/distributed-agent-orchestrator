"""
A small time-boxed cache for search results.

WHY. `ddgs` scrapes HTML rather than calling a supported API, and it throttles
under rapid use - which is the documented open gap in docs/architecture.md. An
eval run puts nine cases through the loop, and `eval/experiment.py repeat` puts
the same case through it eight times, so the same handful of queries are asked
over and over within a few minutes. Every repetition is a fresh scrape, and the
throttling that follows is not a property of the question being asked.

WHAT IS CACHED, AND WHAT IS DELIBERATELY NOT. Only outcomes that carry real
results. Every other outcome this agent produces - rate-limited, failed, no
results, everything-was-an-advertisement - is a statement about a moment rather
than about the query, and caching one would turn a transient failure into a
sticky one for the whole TTL. The rule is "cache evidence, never absence", and
`SearchOutcome.indexable` already draws exactly that line, so the caller uses it
rather than a second predicate that could drift from it.

The two hard failures do not even reach here: they raise SearchUnavailable.

WHAT THIS DOES NOT CHANGE. It replaces the network call and nothing else. A hit
returns the same SearchOutcome the fetch produced, and the caller goes on to
index and count it exactly as it would have. Re-indexing identical text is
already cheap - the store filters against what it holds before embedding
anything, so a repeat costs a lookup and an early return rather than a round
trip to the embedder - so there was no reason to make a hit take a different
path through the rest of the tool.

PER POD, AND THAT IS VISIBLE. research-agent runs two replicas, so this is two
caches and a repeated query only hits the one that fetched it. With requests
alternating between pods, eight repeats of one query cost two fetches rather
than one - still 6 saved out of 8, but not the 7 a shared cache would give. A
shared cache needs somewhere to share it, which this project deliberately does
not have. The metric distinguishes hits from misses so the real rate is
observable rather than assumed.
"""

import os
import threading
import time

# Sized to the workload rather than to how fast the web changes, and that is a
# judgement rather than a measurement - nothing here measures result volatility.
# Fifteen minutes covers one eval run and a set of repeats, which is what the
# cache is for, and is short enough that no answer is built on a result from a
# previous working session.
TTL_SECONDS = float(os.environ.get("SEARCH_CACHE_TTL", "900"))

# A bound, because this lives in a long-running pod. The eval suite asks nine
# distinct queries and an experiment repeats one, so 128 is far more than a
# session needs; at roughly 3KB per entry the whole cache is well under a
# megabyte even when full.
MAX_ENTRIES = int(os.environ.get("SEARCH_CACHE_MAX_ENTRIES", "128"))


class SearchCache:
    """
    Query -> value, for a while.

    Deliberately not functools.lru_cache: that has no expiry, and a search
    result that never expires is a worse failure than a slow search. It also
    could not be emptied or inspected, and a cache you cannot see is one more
    thing that can be wrong without saying so.
    """

    def __init__(self, ttl: float = TTL_SECONDS, max_entries: int = MAX_ENTRIES):
        self._ttl = ttl
        self._max_entries = max_entries
        # FastMCP runs sync tool functions in a thread pool, so two searches can
        # be in here at once. The same reasoning as the retrieval store opening
        # a connection per operation.
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, object]] = {}

    @property
    def enabled(self) -> bool:
        """A ttl of zero turns it off, for when live results are the point."""
        return self._ttl > 0 and self._max_entries > 0

    def _key(self, query: str) -> str:
        # Casefolded and whitespace-collapsed: "MCP adoption" and "mcp  adoption"
        # are the same request to a search engine, and treating them as different
        # would quietly halve the hit rate on queries a model rephrases slightly.
        return " ".join(query.split()).casefold()

    def get(self, query: str):
        """The cached value, or None if absent, expired or disabled."""
        if not self.enabled:
            return None

        key = self._key(query)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            stored_at, value = entry
            if time.time() - stored_at >= self._ttl:
                # Dropped on read rather than swept on a timer: there is no
                # other thread to run the sweep, and an expired entry costs
                # nothing until somebody asks for it.
                del self._entries[key]
                return None

            return value

    def put(self, query: str, value) -> None:
        if not self.enabled:
            return

        key = self._key(query)
        with self._lock:
            # Re-inserting moves it to the end, so the eviction below drops the
            # least recently STORED entry. Insertion order is guaranteed for
            # dicts since 3.7.
            self._entries.pop(key, None)
            self._entries[key] = (time.time(), value)

            while len(self._entries) > self._max_entries:
                self._entries.pop(next(iter(self._entries)))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
