"""
The vector store behind the retrieval agent.

Kept in its own module rather than inside server.py (where the other agents put
their service class) for two reasons: it's substantially more logic than a stub,
and it's the part worth unit-testing on its own. The MCP wrapper in server.py
stays a thin adapter exactly like the other agents.

WHAT AN EMBEDDING IS. An embedding is a fixed-length list of numbers - 768 of
them for nomic-embed-text - produced by a model trained so that texts with
similar meaning end up near each other in that 768-dimensional space. Retrieval
is then geometry: embed the query, find the stored vectors closest to it, return
the text they came from. That's why this can surface a relevant document sharing
no keywords with the query - closeness is semantic, not lexical.

WHY THIS AGENT IS DIFFERENT. Every other agent in this project is stateless: it
takes a request, computes, replies, and remembers nothing. This one owns a file
that must outlive the process. That single difference is what makes the
multi-server architecture load-bearing rather than decorative - it needs a
volume, it survives restarts, and it cannot be casually scaled to N replicas the
way the stateless agents can (N replicas would mean N divergent indexes, since
each pod writes to its own file).
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import sqlite_vec
from sqlite_vec import serialize_float32

logger = logging.getLogger(__name__)

# nomic-embed-text produces 768-dimensional vectors. The vec0 table declares this
# up front and enforces it, so a mismatched vector is rejected at insert rather
# than silently corrupting search results.
EMBEDDING_DIM = 768

# In Kubernetes this path lives on a PersistentVolume; locally it's just a file.
# Same code either way - the difference is entirely in where the path points.
DB_PATH = os.environ.get("RETRIEVAL_DB_PATH", "data/retrieval.db")


def chunk(texts: list[str]) -> list[str]:
    """
    Split input into paragraph-sized documents on blank lines.

    WHY CHUNKING MATTERS. One embedding is a single point in 768-dimensional
    space, so it can only represent one coherent idea well. Embedding five
    unrelated search results as a single document averages them into a vector
    that sits near none of them - an early run scored a correct match at
    distance 0.915 for exactly this reason; chunking first brought it to 0.787.
    Splitting means each vector represents one thing, which is what makes the
    distances meaningful.

    Blank lines are the boundary because that's how the research agent separates
    its results, and paragraphs are a reasonable default for arbitrary prose too.

    This is a free function rather than a VectorStore method on purpose: the
    store's job is to store exactly what it is handed, predictably. Deciding how
    to carve text up is a separate policy, applied by the caller.
    """
    chunks = []
    for text in texts:
        chunks.extend(block.strip() for block in text.split("\n\n") if block.strip())
    return chunks


class VectorStore:
    """
    Stores document text alongside its embedding, and finds the nearest matches
    for a query.

    The embedding function is injected rather than imported. That keeps this
    class testable without Ollama running (the tests pass a deterministic fake),
    and it's the same dependency-injection reasoning as build_graph() taking its
    LLM provider.
    """

    def __init__(self, embed_fn, db_path: str = DB_PATH, dim: int = EMBEDDING_DIM):
        self._embed = embed_fn
        self._db_path = db_path
        self._dim = dim

        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with self._connect() as db:
            self._create_schema(db)

    @contextmanager
    def _connect(self):
        """
        Open a connection per operation.

        sqlite3 connections are not safe to share across threads by default, and
        FastMCP runs sync tool functions in a thread pool - so a single shared
        connection would eventually be used from the wrong thread. Connecting per
        operation sidesteps that entirely, and it's the same discipline the MCP
        client uses for sessions. SQLite opens are cheap; this is not the
        bottleneck (the embedding call is).
        """
        db = sqlite3.connect(self._db_path)
        try:
            # sqlite-vec ships as a loadable extension - vec0 tables and the
            # distance functions don't exist until it's loaded into *this*
            # connection, so it has to happen on every one.
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            yield db
            db.commit()
        finally:
            db.close()

    def _create_schema(self, db) -> None:
        """
        Two tables, joined on rowid.

        vec0 is a virtual table that only holds vectors. The document text lives
        in an ordinary table and the two are matched by rowid. Newer sqlite-vec
        supports auxiliary columns inside vec0, but keeping text out of it means
        this works across versions and the split is easier to reason about: one
        table answers "which rows are nearest", the other answers "what were they".
        """
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id         INTEGER PRIMARY KEY,
                text       TEXT NOT NULL,
                source     TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS document_vectors USING vec0(
                embedding float[{self._dim}]
            )
            """
        )

    # SQLite's default limit on host parameters is 999, so the IN clause is
    # asked in batches rather than assuming the caller sends a small list.
    _PARAM_BATCH = 500

    def _new_texts_only(self, texts: list[str]) -> list[str]:
        """
        Drop texts already in the corpus, and duplicates within the batch.

        WHY THIS EXISTS. index() used to INSERT unconditionally, so indexing
        the same content twice stored it twice. That is not merely untidy:
        retrieve() returns the k NEAREST rows, and identical rows are equally
        near, so duplicates crowd each other out of the results. Observed in
        this corpus, which accumulated three copies of one test document - a
        top-2 retrieval came back with the same text as both [1] and [2],
        spending the entire result set on one document.

        Filtering BEFORE embedding is deliberate: the round trip to the
        embedding model dominates index(), so re-indexing known content now
        costs no inference at all.

        Exact text match rather than a fuzzy or semantic one. Near-duplicates
        are a genuinely harder problem, and a store whose rule for what it
        keeps is "identical string" is one you can reason about.
        """
        seen: set[str] = set()
        candidates = []
        for text in texts:
            if text not in seen:
                seen.add(text)
                candidates.append(text)

        existing: set[str] = set()
        with self._connect() as db:
            for start in range(0, len(candidates), self._PARAM_BATCH):
                batch = candidates[start : start + self._PARAM_BATCH]
                placeholders = ",".join("?" * len(batch))
                rows = db.execute(
                    f"SELECT text FROM documents WHERE text IN ({placeholders})",
                    batch,
                )
                existing.update(row[0] for row in rows)

        return [t for t in candidates if t not in existing]

    def index(self, texts: list[str], source: str) -> int:
        """
        Embed and store documents. Returns how many were NEWLY stored.

        Text already in the corpus is skipped - see _new_texts_only. The
        return value is therefore a count of what changed, not of what was
        submitted, which is what a caller checking "did this do anything"
        needs.

        Embedding happens in one batched call rather than one call per document -
        the round trip to Ollama dominates, so batching is most of the speed.
        """
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            return 0

        texts = self._new_texts_only(texts)
        if not texts:
            logger.info("nothing to index from %s - all already present", source)
            return 0

        embeddings = self._embed(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"embedder returned {len(embeddings)} vectors for {len(texts)} texts"
            )

        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as db:
            for text, embedding in zip(texts, embeddings):
                if len(embedding) != self._dim:
                    raise ValueError(
                        f"embedding has {len(embedding)} dimensions, expected {self._dim}"
                    )
                cursor = db.execute(
                    "INSERT INTO documents (text, source, created_at) VALUES (?, ?, ?)",
                    (text, source, now),
                )
                # Reuse the documents row id as the vector rowid so the two
                # tables line up without a separate mapping.
                db.execute(
                    "INSERT INTO document_vectors (rowid, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, serialize_float32(embedding)),
                )

        logger.info("indexed %d document(s) from %s", len(texts), source)
        return len(texts)

    def retrieve(self, query: str, k: int = 5, max_distance: float | None = None) -> list[dict]:
        """
        Return the k documents closest to the query, nearest first.

        sqlite-vec does exact brute-force search: every stored vector is compared
        to the query. That's linear in corpus size, which sounds bad and is
        completely fine here - at this scale it's milliseconds, and the results
        are exact rather than approximate. Swapping to an approximate index is a
        problem to solve when there's a corpus big enough to need it.

        max_distance drops neighbours that are merely nearest rather than
        actually similar. Nearest-neighbour search always returns k rows if the
        corpus holds k documents, so without a floor "closest" reads as "match"
        and a question about something absent comes back with whatever happened
        to be least unlike it. Defaults to None - the store stays policy-free and
        the agent decides what counts as a match, because the right cutoff
        depends on the embedding model rather than on storage.
        """
        query_embedding = self._embed([query])[0]

        with self._connect() as db:
            rows = db.execute(
                """
                SELECT d.text, d.source, v.distance
                FROM document_vectors v
                JOIN documents d ON d.id = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (serialize_float32(query_embedding), k),
            ).fetchall()

        hits = [
            {"text": text, "source": source, "distance": distance}
            for text, source, distance in rows
        ]
        if max_distance is None:
            return hits
        return [hit for hit in hits if hit["distance"] <= max_distance]

    def count(self) -> int:
        """Number of documents currently indexed - used by tests and diagnostics."""
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
