"""
Put a corpus back, after checking it is one.

Runs inside the retrieval pod, because it needs the volume mounted and
sqlite-vec loaded:

    # 1. stream the file in. base64 because kubectl exec's stdin crosses a
    #    Windows shell here, and a binary pipe that mangles one byte produces
    #    a database that opens and is subtly wrong.
    base64 -w0 corpus.db | kubectl exec -i retrieval-agent-0 -- \\
        sh -c 'base64 -d > /data/retrieval.db.incoming'

    # 2. check it, touching nothing
    kubectl exec -i retrieval-agent-0 -- python - < agents/retrieval_agent/restore_corpus.py

    # 3. snapshot the current corpus, then swap
    kubectl exec -i retrieval-agent-0 -- python - < agents/retrieval_agent/restore_corpus.py apply

    # 4. only when the check reported a migration NOTE - see below
    kubectl delete pod retrieval-agent-0

WHY THIS EXISTS. The docs described restoring a corpus in three places - the
backfill note, the architecture note on re-running it, and the disaster-recovery
note saying a rescued volume "restores into a fresh cluster with the same
kubectl exec pipe used for any other backup". There was no such pipe. Every
`kubectl exec -i ... python -` in this repo streams a SCRIPT in; none of them
writes a database. Three recovery paths ended in a step that had never been
written or run, which is the same failure as an alert rule that loads healthy
and cannot fire: it reads as coverage and is not.

WHAT BLOCKS AND WHAT ONLY WARNS. A corpus that predates the provenance split
has no `claimed_source` column, and restoring one is a supported operation -
store.py adds the column when the agent next opens the database, and
backfill_claims.py moves the old labels across afterwards. The first draft of
this script refused such a corpus outright, which contradicted the runbook and
would have blocked the exact recovery the docs describe. It was found by
running the restore against a real backup from before the split rather than by
reading. Missing columns warn; only a corpus that cannot be read, or has
nothing in it, blocks.

WHY IT VERIFIES FIRST. The failure worth designing against is not a missing
backup, it is a restore that quietly replaces a good corpus with a broken one.
A truncated transfer still produces a file, and SQLite will open the intact
prefix of one without complaint. So nothing is moved until the incoming file
has been opened, integrity-checked, confirmed to carry the schema this agent
expects, and confirmed to contain documents. The live corpus is snapshotted
before the swap, so the restore is itself reversible.

WHEN THE POD HAS TO RESTART, WHICH IS NOT WHAT IT LOOKS LIKE. The first version
of this script said the agent holds the database open, keeps the old inode
after the swap, and therefore serves the old corpus until restarted. That is
wrong, and checking it is what showed why. store.py opens a connection PER
OPERATION and closes it - deliberately, because FastMCP runs sync tools in a
thread pool and a shared sqlite3 connection would eventually be used from the
wrong thread. There is no long-lived handle, and the restored data is live
immediately. Measured by swapping a 391-document corpus for a 170-document one
and seeing the running agent report 170 with no restart at all.

What the restart is actually for is the SCHEMA. `_create_schema` runs once, in
`__init__`, so the `ALTER TABLE ... ADD COLUMN claimed_source` migration happens
at startup and nowhere else. Restore a corpus from before that column existed
and the agent keeps running against a database missing it: reads still work,
while `index_documents` and the corpus audit fail with `no such column:
claimed_source`. Verified by doing exactly that - the query raised, the pod was
deleted, and the column was present on the next start.

So: restart when the check reported a migration NOTE. Otherwise the data is
already live.
"""

import os
import sqlite3
import sys

LIVE = os.environ.get("RETRIEVAL_DB_PATH", "/data/retrieval.db")
INCOMING = os.environ.get("RETRIEVAL_INCOMING_PATH", LIVE + ".incoming")

# The tables this agent needs. document_vectors is a sqlite-vec virtual table;
# its four shadow tables come with it and are not checked separately.
REQUIRED_TABLES = ("documents", "document_vectors")

# claimed_source arrived with the provenance split. A corpus predating it will
# restore, but backfill_claims.py has to run afterwards, so say so rather than
# letting the agent hit a missing column later.
EXPECTED_COLUMNS = ("id", "text", "source", "created_at", "claimed_source")

# Columns store.py adds on startup if they are absent. Their absence dates a
# corpus; it does not make it unusable, so it must not block a recovery.
MIGRATED_COLUMNS = ("claimed_source",)


def connect(path: str) -> sqlite3.Connection:
    """A connection with sqlite-vec loaded, as the agent itself opens it."""
    con = sqlite3.connect(path)
    con.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def inspect(path: str) -> dict:
    """
    What is actually in the file at `path`.

    Every check runs against a real connection rather than the file's size or
    header, because the interesting corruption - a transfer that stopped
    partway - leaves a file whose header is perfectly valid.
    """
    report: dict = {"path": path, "problems": [], "warnings": [], "stats": {}}

    if not os.path.exists(path):
        report["problems"].append(f"{path} does not exist - was step 1 run?")
        return report

    size = os.path.getsize(path)
    report["stats"]["bytes"] = size
    if size == 0:
        report["problems"].append(f"{path} is empty")
        return report

    try:
        con = connect(path)
    except Exception as exc:  # noqa: BLE001 - any failure here means unusable
        report["problems"].append(f"cannot open as a database: {exc}")
        return report

    try:
        # PRAGMA integrity_check reads every page; a truncated file fails here
        # rather than on the first query that happens to touch a lost page.
        result = con.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            report["problems"].append(f"integrity_check: {result}")

        present = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        for table in REQUIRED_TABLES:
            if table not in present:
                report["problems"].append(f"no {table} table")

        if "documents" in present:
            columns = [row[1] for row in con.execute("PRAGMA table_info(documents)")]
            report["stats"]["columns"] = columns
            for column in EXPECTED_COLUMNS:
                if column in columns:
                    continue
                if column in MIGRATED_COLUMNS:
                    report["warnings"].append(
                        f"no {column} column: this corpus predates the provenance "
                        "split. The agent adds the column on startup; run "
                        "backfill_claims.py afterwards to move the old labels across"
                    )
                else:
                    report["problems"].append(f"documents has no {column} column")

            total = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            report["stats"]["documents"] = total
            if total == 0:
                report["problems"].append(
                    "the corpus is empty. Restoring it would replace a working "
                    "index with nothing, which is not a restore"
                )

            if "source" in columns:
                report["stats"]["url_sourced"] = con.execute(
                    "SELECT COUNT(*) FROM documents WHERE source LIKE 'http%'"
                ).fetchone()[0]
                report["stats"]["unattributed"] = con.execute(
                    "SELECT COUNT(*) FROM documents WHERE source = 'unattributed'"
                ).fetchone()[0]

        if "document_vectors" in present:
            # Needs the extension: without sqlite-vec this raises "no such
            # module: vec0", which is itself the answer to whether the file is
            # usable by this agent.
            vectors = con.execute("SELECT COUNT(*) FROM document_vectors").fetchone()[0]
            report["stats"]["vectors"] = vectors
            documents = report["stats"].get("documents")
            if documents and vectors != documents:
                report["problems"].append(
                    f"{documents} documents but {vectors} vectors - retrieval would "
                    "silently miss the difference"
                )
    except Exception as exc:  # noqa: BLE001
        report["problems"].append(f"unreadable: {exc}")
    finally:
        con.close()

    return report


def snapshot(path: str) -> str:
    """
    Copy the live corpus aside before replacing it, so this is reversible.

    VACUUM INTO rather than a file copy: it works while the agent holds the
    database open, and writes a consistent image rather than whatever the page
    cache happened to have flushed.
    """
    import datetime

    destination = f"{path}.bak-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    con = connect(path)
    try:
        con.execute("VACUUM INTO ?", (destination,))
    finally:
        con.close()
    return destination


def describe(report: dict) -> None:
    for key, value in report["stats"].items():
        print(f"  {key}: {value}")
    for warning in report["warnings"]:
        print(f"  NOTE: {warning}")
    for problem in report["problems"]:
        print(f"  PROBLEM: {problem}")


def main(apply: bool) -> int:
    print(f"incoming: {INCOMING}")
    incoming = inspect(INCOMING)
    describe(incoming)

    if incoming["problems"]:
        print("\nREFUSING TO RESTORE. The live corpus has not been touched.")
        return 1

    print(f"\nlive: {LIVE}")
    live = inspect(LIVE)
    describe(live)

    if not apply:
        print("\nDRY RUN - nothing written. Pass 'apply' to commit.")
        return 0

    # Only snapshot a live corpus worth keeping. On a fresh cluster there is
    # nothing here yet, and refusing to restore because the thing being
    # restored into is empty would be backwards.
    if not live["problems"]:
        kept = snapshot(LIVE)
        print(f"\ncurrent corpus snapshotted to {kept}")
    else:
        print("\nno usable live corpus to snapshot; restoring into a fresh volume")

    # Atomic: same filesystem, so the rename cannot leave a half-written file
    # in place. A reader holding the old inode keeps it until it reopens.
    os.replace(INCOMING, LIVE)
    print(f"restored {incoming['stats'].get('documents')} documents to {LIVE}")

    # store.py connects per operation, so reads see this immediately. Only a
    # schema migration needs the process to restart - see the module docstring.
    if incoming["warnings"]:
        print(
            "\nRESTART REQUIRED. This corpus needs a schema migration, which "
            "only runs when\nthe agent starts. Until then index_documents and "
            "the corpus audit fail with\n'no such column'. Restart, then run "
            "backfill_claims.py:\n"
            "    kubectl delete pod retrieval-agent-0"
        )
    else:
        print(
            "\nThe agent connects per operation, so this is already live - no "
            "restart needed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(apply=len(sys.argv) > 1 and sys.argv[1] == "apply"))
