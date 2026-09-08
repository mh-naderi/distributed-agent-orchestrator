"""
The restore has to refuse a corpus that is not one.

WHY THIS EXISTS. A restore is the one operation whose failure mode is worse
than not running it. The dangerous case is not a missing backup - that is
obvious the moment you look - but a restore that completes and replaces a
working index with a broken one, because a transfer that stopped partway still
produces a file, and SQLite opens the intact prefix of a truncated database
without complaint.

So the checks in restore_corpus.inspect are the safety-critical part, and they
are the part that is never exercised in practice: they only run for real on the
day something has already gone wrong. These tests run them against databases
broken on purpose.

The script itself lives in the agent directory but is not shipped in the image
- see test_agent_images.py - so it is loaded here by path rather than imported.
"""

import importlib.util
import sqlite3
import struct
from pathlib import Path

import pytest
import sqlite_vec

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "agents" / "retrieval_agent" / "restore_corpus.py"

DIM = 4  # the real corpus uses 768; the checks here are count-based


def load_script():
    spec = importlib.util.spec_from_file_location("restore_corpus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


restore = load_script()


def make_corpus(path: Path, documents: int = 3, vectors: int | None = None,
                claimed_source: bool = True, vector_table: bool = True) -> Path:
    """A corpus shaped like the real one, with the parts under test optional."""
    con = sqlite3.connect(path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)

    con.execute(
        "CREATE TABLE documents ("
        " id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL,"
        " created_at TEXT NOT NULL"
        ")"
    )
    if claimed_source:
        con.execute("ALTER TABLE documents ADD COLUMN claimed_source TEXT")
    if vector_table:
        con.execute(
            f"CREATE VIRTUAL TABLE document_vectors USING vec0(embedding float[{DIM}])"
        )

    for i in range(documents):
        con.execute(
            "INSERT INTO documents (id, text, source, created_at) VALUES (?, ?, ?, ?)",
            (i + 1, f"document {i}", "https://example.test/a" if i % 2 else "unattributed",
             "2026-01-01T00:00:00"),
        )
    if vector_table:
        for i in range(documents if vectors is None else vectors):
            con.execute(
                "INSERT INTO document_vectors (rowid, embedding) VALUES (?, ?)",
                (i + 1, struct.pack(f"{DIM}f", *([0.1] * DIM))),
            )

    con.commit()
    con.close()
    return path


# ---------------------------------------------------------------------------
# The corpus that should restore
# ---------------------------------------------------------------------------


def test_a_healthy_corpus_has_no_problems(tmp_path):
    report = restore.inspect(str(make_corpus(tmp_path / "good.db")))

    assert report["problems"] == []
    assert report["stats"]["documents"] == 3
    assert report["stats"]["vectors"] == 3
    # The provenance split the corpus audit depends on.
    assert report["stats"]["url_sourced"] == 1
    assert report["stats"]["unattributed"] == 2


# ---------------------------------------------------------------------------
# ...and every corpus that should not
# ---------------------------------------------------------------------------


def test_a_missing_file_is_refused(tmp_path):
    report = restore.inspect(str(tmp_path / "absent.db"))
    assert any("does not exist" in p for p in report["problems"])


def test_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty.db"
    path.write_bytes(b"")
    report = restore.inspect(str(path))
    assert any("is empty" in p for p in report["problems"])


def test_a_truncated_transfer_is_refused(tmp_path):
    """
    The case worth designing against. A half-transferred database keeps a valid
    SQLite header, so anything that checks the first bytes calls it fine.
    """
    path = make_corpus(tmp_path / "truncated.db", documents=200)
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) // 3])

    report = restore.inspect(str(path))

    assert report["problems"], "a truncated database was accepted"
    # It still looks like SQLite - that is the point.
    assert path.read_bytes().startswith(b"SQLite format 3")


def test_a_barely_truncated_corpus_still_fails_integrity(tmp_path):
    """
    The case that justifies reading every page instead of trusting counts.

    Cut a corpus at 99% and it still opens, still reports the full document
    count, still reports a matching vector count - every cheap check passes.
    Measured on a 400-document corpus: 400 documents, 400 vectors, and only
    PRAGMA integrity_check noticed, reporting rowids out of order.
    """
    path = make_corpus(tmp_path / "nearly.db", documents=400)
    whole = path.read_bytes()
    path.write_bytes(whole[: int(len(whole) * 0.99)])

    report = restore.inspect(str(path))

    # The counts are intact and useless as a signal...
    assert report["stats"]["documents"] == 400
    # ...and this is what actually catches it.
    assert any("integrity_check" in p for p in report["problems"])


def test_a_file_that_is_not_a_database_is_refused(tmp_path):
    path = tmp_path / "notadb.db"
    path.write_bytes(b"this is not a database, it is a fragment of an HTML error page")
    report = restore.inspect(str(path))
    assert report["problems"]


def test_an_empty_corpus_is_refused(tmp_path):
    """
    Restoring nothing over something is not a restore. This is the plausible
    accident: a snapshot taken from a fresh cluster before anything indexed.
    """
    report = restore.inspect(str(make_corpus(tmp_path / "nodocs.db", documents=0)))
    assert any("corpus is empty" in p for p in report["problems"])


def test_a_corpus_with_no_vector_table_is_refused(tmp_path):
    report = restore.inspect(
        str(make_corpus(tmp_path / "novec.db", vector_table=False))
    )
    assert any("document_vectors" in p for p in report["problems"])


def test_vectors_that_do_not_match_the_documents_are_refused(tmp_path):
    """
    Retrieval would work and quietly never return the documents with no vector,
    which is the failure this project keeps finding: a wrong answer that looks
    like a correct one.
    """
    report = restore.inspect(
        str(make_corpus(tmp_path / "mismatch.db", documents=5, vectors=2))
    )
    assert any("vectors" in p and "silently" in p for p in report["problems"])


def test_a_corpus_predating_the_provenance_split_warns_but_restores(tmp_path):
    """
    Restoring a pre-split backup is a SUPPORTED recovery, not an error. The
    first version of the script refused it, which would have blocked the exact
    procedure the runbook documents - store.py adds claimed_source when the
    agent next opens the database, and backfill_claims.py moves the old labels
    across afterwards.

    Found by running the restore against the real 170-document backup from
    2026-09-03, not by reading the code.
    """
    report = restore.inspect(
        str(make_corpus(tmp_path / "old.db", claimed_source=False))
    )

    assert report["problems"] == [], "a pre-split corpus must not be refused"
    warning = next(w for w in report["warnings"] if "claimed_source" in w)
    assert "backfill_claims.py" in warning


def test_a_warning_alone_does_not_stop_the_restore(tmp_path, monkeypatch, capsys):
    """The distinction only matters if apply actually proceeds."""
    live = make_corpus(tmp_path / "live.db", documents=2)
    incoming = make_corpus(tmp_path / "live.db.incoming", documents=6,
                           claimed_source=False)

    monkeypatch.setattr(restore, "LIVE", str(live))
    monkeypatch.setattr(restore, "INCOMING", str(incoming))

    assert restore.main(apply=True) == 0
    out = capsys.readouterr().out
    assert "NOTE:" in out and "REFUSING" not in out
    assert restore.inspect(str(live))["stats"]["documents"] == 6

    # ...and this is the case that genuinely needs the restart, because
    # _create_schema runs once in __init__. Without it index_documents and the
    # corpus audit fail with 'no such column: claimed_source'.
    assert "RESTART REQUIRED" in out
    assert "kubectl delete pod retrieval-agent-0" in out


# ---------------------------------------------------------------------------
# The swap
# ---------------------------------------------------------------------------


def test_the_live_corpus_is_snapshotted_before_being_replaced(tmp_path):
    live = make_corpus(tmp_path / "live.db", documents=7)
    kept = Path(restore.snapshot(str(live)))

    assert kept.exists() and kept != live
    assert restore.inspect(str(kept))["stats"]["documents"] == 7
    # VACUUM INTO must not disturb the original.
    assert restore.inspect(str(live))["stats"]["documents"] == 7


def test_a_refused_restore_leaves_the_live_corpus_alone(tmp_path, monkeypatch, capsys):
    live = make_corpus(tmp_path / "live.db", documents=9)
    incoming = tmp_path / "live.db.incoming"
    incoming.write_bytes(b"truncated rubbish")

    monkeypatch.setattr(restore, "LIVE", str(live))
    monkeypatch.setattr(restore, "INCOMING", str(incoming))

    assert restore.main(apply=True) == 1
    assert "REFUSING TO RESTORE" in capsys.readouterr().out
    assert restore.inspect(str(live))["stats"]["documents"] == 9
    assert incoming.exists(), "a rejected file should stay for inspection"


def test_a_good_restore_replaces_the_corpus_and_keeps_the_old_one(tmp_path, monkeypatch, capsys):
    live = make_corpus(tmp_path / "live.db", documents=2)
    incoming = make_corpus(tmp_path / "live.db.incoming", documents=11)

    monkeypatch.setattr(restore, "LIVE", str(live))
    monkeypatch.setattr(restore, "INCOMING", str(incoming))

    assert restore.main(apply=True) == 0

    assert restore.inspect(str(live))["stats"]["documents"] == 11
    assert not incoming.exists(), "the incoming file should have been moved, not copied"

    backups = list(tmp_path.glob("live.db.bak-*"))
    assert len(backups) == 1, "the replaced corpus must still be recoverable"
    assert restore.inspect(str(backups[0]))["stats"]["documents"] == 2

    # A corpus that needs no migration is live immediately: store.py opens a
    # connection per operation, so there is no stale handle to flush. Telling
    # the operator to restart here would be cargo cult - it was in the first
    # draft, and measuring the running agent is what removed it.
    out = capsys.readouterr().out
    assert "already live - no restart needed" in out
    assert "RESTART REQUIRED" not in out


def test_a_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    live = make_corpus(tmp_path / "live.db", documents=2)
    incoming = make_corpus(tmp_path / "live.db.incoming", documents=11)

    monkeypatch.setattr(restore, "LIVE", str(live))
    monkeypatch.setattr(restore, "INCOMING", str(incoming))

    assert restore.main(apply=False) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert restore.inspect(str(live))["stats"]["documents"] == 2
    assert incoming.exists()
    assert not list(tmp_path.glob("live.db.bak-*"))


def test_restoring_into_a_fresh_volume_needs_no_snapshot(tmp_path, monkeypatch, capsys):
    """A new cluster has no live corpus; that must not block the restore."""
    incoming = make_corpus(tmp_path / "live.db.incoming", documents=4)
    monkeypatch.setattr(restore, "LIVE", str(tmp_path / "live.db"))
    monkeypatch.setattr(restore, "INCOMING", str(incoming))

    assert restore.main(apply=True) == 0
    assert "fresh volume" in capsys.readouterr().out
    assert restore.inspect(str(tmp_path / "live.db"))["stats"]["documents"] == 4


@pytest.mark.parametrize("name", ["documents", "document_vectors"])
def test_the_required_tables_are_the_ones_the_agent_uses(name):
    """
    A guard on the guard: if store.py renames a table, these checks would pass
    against a corpus this agent cannot read.
    """
    store_source = (ROOT / "agents" / "retrieval_agent" / "store.py").read_text(
        encoding="utf-8"
    )
    assert name in store_source
    assert name in restore.REQUIRED_TABLES
