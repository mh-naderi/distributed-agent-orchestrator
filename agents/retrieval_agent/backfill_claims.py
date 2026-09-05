"""
Move existing caller claims out of `source` and into `claimed_source`.

Runs inside the retrieval pod, because it needs sqlite-vec loaded and the volume
mounted:

    kubectl exec -i retrieval-agent-0 -- python - < agents/retrieval_agent/backfill_claims.py
    kubectl exec -i retrieval-agent-0 -- python - < agents/retrieval_agent/backfill_claims.py apply

WHAT IT FIXES. Until the claim and the source were separated, a document with no
`Source:` line of its own was stored under whatever the caller said - and the
model is one of the callers. The corpus accumulated 34 documents filed under
"Quazzlemint Foundation 2019 report", a foundation that does not exist, plus
smaller piles under "FIFA World Cup", "www.umfoundation.com" and
"Report 2019 | Heart and Stroke Foundation". New documents are stored correctly
now; these are the ones written before that.

WHAT IT DOES NOT DO. It deletes nothing. A document whose text is real is worth
keeping whatever was written on the front of it; what changes is that the label
stops being presented as where the document came from. Documents that name their
own origin are left alone entirely.

Idempotent: a second run finds nothing to do.
"""

import os
import re
import sqlite3
import sys

sys.path.insert(0, "/app")

from store import UNATTRIBUTED  # noqa: E402

APPLY = len(sys.argv) > 1 and sys.argv[1] == "apply"
SOURCE_LINE = re.compile(r"^Source:\s*(\S+)\s*$", re.MULTILINE)

db_path = os.environ.get("RETRIEVAL_DB_PATH", "/data/retrieval.db")
con = sqlite3.connect(db_path)

columns = {row[1] for row in con.execute("PRAGMA table_info(documents)")}
if "claimed_source" not in columns:
    sys.exit(
        "this corpus predates claimed_source. Restart the agent once so the "
        "schema migration runs, then run this again."
    )

rows = con.execute("SELECT id, source, text FROM documents").fetchall()
print(f"{len(rows)} documents")

moves = []
for doc_id, source, text in rows:
    if source == UNATTRIBUTED:
        continue  # already done
    if source.startswith(("http://", "https://")):
        continue  # derived from the document itself
    if SOURCE_LINE.search(text):
        continue  # names its own origin; the source column is already right
    moves.append((doc_id, source))

by_label: dict[str, int] = {}
for _, source in moves:
    by_label[source] = by_label.get(source, 0) + 1

print(f"\n{len(moves)} document(s) whose source is a caller's claim:")
for label, count in sorted(by_label.items(), key=lambda kv: -kv[1]):
    print(f"  {count:4d}  {label[:64]}")

if not moves:
    print("\nnothing to do")
    raise SystemExit

if not APPLY:
    print("\nDRY RUN - nothing written. Pass 'apply' to commit.")
    raise SystemExit

with con:
    con.executemany(
        "UPDATE documents SET source = ?, claimed_source = COALESCE(claimed_source, ?) "
        "WHERE id = ?",
        [(UNATTRIBUTED, source, doc_id) for doc_id, source in moves],
    )

remaining = con.execute(
    "SELECT COUNT(*) FROM documents WHERE lower(source) LIKE '%quazzlemint%'"
).fetchone()[0]
total = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
claimed = con.execute(
    "SELECT COUNT(*) FROM documents WHERE claimed_source IS NOT NULL"
).fetchone()[0]

print(f"\nAPPLIED. documents={total} (none deleted), claims recorded={claimed}")
print(f"documents still sourced to the fiction: {remaining}")
