"""
Every module an agent imports has to be in its image.

WHY THIS EXISTS. Each agent builds from its own directory, and its Dockerfile
names the files to copy one by one. Adding instrumentation.py needed a matching
COPY line, and so did coverage.py; both were remembered by hand. The third time
would not fail here - the image builds perfectly well without a file nothing in
the build references - it would fail on deploy, as a crash loop on import, or
later still if the missing module were only reached when a particular tool ran.

Checked by reading rather than by building, because the question is not "does
this image build" but "does it contain what the code needs". That needs no
Docker, runs in milliseconds, and says which file is missing from which
Dockerfile instead of leaving a traceback in a pod log.

The imports are followed transitively: server.py imports store.py, and if
store.py imported a third local module then that one is needed in the image
too, though nothing in server.py mentions it.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

AGENTS = sorted(
    p for p in (Path(__file__).resolve().parents[1] / "agents").iterdir() if p.is_dir()
)


def local_modules(agent: Path) -> set[str]:
    return {p.stem for p in agent.glob("*.py")}


def imports_of(path: Path, available: set[str]) -> set[str]:
    """Local modules this file imports directly."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # level 0 is absolute; a relative import inside an agent would not
            # resolve anyway, since the image has no package around these files.
            if node.level == 0 and node.module in available:
                found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in available:
                    found.add(alias.name)
    return found


def required_modules(agent: Path) -> set[str]:
    """Everything reachable from server.py, following local imports."""
    available = local_modules(agent)
    needed: set[str] = set()
    frontier = ["server"]

    while frontier:
        name = frontier.pop()
        if name in needed:
            continue
        needed.add(name)
        module_path = agent / f"{name}.py"
        if module_path.exists():
            frontier.extend(imports_of(module_path, available) - needed)

    return needed


def copied_modules(agent: Path) -> set[str]:
    copied = set()
    for line in (agent / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY"):
            for token in line.split()[1:]:
                if token.endswith(".py"):
                    copied.add(token[:-3])
    return copied


@pytest.mark.parametrize("agent", AGENTS, ids=lambda p: p.name)
def test_the_image_carries_every_module_the_agent_imports(agent):
    missing = required_modules(agent) - copied_modules(agent)

    assert not missing, (
        f"{agent.name}/Dockerfile does not COPY {sorted(missing)}. "
        f"The image would build and then fail on import at deploy time."
    )


@pytest.mark.parametrize("agent", AGENTS, ids=lambda p: p.name)
def test_the_image_does_not_carry_files_that_are_not_there(agent):
    """
    A COPY naming a file that has since been renamed or deleted fails the build
    outright, which is a better failure than a silent omission - but it fails at
    deploy time, and this says so in a second.
    """
    for name in copied_modules(agent):
        assert (agent / f"{name}.py").exists(), (
            f"{agent.name}/Dockerfile copies {name}.py, which no longer exists"
        )


@pytest.mark.parametrize("agent", AGENTS, ids=lambda p: p.name)
def test_maintenance_scripts_are_not_shipped(agent):
    """
    backfill_claims.py is run through `kubectl exec -i ... python -` and reads
    the volume, not the image. Copying it in would be harmless but misleading:
    a file in the image reads as something the service runs.
    """
    for script in agent.glob("*.py"):
        if script.stem in required_modules(agent):
            continue
        assert script.stem not in copied_modules(agent), (
            f"{script.name} is not imported by {agent.name}'s server but is copied "
            "into the image; either it is dead weight or something imports it "
            "in a way this test cannot see"
        )


def test_the_check_would_notice_a_missing_copy(tmp_path):
    """
    The guard needs a guard. A test that passes because it looks at the wrong
    thing is worse than no test, and this one is all path handling.
    """
    agent = tmp_path / "fake_agent"
    agent.mkdir()
    (agent / "server.py").write_text("from helper import thing\n", encoding="utf-8")
    (agent / "helper.py").write_text("thing = 1\n", encoding="utf-8")
    (agent / "Dockerfile").write_text("COPY server.py .\n", encoding="utf-8")

    assert required_modules(agent) == {"server", "helper"}
    assert copied_modules(agent) == {"server"}
    assert required_modules(agent) - copied_modules(agent) == {"helper"}


def test_the_check_follows_imports_more_than_one_deep(tmp_path):
    agent = tmp_path / "fake_agent"
    agent.mkdir()
    (agent / "server.py").write_text("import middle\n", encoding="utf-8")
    (agent / "middle.py").write_text("from deep import x\n", encoding="utf-8")
    (agent / "deep.py").write_text("x = 1\n", encoding="utf-8")
    (agent / "Dockerfile").write_text("COPY server.py middle.py .\n", encoding="utf-8")

    assert required_modules(agent) == {"server", "middle", "deep"}
    assert required_modules(agent) - copied_modules(agent) == {"deep"}


# ---------------------------------------------------------------------------
# ...and the dependencies those modules need
# ---------------------------------------------------------------------------
# The other half of "would this image actually run". Each agent carries its own
# requirements.txt, deliberately - the images stay small and independent - and
# the tests import from the working tree, where the dev virtualenv has every
# package any agent might want. So an import added without a matching
# requirements line passes every test here and fails on `pip install` in the
# build, or worse, at import time in the pod if the package happens to be a
# transitive dependency of something else.


def third_party_imports(agent: Path) -> set[str]:
    """Non-stdlib, non-local modules imported by anything shipped in the image."""
    local = local_modules(agent)
    found: set[str] = set()

    for name in required_modules(agent):
        path = agent / f"{name}.py"
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])

    return {m for m in found if m not in sys.stdlib_module_names and m not in local}


def declared_requirements(agent: Path) -> set[str]:
    """
    Package names from requirements.txt, normalised the way PyPI does.

    Extras and version specifiers are stripped, so `mcp[cli]>=1.27,<2` is `mcp`,
    and underscores become hyphens, so the import `sqlite_vec` lines up with the
    distribution `sqlite-vec`.
    """
    declared = set()
    for line in (agent / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[\[<>=!~;\s]", line, maxsplit=1)[0]
        if name:
            declared.add(name.lower().replace("_", "-"))
    return declared


@pytest.mark.parametrize("agent", AGENTS, ids=lambda p: p.name)
def test_every_third_party_import_is_declared(agent):
    imported = {m.lower().replace("_", "-") for m in third_party_imports(agent)}
    missing = imported - declared_requirements(agent)

    assert not missing, (
        f"{agent.name} imports {sorted(missing)} but its requirements.txt does not "
        "declare them. The tests pass because the dev virtualenv has them; the "
        "image would not."
    )


def test_the_dependency_check_would_notice_an_undeclared_import(tmp_path):
    """
    The same reasoning as the COPY guard's own guard: this is string handling
    over two file formats, and a check that quietly matches nothing would report
    success forever.
    """
    agent = tmp_path / "fake_agent"
    agent.mkdir()
    (agent / "server.py").write_text("import requests\nimport os\n", encoding="utf-8")
    (agent / "Dockerfile").write_text("COPY server.py .\n", encoding="utf-8")
    (agent / "requirements.txt").write_text("# nothing declared\n", encoding="utf-8")

    assert third_party_imports(agent) == {"requests"}, "stdlib must not be flagged"
    assert declared_requirements(agent) == set()


def test_extras_and_pins_do_not_hide_a_declared_package(tmp_path):
    """`mcp[cli]>=1.27,<2` declares `mcp`; a naive comparison would miss it."""
    agent = tmp_path / "fake_agent"
    agent.mkdir()
    (agent / "server.py").write_text("from mcp import ClientSession\n", encoding="utf-8")
    (agent / "Dockerfile").write_text("COPY server.py .\n", encoding="utf-8")
    (agent / "requirements.txt").write_text(
        "mcp[cli]>=1.27,<2  # comment\nsqlite_vec>=0.1.9\n", encoding="utf-8"
    )

    assert declared_requirements(agent) == {"mcp", "sqlite-vec"}
    assert third_party_imports(agent) - declared_requirements(agent) == set()
