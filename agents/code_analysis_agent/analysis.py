"""
Static analysis behind the code-analysis agent.

Kept in its own module rather than inside server.py, for the same reason
retrieval_agent/store.py is: it is real logic worth unit-testing on its own,
and the MCP wrapper stays a thin adapter like the other agents.

WHAT THIS REPLACED, AND WHY IT MATTERS. This agent used to return

    "[stub analysis] Reviewed N chars of code, no issues found (stub)."

Asked to review `def divide(a, b): return a / b`, the system answered "No issues
were found in the provided code." The model was not wrong to do that - it is
instructed to use only what the tools return, and the tool claimed a clean
result. A well-formed success carrying no information is undetectable
downstream. That is why the stub was made to raise, and why this exists.

THE DESIGN CONSTRAINT THAT FOLLOWS. Having been burned by a tool that implied
more than it knew, the failure mode to avoid is not "misses a bug" but "reports
something that is not true". A false positive gets relayed to the user as fact
by a model that has been told to trust its tools. So:

  - the checks here are chosen for PRECISION over coverage
  - a clean result states WHAT WAS CHECKED and explicitly declines to claim the
    code is correct

THREE LAYERS, IN ORDER:

1. Parse. A SyntaxError makes every later check meaningless, so it
   short-circuits and is reported on its own.
2. pyflakes. It is a NAME AND FLOW checker - undefined names, unused imports,
   shadowed bindings - not a bug finder. Pure Python, no config, no plugins,
   which keeps the image slim and the output stable.
3. AST rules for bug patterns pyflakes deliberately does not cover, because
   they are style or semantics rather than name resolution. Only patterns
   pyflakes does NOT already report - an assert-on-a-tuple rule was written
   here and then removed once pyflakes was found to flag it, because two
   findings for one problem reads as two problems.
"""

import ast
import io
import logging

from pyflakes.api import check
from pyflakes.reporter import Reporter

logger = logging.getLogger(__name__)

# Capped so a pathological input cannot pin the CPU. Tool arguments arrive from
# a model, which means they are not necessarily sane.
MAX_CODE_BYTES = 100_000

# Named here so a clean report can list them. If a check is added, this is the
# line that has to change with it - the report must never describe coverage it
# does not have.
CHECK_NAMES = (
    "syntax",
    "undefined and unused names (pyflakes)",
    "mutable default arguments",
    "bare and silenced excepts",
    "identity comparisons against None/True/False",
    "division by an unguarded parameter",
    "unreachable code",
)

DISCLAIMER = (
    "This is static analysis only. It does not execute the code and does not "
    "check logic, algorithmic correctness, performance or security. A clean "
    "result means these specific checks found nothing - it is NOT evidence "
    "that the code is correct."
)


class Finding:
    """One reported problem. Ordering is by position so output is stable."""

    def __init__(self, line: int, message: str, source: str):
        self.line = line
        self.message = message
        self.source = source

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Finding(line={self.line}, message={self.message!r})"

    def __eq__(self, other) -> bool:
        return (self.line, self.message, self.source) == (
            other.line,
            other.message,
            other.source,
        )

    def format(self) -> str:
        return f"  line {self.line}: {self.message}  [{self.source}]"


def _pyflakes_findings(code: str) -> list[Finding]:
    """
    Run pyflakes and parse its reported lines.

    pyflakes has no structured API - check() writes text to a Reporter - so the
    output is parsed back. The format is "filename:line:col: message", and the
    filename is fixed below so splitting on it is unambiguous.
    """
    out, err = io.StringIO(), io.StringIO()
    check(code, "<snippet>", Reporter(out, err))

    findings = []
    for raw in out.getvalue().splitlines():
        if not raw.startswith("<snippet>:"):
            continue
        parts = raw.split(":", 3)
        try:
            line = int(parts[1])
        except (IndexError, ValueError):
            continue
        message = parts[-1].strip()
        findings.append(Finding(line, message, "pyflakes"))
    return findings


class _Rules(ast.NodeVisitor):
    """
    AST checks for patterns pyflakes does not look for.

    Every rule here reports a FACT ABOUT THE CODE rather than a judgement about
    intent - "division by parameter b, which is never checked against zero"
    rather than "this will crash". The distinction matters because a model
    relays these to a person as authoritative.
    """

    def __init__(self):
        self.findings: list[Finding] = []

    # -- def f(x=[]) ------------------------------------------------------
    def _check_mutable_defaults(self, node):
        for default in list(node.args.defaults) + [
            d for d in node.args.kw_defaults if d is not None
        ]:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                kind = type(default).__name__.lower()
                self.findings.append(
                    Finding(
                        default.lineno,
                        f"mutable default argument ({kind}) - it is created once at "
                        f"definition and shared across every call",
                        "ast",
                    )
                )

    # -- division by a parameter with no zero check ------------------------
    def _check_unguarded_division(self, node):
        params = {a.arg for a in node.args.args + node.args.kwonlyargs}
        params |= {a.arg for a in node.args.posonlyargs}
        if not params:
            return

        # Any comparison or truth test mentioning the name counts as a guard.
        # Deliberately generous: missing a real bug is preferable to inventing
        # one, per the precision constraint in this module's docstring.
        guarded = set()
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Compare, ast.If, ast.Assert, ast.IfExp)):
                for inner in ast.walk(sub):
                    if isinstance(inner, ast.Name):
                        guarded.add(inner.id)

        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.BinOp)
                and isinstance(sub.op, (ast.Div, ast.FloorDiv, ast.Mod))
                and isinstance(sub.right, ast.Name)
                and sub.right.id in params
                and sub.right.id not in guarded
            ):
                self.findings.append(
                    Finding(
                        sub.lineno,
                        f"divides by parameter '{sub.right.id}', which is never "
                        f"compared against zero in this function",
                        "ast",
                    )
                )

    def _visit_function(self, node):
        self._check_mutable_defaults(node)
        self._check_unguarded_division(node)
        self._check_unreachable(node.body)
        self.generic_visit(node)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- except: / except Exception: pass ---------------------------------
    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.findings.append(
                Finding(
                    node.lineno,
                    "bare 'except:' also catches KeyboardInterrupt and SystemExit",
                    "ast",
                )
            )
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.findings.append(
                Finding(
                    node.lineno,
                    "exception is caught and silently discarded",
                    "ast",
                )
            )
        self.generic_visit(node)

    # -- x == None / x == True --------------------------------------------
    def visit_Compare(self, node):
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(
                comparator, ast.Constant
            ):
                if comparator.value is None or isinstance(comparator.value, bool):
                    self.findings.append(
                        Finding(
                            node.lineno,
                            f"compares against {comparator.value!r} with '==' - "
                            f"'is' is the identity test",
                            "ast",
                        )
                    )
        self.generic_visit(node)

    # -- code after return/raise/continue/break ---------------------------
    def _check_unreachable(self, body):
        for index, statement in enumerate(body):
            terminal = isinstance(
                statement, (ast.Return, ast.Raise, ast.Continue, ast.Break)
            )
            if terminal and index + 1 < len(body):
                self.findings.append(
                    Finding(
                        body[index + 1].lineno,
                        f"unreachable: preceded by "
                        f"{type(statement).__name__.lower()} on line "
                        f"{statement.lineno}",
                        "ast",
                    )
                )
                break


def analyse(code: str) -> list[Finding]:
    """
    Run every check. Raises ValueError for input that cannot be analysed.

    A SyntaxError is returned as a single finding rather than raised: it is a
    legitimate ANALYSIS RESULT about the submitted code, not a failure of this
    agent. Distinguishing those matters, because the tool wrapper counts raised
    exceptions as tool_calls_total{status="error"}, and "the code you gave me
    does not parse" is a successful analysis.
    """
    if not code or not code.strip():
        raise ValueError("no code supplied")
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError(
            f"code is larger than the {MAX_CODE_BYTES} byte limit for one call"
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [
            Finding(
                exc.lineno or 0,
                f"syntax error: {exc.msg}",
                "syntax",
            )
        ]

    rules = _Rules()
    rules.visit(tree)

    findings = _pyflakes_findings(code) + rules.findings
    findings.sort(key=lambda f: (f.line, f.source, f.message))
    return findings


def report(code: str) -> str:
    """Render findings as the text the orchestrating model will read."""
    findings = analyse(code)

    if not findings:
        checked = "\n".join(f"  - {name}" for name in CHECK_NAMES)
        return (
            f"No issues found by the checks that were run.\n\n"
            f"Checks applied:\n{checked}\n\n{DISCLAIMER}"
        )

    lines = "\n".join(f.format() for f in findings)
    plural = "issue" if len(findings) == 1 else "issues"
    return f"Found {len(findings)} {plural}:\n{lines}\n\n{DISCLAIMER}"
