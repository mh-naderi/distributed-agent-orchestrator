"""
A restricted expression evaluator. NOT a sandbox, and the distinction matters.

A sandbox runs arbitrary code and tries to contain what it does. This runs a
deliberately tiny language in which nothing dangerous is expressible - there is
no import, no attribute access, no assignment, no function definition and no
name that was not put there on purpose. The safety comes from absence rather
than from containment, which is why it needs no privileges, no extra
infrastructure, and behaves identically on Windows and in the cluster.

WHY NOT A REAL SANDBOX. See "Design: the code execution sandbox, and why it is
not built" in docs/architecture.md. Briefly, all three standard answers were
measured and rejected on this hardware: /var/run/docker.sock is not present in a
pod so container-per-execution would mean handing a pod root on the host;
RLIMIT_AS and RLIMIT_CPU exist in the Linux containers but `import resource`
fails on the Windows host that the runbook documents as the normal way to
develop, so rlimit-based containment would silently do nothing there; and gVisor
and Kata do not run under kind on Docker Desktop at all.

WHY AN INTERPRETER RATHER THAN compile() + eval(). Whitelisting node types and
then handing the tree to eval() checks the SHAPE of an expression and nothing
about what it does. Walking it here means every individual operation can be
inspected before it is performed, which is what makes the limits below
enforceable rather than aspirational.

THE ATTACKS THAT MATTER ARE NOT THE OBVIOUS ONES. `import os` is trivially
rejected by a whitelist. The interesting ones are expressions built entirely
from allowed pieces:

    ().__class__.__bases__[0].__subclasses__()   attribute access is the classic
                                                 escape hatch, so Attribute is
                                                 rejected outright
    10 ** 10 ** 10                               a whitelisted operator that
                                                 hangs the process
    "a" * 10**9                                  a whitelisted operator that
                                                 exhausts memory
    [0 for _ in range(10**9)]                    comprehensions loop, so they
                                                 are rejected too

Each is refused by an explicit rule below rather than by hoping the step limit
catches it first - a limit that fires after the machine has already allocated a
gigabyte is not a limit.
"""

import ast
import time

# Wall clock and work ceilings. Both exist because either alone is escapable:
# a tight loop of cheap operations exhausts the step budget slowly, and a single
# enormous operation blows the time budget in one step.
MAX_STEPS = 10_000
MAX_SECONDS = 1.0

# Any sequence produced by an operation. Bounds "a" * n and list * n, which are
# whitelisted operators that allocate without looping.
MAX_SEQUENCE_LEN = 10_000

# Bounds the exponent of **. 2 ** 4096 is a 1234-digit number and costs nothing;
# 10 ** 10 ** 10 never returns. Checked BEFORE the operation, since afterwards
# is too late.
MAX_POW_EXPONENT = 4096

# Bounds integer size generally, so a chain of multiplications cannot grow
# without limit even though no single one exceeds the sequence cap.
MAX_INT_BITS = 16_384

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
    ast.Not: lambda a: not a,
}

_COMPARE_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

# Deliberately small, and every entry is a pure function over the values this
# evaluator can already produce. Nothing here opens a file, imports a module, or
# returns an object with interesting attributes - which matters because
# attribute access is rejected anyway, so a returned object would be inert.
_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "divmod": divmod,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
}

# Named so a refusal can say what IS allowed rather than only what is not.
CAPABILITIES = (
    "arithmetic on numbers",
    "comparisons and boolean logic",
    "list, tuple, dict and set literals",
    "indexing and slicing literals",
    "conditional expressions (a if c else b)",
    f"these functions only: {', '.join(sorted(_BUILTINS))}",
)

LIMITS = (
    f"at most {MAX_STEPS} operations",
    f"at most {MAX_SECONDS}s of wall clock",
    f"sequences no longer than {MAX_SEQUENCE_LEN}",
    f"exponents no larger than {MAX_POW_EXPONENT}",
)


class EvaluationError(Exception):
    """Raised for anything refused or exceeded. Carries a readable reason."""


class _Budget:
    """Shared step and time ceilings for one evaluation."""

    def __init__(self):
        self.steps = 0
        self.deadline = time.monotonic() + MAX_SECONDS

    def tick(self) -> None:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise EvaluationError(f"gave up after {MAX_STEPS} operations")
        # Checked every step rather than once, because a single operation can
        # be slow even when the step count is tiny.
        if time.monotonic() > self.deadline:
            raise EvaluationError(f"gave up after {MAX_SECONDS}s")


def _check_size(value):
    """
    Refuse a value that is too large to keep working with.

    Applied to results rather than only to operands: multiplication is
    whitelisted, so a chain of individually reasonable operations is the way a
    value grows without any single step looking suspicious.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise EvaluationError(
            f"result is too large (over {MAX_INT_BITS} bits)"
        )
    if isinstance(value, (str, bytes, list, tuple, set, dict)) and len(value) > MAX_SEQUENCE_LEN:
        raise EvaluationError(
            f"result is longer than {MAX_SEQUENCE_LEN} items"
        )
    return value


def _guard_binop(op, left, right) -> None:
    """
    Refuse operations that would be expensive BEFORE performing them.

    Checking the result afterwards is useless for these two: `10 ** 10 ** 10`
    never produces a result to check, and `"a" * 10**9` has already allocated
    the memory by the time anyone could look.
    """
    if isinstance(op, ast.Pow):
        if isinstance(right, int) and right > MAX_POW_EXPONENT:
            raise EvaluationError(
                f"exponent {right} is larger than the limit of {MAX_POW_EXPONENT}"
            )
        if isinstance(right, float) and right > MAX_POW_EXPONENT:
            raise EvaluationError("exponent is larger than the limit")

    if isinstance(op, ast.Mult):
        for a, b in ((left, right), (right, left)):
            if isinstance(a, (str, bytes, list, tuple)) and isinstance(b, int):
                if len(a) * max(b, 0) > MAX_SEQUENCE_LEN:
                    raise EvaluationError(
                        f"that would build a sequence longer than {MAX_SEQUENCE_LEN}"
                    )


def _refuse(node) -> EvaluationError:
    """A refusal that names the construct, so the model can adapt rather than retry."""
    friendly = {
        ast.Attribute: "attribute access",
        ast.Import: "imports",
        ast.ImportFrom: "imports",
        ast.Lambda: "lambdas",
        ast.ListComp: "comprehensions",
        ast.SetComp: "comprehensions",
        ast.DictComp: "comprehensions",
        ast.GeneratorExp: "generator expressions",
        ast.Await: "await",
        ast.Yield: "yield",
        ast.NamedExpr: "assignment",
        ast.Starred: "argument unpacking",
        ast.JoinedStr: "f-strings",
    }.get(type(node), type(node).__name__)
    return EvaluationError(f"{friendly} is not allowed here")


def _eval(node, budget: _Budget):
    budget.tick()

    if isinstance(node, ast.Expression):
        return _eval(node.body, budget)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, bool)) or node.value is None:
            return _check_size(node.value)
        raise EvaluationError(f"{type(node.value).__name__} literals are not allowed here")

    if isinstance(node, ast.BinOp):
        left = _eval(node.left, budget)
        right = _eval(node.right, budget)
        handler = _BIN_OPS.get(type(node.op))
        if handler is None:
            raise _refuse(node.op)
        _guard_binop(node.op, left, right)
        try:
            return _check_size(handler(left, right))
        except EvaluationError:
            raise
        except ZeroDivisionError:
            raise EvaluationError("division by zero") from None
        except Exception as exc:
            raise EvaluationError(f"{type(exc).__name__}: {exc}") from None

    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPS.get(type(node.op))
        if handler is None:
            raise _refuse(node.op)
        return _check_size(handler(_eval(node.operand, budget)))

    if isinstance(node, ast.BoolOp):
        # Short-circuits, like Python does, so the unevaluated branch costs
        # nothing and cannot raise.
        if isinstance(node.op, ast.And):
            result = True
            for value in node.values:
                result = _eval(value, budget)
                if not result:
                    return result
            return result
        result = False
        for value in node.values:
            result = _eval(value, budget)
            if result:
                return result
        return result

    if isinstance(node, ast.Compare):
        left = _eval(node.left, budget)
        for op, comparator in zip(node.ops, node.comparators):
            handler = _COMPARE_OPS.get(type(op))
            if handler is None:
                raise _refuse(op)
            right = _eval(comparator, budget)
            if not handler(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.IfExp):
        return _eval(node.body, budget) if _eval(node.test, budget) else _eval(node.orelse, budget)

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_eval(element, budget) for element in node.elts]
        built = {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
        return _check_size(built)

    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise EvaluationError("dict unpacking is not allowed here")
        built = {
            _eval(key, budget): _eval(value, budget)
            for key, value in zip(node.keys, node.values)
        }
        return _check_size(built)

    if isinstance(node, ast.Subscript):
        container = _eval(node.value, budget)
        index = _eval(node.slice, budget)
        try:
            return _check_size(container[index])
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"{type(exc).__name__}: {exc}") from None

    if isinstance(node, ast.Slice):
        return slice(
            _eval(node.lower, budget) if node.lower else None,
            _eval(node.upper, budget) if node.upper else None,
            _eval(node.step, budget) if node.step else None,
        )

    if isinstance(node, ast.Call):
        # Only a bare name may be called. This is the rule that closes the
        # classic escape: with Attribute rejected there is no way to reach
        # __class__, and with only bare names callable there is no way to reach
        # anything that was not put in _BUILTINS deliberately.
        if isinstance(node.func, ast.Attribute):
            # Reported as attribute access rather than as a bad call. Both
            # refuse it, but the model can only act on the accurate reason:
            # the problem with ().__class__.__subclasses__() is the chain, not
            # the parentheses at the end.
            raise _refuse(node.func)
        if not isinstance(node.func, ast.Name):
            raise EvaluationError("only the built-in functions may be called")
        if node.keywords:
            raise EvaluationError("keyword arguments are not allowed here")
        function = _BUILTINS.get(node.func.id)
        if function is None:
            raise EvaluationError(f"{node.func.id!r} is not one of the allowed functions")
        arguments = [_eval(argument, budget) for argument in node.args]
        try:
            return _check_size(function(*arguments))
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"{type(exc).__name__}: {exc}") from None

    if isinstance(node, ast.Name):
        # Names are not variables here - there is no assignment - so the only
        # legitimate use is naming a function, which Call handles above.
        raise EvaluationError(f"{node.id!r} is not defined here")

    raise _refuse(node)


def evaluate(expression: str):
    """
    Evaluate one expression. Raises EvaluationError for anything refused.

    Parses in "eval" mode, which rejects statements at the parser rather than
    leaving it to the whitelist: no assignment, no loops, no function or class
    definitions, no imports, because none of them are expressions.
    """
    if not expression or not expression.strip():
        raise EvaluationError("no expression supplied")
    if len(expression) > 4_000:
        raise EvaluationError("expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise EvaluationError(f"syntax error: {exc.msg}") from None

    return _eval(tree, _Budget())


def report(expression: str) -> str:
    """Render the result as the text the orchestrating model will read."""
    try:
        value = evaluate(expression)
    except EvaluationError as exc:
        allowed = "\n".join(f"  - {c}" for c in CAPABILITIES)
        return (
            f"Refused: {exc}\n\n"
            f"This evaluates a restricted subset of Python expressions, not "
            f"arbitrary code. What is available:\n{allowed}"
        )

    limits = ", ".join(LIMITS)
    return (
        f"{expression.strip()} = {value!r}\n\n"
        f"Evaluated in a restricted expression subset - no imports, no attribute "
        f"access, no assignment, no function definitions. Limits: {limits}."
    )
