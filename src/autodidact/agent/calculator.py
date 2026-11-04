"""A small, safe calculator the agent can call instead of doing arithmetic in its head.

Language models make arithmetic slips (one baseline run answered ₹114,556.55
instead of ₹114,556.50). Those slips have nothing to do with learning how an
app works, so the agent gets a calculator and the evaluation measures what it
should.

Safety: the expression is parsed with ``ast`` and only numbers, + - * / // %,
parentheses and a few named functions are allowed. Nothing is ever passed to
``eval``. Arithmetic is exact (``Decimal``), so money sums do not drift.
"""

from __future__ import annotations

import ast
import decimal
from decimal import ROUND_HALF_UP, Decimal

MAX_CHARS = 2000
MAX_DIGITS = 60

CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate an arithmetic expression exactly. Supports + - * / // %, parentheses, "
            "and the functions round(x, digits), sum(a, b, ...), min, max and abs. "
            "Use it for any arithmetic instead of computing in your head."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "For example: (19350 * 12 + 4500 * 3) / 100",
                }
            },
            "required": ["expression"],
        },
    },
}


class CalcError(Exception):
    pass


def _round(x: Decimal, digits: Decimal = Decimal(0)) -> Decimal:
    if digits != digits.to_integral_value():
        raise CalcError("round() needs a whole number of digits")
    return x.quantize(Decimal(1).scaleb(-int(digits)), rounding=ROUND_HALF_UP)


FUNCTIONS = {
    "round": _round,
    "sum": lambda *xs: sum(xs, Decimal(0)),
    "min": lambda *xs: min(xs),
    "max": lambda *xs: max(xs),
    "abs": lambda x: abs(x),
}

BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}


def evaluate(expression: str) -> Decimal:
    if len(expression) > MAX_CHARS:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(expression.replace("₹", ""), mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"could not parse: {exc.msg}") from exc
    with decimal.localcontext() as ctx:
        ctx.prec = 40
        ctx.traps[decimal.DivisionByZero] = True
        ctx.traps[decimal.InvalidOperation] = True
        try:
            return _eval(tree.body)
        except (decimal.DivisionByZero, ZeroDivisionError) as exc:
            raise CalcError("division by zero") from exc
        except decimal.InvalidOperation as exc:
            raise CalcError("invalid operation") from exc


def _eval(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = Decimal(repr(node.value)) if isinstance(node.value, float) else Decimal(node.value)
        if len(value.as_tuple().digits) > MAX_DIGITS:
            raise CalcError("number too large")
        return value
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        result = BINARY[type(node.op)](_eval(node.left), _eval(node.right))
        if result.adjusted() > MAX_DIGITS:
            raise CalcError("result too large")
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = _eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Tuple):
        raise CalcError("write numbers without thousands separators, e.g. 124350 not 1,24,350")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        args = [_eval(a) for a in node.args]
        try:
            return FUNCTIONS[node.func.id](*args)
        except TypeError as exc:
            raise CalcError(f"wrong arguments for {node.func.id}()") from exc
    raise CalcError(f"not allowed: {ast.unparse(node)[:60]}")


def calculate(expression: str) -> str:
    """What the agent sees: the exact result, or an ERROR line."""
    try:
        value = evaluate(expression)
    except CalcError as exc:
        return f"ERROR: {exc}"
    text = (
        format(value.normalize(), "f") if value == value.to_integral_value() else format(value, "f")
    )
    return text
