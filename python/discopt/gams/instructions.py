"""GAMS nonlinear instruction set and the opcode -> discopt expression translator.

A GAMS Modeling Object (GMO) stores every nonlinear function (the objective and
each constraint row) as a *list of stack-machine instructions* in reverse-Polish
order.  Each instruction is a ``(opcode, field)`` pair:

* ``opcode`` is one of :class:`GamsOpCode` -- push a variable, push an immediate
  constant, add the top two stack entries, call a one-argument function, ...
* ``field`` is the instruction's operand.  Its meaning depends on the opcode:
  for the ``*V`` opcodes it is a (1-based) GMO column index, for the ``*I``
  opcodes it is a (1-based) index into the model's constant pool, and for the
  ``nlCallArg*`` opcodes it is a :class:`GamsFuncCode` selecting the intrinsic
  function to apply.

:func:`translate_instructions` walks such a list with an explicit operand stack
and rebuilds the equivalent :mod:`discopt` expression DAG.  It is deliberately
free of any dependency on the GAMS libraries so it can be unit-tested directly.

The opcode and function-code tables are reproduced from the public COIN-OR
GAMSlinks / SHOT ``GamsNLinstr`` headers, which mirror the GAMS ``gmomcc`` API.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Callable

from discopt.modeling import core as dm
from discopt.modeling.core import Expression


class GamsTranslationError(Exception):
    """Raised when a GAMS nonlinear instruction cannot be translated to discopt."""


# ── Opcodes ────────────────────────────────────────────────────────────────
# Order/values are fixed by the GAMS API (gmomcc); do not renumber.
class GamsOpCode(IntEnum):
    nlNoOp = 0
    nlPushV = 1  # push variable field
    nlPushI = 2  # push immediate constant field
    nlStore = 3  # store result (end of one function)
    nlAdd = 4  # t2 + t1
    nlAddV = 5  # t1 + var(field)
    nlAddI = 6  # t1 + const(field)
    nlSub = 7  # t2 - t1
    nlSubV = 8  # t1 - var(field)
    nlSubI = 9  # t1 - const(field)
    nlMul = 10  # t2 * t1
    nlMulV = 11  # t1 * var(field)
    nlMulI = 12  # t1 * const(field)
    nlDiv = 13  # t2 / t1
    nlDivV = 14  # t1 / var(field)
    nlDivI = 15  # t1 / const(field)
    nlUMin = 16  # -t1
    nlUMinV = 17  # -var(field)
    nlHeader = 18  # header (stack depth hint) -- ignored
    nlEnd = 19  # end of instruction list
    nlCallArg1 = 20  # unary function func(field)
    nlCallArg2 = 21  # binary function func(field)
    nlCallArgN = 22  # n-ary function func(field)
    nlFuncArgN = 23  # number of args for the following nlCallArgN
    nlMulIAdd = 24  # t2 + t1 * const(field)
    nlPushZero = 25  # push 0
    nlChk = 26
    nlAddO = 27
    nlPushO = 28
    nlInvoc = 29
    nlStackIn = 30


# ── Function codes ─────────────────────────────────────────────────────────
# Verbatim ordering from the GAMS GamsFuncCode enum (fnmapval == 0).  The
# integer value of every name is therefore its position in this list, which is
# how GMO encodes the ``field`` of an nlCallArg* instruction.
_FUNC_CODE_NAMES: tuple[str, ...] = (
    "mapval",
    "ceil",
    "floor",
    "round",
    "mod",
    "trunc",
    "sign",
    "min",
    "max",
    "sqr",
    "exp",
    "log",
    "log10",
    "sqrt",
    "abs",
    "cos",
    "sin",
    "arctan",
    "errf",
    "dunfm",
    "dnorm",
    "power",
    "jdate",
    "jtime",
    "jstart",
    "jnow",
    "error",
    "gyear",
    "gmonth",
    "gday",
    "gdow",
    "gleap",
    "ghour",
    "gminute",
    "gsecond",
    "curseed",
    "timest",
    "timeco",
    "timeex",
    "timecl",
    "frac",
    "errorl",
    "heaps",
    "fact",
    "unfmi",
    "pi",
    "ncpf",
    "ncpcm",
    "entropy",
    "sigmoid",
    "log2",
    "boolnot",
    "booland",
    "boolor",
    "boolxor",
    "boolimp",
    "booleqv",
    "relopeq",
    "relopgt",
    "relopge",
    "reloplt",
    "relople",
    "relopne",
    "ifthen",
    "rpower",
    "edist",
    "div",
    "div0",
    "sllog10",
    "sqlog10",
    "slexp",
    "sqexp",
    "slrec",
    "sqrec",
    "cvpower",
    "vcpower",
    "centropy",
    "gmillisec",
    "maxerror",
    "timeel",
    "gamma",
    "loggamma",
    "beta",
    "logbeta",
    "gammareg",
    "betareg",
    "sinh",
    "cosh",
    "tanh",
    "mathlastrc",
    "mathlastec",
    "mathoval",
    "signpower",
    "handle",
    "ncpvusin",
    "ncpvupow",
    "binomial",
    "rehandle",
    "gamsver",
    "delhandle",
    "tan",
    "arccos",
    "arcsin",
    "arctan2",
    "sleep",
    "heapf",
    "cohandle",
    "gamsrel",
    "poly",
    "licensestatus",
    "licenselevel",
    "heaplimit",
    "linear",
    "triangle",
    "forceerror",
    "forceerrorcount",
    "randbinomial",
    "jobhandle",
    "jobstatus",
    "jobkill",
    "jobterminate",
    "numcores",
    "embeddedhandle",
    "platformcode",
    "logit",
    "lsemax",
    "lsemaxsc",
    "lsemin",
    "lseminsc",
    "dummy",
)

# name -> integer function code
FUNC_CODE: dict[str, int] = {name: i for i, name in enumerate(_FUNC_CODE_NAMES)}
# integer function code -> name
FUNC_NAME: dict[int, str] = {i: name for i, name in enumerate(_FUNC_CODE_NAMES)}

# Opcodes whose ``field`` is a (1-based) GMO column index rather than a constant
# index or function code.  GMO stores those fields in the *original model*
# column space, so a solver link reading instructions through GMO must remap
# them into its own (reduced) column space before translation.
VAR_FIELD_OPCODES: frozenset[int] = frozenset(
    {
        GamsOpCode.nlPushV,
        GamsOpCode.nlAddV,
        GamsOpCode.nlSubV,
        GamsOpCode.nlMulV,
        GamsOpCode.nlDivV,
        GamsOpCode.nlUMinV,
    }
)


# ── Function-code -> discopt expression builders ────────────────────────────
# These mirror discopt.modeling.gams_parser._map_func so that a model produced
# from a GMO object is identical to one produced by from_gams() on the same .gms.
_UNARY: dict[str, Callable[[Expression], Expression]] = {
    "exp": dm.exp,
    "log": dm.log,
    "log2": dm.log2,
    "log10": dm.log10,
    "sqrt": dm.sqrt,
    "abs": dm.abs_,
    "sin": dm.sin,
    "cos": dm.cos,
    "tan": dm.tan,
    "tanh": dm.tanh,
    "sigmoid": dm.sigmoid,
    "sign": dm.sign,
    "sqr": lambda a: a**2,
    "errf": dm.erf,
    "arcsin": lambda a: dm.FunctionCall("asin", a),
    "arccos": lambda a: dm.FunctionCall("acos", a),
    "arctan": lambda a: dm.FunctionCall("atan", a),
    "sinh": lambda a: dm.FunctionCall("sinh", a),
    "cosh": lambda a: dm.FunctionCall("cosh", a),
}

_BINARY: dict[str, Callable[[Expression, Expression], Expression]] = {
    "power": lambda a, b: a**b,
    "rpower": lambda a, b: a**b,
    "cvpower": lambda a, b: a**b,
    "vcpower": lambda a, b: a**b,
    "div": lambda a, b: a / b,
    "div0": lambda a, b: a / b,
    "min": dm.minimum,
    "max": dm.maximum,
    "arctan2": lambda a, b: dm.FunctionCall("atan2", a, b),
}


def _const(value: float) -> Expression:
    return dm.Constant(float(value))


def _resolve(seq, field: int, kind: str):
    """Look up a 1-based GMO ``field`` in ``seq`` (variables or constants)."""
    idx = field - 1  # GMO fields are 1-based
    if idx < 0 or idx >= len(seq):
        raise GamsTranslationError(f"{kind} field {field} out of range (have {len(seq)} {kind}s)")
    return seq[idx]


def translate_instructions(
    opcodes: list[int],
    fields: list[int],
    variables: list[Expression],
    constants: list[float],
) -> Expression:
    """Translate one GMO nonlinear instruction list to a discopt expression.

    Parameters
    ----------
    opcodes, fields :
        Parallel lists from ``gmoGetObjFNLInstr`` / ``gmoGetEquFNLInstr`` --
        the opcode and operand of each instruction, in evaluation order.
    variables :
        Scalar discopt expressions for the model columns, indexed 0-based
        (GMO column ``j`` -> ``variables[j - 1]``).
    constants :
        The model constant pool, indexed 0-based (GMO constant ``k`` ->
        ``constants[k - 1]``).

    Returns
    -------
    Expression
        The reconstructed nonlinear expression.

    Raises
    ------
    GamsTranslationError
        On a malformed list or an unsupported opcode / function code.
    """
    if len(opcodes) != len(fields):
        raise GamsTranslationError("opcodes and fields must have equal length")

    stack: list[Expression] = []
    result: Expression | None = None

    def pop() -> Expression:
        if not stack:
            raise GamsTranslationError("instruction stack underflow")
        return stack.pop()

    i = 0
    n = len(opcodes)
    while i < n:
        try:
            op = GamsOpCode(opcodes[i])
        except ValueError as exc:
            raise GamsTranslationError(f"unknown opcode {opcodes[i]}") from exc
        field = fields[i]

        if op in (GamsOpCode.nlHeader, GamsOpCode.nlNoOp):
            pass
        elif op == GamsOpCode.nlPushV:
            stack.append(_resolve(variables, field, "variable"))
        elif op == GamsOpCode.nlPushI:
            stack.append(_const(_resolve(constants, field, "constant")))
        elif op == GamsOpCode.nlPushZero:
            stack.append(_const(0.0))
        elif op == GamsOpCode.nlAdd:
            b = pop()
            stack.append(pop() + b)
        elif op == GamsOpCode.nlAddV:
            stack.append(pop() + _resolve(variables, field, "variable"))
        elif op == GamsOpCode.nlAddI:
            stack.append(pop() + _const(_resolve(constants, field, "constant")))
        elif op == GamsOpCode.nlSub:
            b = pop()
            stack.append(pop() - b)
        elif op == GamsOpCode.nlSubV:
            stack.append(pop() - _resolve(variables, field, "variable"))
        elif op == GamsOpCode.nlSubI:
            stack.append(pop() - _const(_resolve(constants, field, "constant")))
        elif op == GamsOpCode.nlMul:
            b = pop()
            stack.append(pop() * b)
        elif op == GamsOpCode.nlMulV:
            stack.append(pop() * _resolve(variables, field, "variable"))
        elif op == GamsOpCode.nlMulI:
            stack.append(pop() * _const(_resolve(constants, field, "constant")))
        elif op == GamsOpCode.nlMulIAdd:
            # t2 + t1 * const(field)
            term = pop() * _const(_resolve(constants, field, "constant"))
            stack.append(pop() + term)
        elif op == GamsOpCode.nlDiv:
            b = pop()
            stack.append(pop() / b)
        elif op == GamsOpCode.nlDivV:
            stack.append(pop() / _resolve(variables, field, "variable"))
        elif op == GamsOpCode.nlDivI:
            stack.append(pop() / _const(_resolve(constants, field, "constant")))
        elif op == GamsOpCode.nlUMin:
            stack.append(-pop())
        elif op == GamsOpCode.nlUMinV:
            stack.append(-_resolve(variables, field, "variable"))
        elif op == GamsOpCode.nlCallArg1:
            stack.append(_apply_func(field, [pop()]))
        elif op == GamsOpCode.nlCallArg2:
            b = pop()
            stack.append(_apply_func(field, [pop(), b]))
        elif op == GamsOpCode.nlFuncArgN:
            # number of arguments for the following nlCallArgN
            pending_argn = field
            i += 1
            if i >= n or GamsOpCode(opcodes[i]) != GamsOpCode.nlCallArgN:
                raise GamsTranslationError("nlFuncArgN not followed by nlCallArgN")
            args = [pop() for _ in range(pending_argn)][::-1]
            stack.append(_apply_func(fields[i], args))
        elif op == GamsOpCode.nlCallArgN:
            raise GamsTranslationError("nlCallArgN without preceding nlFuncArgN")
        elif op in (GamsOpCode.nlStore, GamsOpCode.nlEnd):
            result = pop() if stack else result
            if op == GamsOpCode.nlEnd:
                break
        else:
            raise GamsTranslationError(f"unsupported opcode {op.name}")
        i += 1

    if result is None:
        if len(stack) == 1:
            return stack[0]
        raise GamsTranslationError("instruction list produced no result")
    return result


def _apply_func(func_code: int, args: list[Expression]) -> Expression:
    name = FUNC_NAME.get(func_code)
    if name is None:
        raise GamsTranslationError(f"unknown function code {func_code}")
    if len(args) == 1 and name in _UNARY:
        return _UNARY[name](args[0])
    if len(args) == 2 and name in _BINARY:
        return _BINARY[name](args[0], args[1])
    if name in _BINARY and len(args) >= 2:
        # n-ary min/max etc. folded pairwise
        expr = args[0]
        for nxt in args[1:]:
            expr = _BINARY[name](expr, nxt)
        return expr
    raise GamsTranslationError(
        f"GAMS function '{name}' (code {func_code}, {len(args)} args) "
        "is not supported by the discopt GAMS link"
    )
