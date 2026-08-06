"""Compile a discopt expression DAG into a POUNCE ``NlExpr`` tape expression.

This is the shared prerequisite for taking JAX off the solve path (issue #75).
Both remaining JAX jobs need the same thing — the value and gradient of a scalar
expression at a point — and POUNCE's Rust AD tape provides exactly that:

* **separation tangents** (``_jax/uniform_relax.py``) need ``g(x0)`` and
  ``grad g(x0)`` for the Kelley cutting-plane loop;
* **NLP subsolve derivatives** (``_jax/nlp_evaluator.py``) need ``f``, ``grad f``,
  ``g``, ``J``, and the Lagrangian Hessian.

Previously those were headed for two *different* replacement backends. They no
longer need to be: ``pounce.NlExpr`` covers discopt's *scalar* DAG operators — 20
natively and 9 by the exact rewrites below — where the in-tree interval-AD engine
covers 6, and it agrees with analytic truth exactly on expressions the interval
engine cannot evaluate at all (``tanh``, ``erf``).

The **array reductions** are the exception and are refused, not approximated:
``prod`` (``jnp.prod`` of one array argument), ``norm1``/``norm2``/``norminf``,
``SumExpression``, ``IndexExpression`` and ``MatMulExpression``. A tape node is a
scalar; there is no array to reduce. An earlier revision lowered ``prod`` as a
variadic ``*`` chain, which silently computed a different function.

*Coverage claims here must be checked against operators, not instances.* ``.nl``
has no opcode for ``sigmoid``/``softplus``/``entropy``/``centropy``/``signpower``,
so a MINLPLib corpus sweep — measured across 316 instances — exercises exactly six
of these (``log``, ``sqrt``, ``exp``, ``abs``, ``sin``, ``cos``) and can never
reach the rest. Three defects lived in the unreached rewrites: ``entropy`` had an
inverted sign, ``_sign`` passed ``compare``'s arguments in the wrong order, and
``prod`` was the wrong function entirely. See
``python/tests/test_75_nl_expr_compiler.py`` for the per-operator differential
that catches this class.

Why not ``.nl`` as the intermediate? Because it is lossy in the wrong direction:
discopt's ``.nl`` writer refuses ``atan2``, ``min``, ``max``, ``erf`` and ``sign``
(``export/nl.py``), several of which the tape *does* differentiate. Building the
tape expression directly keeps them.

Deliberately NOT under ``_jax/``: nothing here touches JAX.
"""

from __future__ import annotations

import math
from typing import Any

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)


class UnsupportedForTape(Exception):
    """``expr`` contains a node the tape cannot represent.

    Raised rather than returned so a caller cannot mistake "no tape" for "a tape
    that evaluates to nothing". Callers are expected to catch this and fall back
    to their existing path; the only expected trigger is :class:`CustomCall`,
    whose body is an opaque JAX callable by contract (``modeling/core.py``) and
    therefore has no tape equivalent.
    """


def _scalar(value: Any, what: str) -> float:
    """``float(value)``, but raising :class:`UnsupportedForTape` on an array.

    ``float()`` on a multi-element array raises ``TypeError``, which is NOT
    caught by :func:`try_compile` and so escapes the fallback and crashes the
    caller. Array-valued constants and parameters are a representability limit
    of a scalar tape, not a bug, and must report as one.
    """
    import numpy as _np

    arr = _np.asarray(value)
    if arr.size != 1:
        raise UnsupportedForTape(f"array-valued {what} (size {arr.size}); the tape path is scalar")
    return float(arr.reshape(()))


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Start of ``var`` in the flat ``x`` vector.

    Delegates to the model's memoized prefix-sum table, so per-leaf resolution
    during the DAG walk is O(1). A linear scan here would be O(n) per leaf and
    O(n^2) over the build — the regression issue #654 fixed, and the reason
    ``dag_compiler`` routes through the same table.
    """
    return model._flat_var_offset(var)


def compile_to_nl_expr(expr: Expression, model: Model) -> Any:
    """Compile ``expr`` to a ``pounce.NlExpr`` over the model's flat variable layout.

    The returned object supports ``.eval(x)`` and ``.gradient(x)`` for a flat ``x``
    ordered exactly like the JAX path's ``x_flat``, so the two are directly
    comparable point-for-point.

    Raises:
        UnsupportedForTape: if the DAG contains a node with no tape equivalent.
    """
    import pounce

    E = pounce.NlExpr
    # id(node) -> NlExpr. The DAG is a DAG, not a tree: a node reachable by k
    # references must be built once, or a linear DAG lowers in time exponential in
    # its sharing depth (the same trap `dag_compiler` documents as issue #383).
    memo: dict[int, Any] = {}
    return _lower(expr, model, E, memo)


def _lower(expr: Expression, model: Model, E: Any, memo: dict[int, Any]) -> Any:
    key = id(expr)
    hit = memo.get(key)
    if hit is not None:
        return hit
    built = _lower_uncached(expr, model, E, memo)
    memo[key] = built
    return built


def _lower_uncached(expr: Expression, model: Model, E: Any, memo: dict[int, Any]) -> Any:
    def rec(child: Expression) -> Any:
        return _lower(child, model, E, memo)

    if isinstance(expr, Constant):
        return E.const_(_scalar(expr.value, "Constant"))

    if isinstance(expr, Variable):
        if expr.size != 1:
            raise UnsupportedForTape(
                f"array variable {expr.name!r} (size {expr.size}); the tape path is scalar"
            )
        return E.var(_compute_var_offset(expr, model))

    if isinstance(expr, Parameter):
        # Parameters are constants at compile time, matching the legacy
        # `compile_expression` behaviour (the params-as-runtime-args variant has
        # no tape analogue -- a tape is built for fixed structure). Because the
        # value is BAKED IN, a caller holding a tape across a `Parameter.value`
        # re-bind gets stale derivatives; `_tape_nlp_evaluator` rebuilds on
        # change, and `evaluator_fingerprint` deliberately does NOT cover this.
        return E.const_(_scalar(expr.value, "Parameter"))

    if isinstance(expr, BinaryOp):
        left, right, op = rec(expr.left), rec(expr.right), expr.op
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "**":
            return left**right
        raise UnsupportedForTape(f"binary operator {op!r}")

    if isinstance(expr, UnaryOp):
        operand, op = rec(expr.operand), expr.op
        if op == "neg":
            return -operand
        if op == "abs":
            return _abs(E, operand)
        raise UnsupportedForTape(f"unary operator {op!r}")

    if isinstance(expr, FunctionCall):
        return _lower_function(expr, E, [rec(a) for a in expr.args])

    if isinstance(expr, SumExpression):
        # An ARRAY reduction (``jnp.sum(operand, axis=...)``), not a list of scalar
        # terms -- ``.operand``/``.axis``, per ``dag_compiler``. The tape is scalar,
        # so refuse rather than guess a flattening.
        raise UnsupportedForTape("SumExpression (array reduction) has no scalar tape lowering")

    if isinstance(expr, SumOverExpression):
        terms = [rec(t) for t in expr.terms]
        if not terms:
            return E.const_(0.0)
        acc = terms[0]
        for t in terms[1:]:
            acc = acc + t
        return acc

    if isinstance(expr, IndexExpression):
        raise UnsupportedForTape("IndexExpression (array indexing) is not yet lowered")

    if isinstance(expr, MatMulExpression):
        raise UnsupportedForTape("MatMulExpression is not yet lowered")

    if isinstance(expr, CustomCall):
        # By contract (`modeling/core.py`) a dm.custom body is an opaque callable
        # that must be JAX-differentiable. There is no tape equivalent, and there
        # should not be a silent one -- relaxing an opaque callable needs AD
        # *through* it, which is exactly what the tape cannot do.
        raise UnsupportedForTape("CustomCall (dm.custom) has no tape equivalent")

    raise UnsupportedForTape(f"expression node {type(expr).__name__}")


def _abs(E: Any, a: Any) -> Any:
    """``abs(a) = max(a, -a)``. The tape has ``max``; ASL/Ipopt treat the kink as a
    non-smooth event and route derivatives through the active branch."""
    return E.max(a, -a)


#: Argument floor for ``entropy``/``centropy``, matching `_jax/dag_compiler.py`
#: (``jnp.maximum(x, 1e-300)``). It regularizes the ``x -> 0+`` limit: the true
#: derivative of ``x*log(x)`` at 0 is ``-inf``, and both backends deliberately
#: report a large finite number instead so a solver evaluating a box pinned at
#: ``[0, 0]`` does not propagate a non-finite into a bound.
_XLOG_FLOOR = 1e-300

#: ``log(_XLOG_FLOOR)``, folded in Python rather than emitted as ``log(const)``.
#:
#: Not cosmetic. Below the floor the clamped branch is ``x * log(floor/y)``, and
#: routing that through a tape ``Log`` node makes its second derivative
#: ``-1/q**2`` with ``q = floor/y = 1e-300``, i.e. ``-1e600`` -- inf. Measured at
#: ``(x, y) = (1e-320, 1.0)``: the ``log(const/y)`` form returns
#: ``[[0, -inf], [-inf, nan]]`` where the truth ``[[0, -1], [-1, 1e-320]]`` is
#: entirely representable. Emitting the constant and subtracting ``log(y)``
#: returns exactly that truth. Same structural trap as the fused opcodes solve --
#: a composite in range built from an intermediate that is not.
_LOG_XLOG_FLOOR = math.log(_XLOG_FLOOR)


def _clamped_xlog(E: Any, x: Any, fused: Any, log_of_floor: Any) -> Any:
    """``entropy``/``centropy`` on the fused opcode, keeping the ``_XLOG_FLOOR`` clamp.

    The two lowerings differ only in which fused opcode carries the unclamped
    branch and what the log factor is once the argument is clamped, so both go
    through here. ``fused`` receives the (clamped) first argument; the authority
    for ``x < floor`` is ``x * log_of_floor`` -- ``x`` times a constant-in-``x``
    factor, NOT the fused op evaluated at the floor.

    That distinction is why this is a ``select`` and not just ``fused(max(x,
    floor))``. Below the floor the authority stays LINEAR in ``x`` with slope
    ``log(1e-300) = -690.78``, while ``xlogx(max(x, floor))`` would be constant
    with slope ``0``. The regularized-but-large derivative is the whole point of
    the clamp -- ``factorable_reform._try_entropy`` refuses only ``lo < 0.0``, so
    a box pinned at ``[0, 0]`` is admitted by design -- and reporting ``0`` there
    would tell the NLP that moving ``x`` off zero does not change the entropy,
    which is false.

    ``max(x, floor)`` still guards the fused argument. The tape's forward sweep
    evaluates EVERY slot, so at ``x < 0`` the inactive branch would otherwise
    compute ``xlogx(-1) = nan``; pounce's
    ``cond_does_not_leak_a_non_finite_from_its_inactive_branch`` measures that all
    three sweeps (``gradient_seed``, ``hessian_accumulate``,
    ``hessian_directional``) route around it, and the ``max`` means this lowering
    does not have to rely on that invariant to stay finite by construction.

    Cost: the ``select`` makes the node opaque to pounce's FBBT translator, which
    emits ``Opaque`` for ``Expr::Cond``. Nothing is actually lost -- ``E.max``
    lowers to ``Expr::MaxList``, which was ALREADY ``Opaque``, so the previous
    ``x * log(max(x, floor))`` had an opaque interior too and could not tighten
    through the log either way.
    """
    floor = E.const_(_XLOG_FLOOR)
    return E.select(E.compare("<", x, floor), x * log_of_floor, fused(E.max(x, floor)))


def _require_fused_xlog(E: Any, name: str) -> None:
    """Refuse ``entropy``/``centropy`` on a POUNCE predating the fused opcodes.

    ``pounce_usable()`` gates on ``NlExpr`` existing at all (pounce #470); these
    two opcodes landed later (pounce #489), so a build in between imports fine and
    then raises ``AttributeError`` deep inside a compile. Raise the
    ``UnsupportedForTape`` the caller already handles, which degrades this one
    model to the JAX evaluator rather than failing the solve.

    Deliberately NOT a fallback to the old ``x * log(x)`` lowering: that lowering
    is the defect this replaces, and quietly routing back to it would turn a stale
    extension into wrong derivatives instead of a visible capability miss.
    """
    missing = [n for n in ("xlogx", "centropy") if not hasattr(E, n)]
    if missing:
        raise UnsupportedForTape(
            f"{name} needs the fused NlExpr.{'/'.join(missing)} opcode(s) from "
            "pounce #489; rebuild the pounce extension to use the tape here"
        )


#: Below this ``|a|``, ``log1p`` uses a truncated series rather than Kahan's
#: compensated form. Both the term count and the crossover are set by the SECOND
#: derivative, which is the binding constraint here -- the value and gradient are
#: at one ulp across this whole region either way.
#:
#: The series runs to ``a**5/5``, not ``a**4/4``. Truncating at ``a**4`` leaves a
#: second derivative of ``-1 + 2a - 3a**2`` against a true expansion
#: ``-1 + 2a - 3a**2 + 4a**3 - ...``, an error of ``~4a**3``; the extra term moves
#: the leading error to ``~5a**4``. Measured at a=5e-5: 5.00e-13 before, 7.61e-19
#: after.
#:
#: The crossover is 5e-4 by MEASUREMENT, not algebra. Kahan's second-derivative
#: error decays like ``eps/a`` (its ``u - 1`` differs from ``a`` by a rounding gap
#: that double differentiation amplifies) while the series' grows like ``5a**4``,
#: so the worst case over the line is minimized where they cross. Scanning
#: candidate crossovers over a 69-point grid spanning 1e-17..1, worst-case
#: relative Hessian error was:
#:
#:     1e-4 -> 1.092e-11   2e-4 -> 4.552e-13   5e-4 -> 2.281e-13
#:     1e-3 -> 5.001e-13   2e-3 -> 5.007e-11
#:
#: with value/gradient flat at 1.336e-16 / 2.979e-16 for every candidate up to
#: 1e-3 and degrading at 2e-3 (3.002e-15 / 1.773e-14) -- so 5e-4 is the minimax
#: point and the scan would have shown a bad trade if one existed.
_LOG1P_TAYLOR = 5e-4


def _log1p(E: Any, a: Any) -> Any:
    """``log1p(a)`` accurately for small ``|a|``; the tape has no ``log1p`` opcode.

    ``log(1 + a)`` loses every significant digit once ``a`` falls below the
    rounding gap of 1.0: measured, ``a = 1e-17`` returns exactly ``0.0`` where
    ``jnp.log1p`` returns ``1e-17``. This is Kahan's compensated form — with
    ``u = 1 + a`` computed in floating point, ``log(u) / (u - 1)`` is the
    correction factor for the representation error in ``u``, and it is accurate
    precisely where the naive form is not.

    ``select`` is a data-flow node: BOTH arms are evaluated and both contribute
    partials to the reverse sweep, so it is not enough for the selected arm to be
    finite — each arm must be fed an argument inside its own safe range. Two
    guards, both load-bearing, both from a measurement:

    * ``d_safe`` is never zero, so the correction never forms ``0/0``.
    * the Kahan arm sees ``a_small``, clamped to ``[-0.5, 0.5]``. Applying the
      correction at large ``a`` overflows in the *derivative*, not the value: the
      quotient rule squares ``d``, and at ``a = 1e300`` that term is ``1e600 ->
      inf``, which silently drops the second term and returned a gradient of
      6.918e-298 against a true ``1e-300``. Bounding the arm's argument bounds
      ``d <= 1.5``, so ``d**2`` cannot overflow.

    Outside the small range the naive ``log(1 + a)`` is already accurate, and it
    is likewise fed a constant when it is not the selected arm.
    """
    one = E.const_(1.0)
    small = E.compare("<", _abs(E, a), E.const_(0.5))

    a_small = E.select(small, a, E.const_(0.0))
    u = one + a_small
    at_one = E.compare("==", u, one)
    d_safe = E.select(at_one, one, u - one)
    # `u == 1` means `a` vanished entirely in the addition, so `log(u)` is 0 and
    # the correction would return 0 rather than `a`. That arm returns `a_small`
    # unchanged -- which IS log1p to full precision once `a` is below the
    # rounding gap of 1.0.
    kahan = E.select(at_one, a_small, a_small * E.log(u) / d_safe)

    naive = E.log(E.select(small, E.const_(2.0), one + a))
    kahan_or_naive = E.select(small, kahan, naive)

    # Below `_LOG1P_TAYLOR`, a truncated series -- NOT the Kahan form. Kahan
    # corrects the VALUE; it does not correct the derivatives AD then takes of
    # it, because `u - 1` differs from `a` in floating point and double
    # differentiation amplifies that gap. Measured against the analytic
    # `-1/(1+a)**2`, the arms above give a second derivative of exactly 0.0 at
    # a=1e-17 (the `at_one` arm is LINEAR, so it has no curvature at all) and
    # -1.0039 at a=1e-13, against a true -1. The series is accurate in all three
    # orders at once: its second derivative is `-1 + 2a - 3a**2 + 4a**3`, which
    # is the leading expansion of the true `-1/(1+a)**2`. See `_LOG1P_TAYLOR` for
    # why it runs to `a**5/5` -- the SECOND derivative sets the term count, not
    # the value.
    #
    # `a_tiny` is clamped for the usual reason -- both arms of a `select`
    # evaluate, and `a**4` at a=1e300 is `inf`, which would poison the sweep.
    a_tiny = E.select(E.compare("<", _abs(E, a), E.const_(_LOG1P_TAYLOR)), a, E.const_(0.0))
    taylor = a_tiny * (
        one
        - a_tiny
        * (
            E.const_(1.0 / 2.0)
            - a_tiny
            * (E.const_(1.0 / 3.0) - a_tiny * (E.const_(1.0 / 4.0) - a_tiny / E.const_(5.0)))
        )
    )
    return E.select(E.compare("<", _abs(E, a), E.const_(_LOG1P_TAYLOR)), taylor, kahan_or_naive)


def _sign(E: Any, a: Any) -> Any:
    """``sign(a)`` via nested selects; zero maps to 0.0 as in ``jnp.sign``.

    ``compare`` takes the operator FIRST (``compare(op, lhs, rhs)``). Passing it
    last raises ``TypeError``, not ``UnsupportedForTape``, so it escapes
    ``try_compile``'s fallback and crashes the caller.
    """
    zero = E.const_(0.0)
    pos = E.compare(">", a, zero)
    neg = E.compare("<", a, zero)
    return E.select(pos, E.const_(1.0), E.select(neg, E.const_(-1.0), zero))


def _lower_function(expr: FunctionCall, E: Any, args: list) -> Any:
    """Lower a ``FunctionCall``. 20 operators are native to the tape; the rest are
    exact rewrites into operators that are."""
    name = expr.func_name

    def arg0() -> Any:
        """First argument, after ``_require`` has proven it exists."""
        return args[0]

    native_unary = {
        "exp",
        "log",
        "log10",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "erf",
    }
    if name in native_unary:
        _require(args, 1, name)
        return getattr(E, name)(arg0())

    if name in ("min", "max", "atan2"):
        _require(args, 2, name)
        return getattr(E, name)(args[0], args[1])

    # --- exact rewrites (no tape opcode, but expressible) ---------------------
    if name == "abs":
        _require(args, 1, name)
        return _abs(E, arg0())
    if name == "sign":
        _require(args, 1, name)
        return _sign(E, arg0())
    if name == "log1p":
        _require(args, 1, name)
        return _log1p(E, arg0())
    if name == "log2":
        _require(args, 1, name)
        return E.log(arg0()) * E.const_(1.0 / math.log(2.0))
    if name == "sigmoid":
        # Branch-stable form. `t = exp(-|a|)` is in (0, 1] for EVERY input, so
        # neither arm can overflow, and each arm is the algebraically exact
        # sigmoid on its own side:
        #
        #     a >= 0:  1/(1+exp(-a))  = 1/(1+t)
        #     a <  0:  exp(a)/(1+exp(a)) = t/(1+t)
        #
        # The `|a|` kink at 0 cancels rather than leaking into the derivative:
        # each arm pairs with the matching sign of `dt/da`, both arms give 1/2
        # and 1/4 there, so the result is smooth across the switch.
        #
        # The naive `1/(1+exp(-a))` this replaces was NOT safe, though its first
        # two orders looked it. A prior version of this comment claimed "measured
        # over -1e300..1e300: zero non-finite values, zero non-finite gradients"
        # and concluded "do not harden this one". That measurement was real but
        # only covered orders 0 and 1. At order 2 the left tail is where it dies:
        # `exp(745)` overflows, the quotient rule then forms `inf/inf`, and the
        # LAGRANGIAN HESSIAN -- which the NLP subsolve actually consumes --
        # came back `nan` at a=-745 and SIGN-FLIPPED at a=-300 (-5.148e-131
        # against a true +5.148e-131). `dm.sigmoid` is the SIGMOID activation in
        # all four `nn/formulations/`, so that is a live user path.
        #
        # The upper-tail accuracy that motivated keeping the naive form IS
        # preserved: at a=40 this still gives 4.248e-18 for the derivative where
        # `jax.nn.sigmoid` underflows to 0.0, because arm 1 is `1/(1+t)` with the
        # same `t`. Rewriting instead to `0.5*(1 + tanh(a/2))` would still be
        # wrong -- `tanh(-20)` rounds to exactly -1.0, giving 0.0 for a value
        # whose true magnitude is 4e-18.
        _require(args, 1, name)
        a = arg0()
        one = E.const_(1.0)
        t = E.exp(-_abs(E, a))
        return E.select(E.compare(">=", a, E.const_(0.0)), one / (one + t), t / (one + t))
    if name == "softplus":
        # `log(1 + exp(a))` OVERFLOWS: `exp(710)` is `inf`, so softplus(745)
        # returned `inf` where the true value is 745 (measured: 2 non-finite
        # values over the sampled domain). The shifted form never exponentiates
        # a positive argument, so `exp` is confined to (0, 1] and cannot
        # overflow. It also recovers the lower tail exactly -- at `a = -300`
        # this returns 5.148e-131, matching `jnp.logaddexp`, where the naive
        # form collapsed to 0.0.
        _require(args, 1, name)
        a = arg0()
        return E.max(a, E.const_(0.0)) + _log1p(E, E.exp(-_abs(E, a)))
    if name == "entropy":
        # x*log(x) -- discopt's DAG semantics, NOT the information-theory
        # convention -x*log(x). The authority is `_jax/dag_compiler.py`, which
        # this must reproduce bit-for-bit: `lambda x: x * jnp.log(jnp.maximum(x,
        # 1e-300))`. Getting the sign from the name instead of the reference
        # cost a silent factor of -1 (reldiff 2.0) that no corpus instance could
        # have caught -- `.nl` has no entropy opcode, so 316 MINLPLib instances
        # exercise this line zero times.
        #
        # The floor is part of the semantics, not a nicety. Without it, `x = 0`
        # gives `0 * log(0) = 0 * -inf = nan` for the value and `-inf` for the
        # derivative, where the authority returns -0.0 and -690.78. That point is
        # REACHABLE: `factorable_reform._try_entropy` refuses only `lo < 0.0`, so
        # a box whose lower bound is exactly 0 is admitted by design. The comment
        # above already named `jnp.maximum(x, 1e-300)` as the authority; the sign
        # was carried across in `08e1e0a1` and the clamp was not.
        #
        # Lowered onto the FUSED `xlogx` opcode (pounce #489), not onto
        # `x * log(x)`. This is a correctness requirement, not a speed one:
        # `(x log x)'' = 1/x` is finite for every positive x down to 1e-308 --
        # at x = 1e-299 it is 1e299, an ordinary number -- but every chain-rule
        # decomposition routes through `log''(x) = -1/x**2 = -1e598`, past
        # `f64::MAX`. A composite in range built from a factor out of range is
        # unreachable however carefully the product and log rules are written;
        # only an opcode that never forms `1/x**2` gets there. Measured: the
        # fused second derivative returns 1e299 where `x * log(x)` returns inf.
        _require(args, 1, name)
        _require_fused_xlog(E, name)
        return _clamped_xlog(E, arg0(), lambda a: E.xlogx(a), E.const_(_LOG_XLOG_FLOOR))
    if name == "centropy":
        # x*log(x/y), matching dag_compiler's GAMS centropy -- including the same
        # floor on the NUMERATOR only (`x * log(max(x, 1e-300) / y)`). Same
        # defect and same fix as entropy above: measured 3 non-finite values and
        # 4 non-finite gradients over the sampled domain before the clamp; 0 and
        # 1 after it.
        #
        # Lowered onto the FUSED `centropy` opcode for entropy's reason
        # (`d2/dx2` is `1/x`, unreachable through `log''`) plus one of its own:
        # `d2/dy2` is `x/y**2`, and the quotient rule materializes `y**2`, which
        # overflows for `|y| > 1.3e154` while `x/y**2` itself stays in range.
        # That was the KNOWN RESIDUAL this comment used to record: the gradient
        # of `centropy(1e300, 1e300)` came back `[1, nan]` because `y**2` is
        # 1e600. The fused rule computes `q = x/y` once and expresses every
        # second-order term as a division by `y`, never by `y**2`.
        #
        # The old note here warned against "fixing" it as
        # `log(max(x, floor)) - log(y)`, because that trades an overflow no real
        # model reaches for catastrophic cancellation at `x ~= y`, which models
        # hit routinely. That warning is now DISCHARGED rather than ignored: the
        # opcode's `ln_ratio` picks among three regimes -- `ln_1p((x-y)/y)` near
        # `x = y` (Sterbenz makes `x - y` exact for `y/2 <= x <= 2y`), `q.ln()`
        # for finite positive q, and the difference form only when the ratio
        # itself leaves f64 range. Measured at `x = 1e10 + 1, y = 1e10`: naive
        # error 8.27e-8, fused error exactly 0.
        _require(args, 2, name)
        _require_fused_xlog(E, name)
        y = args[1]
        return _clamped_xlog(
            E, args[0], lambda a: E.centropy(a, y), E.const_(_LOG_XLOG_FLOOR) - E.log(y)
        )
    if name == "signpower":
        # sign(x) * |x|**p -- the standard smooth-away-from-zero signed power.
        _require(args, 2, name)
        return _sign(E, args[0]) * (_abs(E, args[0]) ** args[1])
    if name == "prod":
        # NOT a variadic multiply. `dag_compiler` compiles prod as
        # `jnp.prod(arg)` -- ONE argument, which is an ARRAY, reduced to a
        # scalar. Lowering it as a `*` chain over `args` silently computed a
        # different function (measured: reldiff 0.90 on value, 1.70 on
        # gradient). The scalar tape has no array to reduce, so refuse, exactly
        # as SumExpression does. `norm1`/`norm2`/`norminf` are array reductions
        # too and fall through to the same refusal below.
        raise UnsupportedForTape(
            "prod is an array reduction (jnp.prod of one array argument); "
            "the scalar tape path has no array to reduce"
        )

    raise UnsupportedForTape(f"function {name!r}")


def _require(args: list, n: int, name: str) -> None:
    if len(args) != n:
        raise UnsupportedForTape(f"{name} expects {n} argument(s), got {len(args)}")


def try_compile(expr: Expression, model: Model) -> Any | None:
    """``compile_to_nl_expr`` returning ``None`` instead of raising.

    For call sites that already have a working fallback and want the tape only
    when it is available. The distinction matters: this returns ``None`` only for
    *representability*, never for a numerical failure, so a caller cannot confuse
    "unsupported operator" with "bad point".
    """
    try:
        return compile_to_nl_expr(expr, model)
    except UnsupportedForTape:
        return None
