"""Reduced-space McCormick relaxation + Kelley-cut LP bound, on the MCBox type.

MAiNGO-parity plan P0.3: this reimplements the #572 reduced-space evaluator on top of
:mod:`discopt._jax.mcbox`. Where the original walked the model AST into hand-rolled
``(cv, cc)`` closures (affine / QP scope, ``jax.grad`` subgradients), this interprets
the AST into :class:`~discopt._jax.mcbox.MCBox` operations — so the relaxation and its
subgradients propagate by rule, and the scope widens to everything MCBox supports
(general products, transcendental composition), still sound-or-refuse.

Public API is unchanged from #572: :class:`UnsupportedRelaxation`,
:func:`build_reduced_relaxation`, :func:`reduced_mccormick_lp_bound`,
:class:`ReducedRelaxation`, :class:`ReducedBound`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from discopt._jax import mcbox as _mcbox
from discopt._jax.mcbox import UnsupportedMcboxOp, mcbox_leaves
from discopt._jax.relaxation_compiler import _resolve_scalar_var_offset
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    CustomCall,
    FunctionCall,
    ObjectiveSense,
    Parameter,
    UnaryOp,
    Variable,
)


class UnsupportedRelaxation(Exception):
    """Model is outside the sound reduced-space scope — caller falls back (α-BB/lifted)."""


def _require_finite_box(lb, ub) -> None:
    """McCormick envelopes require a FINITE box: over an unbounded variable the
    product/composition envelopes degrade to non-information ``(-inf, +inf)`` and the
    linearized cuts carry NaN/inf, which can corrupt (or crash) the LP and, worse,
    let a node be fathomed on a bogus bound. Refuse rather than approximate — the
    caller falls back to the lifted path (which handles unbounded columns via its own
    guards). This is the sound-or-refuse contract."""
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        raise UnsupportedRelaxation("reduced-space McCormick requires a finite box")


def _scalar_const(expr) -> float:
    v = np.asarray(expr.value, dtype=float)
    if v.size != 1:
        raise UnsupportedRelaxation("non-scalar constant/parameter")
    return float(v.reshape(()))


def _is_affine_ast(expr, model) -> bool:
    """True iff ``expr`` is affine in the original variables over the whole box.

    Static, structural check on the AST (constants/params/scalar-vars are affine;
    ``+``/``-``/unary-neg preserve affine-ness; ``*``//``/`` are affine only when one
    side is a constant; ``**`` only for exponent 1 over an affine base). Used to gate
    division: ``a / b`` is SOUND when the denominator ``b`` is affine (then ``1/b`` is a
    univariate reciprocal of an affine expression — the classic McCormick case). For a
    NON-affine denominator the general reciprocal-then-product composition
    (``x/y = x·(1/y)``) is NOT validated — it produced an INVALID ``cc`` subgradient on
    nvs22 con2 (``(A·x6)/((x2·x3)·(sum-of-squares))``), a cut that excluded the true
    feasible optimum by ~1.7e5 -> false-infeasible -> false optimal (task #69). Refuse
    rather than emit an invalid bound (sound-or-refuse, CLAUDE.md §3)."""
    if isinstance(expr, (Constant, Parameter)):
        return True
    if _resolve_scalar_var_offset(expr, model) is not None:
        return True
    if isinstance(expr, Variable):
        return False
    if isinstance(expr, UnaryOp):
        return expr.op == "neg" and _is_affine_ast(expr.operand, model)
    if isinstance(expr, BinaryOp):
        lo_const = isinstance(expr.left, (Constant, Parameter))
        ro_const = isinstance(expr.right, (Constant, Parameter))
        if expr.op in ("+", "-"):
            return _is_affine_ast(expr.left, model) and _is_affine_ast(expr.right, model)
        if expr.op == "*":
            if lo_const:
                return _is_affine_ast(expr.right, model)
            if ro_const:
                return _is_affine_ast(expr.left, model)
            return False
        if expr.op == "/":
            return ro_const and _is_affine_ast(expr.left, model)
        if expr.op == "**":
            return (
                ro_const and _scalar_const(expr.right) == 1.0 and _is_affine_ast(expr.left, model)
            )
        return False
    return False


def _to_mcbox(expr, leaves, model):
    """Interpret a scalar model expression into an MCBox over the current leaves."""
    if isinstance(expr, (Constant, Parameter)):
        return _mcbox._const(_scalar_const(expr), len(leaves))

    off = _resolve_scalar_var_offset(expr, model)
    if off is not None:
        return leaves[off]
    if isinstance(expr, Variable):
        raise UnsupportedRelaxation("non-scalar variable leaf in scalar expression")

    if isinstance(expr, UnaryOp):
        a = _to_mcbox(expr.operand, leaves, model)
        if expr.op == "neg":
            return -a
        raise UnsupportedRelaxation(f"unary op '{expr.op}'")

    if isinstance(expr, BinaryOp):
        op = expr.op
        if op == "+":
            return _to_mcbox(expr.left, leaves, model) + _to_mcbox(expr.right, leaves, model)
        if op == "-":
            return _to_mcbox(expr.left, leaves, model) - _to_mcbox(expr.right, leaves, model)
        if op == "*":
            return _to_mcbox(expr.left, leaves, model) * _to_mcbox(expr.right, leaves, model)
        if op == "/":
            if isinstance(expr.right, (Constant, Parameter)):
                c = _scalar_const(expr.right)
                if c == 0.0:
                    raise UnsupportedRelaxation("division by zero constant")
                return _to_mcbox(expr.left, leaves, model) * (1.0 / c)
            # Variable division via the sign-definite reciprocal (P1.2). SOUND only
            # for an AFFINE denominator (1/affine is a univariate reciprocal of an
            # affine arg — validated). A NON-affine denominator routes through the
            # general reciprocal-then-bilinear composition whose cc subgradient is
            # NOT validated and was shown INVALID on nvs22 con2 (task #69) -> refuse.
            if not _is_affine_ast(expr.right, model):
                raise UnsupportedRelaxation(
                    "division by a non-affine denominator (unvalidated reduced-space "
                    "reciprocal composition; P1.2 follow-up)"
                )
            return _to_mcbox(expr.left, leaves, model) / _to_mcbox(expr.right, leaves, model)
        if op == "**":
            if not isinstance(expr.right, (Constant, Parameter)):
                raise UnsupportedRelaxation("non-constant exponent")
            p = _scalar_const(expr.right)
            try:
                return _to_mcbox(expr.left, leaves, model) ** p
            except UnsupportedMcboxOp as e:
                raise UnsupportedRelaxation(str(e)) from e
        raise UnsupportedRelaxation(f"binary op '{op}'")

    if isinstance(expr, FunctionCall):
        if len(expr.args) != 1:
            raise UnsupportedRelaxation(f"multi-arg function '{expr.func_name}'")
        a = _to_mcbox(expr.args[0], leaves, model)
        fn = getattr(_mcbox, expr.func_name, None)
        if fn is None or not callable(fn):
            raise UnsupportedRelaxation(f"function '{expr.func_name}'")
        try:
            return fn(a)
        except UnsupportedMcboxOp as e:
            raise UnsupportedRelaxation(str(e)) from e

    if isinstance(expr, CustomCall):
        # P3.1: a hidden-DOF ``CustomCall`` becomes globally relaxable when its opaque
        # jax callable traces through MCBox — the relaxation and its subgradients then
        # fall out of the same rule-based propagation as any other operator, and the
        # model can be certified while branching only on the true degrees of freedom
        # (the CustomCall's internal intermediates never become branching variables).
        #
        # Sound-or-refuse (plan §0.3, §5-P3.1): the trace runs under
        # ``strict_division`` — an opaque body offers no AST for the ``_is_affine_ast``
        # guard, so variable-denominator division (the unvalidated non-affine reciprocal,
        # nvs22) refuses rather than risk an invalid bound. A body written against
        # anything MCBox does not support (raw ``jnp`` intrinsics on an MCBox, an
        # unsupported op) raises here and routes to the caller's fallback (local-NLP /
        # lifted) exactly as an opaque CustomCall does today — never a partial bound.
        arg_boxes = [_to_mcbox(a, leaves, model) for a in expr.args]
        try:
            with _mcbox.strict_division():
                result = expr.fn(*arg_boxes)
        except UnsupportedRelaxation:
            raise
        except (UnsupportedMcboxOp, TypeError, ValueError, AttributeError) as e:
            raise UnsupportedRelaxation(
                f"CustomCall '{expr.name}' is not soundly MCBox-relaxable ({type(e).__name__}: {e})"
            ) from e
        if not isinstance(result, _mcbox.MCBox):
            raise UnsupportedRelaxation(
                f"CustomCall '{expr.name}' did not return a scalar MCBox "
                f"(got {type(result).__name__}); reduce it to a scalar relaxation"
            )
        return result

    raise UnsupportedRelaxation(f"node type {type(expr).__name__}")


@dataclass
class ReducedRelaxation:
    obj_under: Callable  # convex underestimator of the objective (minimize form)
    con_feas: list  # convex h_i(x) with h_i<=0 the relaxed feasibility
    negate: bool
    n: int


def build_reduced_relaxation(model, node_lb, node_ub) -> ReducedRelaxation:
    """Build the sound reduced-space McCormick relaxation (MCBox), or raise.

    Raises :class:`UnsupportedRelaxation` if any part is outside MCBox scope. The
    objective/constraint closures interpret the AST into MCBox at each point (jit-able);
    a concrete trial build at the box midpoint surfaces unsupported structure eagerly.
    """
    lb = np.asarray(node_lb, dtype=float)
    ub = np.asarray(node_ub, dtype=float)
    _require_finite_box(lb, ub)
    n = int(lb.size)
    negate = model._objective.sense == ObjectiveSense.MAXIMIZE
    obj_expr = model._objective.expression

    def obj_under(x):
        z = _to_mcbox(obj_expr, mcbox_leaves(x, lb, ub), model)
        return -z.cc if negate else z.cv

    con_feas = []
    for c in model._constraints:
        if not isinstance(c, Constraint):
            raise UnsupportedRelaxation(f"non-Constraint {type(c).__name__}")
        body, sense = c.body, c.sense

        if sense in ("<=", "=="):
            con_feas.append(lambda x, b=body: _to_mcbox(b, mcbox_leaves(x, lb, ub), model).cv)
        if sense in (">=", "=="):
            con_feas.append(lambda x, b=body: -_to_mcbox(b, mcbox_leaves(x, lb, ub), model).cc)

    # eager buildability check (concrete midpoint) so the caller can fall back
    mid = jnp.asarray(0.5 * (lb + ub))
    _ = float(obj_under(mid))
    for h in con_feas:
        _ = float(h(mid))
    return ReducedRelaxation(obj_under=obj_under, con_feas=con_feas, negate=negate, n=n)


def _build_jit_evaluator(model, lb, ub):
    """A single jitted fn ``x -> (u, gu, h, gh)`` for the reduced relaxation, using
    MCBox's **explicit** rule-based subgradients (not ``jax.grad`` over the whole
    construction — ~4 orders of magnitude faster, P2.1). ``u``/``gu`` are the objective
    underestimator (minimize form) value/subgradient; ``h`` (m,) / ``gh`` (m, n) are the
    per-constraint feasibility values/subgradients (``h_i <= 0`` relaxed). Raises
    :class:`UnsupportedRelaxation` on first call (compile) if out of MCBox scope."""
    _require_finite_box(lb, ub)
    lbj, ubj = jnp.asarray(lb), jnp.asarray(ub)
    n = int(lb.size)
    negate = model._objective.sense == ObjectiveSense.MAXIMIZE
    obj_expr = model._objective.expression
    cons = []
    for c in model._constraints:
        if not isinstance(c, Constraint):
            raise UnsupportedRelaxation(f"non-Constraint {type(c).__name__}")
        cons.append((c.body, c.sense))

    def ev(x):
        leaves = mcbox_leaves(x, lbj, ubj)
        zo = _to_mcbox(obj_expr, leaves, model)
        u = -zo.cc if negate else zo.cv
        gu = -zo.sub_cc if negate else zo.sub_cv
        hs, ghs = [], []
        for body, sense in cons:
            zb = _to_mcbox(body, leaves, model)
            if sense in ("<=", "=="):
                hs.append(zb.cv)
                ghs.append(zb.sub_cv)
            if sense in (">=", "=="):
                hs.append(-zb.cc)
                ghs.append(-zb.sub_cc)
        h = jnp.stack(hs) if hs else jnp.zeros(0)
        gh = jnp.stack(ghs) if ghs else jnp.zeros((0, n))
        return u, gu, h, gh

    return jax.jit(ev), negate, n


@dataclass
class ReducedBound:
    bound: float
    status: str  # optimal | infeasible | unbounded | unsupported
    rounds: int
    history: list


def reduced_mccormick_lp_bound(
    model, node_lb, node_ub, *, max_rounds: int = 40, tol: float = 1e-7
) -> ReducedBound:
    """MAiNGO-style reduced-space lower bound: Kelley cutting planes on the MCBox
    relaxation, solved as an LP over the original variables (no auxiliary columns).

    The per-round value+subgradient evaluation is a single jitted MCBox pass with
    explicit subgradients (P2.1). The inner LP is solved on discopt's **in-house Rust
    simplex** (P2.2), reusing one LP whose columns are fixed (``n_orig + 1``) and whose
    rows only grow each Kelley round — the dual-simplex warm start the simplex already
    exposes. These ``n_orig``-column bases are exactly the ones that do NOT trigger the
    dense-lifted FT refactor storm (#557). Set ``DISCOPT_REDUCED_LP_BACKEND=scipy`` to
    use scipy/HiGHS instead (debug/fallback); the simplex path also falls back to scipy
    automatically if the Rust binding is unavailable.
    """
    import os

    import scipy.optimize as sopt

    lb = np.asarray(node_lb, dtype=float)
    ub = np.asarray(node_ub, dtype=float)
    try:
        evj, negate, n = _build_jit_evaluator(model, lb, ub)
        evj(jnp.asarray(0.5 * (lb + ub)))  # compile + eager scope check
    except UnsupportedRelaxation:
        return ReducedBound(bound=-np.inf, status="unsupported", rounds=0, history=[])

    c = np.zeros(n + 1)
    c[n] = 1.0  # minimize the epigraph variable t
    # t (epigraph) needs finite bounds for the in-house simplex; the Kelley cuts provide
    # the real lower bound, so a wide inert box suffices (a None here yields NaN there).
    T_BIG = 1e15
    bounds = [(float(lb[i]), float(ub[i])) for i in range(n)] + [(-T_BIG, T_BIG)]
    backend = "scipy" if os.environ.get("DISCOPT_REDUCED_LP_BACKEND") == "scipy" else "simplex"
    milp = None
    A_all: list = []
    rhs_all: list = []
    xk = 0.5 * (lb + ub)
    prev, bound = -np.inf, -np.inf
    history: list = []

    r = 0
    for r in range(1, max_rounds + 1):
        u_j, gu_j, h_j, gh_j = evj(jnp.asarray(xk))
        u0 = float(u_j)
        new_rows: list = []
        new_b: list = []
        g = np.asarray(gu_j, dtype=float)
        if np.all(np.isfinite(g)) and np.isfinite(u0):
            row = np.zeros(n + 1)
            row[:n] = g
            row[n] = -1.0  # t >= u0 + g.(x - xk)
            new_rows.append(row)
            new_b.append(float(g @ xk - u0))
        for h0, gg in zip(np.asarray(h_j, dtype=float), np.asarray(gh_j, dtype=float)):
            if np.all(np.isfinite(gg)) and np.isfinite(h0):
                row = np.zeros(n + 1)
                row[:n] = gg
                new_rows.append(row)
                new_b.append(float(gg @ xk - h0))
        A_all.extend(new_rows)
        rhs_all.extend(new_b)
        if not A_all:  # nothing finite to cut on -> no valid bound
            break

        status, bound_r, xr = "error", None, None
        if backend == "simplex":
            try:
                import scipy.sparse as sp

                from discopt._jax.mccormick_lp import _append_relax_rows
                from discopt._jax.milp_relaxation import MilpRelaxationModel

                R = sp.csr_matrix(np.asarray(new_rows))
                b = np.asarray(new_b, dtype=float)
                if milp is None:
                    milp = MilpRelaxationModel(c, R, b, bounds)
                else:
                    _append_relax_rows(milp, R, b)
                mres = milp.solve(backend="simplex", time_limit=30.0)
                st = str(mres.status).lower()
                if "infeasible" in st:
                    status = "infeasible"
                elif "unbound" in st:
                    status = "unbounded"
                elif mres.objective is not None or mres.bound is not None:
                    bv = mres.objective if mres.objective is not None else mres.bound
                    if bv is None or not np.isfinite(bv) or mres.x is None:
                        raise RuntimeError("simplex returned a non-finite bound")
                    status = "optimal"
                    bound_r = float(bv)
                    xr = np.asarray(mres.x, dtype=float)
                else:
                    raise RuntimeError(f"simplex status {st}")
            except Exception:
                backend = "scipy"  # fall back for this and subsequent rounds
                milp = None
        if backend == "scipy":
            res = sopt.linprog(
                c, A_ub=np.asarray(A_all), b_ub=np.asarray(rhs_all), bounds=bounds, method="highs"
            )
            if res.success:
                status, bound_r, xr = "optimal", float(res.fun), np.asarray(res.x, dtype=float)
            else:
                msg = (res.message or "").lower()
                if "infeasible" in msg:
                    status = "infeasible"
                elif "unbounded" in msg:
                    status = "unbounded"
                else:
                    status = "error"

        if status == "infeasible":
            # SOUND-OR-REFUSE (task #69, P2.3 root cause #2): a reduced-space
            # ``infeasible`` fathoms the node, so it MUST be a rigorous proof — not
            # a numerical artifact. The Kelley LP (box + epigraph + valid cuts) can
            # only be genuinely infeasible if the accumulated valid constraint cuts
            # contradict; but the in-house simplex can spuriously report infeasible
            # on a badly-scaled basis. Cross-check with scipy/HiGHS before trusting
            # the fathom: if HiGHS finds the same LP feasible (or is itself
            # inconclusive), DO NOT fathom — return no bound and let the caller keep
            # any other (lifted) bound source. Only a two-solver agreement fathoms.
            if backend == "simplex":
                try:
                    _vres = sopt.linprog(
                        c,
                        A_ub=np.asarray(A_all),
                        b_ub=np.asarray(rhs_all),
                        bounds=bounds,
                        method="highs",
                    )
                except Exception:
                    _vres = None
                _confirmed = _vres is not None and (
                    "infeasible" in (getattr(_vres, "message", "") or "").lower()
                    or (not _vres.success and getattr(_vres, "status", None) == 2)
                )
                if not _confirmed:
                    # Unconfirmed infeasible => suspected simplex mis-solve. Refuse to
                    # fathom; report no usable reduced bound for this node (the caller
                    # falls back to any other bound source; the node is NOT pruned).
                    return ReducedBound(
                        bound=-np.inf, status="unsupported", rounds=r, history=history
                    )
            return ReducedBound(bound=np.inf, status="infeasible", rounds=r, history=history)
        if status == "unbounded":
            return ReducedBound(bound=-np.inf, status="unbounded", rounds=r, history=history)
        if status != "optimal" or bound_r is None or xr is None:
            break
        bound = float(bound_r)
        history.append(bound)
        xk = xr[:n]
        if bound - prev < tol * (abs(bound) + 1.0):
            break
        prev = bound

    dual = -bound if negate else bound
    return ReducedBound(bound=dual, status="optimal", rounds=r, history=history)
