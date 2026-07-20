"""Convex LP-OA branch-and-cut kernel — production producer + routing gate (#798).

The native Rust kernel (`discopt._rust.solve_convex_tree_py`) certifies convex
MINLPs of the `rsyn*`/`syn*` family far faster than the NLP-BB path (measured:
all 4 panel instances certified in ~24 s vs NLP-BB timing out uncertified at
120 s each). This module is the analyze-once producer + the **soundness gate**
that decides whether a model may be routed to it.

## Soundness gate (do NOT relax)

The kernel outer-approximates every nonlinear constraint by first-order tangents,
which is a VALID relaxation only for CONVEX `≤` rows. A model is routed here ONLY
when ALL of these hold; otherwise `build_convex_spec` returns ``None`` and the
caller keeps the (always-correct) NLP-BB path:

* the objective is LINEAR (its gradient is constant);
* every nonlinear constraint decomposes into composite-of-affine form
  ``g(x) = a·x + b + Σ_t coeff_t·func_t(p_t·x + q_t)``;
* each such term is CONVEX in the constraint's ``≤`` normal form — a convex
  ``func`` (exp) with ``coeff ≥ 0``, or a concave ``func`` (log/sqrt/log1p) with
  ``coeff ≤ 0`` (a ``≥`` row is negated to ``≤`` first, flipping every sign);
* nonlinear EQUALITY constraints are never routed (a nonlinear equality is not a
  convex feasible set).

Routing an unproven-convex model would give an unsound (too-tight) dual bound and
a possible false ``optimal`` — so the gate is conservative by construction: any
unrecognized function, non-affine argument, bilinear term, or wrong-curvature
term makes the whole model fall back.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from discopt.modeling.core import SolveResult

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    UnaryOp,
    Variable,
)

# func -> (numpy value, curvature) where curvature is +1 convex, -1 concave.
# A term coeff*func(affine) is convex iff sign(coeff) * curvature >= 0.
_FUNC = {
    "log": (np.log, -1),
    "log1p": (np.log1p, -1),
    "sqrt": (np.sqrt, -1),
    "exp": (np.exp, +1),
}
# Rust term_func codes (must match ConvexFunc in convex_kernel.rs).
_FUNC_CODE = {"log": 0, "exp": 1, "sqrt": 2, "log1p": 3}


class NotConvexKernel(Exception):
    """The model cannot be soundly routed to the convex kernel (→ NLP-BB)."""


def _flat_offsets(model) -> dict[int, int]:
    off, cur = {}, 0
    for v in model._variables:
        off[v._index] = cur
        cur += v.size
    return off


def _col_of(node, offsets: dict[int, int]) -> int:
    if isinstance(node, Variable):
        if node.size != 1:
            raise NotConvexKernel("array variable used as scalar")
        return offsets[node._index]
    if isinstance(node, IndexExpression) and isinstance(node.base, Variable):
        base, idx = node.base, node.index
        flat = int(np.ravel_multi_index(idx, base.shape)) if isinstance(idx, tuple) else int(idx)
        return offsets[base._index] + flat
    raise NotConvexKernel("non-variable leaf")


class _Decomp:
    __slots__ = ("aff", "const", "terms")

    def __init__(self):
        self.aff: dict[int, float] = {}
        self.const: float = 0.0
        self.terms: list[dict] = []  # {coeff, func, arg_aff, arg_const}

    def scale(self, k: float) -> _Decomp:
        self.const *= k
        for c in list(self.aff):
            self.aff[c] *= k
        for t in self.terms:
            t["coeff"] *= k
        return self

    def add(self, other: _Decomp) -> _Decomp:
        self.const += other.const
        for c, v in other.aff.items():
            self.aff[c] = self.aff.get(c, 0.0) + v
        self.terms.extend(other.terms)
        return self


def _as_const(node) -> Optional[float]:
    if isinstance(node, Constant) and node.value.ndim == 0:
        return float(node.value)
    return None


def _decompose(node, offsets) -> _Decomp:
    """Decompose into affine + composite-univariate terms, or raise."""
    d = _Decomp()
    c = _as_const(node)
    if c is not None:
        d.const = c
        return d
    if isinstance(node, (Variable, IndexExpression)):
        d.aff[_col_of(node, offsets)] = 1.0
        return d
    if isinstance(node, UnaryOp):
        if node.op == "neg":
            return _decompose(node.operand, offsets).scale(-1.0)
        raise NotConvexKernel(f"unary {node.op}")
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return _decompose(node.left, offsets).add(_decompose(node.right, offsets))
        if node.op == "-":
            return _decompose(node.left, offsets).add(_decompose(node.right, offsets).scale(-1.0))
        if node.op == "*":
            lc, rc = _as_const(node.left), _as_const(node.right)
            if lc is not None:
                return _decompose(node.right, offsets).scale(lc)
            if rc is not None:
                return _decompose(node.left, offsets).scale(rc)
            raise NotConvexKernel("bilinear product")
        if node.op == "/":
            rc = _as_const(node.right)
            if rc is not None and rc != 0.0:
                return _decompose(node.left, offsets).scale(1.0 / rc)
            raise NotConvexKernel("division by non-constant")
        raise NotConvexKernel(f"binary {node.op}")
    if isinstance(node, FunctionCall):
        if node.func_name not in _FUNC:
            raise NotConvexKernel(f"unsupported func {node.func_name}")
        if len(node.args) != 1:
            raise NotConvexKernel(f"multi-arg func {node.func_name}")
        arg = _decompose(node.args[0], offsets)
        if arg.terms:
            raise NotConvexKernel("non-affine function argument")
        d.terms.append(
            {"coeff": 1.0, "func": node.func_name, "arg_aff": arg.aff, "arg_const": arg.const}
        )
        return d
    raise NotConvexKernel(f"node {type(node).__name__}")


def _assert_convex_le(d: _Decomp) -> None:
    """Every term of a `≤`-normal-form g must be convex, else raise."""
    for t in d.terms:
        _val, curv = _FUNC[t["func"]]
        # convex iff sign(coeff)*curvature >= 0 (coeff==0 term is trivially fine).
        if t["coeff"] * curv < -1e-15:
            raise NotConvexKernel(
                f"nonconvex term coeff={t['coeff']:+.3g} func={t['func']} (curv={curv:+d})"
            )


def build_convex_spec(model, bounds=None) -> Optional[dict]:
    """Marshal `model` into the flat arrays for `solve_convex_tree_py`, or `None`.

    Returns ``None`` (→ keep the NLP-BB path) whenever the model is not provably a
    convex composite-of-affine MINLP per the soundness gate in the module docstring.
    """
    try:
        return _build(model, bounds)
    except NotConvexKernel:
        return None


def _build(model, bounds) -> dict:
    from discopt._jax.gdp_reformulate import reformulate_gdp
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.modeling.core import VarType

    m = reformulate_gdp(model, method="big-m")
    lb, ub = flat_variable_bounds(m)
    n = len(lb)
    lb = lb.astype(float)
    ub = ub.astype(float)

    is_int = np.zeros(n, bool)
    k = 0
    for v in m._variables:
        for _ in range(v.size):
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                is_int[k] = True
            k += 1

    ev = NLPEvaluator(m)
    senses = [c.sense if isinstance(c.sense, str) else c.sense.value for c in m._constraints]
    if m._objective is None or m._objective.sense.name not in ("MAXIMIZE", "MINIMIZE"):
        raise NotConvexKernel("no usable objective")
    sense_max = m._objective.sense.name == "MAXIMIZE"

    # Objective must be LINEAR (constant gradient) to be an LP objective.
    rng = np.random.default_rng(0)
    lo = np.where(np.isfinite(lb), lb, 0.0)
    hi = np.where(np.isfinite(ub), ub, lo + 5.0)
    xa = lo + rng.random(n) * (hi - lo)
    xb = lo + rng.random(n) * (hi - lo)
    ga = np.asarray(ev.evaluate_gradient(xa), float)
    gb = np.asarray(ev.evaluate_gradient(xb), float)
    if not np.allclose(ga, gb, atol=1e-9):
        raise NotConvexKernel("nonlinear objective")
    negate = bool(getattr(ev, "_negate", sense_max))
    c = (-ga if negate else ga).astype(float)

    # Classify rows linear (constant Jacobian) vs nonlinear.
    ja = ev.evaluate_jacobian(xa)
    jb = ev.evaluate_jacobian(xb)
    lin_rows = np.all(np.isclose(ja, jb, atol=1e-9), axis=1)
    g0 = np.asarray(ev.evaluate_constraints(xa), float)
    const = g0 - ja @ xa
    offsets = _flat_offsets(m)

    le_rows, eq_rows = [], []  # each: (cols, coeffs, rhs)
    nl_specs = []  # convex rows: (lin_aff, lin_const, terms, rhs=0)
    for i in range(ja.shape[0]):
        s = senses[i]
        if lin_rows[i]:
            a = np.asarray(ja[i], float)
            ci = float(const[i])
            if s == "<=":
                le_rows.append((a, -ci))
            elif s == ">=":
                le_rows.append((-a, ci))
            else:
                eq_rows.append((a, -ci))
            continue
        # Nonlinear row: decompose g_i (constraint g_i {sense} 0) and gate convexity.
        if s not in ("<=", ">="):
            raise NotConvexKernel("nonlinear equality constraint")
        expr = _constraint_expr(m, i)
        d = _decompose(expr, offsets)
        if s == ">=":  # normalize g ≥ 0 → (−g) ≤ 0
            d.scale(-1.0)
        _assert_convex_le(d)
        nl_specs.append(d)

    return _marshal(n, c, sense_max, is_int, lb, ub, le_rows, eq_rows, nl_specs)


def _constraint_expr(model, row_idx):
    con = model._constraints[row_idx]
    for attr in ("expr", "body", "lhs"):
        e = getattr(con, attr, None)
        if e is not None:
            return e
    raise NotConvexKernel("cannot locate constraint expression")


def _csr_from_rows(rows, n):
    ptr, cols, vals, rhs = [0], [], [], []
    for a, r in rows:
        a = np.asarray(a, float)
        nz = np.where(np.abs(a) > 1e-13)[0]
        cols.extend(nz.tolist())
        vals.extend(a[nz].tolist())
        ptr.append(len(cols))
        rhs.append(float(r))
    return (
        np.asarray(ptr, np.int64),
        np.asarray(cols, np.int64),
        np.asarray(vals, float),
        np.asarray(rhs, float),
    )


def _affine_csr(items):
    cs = sorted(items)
    return np.asarray(cs, np.int64), np.asarray([items[c] for c in cs], float)


def _marshal(n, c, sense_max, is_int, lb, ub, le_rows, eq_rows, nl_specs) -> dict:
    le_ptr, le_cols, le_coeffs, le_rhs = _csr_from_rows(le_rows, n)
    eq_ptr, eq_cols, eq_coeffs, eq_rhs = _csr_from_rows(eq_rows, n)

    nl_rhs, nl_lin_const = [], []
    nl_lin_ptr, nl_lin_cols, nl_lin_coeffs = [0], [], []
    nl_term_ptr = [0]
    term_coeff, term_func, term_arg_const = [], [], []
    term_arg_ptr, term_arg_cols, term_arg_coeffs = [0], [], []
    for d in nl_specs:
        lc, lk = _affine_csr(d.aff)
        nl_lin_cols.extend(lc.tolist())
        nl_lin_coeffs.extend(lk.tolist())
        nl_lin_ptr.append(len(nl_lin_cols))
        nl_lin_const.append(d.const)
        nl_rhs.append(0.0)
        for t in d.terms:
            term_coeff.append(t["coeff"])
            term_func.append(_FUNC_CODE[t["func"]])
            term_arg_const.append(t["arg_const"])
            ac, ak = _affine_csr(t["arg_aff"])
            term_arg_cols.extend(ac.tolist())
            term_arg_coeffs.extend(ak.tolist())
            term_arg_ptr.append(len(term_arg_cols))
        nl_term_ptr.append(len(term_coeff))

    return dict(
        n=n,
        c=np.asarray(c, float),
        integrality=np.asarray(is_int, np.int64),
        lo=np.asarray(lb, float),
        hi=np.asarray(ub, float),
        sense_max=bool(sense_max),
        le_row_ptr=le_ptr,
        le_cols=le_cols,
        le_coeffs=le_coeffs,
        le_rhs=le_rhs,
        eq_row_ptr=eq_ptr,
        eq_cols=eq_cols,
        eq_coeffs=eq_coeffs,
        eq_rhs=eq_rhs,
        nl_rhs=np.asarray(nl_rhs, float),
        nl_lin_const=np.asarray(nl_lin_const, float),
        nl_lin_ptr=np.asarray(nl_lin_ptr, np.int64),
        nl_lin_cols=np.asarray(nl_lin_cols, np.int64),
        nl_lin_coeffs=np.asarray(nl_lin_coeffs, float),
        nl_term_ptr=np.asarray(nl_term_ptr, np.int64),
        term_coeff=np.asarray(term_coeff, float),
        term_func=np.asarray(term_func, np.int64),
        term_arg_const=np.asarray(term_arg_const, float),
        term_arg_ptr=np.asarray(term_arg_ptr, np.int64),
        term_arg_cols=np.asarray(term_arg_cols, np.int64),
        term_arg_coeffs=np.asarray(term_arg_coeffs, float),
    )


def convex_kernel_enabled() -> bool:
    """`DISCOPT_CONVEX_KERNEL` opt-in (default-OFF)."""
    return os.environ.get("DISCOPT_CONVEX_KERNEL", "0") not in ("0", "", "false", "False")


def solve_convex_tree(spec: dict, *, time_limit_s: Optional[float] = None, **cfg) -> dict:
    """Run the native convex kernel on a marshaled `spec` (from build_convex_spec)."""
    import discopt._rust as _rust

    params = dict(
        max_nodes=cfg.get("max_nodes", 100000),
        gap_tol=cfg.get("gap_tol", 1e-4),
        int_tol=cfg.get("int_tol", 1e-5),
        oa_tol=cfg.get("oa_tol", 1e-6),
        max_oa_rounds=cfg.get("max_oa_rounds", 60),
        max_sep_rounds=cfg.get("max_sep_rounds", 12),
        fbbt_rounds=cfg.get("fbbt_rounds", 20),
        initial_incumbent=cfg.get("initial_incumbent", None),
        time_limit_s=time_limit_s,
    )
    result: dict = dict(_rust.solve_convex_tree_py(**spec, **params))
    return result


def try_convex_solve(
    model, *, time_limit: float = 3600.0, gap_tolerance: float = 1e-4
) -> Optional[SolveResult]:
    """Route `model` to the native convex kernel, or return ``None`` to fall back.

    Scoped to the smaller/quickly-certifiable convex MINLPs (#798): the kernel gets
    a BOUNDED attempt (``min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET)``, default
    120 s) and its result is used ONLY when it fully **certifies optimality** and
    the incumbent is verified feasible against the pristine model (#779). Everything
    else — flag off, non-convex, not-certified-within-budget, no incumbent, or an
    unverifiable incumbent — returns ``None`` so the caller keeps the (always-correct)
    default path with its full time budget. This bounds the kernel's cost on large
    instances it cannot finish (tracked separately for SCIP-parity) and never
    reports an unsound or uncertified result. Proven-infeasible roots are surfaced.
    """
    import os
    import time

    import numpy as np

    from discopt.modeling.core import SolveResult

    if not convex_kernel_enabled():
        return None
    spec = build_convex_spec(model)
    if spec is None:
        return None

    budget = min(time_limit, float(os.environ.get("DISCOPT_CONVEX_KERNEL_BUDGET", "120")))
    t0 = time.perf_counter()
    r = solve_convex_tree(
        spec,
        time_limit_s=budget,
        gap_tol=gap_tolerance,
        initial_incumbent=None,
    )
    wall = time.perf_counter() - t0

    incumbent = r["incumbent"]
    inc_x = np.asarray(r["incumbent_x"], float)

    if r["status"] == "infeasible":
        return SolveResult(status="infeasible", bound=r["bound"], wall_time=wall, nlp_bb=False)
    # Use the kernel result ONLY when it CERTIFIED optimality within budget; any
    # limit / feasible-only / no-incumbent outcome defers to the default path,
    # which then gets the caller's full time budget.
    if r["status"] != "optimal" or incumbent is None or inc_x.size == 0:
        return None
    status = "optimal"

    # Map the flat structural point back onto the ORIGINAL model's variables
    # (reformulation appends aux columns, so the original vars are a prefix), and
    # VERIFY feasibility against the pristine model — the #779 guard. Any violation
    # beyond tolerance ⇒ fall back rather than report an unsound incumbent.
    x_dict, x_flat = _unflatten(model, inc_x)
    if not _incumbent_is_feasible(model, x_flat):
        return None

    gap = None
    if incumbent not in (None, 0.0):
        gap = abs(incumbent - r["bound"]) / max(1.0, abs(incumbent))
    return SolveResult(
        status=status,
        objective=float(incumbent),
        bound=float(r["bound"]),
        gap=gap,
        x=x_dict,
        wall_time=wall,
        node_count=int(r["node_count"]),
        gap_certified=(status == "optimal"),
        nlp_bb=False,
    )


def _unflatten(model, inc_x):
    """(dict name→array, flat original-var vector) from the kernel's structural x."""
    import numpy as np

    x_dict = {}
    flat = []
    off = 0
    for v in model._variables:
        vals = np.asarray(inc_x[off : off + v.size], float)
        off += v.size
        flat.extend(vals.tolist())
        x_dict[v.name] = vals.reshape(v.shape) if v.shape else vals.reshape(())
    return x_dict, np.asarray(flat, float)


def _incumbent_is_feasible(model, x_flat, tol: float = 1e-5) -> bool:
    """#779: evaluate the PRISTINE model's constraints at `x_flat`; True iff feasible."""
    import numpy as np

    from discopt._jax.nlp_evaluator import NLPEvaluator

    try:
        ev = NLPEvaluator(model)
        g = np.asarray(ev.evaluate_constraints(x_flat), float)
    except Exception:
        return False
    for gi, con in zip(g, model._constraints):
        s = con.sense if isinstance(con.sense, str) else con.sense.value
        if s == "<=" and gi > tol:
            return False
        if s == ">=" and gi < -tol:
            return False
        if s not in ("<=", ">=") and abs(gi) > tol:
            return False
    return True
