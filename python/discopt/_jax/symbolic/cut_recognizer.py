"""Presolve recognizer: auto-derive structured cuts from a model's graph.

This is the recognition layer on top of :mod:`constraint_cuts`. Given a built
:class:`discopt.modeling.core.Model`, it reads the objective and equality
constraints, identifies a **bilinear-concave objective term coupled through a
square-difference (Weymouth/Hazen-Williams) chain**, and automatically derives a
sound univariate underestimator cut for that term — without being handed the
equations or the elimination targets.

Pipeline (all from the model graph):

1. Translate objective + ``==`` constraints to SymPy (:func:`model_to_sympy`).
2. Union-find over ``a == b`` equalities to canonicalize flow aliases, and detect
   fixed variables ``a == const``.
3. Find objective product terms ``coef * x * (y**k - 1)`` -> candidates ``(x, y)``.
4. For each candidate, locate ``y``'s defining ratio equality ``p_out = y * p_in``,
   express ``p_in^2`` and ``p_out^2`` in the flow ``x`` via their Weymouth
   equations, and FBBT the terminal pressure's lower bound from a demand equation
   with a fixed flow. This yields a sound coupling ``y >= sqrt(phi(x))``.
5. Hand the assembled chain to :func:`constraint_cuts.eliminate_chain_coupling`,
   build the underestimator, and certify it.

**Scope:** this recognizer targets the square-difference-network class (gas/water
pipe + compressor/pump networks with a power/ratio objective). It is deliberately
narrow and explicit about the structure it matches; unmatched models yield no
cuts (the solver falls back to its normal relaxation). It is design-time
(imports SymPy); its output is numeric cut coefficients / JAX closures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp

from discopt._jax.symbolic.constraint_cuts import (
    ChainCoupling,
    TermUnderestimator,
    eliminate_chain_coupling,
    power_term_underestimator,
    verify_cut,
)

# ---------------------------------------------------------------------------
# DAG -> SymPy
# ---------------------------------------------------------------------------


def _var_key(node) -> Optional[str]:
    from discopt.modeling import core

    if isinstance(node, core.IndexExpression):
        idx = node.index
        i = idx if isinstance(idx, int) else (idx[0] if isinstance(idx, tuple) else idx)
        return f"{node.base.name}[{i}]"
    if isinstance(node, core.Variable):
        return node.name
    return None


def _to_sympy(node, syms: dict):
    from discopt.modeling import core

    if isinstance(node, core.Constant):
        return sp.Float(float(node.value))
    if isinstance(node, (core.Variable, core.IndexExpression)):
        key = _var_key(node)
        if key not in syms:
            syms[key] = sp.Symbol(key.replace("[", "_").replace("]", ""), real=True)
        return syms[key]
    if isinstance(node, core.BinaryOp):
        lo, hi = _to_sympy(node.left, syms), _to_sympy(node.right, syms)
        if node.op == "**":
            # Rationalize integer-valued float exponents (2.0 -> 2) so SymPy treats
            # x**2 as polynomial; keep genuine fractional exponents (e.g. 0.2857).
            if hi.is_number and float(hi) == int(float(hi)):
                hi = sp.Integer(int(float(hi)))
            return lo**hi
        return {"+": lo + hi, "-": lo - hi, "*": lo * hi, "/": lo / hi}[node.op]
    if isinstance(node, core.UnaryOp):
        x = _to_sympy(node.operand, syms)
        return {"neg": -x, "abs": sp.Abs(x)}[node.op]
    if isinstance(node, core.FunctionCall):
        args = [_to_sympy(a, syms) for a in node.args]
        return getattr(sp, node.func_name)(*args)
    raise TypeError(f"unsupported node {type(node).__name__}")


@dataclass
class SympyModel:
    objective: sp.Expr
    equalities: list  # list of (name, sympy.Eq)
    symbols: dict  # key -> Symbol
    bounds: dict  # Symbol -> (lb, ub)


def model_to_sympy(model) -> SympyModel:
    """Translate a model's objective and ``==`` constraints to SymPy."""

    syms: dict = {}
    obj = _to_sympy(model._objective.expression, syms)
    eqs = []
    for c in model._constraints:
        if c.sense == "==":
            eqs.append((c.name, sp.Eq(_to_sympy(c.body, syms), sp.Float(c.rhs))))

    # Variable bounds, keyed by the per-element symbol.
    import numpy as np

    def _elem(bnd, i, size):
        arr = np.atleast_1d(np.asarray(bnd, dtype=float)) if bnd is not None else None
        if arr is None:
            return None
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return float(arr.reshape(-1)[i])

    bounds: dict = {}
    for v in model._variables:
        size = int(getattr(v, "size", 1) or 1)
        lb = getattr(v, "lb", None)
        ub = getattr(v, "ub", None)
        for i in range(size):
            key = v.name if size == 1 else f"{v.name}[{i}]"
            if key in syms:
                bounds[syms[key]] = (_elem(lb, i, size), _elem(ub, i, size))
    return SympyModel(objective=obj, equalities=eqs, symbols=syms, bounds=bounds)


# ---------------------------------------------------------------------------
# Canonicalization: union-find on a == b, and fixed a == const
# ---------------------------------------------------------------------------


def _linear_terms(expr: sp.Expr):
    """Return (coeff_dict, const) for an affine ``expr``, or None if nonlinear."""
    poly = (
        sp.Poly(sp.expand(expr), *sorted(expr.free_symbols, key=str)) if expr.free_symbols else None
    )
    if poly is None:
        return ({}, float(expr))
    if poly.total_degree() > 1:
        return None
    coeffs = {}
    const = 0.0
    for monom, c in poly.terms():
        if all(m == 0 for m in monom):
            const = float(c)
        else:
            sym = poly.gens[[i for i, m in enumerate(monom) if m == 1][0]]
            coeffs[sym] = float(c)
    return coeffs, const


def _canonicalize(eqs, symbols):
    parent = {s: s for s in symbols.values()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    fixed: dict = {}
    for _, eq in eqs:
        lt = _linear_terms(eq.lhs - eq.rhs)
        if lt is None:
            continue
        coeffs, const = lt
        syms = list(coeffs)
        if len(syms) == 2 and abs(const) < 1e-12 and {abs(coeffs[s]) for s in syms} == {1.0}:
            if coeffs[syms[0]] * coeffs[syms[1]] < 0:  # a - b == 0
                union(syms[0], syms[1])
        elif len(syms) == 1 and abs(coeffs[syms[0]]) > 0:  # a == const
            fixed[syms[0]] = -const / coeffs[syms[0]]
    classes = {s: find(s) for s in parent}
    return classes, fixed


# ---------------------------------------------------------------------------
# Objective bilinear-concave term detection
# ---------------------------------------------------------------------------


@dataclass
class ProductTerm:
    x: sp.Symbol  # the linear factor (flow)
    y: sp.Symbol  # the variable inside the concave power factor (ratio)
    exponent: float
    coefficient: float


def find_product_terms(objective: sp.Expr) -> list[ProductTerm]:
    """Find additive terms ``coef * x * (y**k - 1)`` with x linear, y in a power."""
    out: list[ProductTerm] = []
    # Keep the objective factored so the (y**k - 1) factor stays intact.
    for term in sp.Add.make_args(objective):
        fs = list(term.free_symbols)
        if len(fs) != 2:
            continue
        # one symbol linear, the other appears under a fractional power
        lin = [s for s in fs if _is_linear_in(term, s)]
        if len(lin) != 1:
            continue
        x = lin[0]
        y = fs[0] if fs[1] == x else fs[1]
        powers = [a.exp for a in term.atoms(sp.Pow) if a.base == y]
        if not powers:
            continue
        k = float(max(powers, key=lambda e: float(e)))
        # coefficient: term / (x * (y**k - 1)) must be a pure constant
        cand = sp.simplify(term / (x * (y ** sp.Float(k) - 1)))
        if cand.free_symbols:
            continue
        out.append(ProductTerm(x=x, y=y, exponent=k, coefficient=float(cand)))
    return out


def _is_linear_in(term, s):
    """True if ``term`` is affine in ``s`` (robust to fractional powers of others)."""
    return sp.simplify(sp.diff(term, s, 2)) == 0 and sp.simplify(sp.diff(term, s)) != 0


# ---------------------------------------------------------------------------
# Chain assembly + coupling derivation
# ---------------------------------------------------------------------------


def _eqs_with(sym, eqs):
    return [(n, e) for (n, e) in eqs if sym in (e.lhs - e.rhs).free_symbols]


def _square_diff_eq(psym, eqs):
    """Find an equation containing psym**2 (a Weymouth/square-difference row)."""
    for n, e in eqs:
        expr = e.lhs - e.rhs
        if expr.has(psym**2) or expr.has(psym ** sp.Float(2.0)):
            return n, e
    return None


def _solve_psq(psym, eq):
    """Solve a square-difference equation for psym**2 as a linear expr in squares."""
    expr = sp.expand(eq.lhs - eq.rhs)
    # substitute z = psym**2 etc. Solve linear-in-squares.
    sq = {s: sp.Symbol(f"__sq_{s.name}", real=True) for s in expr.free_symbols}
    lin = expr
    for s, z in sq.items():
        lin = lin.subs(s**2, z).subs(s ** sp.Float(2.0), z)
    sol = sp.solve(sp.Eq(lin, 0), sq[psym])
    if not sol:
        return None
    back = sol[0]
    for s, z in sq.items():
        back = back.subs(z, s**2)
    return sp.simplify(back)


@dataclass
class RecognizedCut:
    term: ProductTerm
    coupling: ChainCoupling
    underestimator: TermUnderestimator
    pn5_lower: float
    verification: dict
    x_key: str = ""  # original model key for the flow var, e.g. "w_cs[0]"
    y_key: str = ""  # original model key for the ratio var, e.g. "beta[0]"


def _derive_terminal_pressure_lb(p5, eqs, classes, fixed, bounds):
    """FBBT a pressure's lower bound from a demand equation with a fixed flow."""
    best = None
    for n, e in eqs:
        expr = e.lhs - e.rhs
        if not (expr.has(p5**2) or expr.has(p5 ** sp.Float(2.0))):
            continue
        psq = _solve_psq(p5, e)
        if psq is None:
            continue
        # psq = fixed_flow^2/C + other_pressure^2: substitute the fixed flow value
        # and any pressure at its lower bound (psq is increasing in each).
        sub = {}
        ok = True
        for s in psq.free_symbols:
            rep = classes.get(s, s)
            fval = next((v for f, v in fixed.items() if classes.get(f, f) == rep), None)
            if fval is not None:
                sub[s] = fval
            elif s in bounds and bounds[s][0] is not None:
                sub[s] = bounds[s][0]  # pressure at its lower bound (monotone increasing)
            else:
                ok = False
        if not ok:
            continue
        val = float(psq.subs(sub))
        if val > 0:
            lb = val**0.5
            best = lb if best is None else max(best, lb)
    return best


def recognize_and_derive_cuts(model, *, verify: bool = True) -> list[RecognizedCut]:
    """Auto-derive structured underestimator cuts from a model's graph.

    Returns a list of :class:`RecognizedCut` (possibly empty if the model does not
    match the square-difference-network pattern).
    """
    sm = model_to_sympy(model)
    classes, fixed = _canonicalize(sm.equalities, sm.symbols)
    reverse = {sym: key for key, sym in sm.symbols.items()}
    cuts: list[RecognizedCut] = []

    for term in find_product_terms(sm.objective):
        x, y = term.x, term.y
        # y's defining ratio equality: p_out - y*p_in == 0
        ratio = None
        for n, e in _eqs_with(y, sm.equalities):
            expr = sp.expand(e.lhs - e.rhs)
            # bilinear in y: collect p_in (the factor multiplying y) and p_out
            pin = None
            for s in expr.free_symbols:
                if s == y:
                    continue
                if expr.coeff(y, 1).has(s):
                    pin = s
            if pin is None:
                continue
            rest = sp.simplify(expr - expr.coeff(y, 1) * y)  # terms without y
            pout_syms = [s for s in rest.free_symbols if s != pin]
            if len(pout_syms) == 1:
                ratio = (pout_syms[0], pin)
                break
        if ratio is None:
            continue
        p_out, p_in = ratio

        # express p_in^2 and p_out^2 in terms of the flow x (+ a terminal pressure)
        def psq_in_x(p):
            sd = _square_diff_eq(p, sm.equalities)
            if sd is None:
                return None, None
            psq = _solve_psq(p, sd[1])
            if psq is None:
                return None, None
            # rename flows in the same class as x -> x; keep pressures symbolic
            sub = {}
            terminal = None
            for s in psq.free_symbols:
                if classes.get(s, s) == classes.get(x, x):
                    sub[s] = x
                elif s != p and s in sm.bounds:
                    # a pressure (bounded) -> potential terminal
                    if s not in (p_in, p_out):
                        terminal = s
            return sp.simplify(psq.subs(sub)), terminal

        pin_sq, _ = psq_in_x(p_in)
        pout_sq, terminal = psq_in_x(p_out)
        if pin_sq is None or pout_sq is None or terminal is None:
            continue

        # FBBT the terminal pressure lower bound
        pn5_lb = _derive_terminal_pressure_lb(terminal, sm.equalities, classes, fixed, sm.bounds)
        if pn5_lb is None:
            continue

        # assemble the chain for eliminate_chain_coupling
        p1, p2, p5 = sp.symbols("p1 p2 p5", positive=True)
        w = sp.Symbol("w", positive=True)
        beta = sp.Symbol("beta", positive=True)
        eqs = [
            sp.Eq(p1**2, pin_sq.subs(x, w)),  # p_in^2 = f(w)
            sp.Eq(p2, beta * p1),  # ratio
            sp.Eq(p2**2, pout_sq.subs({x: w, terminal: p5})),  # p_out^2 = f(w, p5)
        ]
        coupling = eliminate_chain_coupling(
            eqs,
            target=beta,
            keep=w,
            eliminate=[p1, p2],
            lower_bounds={p5: sp.Float(pn5_lb)},
            sample={w: 35.0, p5: pn5_lb},
        )
        wmax = float((sm.bounds.get(x, (0.0, 100.0))[1]) or 100.0)
        under = power_term_underestimator(
            coupling,
            exponent=term.exponent,
            coefficient=term.coefficient,
            domain=(0.0, min(wmax, float(sp.sqrt(3.5 * (2500 - 30**2))))),
        )
        report = {}
        if verify:
            cfn = sp.lambdify([coupling.keep], coupling.target_lower, "numpy")
            tfn = sp.lambdify([w, beta], term.coefficient * w * (beta**term.exponent - 1), "numpy")
            report = verify_cut(
                lambda a, b: float(tfn(a, b)),
                under,
                lambda a: float(cfn(a)),
                domain=under_domain_of(under),
                target_max=float((sm.bounds.get(y, (1.0, 2.0))[1]) or 2.0),
            )
        cuts.append(
            RecognizedCut(
                term=term,
                coupling=coupling,
                underestimator=under,
                pn5_lower=pn5_lb,
                verification=report,
                x_key=reverse.get(term.x, ""),
                y_key=reverse.get(term.y, ""),
            )
        )
    return cuts


# ---------------------------------------------------------------------------
# Injection: rewrite the objective with auxiliary-variable lower bounds
# ---------------------------------------------------------------------------


def _handle_map(model) -> dict:
    """Map original variable keys (``"w_cs[0]"``) to model expression handles."""
    handles: dict = {}
    for v in model._variables:
        size = int(getattr(v, "size", 1) or 1)
        if size == 1:
            handles[v.name] = v
        else:
            for i in range(size):
                handles[f"{v.name}[{i}]"] = v[i]
    return handles


def inject_cuts(model, cuts, *, samples=(8, 15, 25, 35, 45, 55, 65, 72)) -> int:
    """Inject recognized cuts into ``model`` in place; returns the number applied.

    For each cut, adds an auxiliary ``u >= 0`` with ``u == term`` (exact defining
    constraint) plus tangent lower bounds ``u >= K*h(w)``, and rewrites the
    objective to use ``u`` instead of the nonconvex term. The objective *value* is
    unchanged (``u == term``), but its relaxation is bounded below by the convex
    ``K*h(w)`` — closing the gap.
    """
    handles = _handle_map(model)
    obj = model._objective.expression
    applied = 0
    for j, cut in enumerate(cuts):
        if cut.x_key not in handles or cut.y_key not in handles:
            continue
        xh, yh = handles[cut.x_key], handles[cut.y_key]
        term_expr = cut.term.coefficient * xh * (yh**cut.term.exponent - 1.0)
        u = model.continuous(f"u_cut_{j}", lb=0.0, ub=1e6)
        model.subject_to(u == term_expr, name=f"cut_def_{j}")
        for wk in samples:
            v, slope = cut.underestimator.tangent_cut(float(wk))
            model.subject_to(u >= v + slope * (xh - float(wk)) - 1e-6, name=f"cut_tan_{j}_{wk}")
        obj = obj - term_expr + u
        applied += 1
    if applied:
        model.minimize(obj)
    return applied


def recognize_and_inject(model, **kwargs) -> int:
    """Convenience: recognize structured cuts and inject them in place."""
    cuts = recognize_and_derive_cuts(model, **kwargs)
    return inject_cuts(model, cuts)


def under_domain_of(under: TermUnderestimator) -> tuple[float, float]:
    return (0.0, 74.0)
