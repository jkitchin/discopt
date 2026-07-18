"""Reference checker for feasibility (Tier 1) and convex/KKT (Tier 2) certificates.

The executable specification of the Lean checker: the same decision procedures,
over :class:`fractions.Fraction` instead of Lean's ``Rat``. It exists to (a)
de-risk the Lean port -- the two must agree on every accept/reject -- and (b)
give CI a runnable end-to-end verification.

* **Tier 1 (feasibility)** -- the incumbent lies in the box, is integral where
  required, satisfies every constraint, and attains the reported objective value.
* **Tier 2 (convex / KKT)** -- additionally proves *global optimality* for the
  exact-rational convex QP/QCQP subclass: the objective and constraint bodies are
  quadratic with PSD/affine (constant) Hessians (so the model is convex), and the
  incumbent satisfies the KKT conditions with the emitted multipliers, hence
  ``dualBound == objectiveValue`` (gap closed). Gradients/Hessians are re-derived
  from the model by exact symbolic differentiation -- the checker trusts only the
  multipliers, not any emitted derivative.

Soundness stance: the checker returns ``True`` only when it has *verified* the
claim over exact arithmetic. Anything it cannot evaluate/differentiate exactly (a
transcendental ``fn``, a non-quadratic body) makes it return ``False`` with a
reason -- it never guesses.
"""

from __future__ import annotations

from fractions import Fraction

from .diff import NotSmooth, differentiate, has_variable
from .linalg import is_psd, is_zero_matrix
from .schema import as_fraction


class _Refuse(Exception):
    """Internal: an expression the exact checker cannot evaluate."""


def _eval(node: dict, x: list[Fraction]) -> Fraction:
    """Evaluate an expression node at incumbent *x* over exact rationals."""
    k = node["k"]
    if k == "const":
        return as_fraction(node["v"])
    if k == "var":
        return x[node["i"]]
    if k == "neg":
        return -_eval(node["x"], x)
    if k == "abs":
        return abs(_eval(node["x"], x))
    if k in ("add", "sub", "mul", "div"):
        left = _eval(node["l"], x)
        right = _eval(node["r"], x)
        if k == "add":
            return left + right
        if k == "sub":
            return left - right
        if k == "mul":
            return left * right
        # div
        if right == 0:
            raise _Refuse("division by zero at incumbent")
        return left / right
    if k == "pow":
        base = _eval(node["l"], x)
        exp = _eval(node["r"], x)
        if exp.denominator != 1:
            raise _Refuse(f"non-integer exponent {exp} (not exact-rational)")
        e = exp.numerator
        if e < 0 and base == 0:
            raise _Refuse("zero base with negative exponent")
        return base**e
    if k == "fn":
        raise _Refuse(f"transcendental function {node.get('name')!r} not exact-rational (Tier 1)")
    raise _Refuse(f"unknown expression node {k!r}")


# ── dispatcher ───────────────────────────────────────────────────────────────
def check_certificate(cert: dict) -> tuple[bool, str]:
    """Check a certificate. Returns ``(ok, reason)``.

    Dispatches on ``tier``: ``feasibility`` (Tier 1) or ``convex`` (Tier 2). A
    Tier-2 check first requires Tier-1 feasibility, then the convexity + KKT
    conditions that certify global optimality.
    """
    try:
        body = cert["certificate"]
        tier = body.get("tier")
    except (KeyError, TypeError) as exc:
        return False, f"malformed certificate: {exc}"
    if tier == "feasibility":
        return _check_feasibility(cert)
    if tier == "convex":
        ok, reason = _check_feasibility(cert)
        if not ok:
            return False, f"primal infeasible: {reason}"
        return _check_convex(cert)
    return False, f"unsupported tier {tier!r}"


# ── Tier 1: feasibility ──────────────────────────────────────────────────────
def _check_feasibility(cert: dict) -> tuple[bool, str]:
    """Verify the incumbent is feasible with the reported objective value."""
    try:
        body = cert["certificate"]
        model = body["model"]
        columns = model["columns"]
        x = [as_fraction(v) for v in body["incumbent"]["x"]]
        feas_tol = as_fraction(body["tolerances"]["feas"]) or Fraction(0)
        int_tol = as_fraction(body["tolerances"]["int"]) or Fraction(0)
    except (KeyError, TypeError) as exc:
        return False, f"malformed certificate: {exc}"

    if len(x) != len(columns):
        return False, f"incumbent has {len(x)} columns, model declares {len(columns)}"

    # (a) bounds + (b) integrality
    for j, col in enumerate(columns):
        xj = x[j]
        lb = as_fraction(col.get("lb"))
        ub = as_fraction(col.get("ub"))
        if lb is not None and xj < lb - feas_tol:
            return False, f"column {j} ({col['name']}): {xj} < lb {lb} (tol {feas_tol})"
        if ub is not None and xj > ub + feas_tol:
            return False, f"column {j} ({col['name']}): {xj} > ub {ub} (tol {feas_tol})"
        if col["type"] in ("integer", "binary"):
            nearest = Fraction(round(xj))
            if abs(xj - nearest) > int_tol:
                return False, (
                    f"column {j} ({col['name']}): {float(xj):.6g} not integral "
                    f"(tol {float(int_tol):.0e})"
                )
            if col["type"] == "binary" and nearest not in (0, 1):
                return False, f"column {j} ({col['name']}): binary value {nearest} not in {{0,1}}"

    # (c) constraints
    for c in model["constraints"]:
        try:
            val = _eval(c["body"], x)
        except _Refuse as r:
            return False, f"constraint {c['name']!r}: {r}"
        rhs = as_fraction(c["rhs"])
        sense = c["sense"]
        if sense == "le" and val > rhs + feas_tol:
            return False, f"constraint {c['name']!r}: {float(val):.6g} > {float(rhs):.6g} (le)"
        if sense == "ge" and val < rhs - feas_tol:
            return False, f"constraint {c['name']!r}: {float(val):.6g} < {float(rhs):.6g} (ge)"
        if sense == "eq" and abs(val - rhs) > feas_tol:
            return (
                False,
                f"constraint {c['name']!r}: |{float(val):.6g} - {float(rhs):.6g}| > tol (eq)",
            )

    # (d) objective value
    try:
        obj = _eval(model["objective"]["body"], x)
    except _Refuse as r:
        return False, f"objective: {r}"
    claimed = as_fraction(body["incumbent"]["objectiveValue"])
    if abs(obj - claimed) > feas_tol:
        return False, (
            f"objective value mismatch: evaluated {float(obj):.8g} vs claimed {float(claimed):.8g}"
        )

    return True, "feasible: incumbent satisfies all bounds, integrality, and constraints"


# ── Tier 2: convex / KKT global optimality ───────────────────────────────────
def _constant_hessian(
    expr: dict, n: int, x: list[Fraction]
) -> tuple[list[list[Fraction]] | None, str]:
    """Exact Hessian of *expr* over ``n`` columns, requiring it to be constant.

    Returns ``(H, "")`` for a quadratic (constant-Hessian) body, or
    ``(None, reason)`` if the body is non-smooth (abs/fn) or non-quadratic
    (Hessian depends on x) -- the checker then refuses.
    """
    h = [[Fraction(0)] * n for _ in range(n)]
    for j in range(n):
        try:
            dj = differentiate(expr, j)
        except NotSmooth as e:
            return None, str(e)
        for i in range(j, n):
            try:
                dij = differentiate(dj, i)
            except NotSmooth as e:
                return None, str(e)
            if has_variable(dij):
                return None, "non-constant Hessian (body is not quadratic)"
            try:
                v = _eval(dij, x)
            except _Refuse as r:
                return None, str(r)
            h[i][j] = v
            h[j][i] = v
    return h, ""


def _grad(expr: dict, n: int, x: list[Fraction]) -> tuple[list[Fraction] | None, str]:
    """Exact gradient of *expr* at *x* over ``n`` columns (or refuse)."""
    g: list[Fraction] = []
    for j in range(n):
        try:
            g.append(_eval(differentiate(expr, j), x))
        except (NotSmooth, _Refuse) as e:
            return None, str(e)
    return g, ""


def _check_convex(cert: dict) -> tuple[bool, str]:
    """Verify convexity (constant PSD/affine Hessians) + the KKT conditions."""
    body = cert["certificate"]
    model = body["model"]
    n = len(model["columns"])
    x = [as_fraction(v) for v in body["incumbent"]["x"]]
    feas_tol = as_fraction(body["tolerances"]["feas"]) or Fraction(0)
    kkt_tol = as_fraction(body["tolerances"].get("kkt")) or feas_tol
    kkt = body.get("kkt")
    if kkt is None:
        return False, "convex certificate missing kkt block"
    lam = [as_fraction(v) for v in kkt["constraint_multipliers"]]
    bl = [as_fraction(v) or Fraction(0) for v in kkt["bound_lower"]]
    bu = [as_fraction(v) or Fraction(0) for v in kkt["bound_upper"]]
    cons = model["constraints"]
    if len(lam) != len(cons):
        return False, "kkt multiplier count != constraint count"

    # (1) Convexity: objective PSD; le body PSD; ge body NSD; eq body affine.
    hf, reason = _constant_hessian(model["objective"]["body"], n, x)
    if hf is None:
        return False, f"objective not exact-quadratic: {reason}"
    ok, why = is_psd(hf, tol=kkt_tol)
    if not ok:
        return False, f"objective not convex: {why}"
    for c in cons:
        h, reason = _constant_hessian(c["body"], n, x)
        if h is None:
            return False, f"constraint {c['name']!r} not exact-quadratic: {reason}"
        if c["sense"] == "le":
            ok, why = is_psd(h, tol=kkt_tol)
            if not ok:
                return False, f"constraint {c['name']!r} (le) body not convex: {why}"
        elif c["sense"] == "ge":
            neg = [[-v for v in row] for row in h]
            ok, why = is_psd(neg, tol=kkt_tol)
            if not ok:
                return False, f"constraint {c['name']!r} (ge) body not concave: {why}"
        else:  # eq
            if not is_zero_matrix(h, tol=kkt_tol):
                return False, f"constraint {c['name']!r} (eq) body not affine"

    # (2) KKT stationarity: grad_f + Σ λ_i s_i grad_g_i - ρ^L + ρ^U = 0.
    gf, reason = _grad(model["objective"]["body"], n, x)
    if gf is None:
        return False, f"objective gradient: {reason}"
    grads = []
    for c in cons:
        g, reason = _grad(c["body"], n, x)
        if g is None:
            return False, f"constraint {c['name']!r} gradient: {reason}"
        grads.append(g)
    for j in range(n):
        s = gf[j] - bl[j] + bu[j]
        for i, c in enumerate(cons):
            sign = 1 if c["sense"] in ("le", "eq") else -1  # ge normalized as -body
            s += lam[i] * sign * grads[i][j]
        if abs(s) > kkt_tol:
            return False, (
                f"KKT stationarity violated at column {j}: residual {float(s):.3g} > tol"
            )

    # (3) Dual feasibility: inequality multipliers >= 0; bound multipliers >= 0.
    for i, c in enumerate(cons):
        if c["sense"] in ("le", "ge") and lam[i] < -kkt_tol:
            return False, f"dual infeasible: multiplier for {c['name']!r} = {float(lam[i]):.3g} < 0"
    for j in range(n):
        if bl[j] < -kkt_tol or bu[j] < -kkt_tol:
            return False, f"dual infeasible: bound multiplier at column {j} < 0"

    # (4) Complementary slackness: λ_i g_i(x) = 0 (g in <=0 form); bound analogues.
    for i, c in enumerate(cons):
        if c["sense"] == "eq":
            continue
        val = _eval(c["body"], x)
        rhs = as_fraction(c["rhs"])
        gi = (val - rhs) if c["sense"] == "le" else (rhs - val)  # <= 0 form
        if abs(lam[i] * gi) > kkt_tol:
            return False, (
                f"complementary slackness violated for {c['name']!r}: "
                f"|λ·g| = {float(abs(lam[i] * gi)):.3g} > tol"
            )
    for j, col in enumerate(model["columns"]):
        lb = as_fraction(col.get("lb"))
        ub = as_fraction(col.get("ub"))
        if lb is not None and abs(bl[j] * (lb - x[j])) > kkt_tol:
            return False, f"complementary slackness violated at lower bound of column {j}"
        if ub is not None and abs(bu[j] * (x[j] - ub)) > kkt_tol:
            return False, f"complementary slackness violated at upper bound of column {j}"

    # (5) Gap closed: dualBound == objectiveValue (KKT+convex => x* is global).
    claimed = as_fraction(body.get("dualBound"))
    objval = as_fraction(body["incumbent"]["objectiveValue"])
    if claimed is None:
        return False, "convex certificate missing dualBound"
    if abs(claimed - objval) > feas_tol:
        return False, (
            f"dualBound {float(claimed):.8g} != objectiveValue {float(objval):.8g} (gap not closed)"
        )

    return True, "convex KKT global optimum verified: dualBound == objectiveValue"
