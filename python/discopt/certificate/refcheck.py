"""Reference checker for Tier-1 feasibility certificates (exact rationals).

This is the **executable specification** of the Lean ``checkFeasible`` decision
procedure: the same algorithm, over :class:`fractions.Fraction` instead of Lean's
``Rat``. It exists to (a) de-risk the Lean port -- the two must agree bit for bit
on accept/reject -- and (b) give CI a runnable end-to-end verification while the
Lean project is compiled in a developer environment.

Soundness stance: the checker returns ``True`` only when it has *verified* the
claim over exact arithmetic. Anything it cannot evaluate exactly (a transcendental
``fn`` node) makes it return ``False`` with a reason -- it never guesses.
"""

from __future__ import annotations

from fractions import Fraction

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


def check_certificate(cert: dict) -> tuple[bool, str]:
    """Check a Tier-1 feasibility certificate. Returns ``(ok, reason)``.

    Verifies, over exact rationals, that the incumbent (a) lies within the column
    bounds, (b) is integral on integer/binary columns, (c) satisfies every
    constraint, and (d) attains the reported objective value -- each within the
    certificate's own tolerances. ``ok=False`` with a human-readable ``reason`` on
    the first violation (or on any node the exact checker refuses to evaluate).
    """
    try:
        body = cert["certificate"]
        if body.get("tier") != "feasibility":
            return False, f"unsupported tier {body.get('tier')!r} (reference checker is Tier 1)"
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
