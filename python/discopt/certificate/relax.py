"""Exact McCormick relaxation compiler for the quadratic fragment (Tier-3 untrusted).

For a *fully untrusted* Tier-3 leaf bound, the checker must derive a valid lower
bound on the model over a leaf box **without trusting the solver's bound**. This
module is the trusted relaxation compiler it uses: given the model (in the
certificate's expression encoding) and a box, it builds -- in exact rationals --
the McCormick relaxation linear program

    min  c·z + obj_const   s.t.   A z >= b        (z = [original vars, aux products])

whose optimum is a valid lower bound on the model's objective over the box. A
dual-feasible ``y`` then certifies a bound ``b·y + obj_const`` by weak duality
(:func:`discopt.certificate.bnb.lp_lower_bound`) -- the checker builds ``A,b,c``
itself and only *verifies* the emitted dual, so a wrong dual causes a rejection,
never an unsound accept.

Scope: the **quadratic fragment** -- objective and constraint bodies that are
polynomials of total degree <= 2 (linear + bilinear ``x*y`` + square ``x^2``).
Anything higher-degree, rational (non-constant denominator), ``abs``, or
transcendental raises :class:`NotQuadratic`; the caller then falls back to the
trusted recorded bound for that leaf. This mirrors the McCormick lifting discopt
uses, but is independent code -- its validity (McCormick envelopes are valid
under/over-estimators) is the soundness obligation, provable over an ordered field.
"""

from __future__ import annotations

from fractions import Fraction

from .bnb import mccormick_bilinear, mccormick_square
from .schema import as_fraction


class NotQuadratic(Exception):
    """The expression is outside the linear+bilinear+square (degree<=2) fragment."""


# A quadratic form over the original columns: constant + Σ lin_i x_i + Σ quad_{ij} x_i x_j
# (quad keys are ordered pairs i<=j).
class QuadForm:
    __slots__ = ("const", "lin", "quad")

    def __init__(self, const=Fraction(0), lin=None, quad=None):
        self.const = const
        self.lin: dict[int, Fraction] = lin or {}
        self.quad: dict[tuple[int, int], Fraction] = quad or {}

    def is_constant(self) -> bool:
        return not self.lin and not self.quad

    def is_affine(self) -> bool:
        return not self.quad

    def _add(self, other, sign=1):
        c = self.const + sign * other.const
        lin = dict(self.lin)
        for i, v in other.lin.items():
            lin[i] = lin.get(i, Fraction(0)) + sign * v
        quad = dict(self.quad)
        for k, v in other.quad.items():
            quad[k] = quad.get(k, Fraction(0)) + sign * v
        return QuadForm(c, _prune(lin), _prune(quad))


def _prune(d):
    return {k: v for k, v in d.items() if v != 0}


def _scale(q: QuadForm, s: Fraction) -> QuadForm:
    return QuadForm(
        q.const * s,
        _prune({i: v * s for i, v in q.lin.items()}),
        _prune({k: v * s for k, v in q.quad.items()}),
    )


def _mul(a: QuadForm, b: QuadForm) -> QuadForm:
    """Product of two quadratic forms; raises if the result exceeds degree 2."""
    if a.quad and not b.is_constant():
        raise NotQuadratic("product would exceed degree 2")
    if b.quad and not a.is_constant():
        raise NotQuadratic("product would exceed degree 2")
    # const * anything
    if a.is_constant():
        return _scale(b, a.const)
    if b.is_constant():
        return _scale(a, b.const)
    # both affine (no quad): (ca + Σ la_i x_i)(cb + Σ lb_j x_j)
    const = a.const * b.const
    lin: dict[int, Fraction] = {}
    for i, v in a.lin.items():
        lin[i] = lin.get(i, Fraction(0)) + v * b.const
    for j, v in b.lin.items():
        lin[j] = lin.get(j, Fraction(0)) + v * a.const
    quad: dict[tuple[int, int], Fraction] = {}
    for i, vi in a.lin.items():
        for j, vj in b.lin.items():
            key = (i, j) if i <= j else (j, i)
            quad[key] = quad.get(key, Fraction(0)) + vi * vj
    return QuadForm(const, _prune(lin), _prune(quad))


def extract(node: dict) -> QuadForm:
    """Symbolic quadratic form of a certificate expression node (or raise)."""
    k = node["k"]
    if k == "const":
        return QuadForm(as_fraction(node["v"]))
    if k == "var":
        return QuadForm(Fraction(0), {node["i"]: Fraction(1)}, {})
    if k == "neg":
        return _scale(extract(node["x"]), Fraction(-1))
    if k == "add":
        return extract(node["l"])._add(extract(node["r"]), 1)
    if k == "sub":
        return extract(node["l"])._add(extract(node["r"]), -1)
    if k == "mul":
        return _mul(extract(node["l"]), extract(node["r"]))
    if k == "pow":
        exp = node["r"]
        if exp.get("k") != "const" or as_fraction(exp["v"]).denominator != 1:
            raise NotQuadratic("non-integer exponent")
        n = as_fraction(exp["v"]).numerator
        base = extract(node["l"])
        if n == 0:
            return QuadForm(Fraction(1))
        if n == 1:
            return base
        if n == 2:
            return _mul(base, base)
        raise NotQuadratic(f"exponent {n} exceeds the quadratic fragment")
    if k == "div":
        denom = extract(node["r"])
        if not denom.is_constant() or denom.const == 0:
            raise NotQuadratic("division by a non-constant (rational function)")
        return _scale(extract(node["l"]), Fraction(1) / denom.const)
    raise NotQuadratic(f"node {k!r} is not in the quadratic fragment")


# ── LP assembly ──────────────────────────────────────────────────────────────
def build_leaf_lp(model: dict, lo: list[Fraction], hi: list[Fraction]) -> dict:
    """Build the exact McCormick relaxation LP of *model* over box ``[lo, hi]``.

    Returns ``{"A", "b", "c", "obj_const", "aux", "n_orig", "n_total"}`` for
    ``min c·z + obj_const s.t. A z >= b`` (z free), whose optimum is a valid lower
    bound on the model objective over the box. Raises :class:`NotQuadratic` if any
    body is outside the degree-<=2 fragment.
    """
    n_orig = len(model["columns"])
    objq = extract(model["objective"]["body"])
    consq = [(c, extract(c["body"])) for c in model["constraints"]]

    # Assign an aux column to every distinct quadratic term across obj + constraints.
    aux_index: dict[tuple[int, int], int] = {}
    aux: list[dict] = []

    def _aux_col(key):
        if key not in aux_index:
            w = n_orig + len(aux)
            aux_index[key] = w
            i, j = key
            aux.append(
                {"op": "square", "x": i, "w": w}
                if i == j
                else {"op": "bilinear", "x": i, "y": j, "w": w}
            )
        return aux_index[key]

    for q in [objq] + [q for _, q in consq]:
        for key in q.quad:
            _aux_col(key)
    n_total = n_orig + len(aux)

    def _lift(q: QuadForm) -> list[Fraction]:
        row = [Fraction(0)] * n_total
        for i, v in q.lin.items():
            row[i] += v
        for key, v in q.quad.items():
            row[aux_index[key]] += v
        return row

    c = _lift(objq)
    obj_const = objq.const

    A: list[list[Fraction]] = []
    b: list[Fraction] = []

    def _ge(row, rhs):
        A.append(row)
        b.append(rhs)

    def _le(row, rhs):  # row·z <= rhs  <=>  -row·z >= -rhs
        A.append([-x for x in row])
        b.append(-rhs)

    # Model constraints (lifted): body sense rhs, folding the body's constant.
    for con, q in consq:
        row = _lift(q)
        rhs = as_fraction(con["rhs"]) - q.const
        sense = con["sense"]
        if sense == "le":
            _le(row, rhs)
        elif sense == "ge":
            _ge(row, rhs)
        else:  # eq -> both
            _ge(row, rhs)
            _le(row, rhs)

    # Box bounds on the original variables.
    for i in range(n_orig):
        if lo[i] is not None:
            e = [Fraction(0)] * n_total
            e[i] = Fraction(1)
            _ge(e, lo[i])
        if hi[i] is not None:
            e = [Fraction(0)] * n_total
            e[i] = Fraction(1)
            _le(e, hi[i])

    # McCormick envelope rows for each aux product over the box.
    for t in aux:
        if t["op"] == "square":
            rows = mccormick_square(lo[t["x"]], hi[t["x"]], t["x"], t["w"])
        else:
            rows = mccormick_bilinear(
                lo[t["x"]], hi[t["x"]], lo[t["y"]], hi[t["y"]], t["x"], t["y"], t["w"]
            )
        for coeffs, const, _sense in rows:  # coeffs·z + const >= 0  <=>  coeffs·z >= -const
            row = [Fraction(0)] * n_total
            for idx, v in coeffs.items():
                row[idx] = v
            _ge(row, -const)

    return {
        "A": A,
        "b": b,
        "c": c,
        "obj_const": obj_const,
        "aux": aux,
        "n_orig": n_orig,
        "n_total": n_total,
    }


# ── exact dual recovery (emitter side) ───────────────────────────────────────
def _basic_solution(cols: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
    """A basic solution ``y >= 0`` of ``Σ_k y_k cols[k] = rhs`` (or ``None``).

    Exact rational Gauss–Jordan over the columns: pick pivots to reach ``rhs`` in
    the column span; free variables are set to 0 (a vertex/basic solution). Returns
    the full ``y`` (length ``len(cols)``) if the resulting basic values are all
    ``>= 0``, else ``None``. Used to recover an exact dual from a float active set.
    """
    n = len(rhs)
    m = len(cols)
    # Augmented matrix M (n x (m+1)); column k is cols[k], last column is rhs.
    M = [[cols[k][r] for k in range(m)] + [rhs[r]] for r in range(n)]
    where = [-1] * m  # pivot row for each column, or -1
    row = 0
    for col in range(m):
        if row >= n:
            break
        piv = next((r for r in range(row, n) if M[r][col] != 0), None)
        if piv is None:
            continue
        M[row], M[piv] = M[piv], M[row]
        inv = Fraction(1) / M[row][col]
        M[row] = [x * inv for x in M[row]]
        for r in range(n):
            if r != row and M[r][col] != 0:
                f = M[r][col]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[row])]
        where[col] = row
        row += 1
    # Unsatisfiable rows: all-zero coefficients but nonzero rhs.
    for r in range(n):
        if all(M[r][k] == 0 for k in range(m)) and M[r][m] != 0:
            return None
    y = [Fraction(0)] * m
    for col in range(m):
        if where[col] != -1:
            y[col] = M[where[col]][m]
    if any(v < 0 for v in y):
        return None
    return y


def leaf_dual(lp: dict, tol: float = 1e-6) -> tuple[Fraction, list[Fraction]] | None:
    """Best-effort *exact* dual certifying a lower bound for ``lp`` (or ``None``).

    Solves ``min c·z s.t. A z >= b`` numerically (SciPy/HiGHS) to locate the active
    constraints, then recovers an exact rational dual ``y >= 0`` with ``Aᵀy = c``
    from that active set (:func:`_basic_solution`). Returns ``(bound, y)`` with
    ``bound = b·y + obj_const`` -- a valid lower bound by weak duality -- verified
    exactly before returning. ``None`` if SciPy is unavailable, the solve fails, or
    no exact nonnegative dual is recovered (the caller then keeps the trusted bound).
    """
    try:
        import numpy as np
        from scipy.optimize import linprog
    except Exception:
        return None
    A, b, c = lp["A"], lp["b"], lp["c"]
    n = lp["n_total"]
    Af = np.array([[float(x) for x in row] for row in A], dtype=float)
    bf = np.array([float(x) for x in b], dtype=float)
    cf = np.array([float(x) for x in c], dtype=float)
    try:
        res = linprog(cf, A_ub=-Af, b_ub=-bf, bounds=[(None, None)] * n, method="highs")
    except Exception:
        return None
    if not res.success or res.x is None:
        return None
    z = res.x
    # Prefer the dual *support* from HiGHS marginals: at a degenerate vertex the
    # primal-active set is over-determined and an arbitrary basis can be dual-
    # infeasible (negative), whereas the marginals already identify a feasible
    # support. Fall back to the primal-active set if marginals are unavailable.
    support: list[int] = []
    marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
    if marg is not None:
        support = [i for i in range(len(A)) if abs(float(marg[i])) > tol]
    if not support:
        support = [i for i in range(len(A)) if abs(float(Af[i] @ z) - float(bf[i])) <= tol]
    if not support:
        return None
    # Solve Σ_{i in support} y_i A_i = c exactly (columns are the support rows).
    cols = [[A[i][j] for j in range(n)] for i in support]  # each col length n
    y_support = _basic_solution(cols, list(c))
    if y_support is None:
        return None
    y = [Fraction(0)] * len(A)
    for k, i in enumerate(support):
        y[i] = y_support[k]
    # Exact verification (never trust the numeric path): y >= 0 and Aᵀy = c.
    from .bnb import lp_lower_bound

    ok, bound = lp_lower_bound(A, b, c, y)
    if not ok:
        return None
    return bound + lp["obj_const"], y
