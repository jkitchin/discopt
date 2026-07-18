"""Tier-3 (spatial branch-and-bound) checker primitives — the soundness kernel.

Tier 3 certifies *global optimality for nonconvex models*. The proof is a
branch-and-bound tree whose leaf boxes cover the root box; every leaf is fathomed
either by **infeasibility** (its relaxation is empty — a Farkas ray) or by
**bound** (its relaxation's certified lower bound is >= the incumbent). Then no
feasible point beats the incumbent, and ``min`` of the leaf bounds is a valid
global dual bound.

The load-bearing soundness checks are all **exact-rational and model-agnostic**,
which is what this module provides and unit-tests:

* :func:`check_tree_covers` — the branch tree's leaf boxes cover the root box
  (spatial split ``x<=s | x>=s``; integer split ``x<=floor | x>=ceil``).
* :func:`mccormick_bilinear` / :func:`mccormick_square` — the closed-form McCormick
  envelope rows for ``w = x*y`` / ``w = x^2`` over a box, which the checker
  *recomputes* from the box (so a tampered relaxation coefficient is caught), plus
  a sampling validity witness.
* :func:`lp_lower_bound` — a certified lower bound on an LP ``min c·z s.t. Az>=b``
  from a dual-feasible ``y>=0`` by weak duality (``Aᵀy=c`` ⇒ bound ``b·y``).
* :func:`farkas_infeasible` — a Farkas ray certifying ``Az>=b`` is empty.

These compose into :func:`certified_leaf_bound` and (schema-level)
:func:`check_bnb_certificate`. The *emitter* that records the tree + per-leaf
duals from a real solve is a separate engineering task (a Rust B&B recorder — see
``docs/dev/lean-certificate-plan.md`` §12); this module is the checker it targets,
and the Lean port's Tier-3 soundness theorem must match it.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from .schema import as_fraction

Row = tuple  # (coeffs: dict[int, Fraction], const: Fraction, sense: "ge")


# ── box covering ─────────────────────────────────────────────────────────────
def _box(node_box: dict) -> tuple[list[Optional[Fraction]], list[Optional[Fraction]]]:
    lo = [as_fraction(v) for v in node_box["lb"]]
    hi = [as_fraction(v) for v in node_box["ub"]]
    return lo, hi


def check_tree_covers(tree: dict, root_id, integer_cols: set[int]) -> tuple[bool, str]:
    """Verify every branch node's children exactly split its box (so leaves cover root).

    ``tree`` maps node id -> ``{"box": {"lb":[..], "ub":[..]}, "kind": "branch"|"leaf",
    "branch": {"var": j, "point": [n,d], "children": [l, r]}}``. Covering follows by
    induction: if each branch's two children reproduce the parent box with one
    coordinate split, the frontier of leaves covers the root box.

    Spatial split at point ``s`` on column ``j``: left ``ub[j]=s``, right ``lb[j]=s``.
    Integer split: left ``ub[j]=floor(s)``, right ``lb[j]=floor(s)+1`` (the open gap
    ``(floor(s), floor(s)+1)`` is sound only because column ``j`` is integer).
    """
    seen: set = set()

    def visit(nid) -> tuple[bool, str]:
        if nid in seen:
            return False, f"cycle at node {nid}"
        seen.add(nid)
        node = tree.get(nid)
        if node is None:
            return False, f"missing node {nid}"
        if node["kind"] == "leaf":
            return True, ""
        br = node["branch"]
        j = br["var"]
        s = as_fraction(br["point"])
        plo, phi = _box(node["box"])
        kids = br["children"]
        if len(kids) != 2:
            return False, f"node {nid}: branch must have 2 children"
        llo, lhi = _box(tree[kids[0]]["box"])
        rlo, rhi = _box(tree[kids[1]]["box"])
        # Both children equal the parent box except on column j.
        for i in range(len(plo)):
            if i == j:
                continue
            if llo[i] != plo[i] or lhi[i] != phi[i] or rlo[i] != plo[i] or rhi[i] != phi[i]:
                return False, f"node {nid}: child differs from parent off split column at {i}"
        # Split on column j.
        if j in integer_cols:
            fl = Fraction(s.numerator // s.denominator)  # floor
            want = llo[j] == plo[j] and lhi[j] == fl and rlo[j] == fl + 1 and rhi[j] == phi[j]
            if not want:
                return False, f"node {nid}: integer split on col {j} not [.,{fl}]|[{fl + 1},.]"
        else:
            if not (llo[j] == plo[j] and lhi[j] == s and rlo[j] == s and rhi[j] == phi[j]):
                return False, f"node {nid}: spatial split on col {j} not [.,{s}]|[{s},.]"
        for k in kids:
            ok, why = visit(k)
            if not ok:
                return False, why
        return True, ""

    return visit(root_id)


# ── McCormick envelopes (closed form in the box) ─────────────────────────────
def mccormick_bilinear(
    xl: Fraction, xh: Fraction, yl: Fraction, yh: Fraction, ix: int, iy: int, iw: int
) -> list[Row]:
    """The 4 McCormick rows for ``w = x*y`` over ``[xl,xh]x[yl,yh]`` (all valid).

    Each row is ``(coeffs, const, "ge")`` meaning ``sum(coeffs)·vars + const >= 0``,
    with column indices ``ix`` (x), ``iy`` (y), ``iw`` (w).
    """
    return [
        # w >= yl*x + xl*y - xl*yl   (under)
        ({ix: -yl, iy: -xl, iw: Fraction(1)}, xl * yl, "ge"),
        # w >= yh*x + xh*y - xh*yh   (under)
        ({ix: -yh, iy: -xh, iw: Fraction(1)}, xh * yh, "ge"),
        # w <= yl*x + xh*y - xh*yl   (over)
        ({ix: yl, iy: xh, iw: Fraction(-1)}, -xh * yl, "ge"),
        # w <= yh*x + xl*y - xl*yh   (over)
        ({ix: yh, iy: xl, iw: Fraction(-1)}, -xl * yh, "ge"),
    ]


def mccormick_square(xl: Fraction, xh: Fraction, ix: int, iw: int) -> list[Row]:
    """McCormick rows for ``w = x^2`` over ``[xl,xh]``: two tangents + one secant."""
    return [
        # w >= 2*xl*x - xl^2   (tangent under)
        ({ix: -2 * xl, iw: Fraction(1)}, xl * xl, "ge"),
        # w >= 2*xh*x - xh^2   (tangent under)
        ({ix: -2 * xh, iw: Fraction(1)}, xh * xh, "ge"),
        # w <= (xl+xh)*x - xl*xh   (secant over)
        ({ix: xl + xh, iw: Fraction(-1)}, -xl * xh, "ge"),
    ]


def row_holds(row: Row, point: dict[int, Fraction]) -> bool:
    """Evaluate a ``(coeffs, const, "ge")`` row at a point ``{col: value}`` (>= 0?)."""
    coeffs, const, _ = row
    return sum(c * point[i] for i, c in coeffs.items()) + const >= 0


# ── LP weak duality / Farkas ─────────────────────────────────────────────────
def lp_lower_bound(
    a: list[list[Fraction]], b: list[Fraction], c: list[Fraction], y: list[Fraction]
) -> tuple[bool, object]:
    """Certified lower bound on ``min c·z s.t. A z >= b`` (z free) via weak duality.

    A dual-feasible ``y`` (``y >= 0`` and ``Aᵀy = c``) proves ``c·z >= b·y`` for every
    primal-feasible ``z``, so ``b·y`` is a valid lower bound on the optimum.
    Returns ``(True, bound)`` or ``(False, reason)``. Exact rationals throughout.
    """
    m, n = len(a), len(c)
    if len(y) != m or len(b) != m:
        return False, "dimension mismatch"
    for i in range(m):
        if y[i] < 0:
            return False, f"dual infeasible: y[{i}] = {y[i]} < 0"
    for j in range(n):
        col = sum(a[i][j] * y[i] for i in range(m))
        if col != c[j]:
            return False, f"Aᵀy != c at column {j}: {col} != {c[j]}"
    return True, sum(b[i] * y[i] for i in range(m))


def farkas_infeasible(
    a: list[list[Fraction]], b: list[Fraction], y: list[Fraction]
) -> tuple[bool, str]:
    """Verify a Farkas ray: ``A z >= b`` is empty because ``y>=0, Aᵀy=0, b·y>0``."""
    m, n = len(a), (len(a[0]) if a else 0)
    if len(y) != m or len(b) != m:
        return False, "dimension mismatch"
    for i in range(m):
        if y[i] < 0:
            return False, f"y[{i}] < 0"
    for j in range(n):
        if sum(a[i][j] * y[i] for i in range(m)) != 0:
            return False, f"Aᵀy != 0 at column {j}"
    if sum(b[i] * y[i] for i in range(m)) <= 0:
        return False, "b·y <= 0 (no infeasibility)"
    return True, ""


# ── leaf-level composition ───────────────────────────────────────────────────
def _dense(row: Row, ncols: int) -> tuple[tuple[Fraction, ...], Fraction]:
    """A ``(coeffs, const, "ge")`` row as a dense ``(A_row, b)`` for ``A z >= b``."""
    coeffs, const, _ = row
    a = tuple(coeffs.get(i, Fraction(0)) for i in range(ncols))
    return a, -const  # sum coeffs·z + const >= 0  <=>  A·z >= -const


def _allowed_rows(ncols: int, box: tuple[list, list], aux_terms: list[dict]) -> set:
    """The dense ``(A_row, b)`` rows that are *provably valid* for this leaf.

    Valid rows are (a) the box bounds ``x_j >= lo_j`` / ``x_j <= hi_j`` and (b) the
    closed-form McCormick rows for every declared auxiliary product/square over the
    box. Any LP row outside this set could be an unsound cut, so the checker
    rejects a relaxation that contains one.
    """
    lo, hi = box
    allowed: set = set()
    for j in range(ncols):
        if lo[j] is not None:
            a = [Fraction(0)] * ncols
            a[j] = Fraction(1)
            allowed.add((tuple(a), lo[j]))  # x_j >= lo_j
        if hi[j] is not None:
            a = [Fraction(0)] * ncols
            a[j] = Fraction(-1)
            allowed.add((tuple(a), -hi[j]))  # -x_j >= -hi_j
    for t in aux_terms:
        if t["op"] == "square":
            rows = mccormick_square(lo[t["x"]], hi[t["x"]], t["x"], t["w"])
        elif t["op"] == "bilinear":
            rows = mccormick_bilinear(
                lo[t["x"]], hi[t["x"]], lo[t["y"]], hi[t["y"]], t["x"], t["y"], t["w"]
            )
        else:
            continue
        for r in rows:
            allowed.add(_dense(r, ncols))
    return allowed


def relaxation_rows_valid(
    a: list[list[Fraction]], b: list[Fraction], box: tuple[list, list], aux_terms: list[dict]
) -> tuple[bool, str]:
    """Every LP row is a box bound or a valid McCormick row for a declared term.

    This is the anti-unsoundness gate: it guarantees the relaxation LP removes no
    point of the true feasible set, so its optimum is a genuine lower bound. (The
    complementary obligation -- that the LP's objective and the model's original
    constraints are faithfully lifted -- is supplied by the emitter/recorder; see
    the plan doc §12.)
    """
    ncols = len(box[0])
    allowed = _allowed_rows(ncols, box, aux_terms)
    for i, row in enumerate(a):
        if (tuple(row), b[i]) not in allowed:
            return (
                False,
                f"LP row {i} is not a recognized valid relaxation row (possible unsound cut)",
            )
    return True, ""


def certified_leaf_bound(
    box: tuple[list, list],
    lp: dict,
    dual: list[Fraction],
    aux_terms: list[dict],
) -> tuple[bool, object]:
    """Certify a bound-fathomed leaf: valid relaxation rows + a weak-duality bound.

    ``lp`` is ``{"A", "b", "c"}`` over the lifted variables; ``dual`` is the LP dual.
    Returns ``(True, leaf_lower_bound)`` or ``(False, reason)``.
    """
    ok, why = relaxation_rows_valid(lp["A"], lp["b"], box, aux_terms)
    if not ok:
        return False, why
    return lp_lower_bound(lp["A"], lp["b"], lp["c"], dual)
