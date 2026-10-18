"""Differential soundness battery for GDP lowerings (#1775 G0.3, spec §6.1).

A GDP lowering replaces a disjunctive set ``S`` over the model's columns ``x`` by
algebraic rows over ``(x, a)``, where ``a`` are the auxiliaries it adds (selector
binaries, hull disaggregated copies, simplex weights). It is *exact* on ``x`` when
the projection of the lowered set onto ``x`` equals ``S``. This module checks both
directions on sampled points:

* **no feasible point is cut** — every sampled ``x ∈ S`` extends to some ``a``
  that satisfies every lowered row;
* **no infeasible point is admitted** — no sampled ``x ∉ S`` (by a clear margin)
  extends to any ``a``.

The oracle for ``x ∈ S`` is the shape's own predicate, written by hand from the
declared disjunction. It is cross-checked on every sampled point against
:func:`discopt.validation.feasibility.verify_point` on the *declared* model (which
reads the logic through ``DeclaredLogic``, #1659), so a predicate that disagrees
with the declared model fails loudly instead of testing the wrong set.

The extension question is answered without trusting the solver under test:

1. every integer auxiliary is **enumerated** (refused above ``MAX_ENUM_BITS``);
2. with ``x`` and the integers fixed, continuous auxiliaries are handled by an
   **LP** that minimises the largest row violation ``t`` — exact whenever every
   lowered row is affine in ``a`` at the fixed point, which is checked
   numerically, not assumed (the hull and simplex rows of a linear disjunct are);
3. when a lowered row is nonlinear in a continuous auxiliary (the hull of a
   nonlinear disjunct), the rows that are affine in ``a`` form a relaxation: its
   LP rejects exactly, and when its feasible ``a`` is a single point (checked
   with one min/max LP per auxiliary) that point decides admission. For a hull
   it always is — the bound rows pin an inactive disjunct's copies to 0 and the
   aggregation rows pin the active one's to ``x``. A point that neither decides
   raises; the battery never falls back to the solver it is judging. These
   verdicts are counted as ``nonlinear_verdicts``.

Every check bumps a counter; :func:`run_shape` returns them and the pytest wrapper
asserts each is positive, so a shape that sampled nothing cannot read as a pass
(CLAUDE.md §6). Running this file directly runs the whole battery and exits
non-zero when no comparison executed.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Lowerings ``reformulate_gdp`` implements. Kept literal (not read from
#: ``GDP_REFORMULATION_METHODS``) so a method added there without a battery run
#: shows up as a mismatch in ``test_battery_covers_every_method``.
METHODS: tuple[str, ...] = ("big-m", "mbigm", "hull", "simplex", "auto")

#: Integer auxiliaries are enumerated; above this many the shape is too large for
#: an exhaustive extension check and is refused rather than sampled.
MAX_ENUM_BITS = 14

#: A sampled point counts as infeasible only when the predicate still rejects it
#: with every row relaxed by this much. Points between "feasible exactly" and
#: "infeasible with margin" are not classified (counted as ``skipped``).
INFEASIBLE_MARGIN = 1e-3

#: Largest row violation the extension LP may leave for the point to count as
#: admitted. Matches ``validation.feasibility.ABS_TOL``.
ADMIT_TOL = 1e-6


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

Values = Mapping[str, Any]
Predicate = Callable[[Values, float], bool]


@dataclass(frozen=True)
class Shape:
    """One disjunctive set and its hand-written membership predicate.

    ``build`` returns a fresh declared model. ``member(v, tol)`` decides whether
    the point whose variable values are ``v`` (keyed by variable name; scalars as
    floats, arrays as numpy arrays) lies in the declared set with every row
    relaxed by ``tol``. ``points`` are extra points (same mapping form) added to
    the random sample — equalities and boundaries are hit only that way.

    ``refuses`` maps a method to ``(exception type, message regex)`` when the
    method is *meant* to refuse this shape (an unbounded body, §0.2 item 5). A
    refusal of a shape the method should eventually support is not listed here;
    the pytest wrapper marks it ``xfail(strict=True)`` with its issue instead.
    ``box`` gives a finite *sampling* box for a variable declared unbounded; the
    model itself is not changed. ``crosscheck=False`` skips the ``verify_point``
    cross-check of the predicate; it exists only for a shape on which that
    verifier is itself known broken (so the lowering can still be judged), and
    such a shape must name the issue in its docstring.
    """

    name: str
    build: Callable[[], Any]
    member: Predicate
    points: Sequence[Values] = ()
    refuses: Mapping[str, tuple[type[BaseException], str]] = field(default_factory=dict)
    n_samples: int = 60
    box: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    crosscheck: bool = True


@dataclass
class Report:
    shape: str
    method: str
    feasible_checked: int = 0
    infeasible_checked: int = 0
    skipped: int = 0
    oracle_crosschecks: int = 0
    nonlinear_verdicts: int = 0
    refused: str | None = None
    cut: list[dict] = field(default_factory=list)
    admitted: list[dict] = field(default_factory=list)

    @property
    def comparisons(self) -> int:
        return self.feasible_checked + self.infeasible_checked

    def assert_sound(self) -> None:
        msgs = []
        if self.cut:
            msgs.append(f"{len(self.cut)} feasible point(s) cut, e.g. {self.cut[0]}")
        if self.admitted:
            msgs.append(
                f"{len(self.admitted)} infeasible point(s) admitted, e.g. {self.admitted[0]}"
            )
        if msgs:
            raise AssertionError(f"{self.shape} under {self.method!r}: " + "; ".join(msgs))


# ---------------------------------------------------------------------------
# Point handling
# ---------------------------------------------------------------------------


def _columns(model) -> list[tuple[Any, int, int]]:
    out, off = [], 0
    for v in model._variables:
        n = int(v.size)
        out.append((v, off, off + n))
        off += n
    return out


def _values(model, x: np.ndarray) -> dict[str, Any]:
    vals: dict[str, Any] = {}
    for v, a, b in _columns(model):
        vals[v.name] = float(x[a]) if v.shape == () else x[a:b].reshape(v.shape).copy()
    return vals


def _flat(model, vals: Values) -> np.ndarray:
    parts = []
    for v, a, b in _columns(model):
        parts.append(np.ravel(np.asarray(vals[v.name], dtype=np.float64)))
        if parts[-1].size != b - a:
            raise ValueError(f"point gives {parts[-1].size} values for {v.name!r} ({b - a})")
    return np.concatenate(parts) if parts else np.zeros(0)


def _is_integer(v) -> bool:
    from discopt.modeling.core import VarType

    return v.var_type in (VarType.BINARY, VarType.INTEGER)


def _box(model, override: Mapping[str, tuple[float, float]] | None = None):
    """Flat ``(lb, ub)``; ``override`` replaces a variable's box (sampling only)."""
    override = override or {}
    lbs, ubs = [], []
    for v in model._variables:
        lo, hi = override.get(v.name, (v.lb, v.ub))
        lbs.append(np.ravel(np.broadcast_to(np.asarray(lo, dtype=np.float64), v.shape)))
        ubs.append(np.ravel(np.broadcast_to(np.asarray(hi, dtype=np.float64), v.shape)))
    return np.concatenate(lbs), np.concatenate(ubs)


def _require_finite(lb, ub, what: str) -> None:
    if not (np.all(np.abs(lb) < 1e15) and np.all(np.abs(ub) < 1e15)):
        raise ValueError(f"{what} needs a finite box; give the shape a sampling ``box``")


def sample_points(model, shape: Shape, rng: np.random.Generator) -> list[np.ndarray]:
    """Random box points (half snapped to a 0.5 grid so boundaries are hit) + extras."""
    lb, ub = _box(model, shape.box)
    _require_finite(lb, ub, f"sampling {shape.name!r}")
    int_mask = np.concatenate(
        [np.full(int(v.size), _is_integer(v)) for v in model._variables]
    ).astype(bool)
    pts = []
    for k in range(shape.n_samples):
        x = lb + rng.random(lb.size) * (ub - lb)
        if k % 2:
            x = np.clip(np.round(x * 2.0) / 2.0, lb, ub)
        x[int_mask] = np.round(x[int_mask])
        pts.append(x)
    pts.extend(_flat(model, p) for p in shape.points)
    return pts


# ---------------------------------------------------------------------------
# Extension check: does x extend to a point of the lowered model?
# ---------------------------------------------------------------------------


class _Undefined(Exception):
    """A lowered row is non-finite at ``x`` for every probed auxiliary value.

    The row is undefined there, so no solver can use the point: on a feasible
    ``x`` that is a cut. It is not a proof of rejection, so on an infeasible
    ``x`` the driver raises instead of counting it as rejected.
    """


class _Lowered:
    """The lowered model with the declared columns as a prefix of its columns."""

    def __init__(self, declared, lowered):
        from discopt._tape_nlp_evaluator import make_evaluator

        cols = _columns(lowered)
        n_decl = len(declared._variables)
        if [v for v, _, _ in cols[:n_decl]] != list(declared._variables):
            raise AssertionError("lowering did not keep the declared columns as a prefix")
        self.model = lowered
        self.n_x = cols[n_decl - 1][2] if n_decl else 0
        aux = cols[n_decl:]
        self.int_cols = [i for v, a, b in aux if _is_integer(v) for i in range(a, b)]
        self.cont_cols = [i for v, a, b in aux if not _is_integer(v) for i in range(a, b)]
        if len(self.int_cols) > MAX_ENUM_BITS:
            raise ValueError(
                f"{len(self.int_cols)} integer auxiliaries; the battery enumerates at most "
                f"{MAX_ENUM_BITS}"
            )
        self.lb, self.ub = _box(lowered)
        aux_cols = self.int_cols + self.cont_cols
        _require_finite(self.lb[aux_cols], self.ub[aux_cols], "the auxiliary columns")
        self.ev = make_evaluator(lowered) if lowered._constraints else None
        self.senses, self.rhs = self._rows()

    def _rows(self):
        if self.ev is None:
            return np.zeros(0, dtype=object), np.zeros(0)
        senses, rhs = [], []
        for start, stop, con in self.ev.constraint_row_map():
            senses += [con.sense] * (stop - start)
            rhs += [float(getattr(con, "rhs", 0.0) or 0.0)] * (stop - start)
        return np.array(senses, dtype=object), np.array(rhs)

    def _signed(self, g: np.ndarray) -> np.ndarray:
        """Per row, the violation as a signed list of ``<= 0`` residuals (== gives two)."""
        r = g - self.rhs
        le = self.senses == "<="
        ge = self.senses == ">="
        eq = self.senses == "=="
        if not np.all(le | ge | eq):
            raise ValueError(f"unknown constraint sense in {set(self.senses.tolist())}")
        return np.concatenate([r[le], -r[ge], r[eq], -r[eq]])

    def _signed_jac(self, J: np.ndarray) -> np.ndarray:
        le = self.senses == "<="
        ge = self.senses == ">="
        eq = self.senses == "=="
        return np.vstack([J[le], -J[ge], J[eq], -J[eq]])

    def _max_violation(self, z: np.ndarray) -> float:
        if self.ev is None:
            return 0.0
        g = np.asarray(self.ev.evaluate_constraints(z), dtype=np.float64)
        if not np.all(np.isfinite(g)):
            return float("inf")
        return float(max(0.0, np.max(self._signed(g), initial=0.0)))

    def extends(self, x: np.ndarray, rep: Report) -> tuple[bool, str]:
        """``(True, how)`` when some auxiliary assignment satisfies every row at ``x``."""
        # ``x`` is sampled from the box declared BEFORE lowering. A lowering that
        # tightened a declared column's bounds excludes whatever it cut off.
        lo_x, hi_x = self.lb[: self.n_x], self.ub[: self.n_x]
        tol = ADMIT_TOL * np.maximum(1.0, np.abs(x))
        if np.any(x < lo_x - tol) or np.any(x > hi_x + tol):
            return False, "outside the lowered model's column bounds"
        lo, hi = self.lb[self.int_cols], self.ub[self.int_cols]
        ranges = [range(int(np.ceil(a)), int(np.floor(b)) + 1) for a, b in zip(lo, hi)]
        undefined = None
        for ints in itertools.product(*ranges):
            z = np.empty(self.lb.size)
            z[: self.n_x] = x
            z[self.int_cols] = ints
            if not self.cont_cols:
                if self._max_violation(z) <= ADMIT_TOL:
                    return True, f"integers={list(ints)}"
                continue
            try:
                ok, how = self._continuous(z, rep)
            except _Undefined as exc:
                undefined = f"integers={list(ints)}: {exc}"
                continue
            if ok:
                return True, f"integers={list(ints)} {how}"
        if undefined is not None:
            # Probing is not a proof, so this may only ever report a cut.
            raise _Undefined(f"no assignment extends; {undefined}")
        return False, "no auxiliary assignment"

    def _continuous(self, z: np.ndarray, rep: Report) -> tuple[bool, str]:
        c = self.cont_cols
        a_lo, a_hi = self.lb[c], self.ub[c]
        z0 = z.copy()
        z0[c] = 0.5 * (a_lo + a_hi)
        g0 = np.asarray(self.ev.evaluate_constraints(z0), dtype=np.float64)
        J0 = np.asarray(self.ev.evaluate_jacobian(z0), dtype=np.float64)[:, c]
        rng = np.random.default_rng(0)
        if not np.all(np.isfinite(g0)):
            dead = ~np.isfinite(g0)
            for _ in range(3):
                z1 = z0.copy()
                z1[c] = a_lo + rng.random(len(c)) * (a_hi - a_lo)
                dead &= ~np.isfinite(np.asarray(self.ev.evaluate_constraints(z1), dtype=float))
            if np.any(dead):
                raise _Undefined(
                    f"rows {np.nonzero(dead)[0].tolist()} non-finite at every probed a"
                )
        affine = np.all(np.isfinite(g0)) and np.all(np.isfinite(J0))
        for _ in range(3):
            if not affine:
                break
            z1 = z0.copy()
            z1[c] = a_lo + rng.random(len(c)) * (a_hi - a_lo)
            g1 = np.asarray(self.ev.evaluate_constraints(z1), dtype=np.float64)
            pred = g0 + J0 @ (z1[c] - z0[c])
            affine = np.all(np.isfinite(g1)) and np.allclose(g1, pred, rtol=1e-9, atol=1e-9)
        if affine:
            ok, how, _ = self._lp(z0, g0, J0)
            return ok, how
        rep.nonlinear_verdicts += 1
        # Nonlinear in ``a``. The rows that ARE affine in ``a`` (checked per row
        # below) form a relaxation: if even they cannot be met, the point is
        # rejected exactly. Otherwise their LP solution is tried as the witness
        # (for a hull it is unique: the bound rows pin an inactive disjunct's
        # copies to 0 and the aggregation rows pin the active one's to x).
        rows = self._affine_rows(z0, g0, J0)
        ok, how, a_star = self._lp(z0, g0, J0, rows=rows)
        if not ok:
            return False, f"affine-row relaxation infeasible ({how})"
        z1 = z.copy()
        z1[c] = a_star
        v = self._max_violation(z1)
        if v <= ADMIT_TOL:
            return True, f"affine-row LP witness, max violation {v:.2e}"
        if self._pinned(z0, g0, J0, rows):
            # The affine rows leave exactly one ``a``; it violates a nonlinear row.
            return False, f"unique affine-row witness violates by {v:.2e}"
        raise RuntimeError(
            "a lowered row is nonlinear in the continuous auxiliaries and the affine rows "
            "do not pin them; the battery has no solver-free verdict for this point"
        )

    def _pinned(self, z0, g0, J0, rows) -> bool:
        """True when the affine rows (met exactly) admit a single ``a`` (width <= 1e-9)."""
        from scipy.optimize import linprog

        c = self.cont_cols
        g0 = np.where(rows, g0, self.rhs)
        J0 = np.where(rows[:, None], J0, 0.0)
        A = self._signed_jac(J0)
        b = -(self._signed(g0) - A @ z0[c]) + ADMIT_TOL
        bounds = list(zip(self.lb[c], self.ub[c]))
        for j in range(len(c)):
            e = np.zeros(len(c))
            e[j] = 1.0
            lo = linprog(e, A_ub=A, b_ub=b, bounds=bounds, method="highs")
            hi = linprog(-e, A_ub=A, b_ub=b, bounds=bounds, method="highs")
            if lo.status != 0 or hi.status != 0:
                return False
            # ADMIT_TOL of row slack moves a pinned coordinate by O(ADMIT_TOL).
            if -hi.fun - lo.fun > 10 * ADMIT_TOL:
                return False
        return True

    def _affine_rows(self, z0, g0, J0) -> np.ndarray:
        """Rows whose value is affine in the continuous auxiliaries at fixed ``x``."""
        c = self.cont_cols
        rng = np.random.default_rng(1)
        keep = np.isfinite(g0) & np.all(np.isfinite(J0), axis=1)
        for _ in range(3):
            z1 = z0.copy()
            z1[c] = self.lb[c] + rng.random(len(c)) * (self.ub[c] - self.lb[c])
            g1 = np.asarray(self.ev.evaluate_constraints(z1), dtype=np.float64)
            pred = g0 + J0 @ (z1[c] - z0[c])
            with np.errstate(invalid="ignore"):
                keep &= np.isfinite(g1) & np.isclose(g1, pred, rtol=1e-9, atol=1e-9)
        return keep

    def _lp(self, z0, g0, J0, rows=None) -> tuple[bool, str, np.ndarray]:
        from scipy.optimize import linprog

        c = self.cont_cols
        if rows is not None:
            g0 = np.where(rows, g0, self.rhs)  # dropped rows: residual 0 at every a
            J0 = np.where(rows[:, None], J0, 0.0)
        base = self._signed(g0) - self._signed_jac(J0) @ z0[c]
        A = self._signed_jac(J0)
        # min t  s.t.  A a + base <= t,  a in box,  t >= 0
        n = len(c)
        res = linprog(
            np.r_[np.zeros(n), 1.0],
            A_ub=np.hstack([A, -np.ones((A.shape[0], 1))]),
            b_ub=-base,
            bounds=[*zip(self.lb[c], self.ub[c]), (0.0, None)],
            method="highs",
        )
        if res.status != 0:
            raise RuntimeError(f"max-violation LP failed: {res.message}")
        return bool(res.fun <= ADMIT_TOL), f"LP max violation {res.fun:.2e}", res.x[:n]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_shape(shape: Shape, method: str, *, seed: int = 1775) -> Report:
    """Run ``shape`` under ``method`` and return the counted report (not asserted)."""
    import re

    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt.validation.feasibility import verify_point

    rep = Report(shape.name, method)
    declared = shape.build()
    # Sample against the declared box before lowering, so an in-place bound
    # change by the lowering cannot hide the region it cut from the sample.
    points = sample_points(declared, shape, np.random.default_rng(seed))
    expected = shape.refuses.get(method)
    try:
        lowered = reformulate_gdp(declared, method)
    except Exception as exc:  # noqa: BLE001 - classified right below, never swallowed
        if (
            expected is not None
            and isinstance(exc, expected[0])
            and re.search(expected[1], str(exc))
        ):
            rep.refused = f"{type(exc).__name__}: {exc}"
            return rep
        raise
    if expected is not None:
        raise AssertionError(f"{shape.name}: {method!r} was expected to refuse, but lowered")
    if lowered is declared:
        raise AssertionError(f"{shape.name}: the model declares no GDP; nothing was lowered")

    low = _Lowered(declared, lowered)
    for x in points:
        vals = _values(declared, x)
        inside = shape.member(vals, 0.0)
        outside = not shape.member(vals, INFEASIBLE_MARGIN)
        if not (inside or outside):
            rep.skipped += 1
            continue
        verdict = verify_point(declared, x).ok if shape.crosscheck else inside
        rep.oracle_crosschecks += shape.crosscheck
        if verdict != inside:
            raise AssertionError(
                f"{shape.name}: the predicate says {'in' if inside else 'out'} but the declared "
                f"model's verify_point says {verdict} at {vals}"
            )
        try:
            ok, how = low.extends(x, rep)
        except _Undefined as exc:
            if not inside:
                raise RuntimeError(f"{shape.name}: no verdict at infeasible {vals}: {exc}") from exc
            ok, how = False, f"undefined: {exc}"
        if inside:
            rep.feasible_checked += 1
            if not ok:
                rep.cut.append({"point": vals, "why": how})
        else:
            rep.infeasible_checked += 1
            if ok:
                rep.admitted.append({"point": vals, "how": how})
    return rep


def main(
    shapes: Sequence[Shape],
    expected_failures: Mapping[tuple[str, str], tuple[str, type[BaseException]]],
) -> int:
    """Print one line per (shape, method); exit 1 on any unexpected result or no checks."""
    total = 0
    bad = 0
    for shape in shapes:
        for method in METHODS:
            xfail = expected_failures.get((shape.name, method))
            try:
                rep = run_shape(shape, method)
            except Exception as exc:  # noqa: BLE001 - printed and counted, never hidden
                tag = "xfail" if xfail and isinstance(exc, xfail[1]) else "ERROR"
                print(f"{shape.name:24s} {method:8s} {tag} {type(exc).__name__}: {exc}", flush=True)
                bad += tag == "ERROR"
                continue
            total += rep.comparisons
            status = "refused" if rep.refused else ("UNSOUND" if rep.cut or rep.admitted else "ok")
            if xfail:
                pinned = issubclass(AssertionError, xfail[1])
                status = "xfail" if status == "UNSOUND" and pinned else "XPASS"
            bad += status in ("UNSOUND", "XPASS")
            print(
                f"{shape.name:24s} {method:8s} {status:8s} feas={rep.feasible_checked:3d} "
                f"infeas={rep.infeasible_checked:3d} skipped={rep.skipped:3d} "
                f"nonlinear={rep.nonlinear_verdicts}",
                flush=True,
            )
    print(f"executed comparisons: {total}; unexpected results: {bad}", flush=True)
    return 1 if total == 0 or bad else 0


if __name__ == "__main__":
    from test_gdp_differential_battery import SHAPES, XFAIL

    sys.exit(main(SHAPES, XFAIL))
