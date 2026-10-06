"""HiGHS engine for pure LP / MILP solves (issue #1229).

Binding contract: ``docs/dev/lp-milp-highs-routing-plan.md`` §3. The principle is
that HiGHS's *labels* are never trusted for a certificate:

* every returned point passes a readback guard (no value at the ``1e20`` open-bound
  sentinel -- the #1229 failure class) and a feasibility check on the standard-form
  data before anything is computed from it;
* the objective is recomputed from the point, never read from HiGHS;
* an LP ``optimal`` is certified only by a Neumaier--Shcherbina safe bound computed
  in Rust (double-double ``Aᵀy``) from HiGHS's dual, closing the gap to the point;
  where roundoff leaves that bound at ``-inf``, over a rigorously widened FBBT box and
  then from an exact rational correction of the dual (labeled with its provenance);
* an LP ``infeasible`` needs a Farkas ray discopt verifies (declared or FBBT box) or a
  rigorous phase-1 bound above zero; an LP ``unbounded`` a primal ray and a feasible
  point discopt verifies;
* a MILP dual bound is HiGHS's ``mip_dual_bound`` (floating point, the same class as
  the Rust driver's), cross-checked against the NS-safe root LP bound.

Anything that fails a check is ``status="error"`` with the reason. There is no
fallback to another engine (§4): a fallback would hide exactly the failure class
this route exists to surface.

This module works on the standard form ``min cᵀx + d  s.t.  A x = b, l <= x <= u``
that :func:`discopt._relax.problem_classifier.extract_lp_data` produces; mapping
back to a ``SolveResult`` lives in ``discopt.solver``. ``highspy`` is imported
lazily so a default MINLP solve never loads it.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import logging
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

#: discopt's open-bound sentinel (CLAUDE.md: test the bound, never a product).
INF = 1e20
#: A point or dual at or above this magnitude is read back as a failure unless the
#: value sits exactly on a declared finite bound (the default continuous box is
#: ``±9.999e19``, and ``min -x`` over it is optimal at that corner -- #850/#937).
READBACK_LIMIT = 1e15
#: Per-row feasibility test ``|viol_i| <= FEAS_TOL + FEAS_RTOL * sum_j |A_ij||x_j|``:
#: the ``_matrix_solution_feasible`` convention.
FEAS_TOL = 1e-6
FEAS_RTOL = 1e-9
#: conftest integrality tolerance.
INT_TOL = 1e-5
#: HiGHS drops matrix entries with ``|a| <= small_matrix_value``; this is the smallest
#: value the option accepts (1e-13 is rejected, highspy 1.12).
SMALL_MATRIX_VALUE = 1e-12
#: An LP is ``optimal`` iff ``objective - ns_bound <= CERT_ABS + CERT_REL*|objective|``.
CERT_ABS = 1e-6
CERT_REL = 1e-9
#: Relative margin a ray descent must clear. A Farkas contradiction is checked against an
#: explicit bound on its own floating-point error instead -- see :func:`farkas_verified`.
RAY_REL = 1e-9
#: A ray entry this small (after scaling the ray to unit max-norm) is cleaned to
#: zero *before* verification, so the vector that is verified is the vector used.
RAY_CLEAN = 1e-14
#: HiGHS stores ``mip_max_nodes`` as a 32-bit int (probe: ``10**12`` -> kError).
_MAX_NODES_CAP = 2**31 - 1
#: HiGHS ``presolve_rule_off`` bit for ``kPresolveRuleParallelRowsAndCols`` (rule 13 of
#: ``PresolveRuleType`` in ``lp_data/HConst.h``; HiGHS logs it as "Rule 13 (bit 8192):
#: Parallel rows and columns"). The MILP route switches that rule off (#1634).
#:
#: Its parallel-column dominance test (``HPresolve::detectParallelRowsAndCols``) decides
#: "these two columns cost the same" by ``|c_col * colScale - c_dup| <=
#: dual_feasibility_tolerance``, an ABSOLUTE 1e-7 on a cost measured per unit of the
#: duplicate column, where ``colScale = a_dup / a_col`` is the ratio of the two columns'
#: coefficients. On the tie branch it then fixes the integer column at whichever bound the
#: other column can compensate for -- the direction the tie makes free and the true cost
#: does not. The route's own slack ``a x + s = b`` (cost 0, coefficient 1) is parallel to
#: every column once presolve leaves a single row, so a coefficient ratio of 4.9e5 turns
#: a real 0.0064-per-unit cost into 1.3e-8 per unit of ``s``: a "tie". Measured on the
#: #1634 seed-181 witness: at presolve step 5 HiGHS fixed ``x2 = 3`` (cost 0.0064 each),
#: certified 0.02082 with the true optimum 0.00801 strictly feasible (slack >= 72), and
#: every guard here passed. With this bit off all three #1634 witnesses certify the true
#: optimum and the generator panel recorded in the PR has no false certificate.
#:
#: Not a tolerance tweak: a smaller ``dual_feasibility_tolerance`` only moves the ratio at
#: which the same absolute test misfires. The reduction is unsound on badly scaled columns
#: and the route has no way to re-derive a bound HiGHS's presolve took away, so it is off.
_PRESOLVE_RULE_PARALLEL_ROWS_AND_COLS = 1 << 13
#: HiGHS ``presolve_rule_off`` bit for ``kPresolveRuleSparsify`` (rule 14). The MILP route
#: switches it off (#1667) because it is the trigger of a presolve cycle that never returns.
#:
#: On the #1667 unit-commitment hand-off (2904 rows, 4344 columns of this route's slack
#: form), ``h.run()`` stays in ``HPresolve::fastPresolveLoop`` -> ``rowPresolve`` forever.
#: A stack sample confirmed this. That loop polls neither ``time_limit`` nor the interrupt
#: callbacks, so ``Model.solve(time_limit=20)`` was still running after 120 s and the route
#: had no way to stop it. Bisected over ``presolve_rule_off``: turning off rule 9
#: (doubleton equation), 12 (aggregator) or 14 (sparsify) alone breaks the cycle
#: (``Optimal`` 450623.7692 in 5-6 s), and rules 7, 8, 10, 11, 15, 16 alone do not.
#: Sparsify is the one taken: it only re-expresses rows to cut fill, while 9 and 12
#: eliminate columns. On 23 HiGHS check instances in this route's standard form, sparsify
#: on and off gave the same status and objective on every instance (panel in the #1667 PR).
#: The upstream bug is tracked in #1671: re-measure and drop this bit once HiGHS fixes it.
_PRESOLVE_RULE_SPARSIFY = 1 << 14
#: Every ``presolve_rule_off`` bit the MILP route sets.
MILP_PRESOLVE_RULE_OFF = _PRESOLVE_RULE_PARALLEL_ROWS_AND_COLS | _PRESOLVE_RULE_SPARSIFY
#: The only terminal statuses a limit can produce.
_LIMIT_STATUSES = ("kTimeLimit", "kIterationLimit", "kSolutionLimit", "kInterrupt")


class HighsUnavailable(ImportError):
    """``highspy`` is missing, which for this route means a broken install."""


def require_highspy():
    try:
        import highspy
    except ImportError as err:
        raise HighsUnavailable(
            "the LP/MILP HiGHS route needs highspy>=1.10: pip install 'highspy>=1.10'"
        ) from err
    return highspy


@dataclass(frozen=True)
class StdForm:
    """``min cᵀx + obj_const  s.t.  A x = b, xl <= x <= xu`` with ``x_j`` integer for
    ``j in int_idx``. ``A`` is CSC with sorted indices and no stored zeros."""

    c: np.ndarray
    A: sp.csc_matrix
    b: np.ndarray
    xl: np.ndarray
    xu: np.ndarray
    obj_const: float = 0.0
    int_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def from_arrays(cls, c, A, b, xl, xu, obj_const=0.0, int_idx=None) -> "StdForm":
        c = np.ascontiguousarray(np.asarray(c, dtype=np.float64).ravel())
        n = c.shape[0]
        A = A if sp.issparse(A) else sp.csc_matrix(np.asarray(A, dtype=np.float64).reshape(-1, n))
        A = sp.csc_matrix(A, dtype=np.float64, copy=True)
        A.eliminate_zeros()
        A.sort_indices()
        b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
        xl = np.ascontiguousarray(np.asarray(xl, dtype=np.float64).ravel())
        xu = np.ascontiguousarray(np.asarray(xu, dtype=np.float64).ravel())
        ii = np.zeros(0, dtype=np.int64) if int_idx is None else np.asarray(int_idx, np.int64)
        if A.shape != (b.shape[0], n) or xl.shape != (n,) or xu.shape != (n,):
            raise ValueError(
                f"inconsistent standard form: c {c.shape}, A {A.shape}, b {b.shape}, "
                f"xl {xl.shape}, xu {xu.shape}"
            )
        for name, arr in (("c", c), ("A", A.data), ("b", b), ("xl", xl), ("xu", xu)):
            if np.isnan(arr).any():
                raise ValueError(f"standard form {name} contains NaN")
        if ii.size and (ii.min() < 0 or ii.max() >= n):
            raise ValueError("int_idx out of range")
        return cls(c, A, b, xl, xu, float(obj_const), np.unique(ii))

    @property
    def n(self) -> int:
        return int(self.c.shape[0])

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    def relaxed(self) -> "StdForm":
        return dataclasses.replace(self, int_idx=np.zeros(0, dtype=np.int64))


@dataclass
class HighsOutcome:
    """A verified outcome in the internal (minimize) sense.

    ``status`` is one of ``optimal``, ``feasible``, ``infeasible``, ``unbounded``,
    ``time_limit``, ``node_limit``, ``error``. ``objective`` and ``bound`` include
    ``obj_const``. ``stats`` is numeric (it feeds ``SolveResult.solver_stats``);
    ``labels`` carries provenance strings.
    """

    status: str
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    bound: Optional[float] = None
    gap_certified: bool = False
    row_dual: Optional[np.ndarray] = None
    col_dual: Optional[np.ndarray] = None
    ray: Optional[np.ndarray] = None
    message: str = ""
    highs_status: str = ""
    node_count: int = 0
    iterations: int = 0
    root_bound: Optional[float] = None
    root_time: Optional[float] = None
    wall_time: float = 0.0
    stats: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Solver-independent verification kernels
# ─────────────────────────────────────────────────────────────


def _logical_columns(sf: StdForm) -> np.ndarray:
    """Mask of *logical* columns: continuous, zero cost, in exactly one row. Such a
    column's value is determined by its row and affects nothing else."""
    is_int = np.zeros(sf.n, dtype=bool)
    is_int[sf.int_idx] = True
    return np.asarray((np.diff(sf.A.indptr) == 1) & (sf.c == 0.0) & ~is_int, dtype=bool)


def readback_problem(x: np.ndarray, sf: StdForm, **duals: Optional[np.ndarray]) -> Optional[str]:
    """Why a HiGHS readback cannot be used, or ``None``.

    Runs before any residual so the verifier never multiplies a sentinel-sized value:
    #1229's false "feasible" needed products ~1e20 to survive a 1e-6 tolerance. A
    large ``x_j`` is legitimate only when it equals a declared *finite* bound, or
    when ``j`` is a logical: a row over huge structural values needs a huge logical
    (``z0 + z1 >= 1`` at ``z = 9.999e19``), and the logical's read-back value is
    never trusted -- :func:`_exact_rows_problem` re-derives it exactly.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (sf.n,):
        return f"primal readback has shape {x.shape}, expected ({sf.n},)"
    bad = np.flatnonzero(~np.isfinite(x))
    if bad.size:
        return f"primal readback x[{bad[0]}] = {x[bad[0]]!r} is not finite"
    big = np.abs(x) >= READBACK_LIMIT
    on_bound = ((x == sf.xl) & (sf.xl > -INF)) | ((x == sf.xu) & (sf.xu < INF))
    bad = np.flatnonzero(big & ~on_bound & ~_logical_columns(sf))
    if bad.size:
        j = int(bad[0])
        return (
            f"readback at sentinel magnitude: x[{j}] = {x[j]:.6g} "
            f"(bounds [{sf.xl[j]:.6g}, {sf.xu[j]:.6g}])"
        )
    for name, d in duals.items():
        if d is None:
            continue
        d = np.asarray(d, dtype=np.float64)
        bad = np.flatnonzero(~np.isfinite(d) | (np.abs(d) >= READBACK_LIMIT))
        if bad.size:
            return f"readback at sentinel magnitude: {name}[{bad[0]}] = {d[bad[0]]!r}"
    return None


def feasibility_problem(x: np.ndarray, sf: StdForm, check_integrality: bool) -> Optional[str]:
    """Why ``x`` is not a feasible point of ``sf``, or ``None``. Call only after
    :func:`readback_problem` passed.

    A row is tested as ``|viol_i| <= FEAS_TOL + FEAS_RTOL * sum_j |A_ij||x_j|`` (the
    ``_matrix_solution_feasible`` convention), except a row touching a value at or
    above ``READBACK_LIMIT`` -- by the readback guard, a declared huge finite bound
    such as the default ``±9.999e19`` box. There ``-9.999e19 + 9.999e19`` cancels to
    exactly 0 and ``1e-9 * 2e20`` forgives any violation under ``2e11``; together they
    certified the infeasible ``x + y >= 2, x + y <= 1`` over free columns as optimal.
    Such a row is checked by :func:`_exact_rows_problem` instead, with the huge values
    left out of its tolerance scale.
    """
    x = np.asarray(x, dtype=np.float64)
    if check_integrality and sf.int_idx.size:
        # #1380: integrality FIRST, then test the rows at the point being
        # CLAIMED -- the integral one. Checking the rows at the fractional point
        # and integrality separately lets a column inside INT_TOL buy ``M *
        # INT_TOL`` of row slack: on ``x <= 1e7 z`` a binary read back at 1e-6
        # carries x all the way to 10 while passing both tests, and the route
        # then reports an objective below the model's true optimum.
        seg = x[sf.int_idx]
        frac = np.abs(seg - np.round(seg))
        k = int(np.argmax(frac))
        if frac[k] > INT_TOL:
            return f"integer column {int(sf.int_idx[k])} = {seg[k]:.9g} is fractional"
        from discopt.validation.feasibility import snap_integer_columns

        x = snap_integer_columns(x, sf.int_idx)
    ax = np.abs(x)
    huge = ax >= READBACK_LIMIT
    if sf.m:
        abs_a = abs(sf.A)
        viol = np.abs(sf.A @ x - sf.b)
        thr = FEAS_TOL + FEAS_RTOL * (abs_a @ np.where(huge, 0.0, ax))
        exact = (abs_a @ huge.astype(np.float64)) > 0.0
        bad = np.flatnonzero((viol > thr) & ~exact)
        if bad.size:
            i = int(bad[np.argmax(viol[bad] - thr[bad])])
            return f"row {i} violated by {viol[i]:.3g} (tolerance {thr[i]:.3g})"
        if exact.any():
            why = _exact_rows_problem(np.flatnonzero(exact), x, sf, thr)
            if why:
                return why
    thr = FEAS_TOL + FEAS_RTOL * ax
    # Only a declared side is a bound: the 1e20 sentinel is "no bound", and a
    # legitimate logical above it (2e20 for ``z0 + z1`` at the default box) is not
    # a violation of anything.
    lo = np.flatnonzero((sf.xl > -INF) & (x < sf.xl - thr))
    if lo.size:
        j = int(lo[0])
        return f"x[{j}] = {x[j]:.6g} below its lower bound {sf.xl[j]:.6g}"
    hi = np.flatnonzero((sf.xu < INF) & (x > sf.xu + thr))
    if hi.size:
        j = int(hi[0])
        return f"x[{j}] = {x[j]:.6g} above its upper bound {sf.xu[j]:.6g}"
    return None


def _exact_rows_problem(
    rows: np.ndarray, x: np.ndarray, sf: StdForm, thr: np.ndarray
) -> str | None:
    """Check ``rows`` of ``A x = b`` in exact rational arithmetic (doubles are exact
    rationals, so this has no rounding at all).

    In each row one *logical* -- a continuous, zero-cost column appearing in no other
    row -- is not taken at its read-back value, which may be unable to represent
    what the row needs (``9.999e19 + 1``). The value the row requires of it is
    computed exactly and checked against its bounds; since nothing else depends on
    that column, the point is feasible iff that value is. The largest-magnitude
    logical is chosen, as it is the one whose rounding can matter.
    """
    from fractions import Fraction

    csr = sf.A.tocsr()
    logical = _logical_columns(sf)
    for i in rows:
        i = int(i)
        s, e = int(csr.indptr[i]), int(csr.indptr[i + 1])
        cols, vals = csr.indices[s:e], csr.data[s:e]
        cand = [k for k in range(e - s) if logical[cols[k]]]
        k_free = max(cand, key=lambda k: abs(x[cols[k]])) if cand else -1
        r = Fraction(float(sf.b[i]))
        for k in range(e - s):
            if k != k_free:
                r -= Fraction(float(vals[k])) * Fraction(float(x[cols[k]]))
        if k_free < 0:
            if abs(r) > Fraction(float(thr[i])):
                return (
                    f"row {i} violated by {float(abs(r)):.3g} in exact arithmetic "
                    f"(tolerance {thr[i]:.3g})"
                )
            continue
        j = int(cols[k_free])
        need = r / Fraction(float(vals[k_free]))
        lo, hi = float(sf.xl[j]), float(sf.xu[j])
        if lo > -INF and need < Fraction(lo) - Fraction(FEAS_TOL + FEAS_RTOL * abs(lo)):
            return f"row {i} needs logical x[{j}] = {float(need):.6g} below its bound {lo:.6g}"
        if hi < INF and need > Fraction(hi) + Fraction(FEAS_TOL + FEAS_RTOL * abs(hi)):
            return f"row {i} needs logical x[{j}] = {float(need):.6g} above its bound {hi:.6g}"
    return None


def ns_bound(y: np.ndarray, sf: StdForm) -> Optional[float]:
    """Neumaier--Shcherbina safe lower bound on ``min cᵀx`` (plus ``obj_const``) from
    any dual ``y``, computed in Rust. ``None`` means ``-inf``."""
    from discopt._rust import ns_safe_bound_csc_py

    g = ns_safe_bound_csc_py(
        np.ascontiguousarray(y, dtype=np.float64),
        sf.c,
        sf.m,
        sf.n,
        np.ascontiguousarray(sf.A.indptr, dtype=np.int64),
        np.ascontiguousarray(sf.A.indices, dtype=np.int64),
        np.ascontiguousarray(sf.A.data, dtype=np.float64),
        sf.b,
        sf.xl,
        sf.xu,
    )
    return None if g is None else float(g) + sf.obj_const


def _box_prod(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``t * x`` with the convention ``0 * inf = 0``: a coefficient that is exactly zero
    cannot ride an open bound side, and numpy would make that product ``nan``."""
    out = np.zeros_like(t)
    nz = t != 0.0
    out[nz] = t[nz] * x[nz]
    return out


def _farkas_exact(y: np.ndarray, sf: StdForm) -> bool:
    """Decide the Farkas contradiction in exact rational arithmetic.

    Every entry of ``A``, ``b``, the box and ``y`` is a float64 and therefore an exact
    rational, so ``r = Aᵀy``, the box supremum of ``rᵀx`` and ``yᵀb`` are all computed
    here with no rounding at all: the verdict is the mathematical truth for the declared
    standard form, with no tolerance and no error bound to get wrong.

    This exists because the float error bound in :func:`farkas_verified` is a *bound*, not
    the error. On the huge box a column whose true ``r_j`` is exactly ``0`` still carries a
    positive bound, and ``(nnz_j + 2)·eps·Σ|A_ij y_i|`` against a ``9.999e19`` side is
    ~1.8e5 -- enough to swamp any real margin and refuse every genuine huge-box ray, which
    is how the free-column infeasible LPs lose their certificates. Widening the margin to
    rescue them would put roundoff back inside the certificate; deciding exactly keeps both
    the guard and the proofs.

    A column with ``r_j = 0`` contributes nothing whatever its box side, which is the
    ``0 * inf = 0`` convention of :func:`_box_prod` made exact. A nonzero ``r_j`` pointing
    along an open side makes the supremum ``+inf``, which proves nothing.
    """
    indptr, indices, data = sf.A.indptr, sf.A.indices, sf.A.data
    zero = Fraction(0)
    for s in (1, -1):
        ys = [Fraction(float(v)) * s for v in y]
        yb = sum((Fraction(float(sf.b[i])) * ys[i] for i in range(sf.m)), zero)
        sup = zero
        bounded = True
        for j in range(sf.n):
            rj = sum(
                (
                    Fraction(float(data[k])) * ys[indices[k]]
                    for k in range(indptr[j], indptr[j + 1])
                ),
                zero,
            )
            if rj > 0:
                if sf.xu[j] >= INF:
                    bounded = False
                    break
                sup += rj * Fraction(float(sf.xu[j]))
            elif rj < 0:
                if sf.xl[j] <= -INF:
                    bounded = False
                    break
                sup += rj * Fraction(float(sf.xl[j]))
        if bounded and yb > sup:
            return True
    return False


def farkas_verified(y: np.ndarray, sf: StdForm) -> bool:
    """True iff ``y`` (in either orientation) proves ``A x = b, xl <= x <= xu`` empty.

    For every box point ``yᵀA x <= sup_box (Aᵀy)ᵀx``; if that supremum is finite and
    strictly below ``yᵀb``, no box point satisfies the rows.

    ``r = Aᵀy`` is a floating-point column dot, so it is only known to lie in
    ``[r - e, r + e]`` with ``e = (nnz_j + 2)·eps·Σ_i |A_ij y_i|``. The supremum is taken
    over that interval as well as over the box, and ``yᵀb`` is lowered by its own
    summation error, so the contradiction has to survive the arithmetic that produced it.
    The module corrects for the same absorption in :func:`fbbt_box` and ``_on_open_side``.

    Comparing against a *relative* margin of the rounded total instead was unsound:
    roundoff in a column dot enters the supremum at full magnitude while raising the
    margin by only ``RAY_REL`` times itself. Bounding that error by ``|side|`` is not
    enough either -- a truly nonzero ``r_j`` that rounds to exactly ``0`` contributes no
    term *and* no error -- so the whole box is used. A coefficient that may be nonzero on
    an open side makes the supremum infinite and the ray is refused.

    Clearing this bound proves the contradiction, but failing it proves nothing: the bound
    is conservative, and on the huge box it refuses genuine rays whose ``r_j`` is exactly
    ``0`` (see :func:`_farkas_exact`). So a float refusal falls through to the exact
    rational decision rather than being taken as the answer.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (sf.m,) or not np.all(np.isfinite(y)) or not np.any(y):
        return False
    y = y / np.max(np.abs(y))
    absA = abs(sf.A).tocsc()
    nnz_col = np.diff(absA.indptr)
    lo_x = np.where(sf.xl <= -INF, -np.inf, sf.xl)
    hi_x = np.where(sf.xu >= INF, np.inf, sf.xu)
    for s in (1.0, -1.0):
        ys = s * y
        r = np.asarray(sf.A.T @ ys, dtype=np.float64)
        r_err = (nnz_col + 2) * _EPS * np.asarray(absA.T @ np.abs(ys), dtype=np.float64)
        lo_t, hi_t = r - r_err, r + r_err
        sup_j = np.maximum(
            np.maximum(_box_prod(lo_t, lo_x), _box_prod(lo_t, hi_x)),
            np.maximum(_box_prod(hi_t, lo_x), _box_prod(hi_t, hi_x)),
        )
        if not np.all(np.isfinite(sup_j)):
            continue
        sup = float(np.sum(sup_j))
        sup_err = (sup_j.size + 2) * _EPS * float(np.sum(np.abs(sup_j)))
        yb = float(sf.b @ ys)
        yb_err = (sf.m + 2) * _EPS * float(np.sum(np.abs(sf.b * ys)))
        if np.isfinite(sup) and (yb - yb_err) - (sup + sup_err) > 0.0:
            return True
    return _farkas_exact(y, sf)


def primal_ray_verified(d: np.ndarray, sf: StdForm, *, deadline: Optional[float] = None) -> bool:
    """True iff ``d`` proves a descent direction of the recession cone exists:
    ``A d' = 0``, ``cᵀd' < 0``, and ``d'`` only moves along open bound sides, for an
    exact rational ``d'`` on the support of ``d`` (:func:`_exact_ray`).

    The floating-point screen below is only a filter. Its relative residual test
    ``|A d| <= RAY_REL |A||d|`` cannot see a row whose one coupling to the ray is
    tiny: ``x1 + x2 = 0``, ``x1 + x2 + 1e-12 x3 = 0`` and ``max x3`` accepted
    ``d = (1, -1, 1)`` with residual 1e-12 on the second row and certified
    ``unbounded`` on an LP whose optimum is 0 (#1286). Any nonzero exact residual,
    however small, grows without bound along the ray, so acceptance is decided
    exactly."""
    d = np.asarray(d, dtype=np.float64)
    if d.shape != (sf.n,) or not np.all(np.isfinite(d)) or not np.any(d):
        return False
    d = d / np.max(np.abs(d))
    d = np.where(np.abs(d) <= RAY_CLEAN, 0.0, d)
    if np.any((d > 0.0) & (sf.xu < INF)) or np.any((d < 0.0) & (sf.xl > -INF)):
        return False
    if sf.m:
        res = np.abs(sf.A @ d)
        if np.any(res > RAY_REL * (abs(sf.A) @ np.abs(d)) + RAY_CLEAN):
            return False
    cd = float(sf.c @ d)
    if not cd < -RAY_REL * max(1.0, float(np.abs(sf.c) @ np.abs(d))):
        return False
    return _exact_ray(d, sf, deadline)


def _exact_ray(d: np.ndarray, sf: StdForm, deadline: Optional[float]) -> bool:
    """Whether an exact recession ray with ``cᵀd' < 0`` lives on the support of ``d``.

    Restricted to the support ``S``, ``A_S d'_S = 0`` is solved over the rationals by
    Gauss-Jordan elimination. Each row pivots on its column of largest ``|d_j|``, the
    remaining (free) columns keep their values from ``d``, and the pivot columns are
    solved for. A good float ray changes by its residual only, so its sign pattern
    survives. ``d'`` is accepted only if every nonzero moves along an open side and
    ``cᵀd' < 0`` holds exactly; columns outside ``S`` stay at 0, so rows outside the
    support are satisfied. Refuses (``False``) past ``EXACT_MAX_COLUMNS`` support
    columns, ``EXACT_MAX_WORK`` bit operations or ``deadline``: an undecided ray is
    never a certificate.
    """
    from fractions import Fraction

    support = np.flatnonzero(d)
    if support.size == 0 or support.size > EXACT_MAX_COLUMNS:
        return False
    sub = sf.A[:, support].tocsr()
    sub.eliminate_zeros()
    mag = np.abs(d[support])

    def size(q: Fraction) -> int:
        return int(q.numerator.bit_length() + q.denominator.bit_length())

    pivots: list[tuple[dict, int]] = []
    work = 0
    for i in range(sub.shape[0]):
        lo, hi = sub.indptr[i], sub.indptr[i + 1]
        if lo == hi:
            continue
        if deadline is not None and time.perf_counter() > deadline:
            return False
        row = {int(sub.indices[p]): Fraction(float(sub.data[p])) for p in range(lo, hi)}
        for pv, pc in pivots:
            f = row.get(pc)
            if not f:
                continue
            for k, v in pv.items():
                work += size(f) + size(v)
                nv = row.get(k, Fraction(0)) - f * v
                if nv:
                    row[k] = nv
                else:
                    row.pop(k, None)
            if work > EXACT_MAX_WORK:
                return False
        if not row:
            continue
        pc = max(row, key=lambda k: (mag[k], -k))
        inv = 1 / row[pc]
        row = {k: v * inv for k, v in row.items()}
        for pv, _ in pivots:
            f = pv.get(pc)
            if not f:
                continue
            for k, v in row.items():
                work += size(f) + size(v)
                nv = pv.get(k, Fraction(0)) - f * v
                if nv:
                    pv[k] = nv
                else:
                    pv.pop(k, None)
            if work > EXACT_MAX_WORK:
                return False
        pivots.append((row, pc))
    pivot_cols = {pc for _, pc in pivots}
    dx = [Fraction(float(v)) for v in d[support]]
    for pv, pc in pivots:
        dx[pc] = -sum(
            (v * dx[k] for k, v in pv.items() if k != pc and k not in pivot_cols),
            Fraction(0),
        )
    cd = Fraction(0)
    for k, j in enumerate(support):
        v = dx[k]
        if (v > 0 and sf.xu[j] < INF) or (v < 0 and sf.xl[j] > -INF):
            return False
        if v:
            cd += Fraction(float(sf.c[j])) * v
    return cd < 0


# ─────────────────────────────────────────────────────────────
# Certificate recovery: FBBT box, exact dual correction, phase 1
# ─────────────────────────────────────────────────────────────

#: FBBT sweeps for the certificate box; every sweep keeps a superset of the feasible set.
FBBT_ROUNDS = 20
#: Most columns the exact dual correction zeroes, and most correction rounds. Past
#: either the rational elimination is too slow for a fallback; the bound is then left
#: uncertified (never guessed). Each round is bounded by the column cap and
#: :data:`EXACT_MAX_WORK`, and all rounds together by :data:`EXACT_MAX_TOTAL_WORK`,
#: so the round cap only has to let a correction that CONVERGES finish. #1655's
#: robust-counterpart LP (budget set, Gamma = 1) needed 9-16 rounds -- zeroing one
#: wrong-signed dual column disturbs the next -- and converged to a bound equal to
#: the objective in 0.01 s, where the old cap of 8 left the NS bound 19% below the
#: optimum and the LP uncertified.
EXACT_MAX_COLUMNS = 256
EXACT_MAX_ROUNDS = 64
#: Work cap for one rational elimination, in bit operations (entries updated times the
#: bit length of the operands). Rational elimination can grow its denominators, and a
#: fallback must not hang a solve; a count, not a clock, so whether a bound is certified
#: does not depend on machine speed (#912). Calibrated on dense 60-bit rational systems
#: at 1.0-2.2e8 units/s (n=60: 5.9e8 units, 6.0 s), so the cap is ~1-2 s of elimination.
EXACT_MAX_WORK = 200_000_000
#: #1655: work cap summed over every elimination of one correction (~2-4 s), so the
#: raised round cap cannot multiply the per-elimination worst case.
EXACT_MAX_TOTAL_WORK = 2 * EXACT_MAX_WORK
_EPS = float(np.finfo(np.float64).eps)


def fbbt_box(sf: StdForm, k: Optional[int] = None) -> StdForm:
    """``sf`` with the open bound sides of its first ``k`` columns (default: all) replaced
    by finite sides implied by ``A[:, :k] x = b`` and the declared box.

    With ``k = n`` the box contains ``{A x = b, xl <= x <= xu}``, so an NS bound or a
    Farkas proof over it holds for the declared problem. With ``k < n`` it contains the
    first ``k`` coordinates of every point of ``A[:, :k] x = b`` in the box: the phase-1
    use, where the trailing columns are the row violations.

    A derived side ``(b_i - rest)/a_ij`` is widened outward by a bound on its whole
    floating-point error, ``(nnz_i + 8)·eps·(Σ|terms| + |b_i|)/|a_ij| + 8·eps·|side|``, so
    it holds in exact arithmetic on the data. This matters where a term is huge: ``rest``
    is a row total minus the column's own term, and at ``|term| ~ 1e20`` that cancellation
    alone is off by ~1e4, which a relative widening does not cover.
    """
    kk = sf.n if k is None else int(k)
    A = sf.A[:, :kk].tocoo()
    rows, cols, vals = A.row, A.col, A.data
    if vals.size == 0:
        return sf
    m = sf.m
    lb = np.where(sf.xl[:kk] <= -INF, -np.inf, sf.xl[:kk])
    ub = np.where(sf.xu[:kk] >= INF, np.inf, sf.xu[:kk])
    pos = vals > 0.0
    b_r = sf.b[rows]
    slack_r = (np.bincount(rows, minlength=m)[rows] + 8.0) * _EPS
    absb_r = np.abs(b_r)
    absa = np.abs(vals)

    def rest(use: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inf = ~np.isfinite(use)
        t = vals * np.where(inf, 0.0, use)
        n_inf = np.bincount(rows, weights=inf.astype(np.float64), minlength=m)
        total = np.bincount(rows, weights=t, minlength=m)
        mag = np.bincount(rows, weights=np.abs(t), minlength=m)
        others_finite = (n_inf[rows] - inf) == 0
        return total[rows] - t, others_finite, mag[rows]

    for _ in range(FBBT_ROUNDS):
        # A term at its minimum uses lb for a positive coefficient, ub for a negative one.
        rmin, fin_min, mag_min = rest(np.where(pos, lb[cols], ub[cols]))
        rmax, fin_max, mag_max = rest(np.where(pos, ub[cols], lb[cols]))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            from_max = (b_r - rmax) / vals
            from_min = (b_r - rmin) / vals
            err_max = slack_r * (mag_max + absb_r) / absa + 8.0 * _EPS * np.abs(from_max)
            err_min = slack_r * (mag_min + absb_r) / absa + 8.0 * _EPS * np.abs(from_min)
            lo_c = np.where(pos, from_max - err_max, from_min - err_min)
            hi_c = np.where(pos, from_min + err_min, from_max + err_max)
        lo_ok = np.where(pos, fin_max, fin_min) & np.isfinite(lo_c)
        hi_ok = np.where(pos, fin_min, fin_max) & np.isfinite(hi_c)
        cand_lo = np.full(kk, -np.inf)
        np.maximum.at(cand_lo, cols[lo_ok], lo_c[lo_ok])
        cand_hi = np.full(kk, np.inf)
        np.minimum.at(cand_hi, cols[hi_ok], hi_c[hi_ok])
        new_lb = np.maximum(lb, cand_lo)
        new_ub = np.minimum(ub, cand_hi)
        with np.errstate(invalid="ignore"):
            moved = np.isfinite(new_lb) & ~np.isfinite(lb)
            moved |= np.isfinite(new_ub) & ~np.isfinite(ub)
            moved |= new_lb > lb + 1e-9 * (1.0 + np.abs(lb))
            moved |= new_ub < ub - 1e-9 * (1.0 + np.abs(ub))
        lb, ub = new_lb, new_ub
        if not moved.any():
            break
    xl, xu = sf.xl.copy(), sf.xu.copy()
    xl[:kk] = np.where((sf.xl[:kk] <= -INF) & np.isfinite(lb), lb, sf.xl[:kk])
    xu[:kk] = np.where((sf.xu[:kk] >= INF) & np.isfinite(ub), ub, sf.xu[:kk])
    return dataclasses.replace(sf, xl=xl, xu=xu)


def _exact_reduced_costs(Y: list, sf: StdForm, cols) -> dict:
    from fractions import Fraction

    A = sf.A
    out = {}
    for j in cols:
        j = int(j)
        s = Fraction(float(sf.c[j]))
        for p in range(A.indptr[j], A.indptr[j + 1]):
            yi = Y[A.indices[p]]
            if yi:
                s -= Fraction(float(A.data[p])) * yi
        out[j] = s
    return out


def _on_open_side(r, sf: StdForm, j: int) -> bool:
    """A reduced cost that selects an open or sentinel-scale side.

    On an open side the box term is ``-inf``. On a finite side at ``READBACK_LIMIT`` or
    beyond (the default ``±9.999e19`` box) it is finite but useless: a roundoff reduced
    cost of 1.4e-17 times that side cost 1.9e3 of bound on nlp_cvx_001_010, leaving an
    objective of -2.04 uncertified. Both are corrected to an exact zero.
    """
    return bool((r > 0 and sf.xl[j] <= -READBACK_LIMIT) or (r < 0 and sf.xu[j] >= READBACK_LIMIT))


def _exact_solve(
    M: list, rhs: list, deadline: Optional[float], spent: Optional[list] = None
) -> Optional[list]:
    """``M z = rhs`` over the rationals by Gaussian elimination; ``None`` if singular, past
    ``EXACT_MAX_WORK`` bit operations, or past the caller's ``deadline``. ``spent``, a
    one-element list, accumulates the work across calls and refuses past
    ``EXACT_MAX_TOTAL_WORK`` (#1655)."""
    from fractions import Fraction

    def size(q: Fraction) -> int:
        return int(q.numerator.bit_length() + q.denominator.bit_length())

    n = len(rhs)
    T = [row[:] + [rhs[i]] for i, row in enumerate(M)]
    work = 0
    for col in range(n):
        if deadline is not None and time.perf_counter() > deadline:
            return None
        piv = next((r for r in range(col, n) if T[r][col] != 0), None)
        if piv is None:
            return None
        T[col], T[piv] = T[piv], T[col]
        p = T[col][col]
        row_size = max(size(v) for v in T[col][col:])
        for r in range(col + 1, n):
            if T[r][col] != 0:
                f = T[r][col] / p
                step = (n + 1 - col) * (size(f) + row_size)
                work += step
                if work > EXACT_MAX_WORK:
                    return None
                if spent is not None:
                    spent[0] += step
                    if spent[0] > EXACT_MAX_TOTAL_WORK:
                        return None
                T[r] = [a - f * b for a, b in zip(T[r], T[col])]
    z = [Fraction(0)] * n
    for i in reversed(range(n)):
        z[i] = (T[i][n] - sum((T[i][k] * z[k] for k in range(i + 1, n)), Fraction(0))) / T[i][i]
    return z


def exact_ns_bound(
    y: np.ndarray, sf: StdForm, *, deadline: Optional[float] = None
) -> tuple[Optional[float], str]:
    """A rigorous lower bound on ``min cᵀx + obj_const`` over ``{A x = b, xl <= x <= xu}``
    from a rational correction of ``y``. Returns ``(bound, "")`` or ``(None, reason)``.

    Weak duality holds for every ``y``: the minimum is at least
    ``bᵀy + Σ_j min_{xl_j <= x_j <= xu_j} (c - Aᵀy)_j x_j``. A float ``y`` cannot use it
    when an open-sided column has reduced cost exactly zero in every optimal dual (a
    zero-cost recession direction, measured on 25fv47/e226/scrs8/stair): roundoff puts
    the reduced cost on the open side and the term is ``-inf``. The correction picks, in
    floating point, independent wrong-signed columns ``S`` and pivot rows ``R``, then
    solves ``A_{R,S}ᵀ dy_R = rc_S`` over the rationals so ``rc_S`` becomes exactly zero
    (project-and-shift, Steffy & Wolter), and repeats on the columns that step disturbed.
    The bound is summed exactly and rounded down. The floating-point choices decide only
    whether a bound is found, never whether it is valid.
    """
    from fractions import Fraction

    import scipy.linalg as sla

    y = np.asarray(y, dtype=np.float64)
    if y.shape != (sf.m,) or not np.all(np.isfinite(y)):
        return None, "dual is not a finite vector over the rows"
    Y = [Fraction(float(v)) for v in y]
    open_cols = np.flatnonzero((sf.xl <= -READBACK_LIMIT) | (sf.xu >= READBACK_LIMIT))
    held: list[int] = []
    spent = [0]
    for rnd in range(EXACT_MAX_ROUNDS + 1):
        if deadline is not None and time.perf_counter() > deadline:
            return None, "time limit reached during the exact dual correction"
        rc = _exact_reduced_costs(Y, sf, open_cols)
        wrong = [j for j, r in rc.items() if _on_open_side(r, sf, j)]
        if not wrong:
            break
        if rnd == EXACT_MAX_ROUNDS:
            return None, f"{len(wrong)} reduced costs still select an open side"
        S = sorted(set(held) | set(wrong))
        if len(S) > EXACT_MAX_COLUMNS:
            return None, f"{len(S)} columns need an exact zero reduced cost"
        sub = sf.A[:, S]
        touched = np.unique(sub.indices)
        if touched.size == 0:
            return None, "an open column with a nonzero cost has no rows"
        B = sub[touched].toarray()
        _, R, piv = sla.qr(B, mode="economic", pivoting=True)
        d = np.abs(np.diag(R))
        rank = int(np.count_nonzero(d > 1e-10 * d[0])) if d.size and d[0] > 0.0 else 0
        if rank == 0:
            return None, "the wrong-signed columns have no usable rows"
        indep = [S[i] for i in piv[:rank]]
        P, _, _ = sla.lu(B[:, piv[:rank]])
        prow = touched[np.argmax(P, axis=0)[:rank]]
        block = sf.A[:, indep].tocsr()[prow].toarray()  # rank x rank, block[i, k] = A[prow_i, S_k]
        rcS = _exact_reduced_costs(Y, sf, indep)
        M = [[Fraction(float(block[i, kc])) for i in range(rank)] for kc in range(rank)]
        dy = _exact_solve(M, [rcS[j] for j in indep], deadline, spent)
        if dy is None:
            return None, "pivot block singular in exact arithmetic, or work/time limit reached"
        for i, v in zip(prow, dy):
            Y[int(i)] += v
        held = S

    total = Fraction(float(sf.obj_const))
    for i, v in enumerate(Y):
        if v:
            total += Fraction(float(sf.b[i])) * v
    for j, r in _exact_reduced_costs(Y, sf, range(sf.n)).items():
        if r > 0:
            if sf.xl[j] <= -INF:
                return None, f"column {j} reduced cost selects its open lower side"
            total += r * Fraction(float(sf.xl[j]))
        elif r < 0:
            if sf.xu[j] >= INF:
                return None, f"column {j} reduced cost selects its open upper side"
            total += r * Fraction(float(sf.xu[j]))
    try:
        g = float(total)
    except OverflowError:
        return None, "exact bound is outside the double range"
    if Fraction(g) > total:
        g = float(np.nextafter(g, -np.inf))
    return g, ""


def phase1_infeasibility_proof(sf: StdForm, *, time_limit: Optional[float]) -> tuple[bool, str]:
    """Prove ``{A x = b, xl <= x <= xu}`` empty without a Farkas ray.

    The phase-1 LP ``min 1ᵀ(p + q)  s.t.  A x + p - q = b`` over ``x`` in the FBBT box of
    ``A x = b`` and ``p, q >= 0`` has value 0 if the declared LP has a point (that point,
    with ``p = q = 0``, lies in the box). A rigorous lower bound on it above a relative
    margin therefore proves the LP empty. The bound comes from HiGHS's phase-1 dual
    through ``ns_bound``, then ``exact_ns_bound``. Returns ``(True, provenance)`` or
    ``(False, reason)``.
    """
    highspy = require_highspy()
    t0 = time.perf_counter()
    m, n = sf.m, sf.n
    if m == 0:
        return False, "no rows"
    eye = sp.identity(m, format="csc")
    work = StdForm.from_arrays(
        np.r_[np.zeros(n), np.ones(2 * m)], sp.hstack([sf.A, eye, -eye], format="csc"), sf.b,
        np.r_[sf.xl, np.zeros(2 * m)], np.r_[sf.xu, np.full(2 * m, INF)],
    )  # fmt: skip
    opts: list[tuple[str, Any]] = [
        ("primal_feasibility_tolerance", 1e-7),
        ("dual_feasibility_tolerance", 1e-7),
        ("run_crossover", "on"),
    ]
    if time_limit is not None:
        if time_limit <= 0.0:
            return False, "no time left for the phase-1 LP"
        opts.append(("time_limit", float(time_limit)))
    h = _new_highs(highspy, opts)
    pass_st, pass_why = _pass_model(h, highspy, work, integer=False)
    if pass_st == highspy.HighsStatus.kError:
        return False, f"phase-1 LP: {pass_why}"
    h.run()
    name = _status_name(h)
    if name != "kOptimal":
        return False, f"phase-1 LP {name}"
    y = np.asarray(h.getSolution().row_dual, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        return False, "phase-1 dual is not finite"
    box = fbbt_box(work, n)
    margin = RAY_REL * (1.0 + float(np.abs(sf.b).sum()))
    g = ns_bound(y, box)
    if g is not None and g > margin:
        return True, "phase1-ns-fbbt-box"
    # The caller's own time_limit only; the work is capped by count (EXACT_MAX_WORK).
    deadline = None if time_limit is None else t0 + float(time_limit)
    g2, why = exact_ns_bound(y, box, deadline=deadline)
    if g2 is not None and g2 > margin:
        return True, "phase1-exact-dual-correction"
    best = max((v for v in (g, g2) if v is not None), default=None)
    return False, f"phase-1 bound {best} does not exceed {margin:.3g}{'; ' + why if why else ''}"


def recession_ray(sf: StdForm, *, time_limit: Optional[float]) -> Optional[np.ndarray]:
    """A candidate descent ray of ``{A x = b, xl <= x <= xu}``, or ``None``.

    Solves ``min cᵀd  s.t.  A d = 0`` with ``d_j`` in ``[-1, 0]`` on an open lower side,
    ``[0, 1]`` on an open upper side (both for a free column) and fixed at 0 otherwise.
    That LP is bounded and ``d = 0`` is feasible, so its value is at most 0. A negative
    value gives a ray. The caller still checks it with ``primal_ray_verified``, so this
    decides only whether a ray is found, never whether one is accepted.
    """
    highspy = require_highspy()
    if time_limit is not None and time_limit <= 0.0:
        return None
    lo = np.where(sf.xl <= -INF, -1.0, 0.0)
    hi = np.where(sf.xu >= INF, 1.0, 0.0)
    if not np.any(lo) and not np.any(hi):
        return None
    work = StdForm.from_arrays(sf.c, sf.A, np.zeros(sf.m), lo, hi)
    opts: list[tuple[str, Any]] = [("run_crossover", "on")]
    if time_limit is not None:
        opts.append(("time_limit", float(time_limit)))
    h = _new_highs(highspy, opts)
    if _pass_model(h, highspy, work, integer=False)[0] == highspy.HighsStatus.kError:
        return None
    h.run()
    if _status_name(h) != "kOptimal":
        return None
    d = np.asarray(h.getSolution().col_value, dtype=np.float64)
    return d if float(sf.c @ d) < 0.0 else None


# ─────────────────────────────────────────────────────────────
# HiGHS plumbing
# ─────────────────────────────────────────────────────────────


def _version_number(h) -> float:
    parts = [int(p) for p in str(h.version()).split(".")[:3]]
    parts += [0] * (3 - len(parts))
    return float(parts[0] * 10000 + parts[1] * 100 + parts[2])


def _set_options(h, highspy, opts: list[tuple[str, Any]]) -> None:
    for key, val in opts:
        st = h.setOptionValue(key, val)
        if st != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected option {key}={val!r} (status {st})")


def _huge_box(sf: StdForm) -> tuple[np.ndarray, np.ndarray]:
    """Per-side masks of the finite declared bounds at sentinel-adjacent magnitude (the
    default ±9.999e19 box).

    The sides are returned separately because a column can carry one sentinel-magnitude
    side and one ordinary declared side. Folding them into a single column mask and
    re-deriving the side from the sign discards the declared side too: ``lb=-5`` with a
    default ``ub`` opened to ``[-inf, inf]``, and the MILP route answered ``error`` where
    the declared box has an optimum. ``lb=0`` escaped only because ``0 < 0`` is false.
    """
    lo = (np.abs(sf.xl) >= READBACK_LIMIT) & (sf.xl > -INF)
    hi = (np.abs(sf.xu) >= READBACK_LIMIT) & (sf.xu < INF)
    return np.asarray(lo, dtype=bool), np.asarray(hi, dtype=bool)


#: #1295: an entry on a column with an open (or sentinel-magnitude) side that is smaller
#: than this fraction of its row's largest entry puts the model outside the class whose
#: HiGHS MIP certificate this route accepts. The value is HiGHS's default column-scale
#: cap, ``2**-kDefaultAllowedMatrixPow2Scale`` (``HConst.h``), beyond which scaling cannot
#: equilibrate the entry. Measured on 280 generated badly scaled MILPs: all 97 false
#: certificates sit at ratio <= 5.3e-7 (below the cap); the 26 real MILPs in
#: ``ref/HiGHS/check`` plus i1183 all sit at >= 3.4e-5 (above it).
UNSCALABLE_OPEN_RATIO = 2.0**-20


def open_column_coefficient_ratio(sf: StdForm) -> float:
    """Smallest ``|a_ij| / max_k |a_ik|`` over entries on a column with an open side.

    A column is open when either side is infinite or at least ``READBACK_LIMIT``, the
    bounds this route hands HiGHS as infinite. ``1.0`` when no entry qualifies.
    """
    A = sp.csr_matrix(sf.A)  # noqa: N806
    A.eliminate_zeros()
    if A.nnz == 0:
        return 1.0
    absA = abs(A).tocoo()  # noqa: N806
    open_col = (np.abs(sf.xl) >= READBACK_LIMIT) | (np.abs(sf.xu) >= READBACK_LIMIT)
    on_open = open_col[absA.col]
    if not on_open.any():
        return 1.0
    row_max = np.zeros(A.shape[0])
    np.maximum.at(row_max, absA.row, absA.data)
    return float((absA.data[on_open] / row_max[absA.row[on_open]]).min())


#: Safety factor on the round-off bound for a reduced cost, covering error accumulated
#: inside HiGHS's own factorization and refinement rather than just the final dot product.
#: The measured separation makes the exact value immaterial: on 120 well-scaled LPs the
#: worst relative violation is 7.87e-17, and on the #1410 counterexample it is 1.0, so
#: every threshold between 1e-14 and 1e-6 gives byte-identical behaviour. This is a
#: round-off bound, not a tuned tolerance.
_DUAL_CERT_SAFETY = 64.0


def box_dual_violation(sf: StdForm, x: np.ndarray, row_dual: np.ndarray) -> float:
    """How far HiGHS's returned primal/dual pair is from being a dual-feasible certificate.

    For ``min cᵀx s.t. Ax = b, xl <= x <= xu`` the reduced costs ``d = c - Aᵀy`` must, at
    an optimal basis, satisfy ``d_j >= 0`` where ``x_j`` sits at ``xl_j``, ``d_j <= 0``
    where it sits at ``xu_j``, and ``d_j == 0`` where it is interior. Returned relative to
    each reduced cost's own scale ``|c_j| + (|A|ᵀ|y|)_j``, because ``d_j`` is a *difference*
    and the round-off it can carry is set by the magnitudes cancelling in it, not by
    ``|d_j|`` (the #1392/#1397 yardstick rule: scaling a cancellation residual by the
    result is useless — the scale lives in the inputs).

    A violation far above round-off means the pair HiGHS returned does not certify the LP
    it was handed, which is a statement about HiGHS's arithmetic on *this* matrix and
    therefore about every LP it solves in the tree.

    #1410: measured 7.87e-17 worst over 120 well-scaled LPs, versus 1.0 on the instance
    where the MILP route certified a false optimum (miss 0.0171 on a true optimum of
    -0.01513). Used as a guard signal it caught 8/8 wrong HiGHS simplex optima over 4200
    generated badly scaled LPs at a cost of 11/4192 (0.26%) correct results decertified;
    the pre-existing :data:`UNSCALABLE_OPEN_RATIO` proxy caught the same 8/8 but
    decertified 3197/4192 (76.3%).
    """
    y = np.asarray(row_dual, dtype=np.float64).ravel()
    xv = np.asarray(x, dtype=np.float64).ravel()
    d = sf.c - np.asarray(sf.A.T @ y).ravel()
    absA = abs(sp.csc_matrix(sf.A))  # noqa: N806
    scale = np.maximum(np.abs(sf.c) + np.asarray(absA.T @ np.abs(y)).ravel(), np.finfo(float).tiny)
    # An infinite (or sentinel-magnitude) side is not a bound a column can sit at, and
    # forming the test band against it yields ``inf - inf`` -> NaN, which would make every
    # comparison below false and silently empty the check. Slack columns all carry
    # ``xu = INF``, so this is the common case, not an edge one.
    fin_lo = sf.xl > -INF
    fin_hi = sf.xu < INF
    lo = np.where(fin_lo, sf.xl, 0.0)
    hi = np.where(fin_hi, sf.xu, 0.0)
    at_lo = fin_lo & (xv <= lo + 1e-9 * (1.0 + np.abs(lo)))
    at_hi = fin_hi & (xv >= hi - 1e-9 * (1.0 + np.abs(hi)))
    # A fixed column (xl == xu) is at both bounds at once and prices either way, so it
    # constrains nothing; an interior column must price to zero.
    lo_only = at_lo & ~at_hi
    hi_only = at_hi & ~at_lo
    v = np.zeros_like(d)
    v[lo_only] = np.maximum(0.0, -d[lo_only])
    v[hi_only] = np.maximum(0.0, d[hi_only])
    interior = ~at_lo & ~at_hi
    v[interior] = np.abs(d[interior])
    return float((v / scale).max()) if v.size else 0.0


def root_pair_ns_gap(sf: StdForm, x: np.ndarray, row_dual: np.ndarray) -> Optional[float]:
    """Rigorous duality gap of HiGHS's own root primal/dual pair, in objective units.

    ``cᵀx + obj_const - NS(ŷ)`` where ``NS`` is the Neumaier--Shcherbina safe bound over
    the DECLARED box (no FBBT box, no exact dual correction: the question is whether
    HiGHS's arithmetic certifies the LP, not whether a certificate can be built some
    other way) and ``ŷ`` is HiGHS's row dual with one sign repair: on a column with a
    single entry ``a_ij`` whose reduced cost points at an infinite side (``d_j < 0`` with
    ``xu_j = inf``, or ``d_j > 0`` with ``xl_j = -inf``) -- a slack whose row dual came
    back with the wrong sign at round-off level -- ``ŷ_i`` is set to ``c_j / a_ij``, which
    makes that reduced cost exactly zero. ``NS`` is a valid lower bound for ANY dual, so
    the repair cannot make the bound wrong; it only moves the round-off from a column NS
    cannot charge (infinite box) onto columns it can, where it is charged in full.

    The gap is ``>= 0`` up to NS's outward rounding whenever ``x`` is feasible, and it
    bounds how far ``x`` can be from the LP optimum. ``None`` means the NS bound is
    ``-inf`` (a violated reduced cost on an open side the repair cannot reach).

    #1612: this is the scale-aware form of the #1410 check. A per-column relative
    violation divides by ``|c_j| + (|A|ᵀ|y|)_j``, which collapses to round-off itself on a
    slack whose row dual is round-off zero (``c_j = 0``, ``y_i = 1e-14``) and reports a
    "violation" of 1.0 on a perfectly solved set-cover LP. Measured: the gap here is
    ``6.5e-11`` on that LP and ``2.1e-12`` on the knapsack whose 6.8e-14 violation also
    tripped the check, against ``0.0221`` on the #1410 matrix, whose LP HiGHS mis-solved.
    """
    xv = np.asarray(x, dtype=np.float64).ravel()
    y = np.array(row_dual, dtype=np.float64).ravel()
    A = sp.csc_matrix(sf.A)  # noqa: N806
    A.eliminate_zeros()
    nnz = np.diff(A.indptr)
    for j in np.flatnonzero(nnz == 1):
        k = A.indptr[j]
        i, a = int(A.indices[k]), float(A.data[k])
        d = float(sf.c[j]) - a * y[i]
        if (sf.xu[j] >= INF and d < 0.0) or (sf.xl[j] <= -INF and d > 0.0):
            y[i] = float(sf.c[j]) / a
    ns = ns_bound(y, sf)
    if ns is None:
        return None
    return float(sf.c @ xv) + sf.obj_const - ns


def dual_violation_tolerance(sf: StdForm) -> float:
    """Round-off bound on :func:`box_dual_violation` for ``sf``.

    ``d_j`` sums ``nnz_j + 1`` terms, so its error is bounded by ``(nnz_j + 1) * u`` times
    its own scale; the widest column sets the bound for the whole vector.
    """
    A = sp.csc_matrix(sf.A)  # noqa: N806
    max_col_nnz = int(np.diff(A.indptr).max()) if A.shape[1] else 0
    return _DUAL_CERT_SAFETY * (max_col_nnz + 1) * float(np.finfo(float).eps)


def _relax_huge_box(sf: StdForm, huge_lo: np.ndarray, huge_hi: np.ndarray) -> StdForm:
    """``sf`` with the huge finite bounds opened to infinity, one side at a time: a
    relaxation of ``sf`` that keeps every ordinary declared bound."""
    return dataclasses.replace(
        sf,
        xl=np.where(huge_lo, -INF, sf.xl),
        xu=np.where(huge_hi, INF, sf.xu),
    )


def _absorb_tiny_entries(sf: StdForm) -> tuple[StdForm, np.ndarray, np.ndarray, int]:
    """Remove every ``|a_ij| <= SMALL_MATRIX_VALUE`` entry HiGHS would drop, soundly (#1617).

    HiGHS drops such an entry on ``passModel`` and answers ``kWarning``; the model it
    then solves is a *perturbation* of ``sf``, so neither its infeasible label nor its
    tree bound would be valid for ``sf``. Instead each dropped term ``a_ij x_j`` is
    replaced by its interval over the declared column box ``[xl_j, xu_j]``: row ``i``
    becomes the ranged row ``b_i - hi_i <= sum_{kept} a_ik x_k <= b_i - lo_i``. Every
    feasible point of ``sf`` satisfies the ranged rows, so the passed model is a
    *relaxation* of ``sf``: an infeasible label and a dual bound for it hold for ``sf``,
    and every incumbent is still verified against ``sf`` itself by the caller. A term on
    a column whose side is open (the ``1e20`` sentinel -- tested on the bound, never on
    the product, CLAUDE.md) opens that side of the row.

    Returns ``(sf without those entries, row_lower, row_upper, n_dropped)``.
    """
    A = sf.A.tocsc()
    data = A.data
    tiny = np.abs(data) <= SMALL_MATRIX_VALUE
    n_drop = int(tiny.sum())
    b = np.asarray(sf.b, dtype=np.float64)
    if n_drop == 0:
        return sf, b.copy(), b.copy(), 0
    col_of = np.repeat(np.arange(A.shape[1]), np.diff(A.indptr))
    cols = col_of[tiny]
    rows = A.indices[tiny]
    a = data[tiny]
    xl = np.asarray(sf.xl, dtype=np.float64)[cols]
    xu = np.asarray(sf.xu, dtype=np.float64)[cols]
    lo_open = np.where(a > 0, xl <= -INF, xu >= INF)
    hi_open = np.where(a > 0, xu >= INF, xl <= -INF)
    with np.errstate(invalid="ignore", over="ignore"):
        t_lo = np.where(a > 0, a * xl, a * xu)
        t_hi = np.where(a > 0, a * xu, a * xl)
    t_lo = np.where(lo_open | (a == 0.0), 0.0, t_lo)
    t_hi = np.where(hi_open | (a == 0.0), 0.0, t_hi)
    m = sf.m
    lo_sum = np.bincount(rows, weights=t_lo, minlength=m)
    hi_sum = np.bincount(rows, weights=t_hi, minlength=m)
    lo_inf = np.bincount(rows, weights=(lo_open & (a != 0.0)).astype(np.float64), minlength=m)
    hi_inf = np.bincount(rows, weights=(hi_open & (a != 0.0)).astype(np.float64), minlength=m)
    # Outward: the sums' own rounding error is bounded by n*eps*sum|terms|, and the
    # subtraction from ``b`` adds one more ulp; widen by both, then one ulp more.
    k = np.bincount(rows, minlength=m).astype(np.float64)
    mag = np.bincount(rows, weights=np.abs(t_lo) + np.abs(t_hi), minlength=m)
    err = (k + 4.0) * np.finfo(np.float64).eps * (mag + np.abs(b))
    row_lower = np.nextafter(b - hi_sum - err, -np.inf)
    row_upper = np.nextafter(b - lo_sum + err, np.inf)
    touched = np.bincount(rows, minlength=m) > 0
    row_lower = np.where(touched, row_lower, b)
    row_upper = np.where(touched, row_upper, b)
    row_lower = np.where(hi_inf > 0, -np.inf, row_lower)
    row_upper = np.where(lo_inf > 0, np.inf, row_upper)
    keep = ~tiny
    A2 = sp.csc_matrix((data[keep], (A.indices[keep], col_of[keep])), shape=A.shape)
    return dataclasses.replace(sf, A=A2), row_lower, row_upper, n_drop


def _pass_model(
    h,
    highspy,
    sf: StdForm,
    integer: bool,
    offset: float = 0.0,
    row_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[Any, str]:
    """Hand ``sf`` to HiGHS; returns ``(passModel status, reason)``, ``reason`` empty on kOk.

    kWarning means HiGHS changed the model on the way in -- it drops every matrix entry
    with ``|a| <= small_matrix_value`` -- so the caller decides whether its certificates
    survive that. kError means HiGHS holds no model. Neither is raised: both are
    properties of the input, not defects, and the route reports them as ``error``.

    ``offset`` is HiGHS's objective constant. Every HiGHS objective value and bound
    then includes it (``objective_function_value`` and ``mip_dual_bound`` both do,
    measured), so a caller that passes ``sf.obj_const`` must not add it again. The
    MILP route passes it because ``mip_rel_gap`` is measured on HiGHS's objective:
    with a constant-free one, a change of variables ``x = y - 1e6`` makes ``c . y``
    ~1e6 at every feasible point and HiGHS stops on a gap that is 3.6e-5 of that
    but 35 units of the objective discopt publishes (#1536). The LP callers keep
    the default -- they read no HiGHS objective value.
    """
    from discopt.solvers.milp_highs import _to_highs_inf

    if sf.A.nnz >= 2**31 or sf.n >= 2**31:
        raise ValueError(f"model too large for HiGHS 32-bit indices (nnz={sf.A.nnz})")
    lp = highspy.HighsLp()
    lp.num_col_ = sf.n
    lp.num_row_ = sf.m
    lp.col_cost_ = sf.c
    lp.col_lower_ = _to_highs_inf(sf.xl, highspy)
    lp.col_upper_ = _to_highs_inf(sf.xu, highspy)
    if row_bounds is None:
        lp.row_lower_ = sf.b
        lp.row_upper_ = sf.b
    else:
        # ``_absorb_tiny_entries``'s ranged rows; ``±inf`` is HiGHS's own infinity.
        lp.row_lower_ = np.where(np.isinf(row_bounds[0]), -highspy.kHighsInf, row_bounds[0])
        lp.row_upper_ = np.where(np.isinf(row_bounds[1]), highspy.kHighsInf, row_bounds[1])
    lp.offset_ = float(offset)
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = sf.n
    lp.a_matrix_.num_row_ = sf.m
    lp.a_matrix_.start_ = sf.A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = sf.A.indices.astype(np.int32)
    lp.a_matrix_.value_ = sf.A.data
    if integer and sf.int_idx.size:
        kinds = [highspy.HighsVarType.kContinuous] * sf.n
        for j in sf.int_idx:
            kinds[int(j)] = highspy.HighsVarType.kInteger
        lp.integrality_ = kinds
    st = h.passModel(lp)
    if st == highspy.HighsStatus.kWarning:
        # HiGHS's default 1e-9 dropped a coefficient (``1e-9 x >= 1e-9`` lost its only
        # entry); re-pass at the smallest value it accepts. Only then: the option also
        # moves HiGHS's internal numerics, and set on every model it turned netlib
        # klein1 from a proved infeasible into kUnknown.
        _set_options(h, highspy, [("small_matrix_value", SMALL_MATRIX_VALUE)])
        st = h.passModel(lp)
    if st == highspy.HighsStatus.kOk:
        return st, ""
    a = np.abs(sf.A.data)
    reason = (
        f"HiGHS passModel {str(st).rsplit('.', 1)[-1]}: {int((a <= SMALL_MATRIX_VALUE).sum())} "
        f"|a_ij| <= {SMALL_MATRIX_VALUE:g} dropped, {int((a >= 1e15).sum())} |a_ij| >= 1e15, "
        f"{int((np.abs(sf.b) >= INF).sum())} row right-hand sides at the 1e20 infinity sentinel"
    )
    return st, reason


def _status_name(h) -> str:
    return str(h.getModelStatus()).rsplit(".", 1)[-1]


def _ray(getter) -> Optional[np.ndarray]:
    out = getter()
    if not (isinstance(out, tuple) and len(out) == 3):
        raise RuntimeError(f"unexpected highspy ray API result {type(out)!r}")
    _st, has, vals = out
    return np.asarray(vals, dtype=np.float64) if has else None


#: HiGHS options ``Model.solve(highs_options=...)`` may set (#1620). An allowlist,
#: not a denylist: HiGHS has hundreds of options, and one this route does not know
#: about (``solver="pdlp"``, ``solve_relaxation``, ``user_objective_scale``,
#: ``write_*_to_file``, a new release's additions) must be refused here rather than
#: trusted to be caught by a downstream certificate check. Every entry changes only
#: logging or the *search* -- never a tolerance, a limit, a gap, the problem HiGHS
#: solves, or the process-wide thread scheduler (see :func:`_new_highs`) -- so the
#: route's certificate means the same thing whatever value is passed. Extending it
#: is a review decision: add the option with that argument.
ALLOWED_USER_OPTIONS: frozenset[str] = frozenset(
    {
        # logging
        "output_flag",
        "log_to_console",
        "log_file",
        "log_dev_level",
        "mip_report_level",
        # search strategy
        "presolve",
        "mip_detect_symmetry",
        "mip_allow_restart",
        "mip_heuristic_effort",
        "mip_heuristic_run_rins",
        "mip_heuristic_run_rens",
        "mip_heuristic_run_root_reduced_cost",
        "mip_heuristic_run_zi_round",
        "mip_heuristic_run_shifting",
        "mip_lp_age_limit",
        "mip_pool_age_limit",
        "mip_pool_soft_limit",
        "mip_pscost_minreliable",
        "simplex_scale_strategy",
        "simplex_dual_edge_weight_strategy",
        "simplex_primal_edge_weight_strategy",
        "simplex_price_strategy",
    }
)

#: Why the options a user is most likely to reach for are refused; anything else
#: outside :data:`ALLOWED_USER_OPTIONS` gets the generic message.
RESERVED_USER_OPTIONS: dict[str, str] = {
    "mip_rel_gap": "use Model.solve(gap_tolerance=...)",
    "mip_abs_gap": "use Model.solve(abs_gap_tolerance=...)",
    "mip_max_nodes": "use Model.solve(max_nodes=...)",
    "time_limit": "use Model.solve(time_limit=...)",
    "presolve_rule_off": "the route pins it (#1634: an unsound presolve rule is disabled)",
    "run_crossover": "the route's dual certificates need a crossover basis",
    "small_matrix_value": "the route sets it to pass its standard form exactly",
    "threads": "HiGHS's scheduler is process-wide; a nonzero value breaks later solves",
    "random_seed": "the route pins it for reproducible certificates",
}

_USER_OPTIONS: contextvars.ContextVar[tuple[tuple[str, Any], ...]] = contextvars.ContextVar(
    "discopt_highs_user_options", default=()
)


def validate_user_options(options: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Check ``Model.solve(highs_options=...)`` and return it as ``(key, value)`` pairs.

    Raises ``TypeError`` for a non-mapping or a non-string key and ``ValueError``
    for any option not in :data:`ALLOWED_USER_OPTIONS` (with the reason from
    :data:`RESERVED_USER_OPTIONS` when there is a specific one). A bad *value* for
    an allowed option raises later, from :func:`_set_options`, when the route
    builds its HiGHS instance.
    """
    if not isinstance(options, Mapping):
        raise TypeError(
            "highs_options must be a dict of HiGHS option name -> value, got "
            f"{type(options).__name__}"
        )
    pairs = []
    for key, val in options.items():
        if not isinstance(key, str):
            raise TypeError(f"highs_options keys must be HiGHS option names, got {key!r}")
        if key not in ALLOWED_USER_OPTIONS:
            why = RESERVED_USER_OPTIONS.get(
                key,
                "only logging and search-strategy options are passed through "
                f"(allowed: {', '.join(sorted(ALLOWED_USER_OPTIONS))})",
            )
            raise ValueError(f"highs_options[{key!r}] cannot be set: {why} (#1620)")
        pairs.append((key, val))
    return tuple(pairs)


@contextlib.contextmanager
def user_options(pairs: tuple[tuple[str, Any], ...]) -> Iterator[None]:
    """Apply validated user HiGHS options to every HiGHS instance built in the scope.

    They are set after the route's base options (so ``output_flag=True`` turns the
    HiGHS log on) and before each solve's own options, so a route option -- e.g. the
    #1634 cross-check's ``presolve="off"`` -- always wins over a user one.
    """
    token = _USER_OPTIONS.set(tuple(pairs))
    try:
        yield
    finally:
        _USER_OPTIONS.reset(token)


def _new_highs(highspy, opts: list[tuple[str, Any]]):
    h = highspy.Highs()
    # ``threads`` stays at HiGHS's default (0). HiGHS keeps one process-wide scheduler,
    # sized by the first run in the process, and refuses any later run whose nonzero
    # ``threads`` differs from that size (``run()`` -> kError). The OA/GDP paths and user
    # code run highspy with default options, so pinning a value here made every later
    # LP/MILP solve in the same process return ``error``.
    _set_options(h, highspy, [("output_flag", False), ("random_seed", 0)])
    # #1620: ``Model.solve(highs_options=...)``, checked by validate_user_options.
    _set_options(h, highspy, list(_USER_OPTIONS.get()))
    _set_options(h, highspy, opts)
    return h


# ─────────────────────────────────────────────────────────────
# LP
# ─────────────────────────────────────────────────────────────


def solve_lp_std(sf: StdForm, *, time_limit: Optional[float] = None) -> HighsOutcome:
    """Solve the continuous relaxation of ``sf`` under the §3.1 contract."""
    highspy = require_highspy()
    t0 = time.perf_counter()
    sf = sf.relaxed() if sf.int_idx.size else sf
    stats: dict[str, float] = {"route/lp_milp_backend": 1.0}

    def remaining() -> Optional[float]:
        return None if time_limit is None else float(time_limit) - (time.perf_counter() - t0)

    def done(out: HighsOutcome) -> HighsOutcome:
        out.wall_time = time.perf_counter() - t0
        out.stats = {**stats, **out.stats}
        return out

    # Declared finite bounds at sentinel-adjacent magnitude (the default ±9.999e19
    # box). HiGHS computes in floating point at that scale and can call an infeasible
    # LP optimal; the exact check refuses the point, and one re-solve with those
    # bounds relaxed to infinity then looks for a Farkas ray. A ray of the relaxed
    # problem also proves the declared problem empty, and it is verified against the
    # declared bounds regardless. Nothing else from that re-solve is used.
    huge_lo, huge_hi = _huge_box(sf)
    presolve_off = False
    relaxed = False
    last_reason = ""
    while True:
        rem = remaining()
        if rem is not None and rem <= 0.0:
            return done(HighsOutcome("time_limit", message="LP time budget exhausted"))
        opts: list[tuple[str, Any]] = [
            ("primal_feasibility_tolerance", 1e-7),
            ("dual_feasibility_tolerance", 1e-7),
            ("run_crossover", "on"),
        ]
        if rem is not None:
            opts.append(("time_limit", float(rem)))
        if presolve_off or relaxed:
            opts.append(("presolve", "off"))
        h = _new_highs(highspy, opts)
        stats["highs/version"] = _version_number(h)
        sf_pass = _relax_huge_box(sf, huge_lo, huge_hi) if relaxed else sf
        pass_st, pass_why = _pass_model(h, highspy, sf_pass, integer=False)
        if pass_st == highspy.HighsStatus.kError:
            return done(HighsOutcome("error", message=pass_why))
        if pass_why:
            # kWarning: HiGHS solves a perturbed LP. Every certificate below -- the
            # point's feasibility, the NS bound, the Farkas and primal rays -- is
            # re-verified against ``sf`` itself, so a perturbed answer is refused, not
            # trusted.
            stats["lp/highs_pass_warning"] = 1.0
        run_st = h.run()
        name = _status_name(h)
        info = h.getInfo()
        stats["lp/iters"] = stats.get("lp/iters", 0.0) + float(info.simplex_iteration_count)
        if run_st == highspy.HighsStatus.kError:
            return done(
                HighsOutcome("error", message=f"HiGHS run() error ({name})", highs_status=name)
            )

        if relaxed:
            if name == "kInfeasible":
                y = _ray(h.getDualRay)
                if y is not None and farkas_verified(y, sf):
                    return done(
                        HighsOutcome("infeasible", ray=y, gap_certified=True, highs_status=name)
                    )
            return done(HighsOutcome("error", highs_status=name, message=last_reason))

        if name == "kOptimal":
            sol = h.getSolution()
            x = np.asarray(sol.col_value, dtype=np.float64)
            y = np.asarray(sol.row_dual, dtype=np.float64)
            rc = np.asarray(sol.col_dual, dtype=np.float64)
            why = readback_problem(x, sf, row_dual=y, col_dual=rc) or feasibility_problem(
                x, sf, check_integrality=False
            )
            if why and (huge_lo.any() or huge_hi.any()):
                relaxed = True
                last_reason = f"HiGHS LP optimal: {why}"
                stats["lp/huge_box_relaxed_resolve"] = 1.0
                continue
            if why:
                return done(
                    HighsOutcome("error", message=f"HiGHS LP optimal: {why}", highs_status=name)
                )
            obj = float(sf.c @ x) + sf.obj_const
            thr = CERT_ABS + CERT_REL * abs(obj)
            out = HighsOutcome(
                "feasible", x=x, objective=obj, row_dual=y, col_dual=rc, highs_status=name
            )
            # Every candidate below is a valid lower bound, so the largest one is too.
            ns = ns_bound(y, sf)
            provenance = ""
            if ns is None or obj - ns > thr:
                box = fbbt_box(sf)
                g = ns_bound(y, box)
                if g is not None and (ns is None or g > ns):
                    ns, provenance = g, "ns-fbbt-box"
                if ns is None or obj - ns > thr:
                    # The caller's own time_limit only; the work is capped by count.
                    deadline = None if time_limit is None else t0 + float(time_limit)
                    g, why = exact_ns_bound(y, box, deadline=deadline)
                    if g is not None and (ns is None or g > ns):
                        ns, provenance = g, "ns-exact-dual-correction"
                    elif why:
                        out.message = f"exact dual correction: {why}"
            if provenance:
                out.labels["lp/bound_provenance"] = provenance
                stats["lp/bound_provenance_" + provenance.replace("-", "_")] = 1.0
            if ns is None:
                stats["lp/ns_bound_missing"] = 1.0
                out.message = (
                    "NS safe bound is -inf for the HiGHS dual over the declared box, the FBBT "
                    f"box and after exact dual correction; optimality not certified. {out.message}"
                )
                return done(out)
            gap = obj - ns
            stats["lp/ns_gap"] = float(gap)
            out.bound = min(ns, obj)
            if gap <= thr:
                out.status = "optimal"
                out.gap_certified = True
            else:
                out.message = f"NS safe bound {ns:.12g} is {gap:.3g} below the objective"
            return done(out)

        if name == "kInfeasible":
            y = _ray(h.getDualRay)
            if y is not None and farkas_verified(y, sf):
                return done(
                    HighsOutcome("infeasible", ray=y, gap_certified=True, highs_status=name)
                )
            if y is not None and farkas_verified(y, fbbt_box(sf)):
                out = HighsOutcome("infeasible", ray=y, gap_certified=True, highs_status=name)
                out.labels["lp/infeasible_provenance"] = "farkas-fbbt-box"
                stats["lp/infeasible_provenance_farkas_fbbt_box"] = 1.0
                return done(out)
            last_reason = "infeasible label without a verifiable Farkas ray"
        elif name == "kUnbounded":
            d = _ray(h.getPrimalRay)
            xp: np.ndarray | None = None
            if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
                xp = np.asarray(h.getSolution().col_value, dtype=np.float64)
            point_ok = xp is not None and not (
                readback_problem(xp, sf) or feasibility_problem(xp, sf, check_integrality=False)
            )
            if point_ok and (d is None or not primal_ray_verified(d, sf)):
                # HiGHS can label an LP unbounded with no ray (measured: ``min x`` over a
                # free column and no rows, highspy 1.12 and 1.15). Look for one directly.
                d = recession_ray(sf, time_limit=remaining())
                if d is not None:
                    stats["lp/recession_ray_lp"] = 1.0
            if d is not None and point_ok and primal_ray_verified(d, sf):
                return done(HighsOutcome("unbounded", x=xp, ray=d, highs_status=name))
            last_reason = "unbounded label without a verified ray and feasible point"
        elif name == "kUnboundedOrInfeasible":
            last_reason = "HiGHS could not decide between infeasible and unbounded"
        elif name in ("kTimeLimit", "kIterationLimit"):
            return done(HighsOutcome("time_limit", highs_status=name, message=name))
        elif name == "kModelEmpty" and sf.n == 0 and sf.m == 0:
            # #1385: an LP with no columns and no rows has one feasible point --
            # the empty assignment -- so ``obj_const`` is the optimum and the
            # certificate is exact. This name was not in the dispatch above, so
            # it fell to the catch-all and returned ``error`` with the constant
            # discarded. The shape test is the certificate: we answer only for a
            # model that really is empty, never for a HiGHS status we merely
            # recognise by name.
            return done(
                HighsOutcome(
                    "optimal",
                    x=np.zeros(0, dtype=np.float64),
                    objective=float(sf.obj_const),
                    bound=float(sf.obj_const),
                    gap_certified=True,
                    highs_status=name,
                )
            )
        else:
            return done(HighsOutcome("error", highs_status=name, message=f"HiGHS LP status {name}"))

        if presolve_off:
            if name == "kInfeasible":
                proved, how = phase1_infeasibility_proof(sf, time_limit=remaining())
                stats["lp/phase1_proof_ran"] = 1.0
                if proved:
                    out = HighsOutcome("infeasible", gap_certified=True, highs_status=name)
                    out.labels["lp/infeasible_provenance"] = how
                    stats["lp/infeasible_provenance_" + how.replace("-", "_")] = 1.0
                    return done(out)
                last_reason = f"{last_reason}; {how}"
            return done(HighsOutcome("error", highs_status=name, message=last_reason))
        # One re-solve without presolve: rays and the infeasible/unbounded split are
        # only available on the original model, not a presolved reduction of it.
        presolve_off = True
        stats["lp/presolve_off_resolve"] = 1.0


# ─────────────────────────────────────────────────────────────
# MILP
# ─────────────────────────────────────────────────────────────


def complete_seed(sf: StdForm, x_struct: np.ndarray, n_struct: int) -> Optional[np.ndarray]:
    """Extend a structural point with the logical columns it determines, or ``None``.

    Each logical column must appear in exactly one row, and each row may carry at
    most one logical; then ``s = (b_i - a_i·x) / A_ij`` is exact. HiGHS validates the
    completed start itself, and the result is re-verified after the solve anyway, so
    this is a warm-start hint only.
    """
    x_struct = np.asarray(x_struct, dtype=np.float64).ravel()
    if x_struct.shape != (n_struct,) or not np.all(np.isfinite(x_struct)):
        return None
    full = np.zeros(sf.n)
    full[:n_struct] = x_struct
    if sf.n == n_struct:
        return full
    S = sf.A[:, n_struct:]
    if np.any(np.diff(S.indptr) != 1):
        return None
    rows = S.indices
    if np.unique(rows).size != rows.size:
        return None
    resid = sf.b - sf.A[:, :n_struct] @ x_struct
    full[n_struct:] = resid[rows] / S.data
    return full


#: Node cap on the integer-feasibility probe. Soundness-neutral: a probe that hits
#: it returns no point, and the caller then reports ``error`` exactly as before.
_UNBOUNDED_PROBE_MAX_NODES = 10_000


def _integer_feasible_point(sf: "StdForm", budget, stats: dict):
    """An integer-feasible point of ``sf``, or ``None``. Never raises.

    The witness Meyer's theorem needs to turn "the relaxation has an improving
    recession ray" into "the MILP is unbounded". HiGHS supplies one whenever it
    finds an incumbent, but on a ``kUnbounded`` exit it returns none -- it stopped
    because the objective ran away, not because the system is empty -- so the
    point has to be asked for separately.

    Asking is the same move #1337 makes for ``kInfeasible``: drop the objective,
    keep the rows, the box and the integrality, and solve for feasibility alone.
    The result is a verified incumbent of the same standard form, so it is the
    same kind of evidence an ordinary incumbent would have been.

    The probe keeps its own ``root_check``: without it a ``kInfeasible`` exit would
    come back ``gap_certified`` with no cross-check at all, which is exactly the
    unchecked label #1295/#1320 exist to refuse -- and this function would then
    hand it on as a proof of emptiness.

    Sets ``milp/unbounded_feasibility_probe_infeasible`` when the probe proves the
    integer system EMPTY, which resolves ``kUnboundedOrInfeasible`` the other way.
    A probe that merely runs out of budget or settles nothing sets nothing, and
    the caller reports ``error`` exactly as before.
    """
    if budget is not None and budget <= 0.0:
        return None
    stats["milp/unbounded_feasibility_probe"] = 1.0
    try:
        probe = solve_milp_std(
            dataclasses.replace(sf, c=np.zeros(sf.n, dtype=np.float64), obj_const=0.0),
            time_limit=budget,
            gap_tolerance=1e-4,
            max_nodes=_UNBOUNDED_PROBE_MAX_NODES,
            root_check=True,
        )
    except Exception as exc:  # noqa: BLE001 - the probe may decline; it may not fail a solve
        logger.debug("integer-feasibility probe raised %s: %s", type(exc).__name__, exc)
        return None
    if probe.status == "infeasible" and probe.gap_certified:
        stats["milp/unbounded_feasibility_probe_infeasible"] = 1.0
        return None
    if probe.status in ("optimal", "feasible") and probe.x is not None:
        pt = np.asarray(probe.x, dtype=np.float64)
        # Verified here, not taken on the probe's word: the witness is the whole
        # evidence for an ``unbounded`` certificate, so it is checked against the
        # rows, the box and the integrality of the form actually being decided.
        if _point_is_integer_feasible(sf, pt):
            return pt
        logger.debug("integer-feasibility probe returned a point that fails re-checking")
    return None


def _point_is_integer_feasible(sf: "StdForm", x: np.ndarray, tol: float = 1e-6) -> bool:
    """Whether ``x`` satisfies ``A x = b``, the box, and ``sf``'s integrality."""
    if x.shape != (sf.n,) or not np.all(np.isfinite(x)):
        return False
    resid = np.abs(sf.A @ x - sf.b)
    scale = np.abs(sf.A) @ np.abs(x)
    if np.any(resid > tol * (1.0 + np.abs(sf.b)) + 1e-9 * scale):
        return False
    if np.any(x < sf.xl - tol) or np.any(x > sf.xu + tol):
        return False
    if sf.int_idx.size:
        xi = x[sf.int_idx]
        if np.any(np.abs(xi - np.round(xi)) > INT_TOL):
            return False
    return True


def _verified_mip_point(
    sf: "StdForm", x: Optional[np.ndarray]
) -> Optional[tuple[np.ndarray, float]]:
    """``(point, objective)`` when ``x`` passes the incumbent verifier, else ``None``.

    The same two gates every HiGHS incumbent passes (:func:`readback_problem`, then
    :func:`feasibility_problem` with integrality), and the point is returned at its
    integral realisation -- the one those gates tested (#1380) -- with the objective
    recomputed from it. So a point this returns is evidence of exactly the standing a
    HiGHS incumbent has.
    """
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float64).ravel()
    if readback_problem(x, sf) or feasibility_problem(x, sf, check_integrality=True):
        return None
    from discopt.validation.feasibility import snap_integer_columns

    x = snap_integer_columns(x, sf.int_idx)
    return x, float(sf.c @ x) + sf.obj_const


def _fixed_integer_lp(sf: "StdForm", x: np.ndarray, time_limit: Optional[float]) -> HighsOutcome:
    """The NS-safe LP of ``sf`` with every integer column fixed at ``round(x_j)``.

    #1509: a HiGHS ``kOptimal`` claims in particular that its incumbent is optimal
    for its OWN integer assignment. That claim is checkable at one LP's cost -- and
    the check does not route through HiGHS's MIP presolve/propagation, which is where
    the #1509 false optimum came from.
    """
    xl = sf.xl.copy()
    xu = sf.xu.copy()
    xi = np.round(np.asarray(x, dtype=np.float64)[sf.int_idx])
    xl[sf.int_idx] = xi
    xu[sf.int_idx] = xi
    return solve_lp_std(dataclasses.replace(sf, xl=xl, xu=xu), time_limit=time_limit)


def _repair_refused_incumbent(
    sf: "StdForm",
    x: np.ndarray,
    root: HighsOutcome,
    time_limit: Optional[float],
    stats: dict,
) -> Optional[tuple[np.ndarray, float]]:
    """A verified point carrying the integers of a refused HiGHS incumbent, or ``None``.

    #1654: the LP of ``sf`` with every integer column fixed at ``round(x_j)``
    (:func:`_fixed_integer_lp`), re-verified by :func:`_verified_mip_point`. Only a
    point that passes the same gates as any HiGHS incumbent is returned, so the
    repair can add a feasible point and never an infeasible one. Skipped when the
    root LP already proved the relaxation empty (that branch reports infeasible), or
    when ``x`` is not finite.
    """
    if root.status == "infeasible" or not sf.int_idx.size:
        return None
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.shape != (sf.n,) or not np.all(np.isfinite(x)):
        return None
    if time_limit is not None and time_limit <= 0.0:
        stats["milp/incumbent_repair_skipped"] = 1.0
        return None
    stats["milp/incumbent_repair_ran"] = 1.0
    fx = _fixed_integer_lp(sf, x, time_limit)
    pt = _verified_mip_point(sf, fx.x) if fx.status == "optimal" else None
    if pt is not None:
        stats["milp/incumbent_repaired"] = 1.0
        stats["milp/fixed_int_objective"] = float(pt[1])
    return pt


def logical_column_scales(sf: StdForm, n_struct: Optional[int]) -> Optional[np.ndarray]:
    """Power-of-two factors that equilibrate the route's own logical columns, or ``None``.

    #1537: the standard form ends every inequality row with a unit logical
    ``a x + s = b, s >= 0``. Multiply that row by 1e6 -- the same model -- and the
    logical is 2.5e-7 of its row's largest entry, below :data:`UNSCALABLE_OPEN_RATIO`.
    That is not a false alarm: on the #1295-class panel HiGHS then pruned the true
    optimum on 3/9 such models (caught only by the #1509 refutation), and with the
    logical's coefficient raised to its row's magnitude it certified all of them
    correctly. Before this, ×1e6 rows on clean MILPs withdrew 12/12 correct
    certificates.

    So the logical is rescaled rather than excused: ``s = f s'`` with ``f`` a power of
    two is exact in floating point, changes no other row and (``c_s = 0``) not the
    objective, and puts the column's entry within a factor 2 below its row's largest. Only
    columns at index ``>= n_struct`` (the route's own, never published), with one
    nonzero, zero cost, continuous, and *below* the cap are touched, so a model this
    route certifies today is handed to HiGHS unchanged. A user's own tiny-coefficient
    column is not touched and still decertifies: that class was measured wrong
    (#1295) and is the user's scaling, not the route's.

    #1621: the same holds for a user's tiny entry that shares the logical's row. In a
    big-M row ``x - M z + s = 0`` the logical is small only because ``x`` is: ``x``'s
    own entry sits at ``1/M`` of the row too. Rescaling ``s`` there removes the one
    entry #1295 measures (an open column) and leaves the user's ``1/M`` entry -- the
    actual trap -- in a model now handed to HiGHS as trusted. Measured on the #1621
    blending MILP: certified ``infeasible`` at M = 1e9 and ``optimal`` 0 at M = 1e10
    (true optimum 473,958.43, x = 0 feasible). So a logical is rescaled only when
    every OTHER entry of its row is within the cap of the row's largest -- the row is
    well scaled apart from the route's own column, which is the #1537 premise.
    """
    if n_struct is None or n_struct >= sf.n or sf.A.nnz == 0:
        return None
    A = sp.csc_matrix(sf.A)  # noqa: N806
    absA = abs(A).tocoo()  # noqa: N806
    row_max = np.zeros(sf.m)
    np.maximum.at(row_max, absA.row, absA.data)
    scales = np.ones(sf.n)
    cand = _logical_columns(sf)
    cand[:n_struct] = False
    # Smallest nonzero entry of each row over the columns that are not rescaling
    # candidates (a logical is the only entry of its column, so this is "every other
    # entry of the row" for each candidate).
    keep = ~cand[absA.col] & (absA.data > 0.0)
    row_min_other = np.full(sf.m, np.inf)
    np.minimum.at(row_min_other, absA.row[keep], absA.data[keep])
    lo_open, hi_open = sf.xl <= -INF, sf.xu >= INF
    for j in np.flatnonzero(cand):
        i, a = int(A.indices[A.indptr[j]]), abs(float(A.data[A.indptr[j]]))
        if row_max[i] <= 0.0 or a / row_max[i] >= UNSCALABLE_OPEN_RATIO:
            continue
        if row_min_other[i] / row_max[i] < UNSCALABLE_OPEN_RATIO:
            continue  # #1621: the user's own entry is below the cap; not ours to excuse
        # floor, not round: the rescaled entry lands in (row_max / 2, row_max], so it
        # never becomes its row's new largest and lowers every other entry's ratio.
        f = float(2.0 ** np.floor(np.log2(row_max[i] / a)))
        if f * a > row_max[i]:
            f /= 2.0
        # A finite side must stay an ordinary finite side after ``/ f``; a sentinel
        # side stays the sentinel. Anything else would change what "open" means.
        new_lo, new_hi = sf.xl[j] / f, sf.xu[j] / f
        if not lo_open[j] and (abs(sf.xl[j]) >= READBACK_LIMIT or new_lo * f != sf.xl[j]):
            continue
        if not hi_open[j] and (abs(sf.xu[j]) >= READBACK_LIMIT or new_hi * f != sf.xu[j]):
            continue
        scales[j] = f
    return scales if np.any(scales != 1.0) else None


def _scale_logicals(sf: StdForm, f: np.ndarray) -> StdForm:
    """``sf`` in the variables ``x'_j = x_j / f_j`` (sentinel sides kept)."""
    A = sp.csc_matrix(sf.A) @ sp.diags(f, format="csc")  # noqa: N806
    xl = np.where(sf.xl <= -INF, sf.xl, sf.xl / f)
    xu = np.where(sf.xu >= INF, sf.xu, sf.xu / f)
    return StdForm.from_arrays(sf.c * f, A, sf.b, xl, xu, sf.obj_const, sf.int_idx)


#: #1654: safety factor on the round-off margin of the activity slack ``d`` that
#: coefficient tightening removes. ``d`` is a floating-point sum of the row's terms,
#: whose error is at most ``(k + 2) * eps * sum|terms|`` for ``k`` terms; shrinking ``d``
#: by this factor times that bound only makes the tightening weaker, never invalid
#: (any ``0 <= d' <= d`` is exact). A margin relative to the row scale itself would
#: be set by the big-M being removed: 1e-9 of 1e9 left ``M`` at 11 where 10 is exact.
_COEF_TIGHTEN_SAFETY = 8.0


def coefficient_tightened(sf: StdForm) -> tuple[Optional[StdForm], int]:
    """``sf`` with every binary's big-M coefficient shrunk to its row's activity bound.

    #1654: the classic MIP coefficient tightening (Savelsbergh 1994; Achterberg 2007
    §10.1). Take a one-sided row ``a x <= U`` -- in this standard form, ``a x + s = b``
    with a single-entry logical ``s`` whose range has one finite side -- and a binary
    ``y`` in it with coefficient ``a_k``. With ``maxact`` the row's maximal activity
    over the declared box of every OTHER column:

    * ``a_k > 0`` and ``d = U - maxact > 0``: the row is slack by ``d`` at ``y = 0``,
      so ``a_k -> a_k - d`` and ``U -> U - d``;
    * ``a_k < 0`` and ``d = U - (maxact + a_k) > 0``: the row is slack by ``d`` at
      ``y = 1``, so ``a_k -> a_k + d``.

    Each rewrite leaves the row's restriction at ``y = 0`` and at ``y = 1`` exactly as
    it was over the box, so the set of INTEGER-feasible points -- and therefore the
    MILP's optimum, every feasible point and every valid bound -- is unchanged; only
    the LP relaxation tightens. On a big-M row ``s_i + p_i <= s_j + M (1 - y)`` over
    ``s in [0, 200]`` it replaces ``M = 1e9`` by ``209``, which removes the trap that
    makes the model unsolvable in floating point: HiGHS accepts ``y`` within 1e-6 of
    integral, and ``1e-6 * 1e9`` is a thousand units of row slack. Any ``d' in [0, d]``
    is also exact, so ``d`` is reduced by a round-off margin before use.

    Only binary columns (integer, box ``[0, 1]``) are rewritten; a row with an
    unbounded or sentinel-magnitude column has no finite activity bound and is left
    alone; ranged and equality rows are left alone. Returns ``(form, n_rewritten)``,
    ``form`` ``None`` when nothing changed.
    """
    if not sf.int_idx.size or sf.A.nnz == 0:
        return None, 0
    is_bin = np.zeros(sf.n, dtype=bool)
    ii = sf.int_idx
    is_bin[ii[(sf.xl[ii] == 0.0) & (sf.xu[ii] == 1.0)]] = True
    if not is_bin.any():
        return None, 0
    logical = _logical_columns(sf)
    csc = sp.csc_matrix(sf.A)
    # row -> its logical column (exactly one per one-sided inequality row)
    row_logical = np.full(sf.m, -1, dtype=np.int64)
    n_log = np.zeros(sf.m, dtype=np.int64)
    for j in np.flatnonzero(logical):
        i = int(csc.indices[csc.indptr[j]])
        row_logical[i] = j
        n_log[i] += 1
    finite_box = (np.abs(sf.xl) < READBACK_LIMIT) & (np.abs(sf.xu) < READBACK_LIMIT)
    A = sp.csr_matrix(sf.A, copy=True)  # noqa: N806
    A.sort_indices()
    b = sf.b.copy()
    n_rewritten = 0
    for i in range(sf.m):
        if n_log[i] != 1:
            continue
        j_s = int(row_logical[i])
        lo, hi = A.indptr[i], A.indptr[i + 1]
        cols = A.indices[lo:hi]
        if not np.any(is_bin[cols]):
            continue
        k_s = lo + int(np.flatnonzero(cols == j_s)[0])
        sig = float(A.data[k_s])
        ends = sorted((sig * sf.xl[j_s], sig * sf.xu[j_s]))
        lo_fin = abs(ends[0]) < READBACK_LIMIT
        hi_fin = abs(ends[1]) < READBACK_LIMIT
        if lo_fin == hi_fin:
            continue  # ranged (or free) row: not a one-sided inequality
        # ``a x = b - sig*s``: ``<= b - min(sig*s)`` when that end is finite, else
        # ``>= b - max(sig*s)``. Work in ``<=`` form; ``sgn`` maps back.
        sgn = 1.0 if lo_fin else -1.0
        U = sgn * (b[i] - (ends[0] if lo_fin else ends[1]))  # noqa: N806
        others = np.flatnonzero(cols != j_s) + lo
        if not np.all(finite_box[A.indices[others]]):
            continue
        for k_ent in others:
            kq = int(k_ent)
            if not is_bin[int(A.indices[kq])]:
                continue
            g = sgn * A.data[others]
            cj = A.indices[others]
            terms = np.maximum(g * sf.xl[cj], g * sf.xu[cj])
            gk = sgn * float(A.data[kq])
            maxact_rest = float(terms.sum()) - max(gk, 0.0)
            scale = float(np.abs(terms).sum()) + abs(U) + abs(gk)
            margin = _COEF_TIGHTEN_SAFETY * (terms.size + 2) * np.finfo(float).eps * scale
            if gk > 0.0:
                d = U - maxact_rest - margin
                if not 0.0 < d < gk:
                    continue
                gk_new, U = gk - d, U - d  # noqa: N806
                b[i] -= sgn * d
            else:
                d = U - (maxact_rest + gk) - margin
                if not 0.0 < d < -gk:
                    continue
                gk_new = gk + d
            A.data[kq] = sgn * gk_new
            n_rewritten += 1
    if not n_rewritten:
        return None, 0
    return (
        StdForm.from_arrays(sf.c, sp.csc_matrix(A), b, sf.xl, sf.xu, sf.obj_const, sf.int_idx),
        n_rewritten,
    )


def _coef_tighten_enabled() -> bool:
    """``DISCOPT_MILP_COEF_TIGHTEN`` (default ON; ``=0`` restores the pre-#1654 route).

    Graduated on introduction by the §5 panel recorded in
    ``docs/dev/issue-1654-coef-tighten-panel-2026-10-05.md`` (certified 22/48 -> 48/48
    on three big-M families, 0 soundness violations); the opt-out keeps the
    pre-#1654 route reachable for A/B measurement.
    """
    return os.environ.get("DISCOPT_MILP_COEF_TIGHTEN", "1") != "0"


def _coefficient_tightening_rescue(
    sf: StdForm, out: HighsOutcome, kw: dict[str, Any], t0: float
) -> HighsOutcome:
    """#1654: re-solve a MILP the route could not certify on its coefficient-tightened
    form (:func:`coefficient_tightened`), and adopt what that solve proves.

    Runs only when the primary result is NOT certified, so a model this route
    certifies today is untouched. The tightened form has exactly ``sf``'s
    integer-feasible set, so its certified bound and infeasibility are statements
    about ``sf`` -- but that is a claim about exact arithmetic, so the tightened solve
    goes through the whole certified pipeline (#1295, #1509, #1621, #1634, #1612), its
    incumbent is re-verified on ``sf`` itself, and a verified point of the primary
    below its bound refutes it. Anything less returns the primary, improved at most by
    a better verified incumbent.
    """
    sf_ct, n_rw = coefficient_tightened(sf)
    if sf_ct is None:
        return out
    out.stats["milp/coef_tightened_entries"] = float(n_rw)
    kw2 = dict(kw)
    if kw["time_limit"] is not None:
        kw2["time_limit"] = float(kw["time_limit"]) - (time.monotonic() - t0)
        if kw2["time_limit"] <= 0.0:
            out.stats["milp/coef_tighten_skipped"] = 1.0
            return out
    # The primary's user start is a point of ``sf`` and so of the tightened form too.
    ct = _certified_milp(sf_ct, kw2)
    if ct.gap_certified and ct.status == "optimal":
        ct = _tighten_feasibility_artefact(sf_ct, ct, kw2, time.monotonic())
    out.stats["milp/coef_tighten_ran"] = 1.0
    out.stats["milp/coef_tighten_time"] = float(ct.wall_time)
    out.labels["milp/coef_tighten_status"] = ct.status
    pt = _verified_mip_point(sf, ct.x)
    if pt is None and ct.x is not None:
        rem = (
            None if kw["time_limit"] is None else float(kw["time_limit"]) - (time.monotonic() - t0)
        )
        pt = _repair_refused_incumbent(sf, ct.x, ct, rem, out.stats)
    mine = _verified_mip_point(sf, out.x)
    best = min((p for p in (pt, mine) if p is not None), key=lambda p: p[1], default=None)

    def merged(base: HighsOutcome, primary: HighsOutcome) -> HighsOutcome:
        base.stats = {**primary.stats, **base.stats}
        base.labels = {**primary.labels, **base.labels}
        base.node_count += primary.node_count
        base.iterations += primary.iterations
        base.wall_time = time.monotonic() - t0
        return base

    why = None
    if ct.gap_certified and ct.status == "infeasible":
        if best is not None:
            why = (
                f"tightened form certified infeasible, but a verified point ({best[1]:.12g}) exists"
            )
        else:
            ct.labels["milp/bound_provenance"] = "coef-tightened"
            ct.x = None
            ct = merged(ct, out)
            ct.labels.pop("milp/certificate", None)
            return ct
    elif ct.gap_certified and ct.status == "optimal" and ct.bound is not None:
        claim = float(ct.bound)
        if best is None:
            why = "tightened form certified optimal, but its incumbent fails on the model"
        elif claim - best[1] > CERT_ABS + CERT_REL * abs(claim):
            why = f"tightened bound {claim:.12g} is above a verified point ({best[1]:.12g})"
        else:
            x, obj = best
            bound = min(claim, obj)
            ct.x, ct.objective, ct.bound = x, obj, bound
            if not _gap_closed(obj, bound, kw):
                ct.status, ct.gap_certified = "feasible", False
                ct.labels["milp/certificate"] = "declined"
            ct.labels["milp/bound_provenance"] = (
                f"{ct.labels.get('milp/bound_provenance', 'highs-fp')} on the "
                "coefficient-tightened form (#1654)"
            )
            ct.message = ct.message or "certified on the coefficient-tightened form (#1654)"
            ct = merged(ct, out)
            # The primary's decline is superseded; only the tightened solve's own
            # verdict may label the result.
            if ct.gap_certified:
                ct.labels.pop("milp/certificate", None)
            return ct
    else:
        why = f"tightened solve not certified ({ct.status}: {ct.message or 'no message'})"
    out.labels["milp/coef_tighten_declined"] = why
    # No certificate either way: keep the primary, with the better verified point.
    if best is not None and (out.objective is None or best[1] < out.objective - 1e-12):
        out.x, out.objective = best
        out.stats["milp/incumbent_from_coef_tighten"] = 1.0
        if out.status in ("error", "time_limit", "node_limit"):
            out.status = "feasible"
        bounds = [b for b in (out.bound, ct.root_bound, out.root_bound) if b is not None]
        out.bound = min(max(bounds), best[1]) if bounds else None
    return out


def solve_milp_std(
    sf: StdForm,
    *,
    time_limit: Optional[float],
    gap_tolerance: float,
    abs_gap_tolerance: Optional[float] = None,
    max_nodes: int,
    initial_point: Optional[np.ndarray] = None,
    n_struct: Optional[int] = None,
    root_check: bool = True,
) -> HighsOutcome:
    """Solve the MILP ``sf`` under the §3.2 contract, with the route's own
    under-scaled logical columns rescaled exactly first (#1537,
    :func:`logical_column_scales`), and every certificate held against a second,
    presolve-free HiGHS solve (#1634, :func:`_cross_check_presolve`); see
    :func:`_solve_milp_std`."""
    kw: dict[str, Any] = dict(
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        abs_gap_tolerance=abs_gap_tolerance,
        max_nodes=max_nodes,
        initial_point=initial_point,
        n_struct=n_struct,
        root_check=root_check,
    )
    # ``monotonic``, not ``perf_counter``: this clock only budgets the #1612 re-solve,
    # and must not shift the route's own ``perf_counter`` deadline reads.
    t0 = time.monotonic()
    out = _certified_milp(sf, kw)
    if out.gap_certified and out.status == "optimal":
        out = _tighten_feasibility_artefact(sf, out, kw, t0)
    if not out.gap_certified and out.status != "unbounded" and _coef_tighten_enabled():
        out = _coefficient_tightening_rescue(sf, out, kw, t0)
    return out


def _certified_milp(sf: StdForm, kw: dict[str, Any]) -> HighsOutcome:
    """One HiGHS MILP solve with every route certificate check (#1537/#1621, #1634)."""
    out = _solve_milp_scaled(sf, **kw)
    if out.gap_certified and out.status in ("optimal", "infeasible"):
        return _cross_check_presolve(sf, out, kw)
    return out


def _tighten_feasibility_artefact(
    sf: StdForm, out: HighsOutcome, kw: dict[str, Any], t0: float
) -> HighsOutcome:
    """#1612: re-solve at a tighter feasibility tolerance when HiGHS's tolerance, not its
    search, is what the certified pair's closure rests on.

    HiGHS accepts an incumbent that violates rows by up to ``mip_feasibility_tolerance``,
    so its objective ``o_H`` can be better than any truly feasible point with the same
    integers -- by ``delta = o_fx - o_H``, where ``o_fx`` is the separate fixed-integer
    LP the #1509 check already solves. HiGHS stops on that incumbent, so its bound
    carries the same ``delta``. Once ``Model.solve``'s incumbent repair moves the
    objective back to ``o_fx`` the published gap is ``delta`` wide, and with HiGHS's
    1e-6 tolerance against the default 1e-6 absolute gap it lands on the stop rule's
    edge: measured on the issue's interdiction model at ``gap_tolerance=1e-9``,
    ``delta = 9.99999997e-7`` against ``abs_gap = 1e-6``, and the #1551 evaluation-error
    check (``1.4e-14``) then withdrew the certificate and the bound.

    The trigger: ``delta > 0`` and the stop rule no longer closes once ``delta`` is
    charged against the repaired pair a second time -- the margin left is smaller than
    the tolerance artefact itself, so the closure is the artefact's doing. The whole
    certified pipeline is then run again at :data:`CROSS_TIGHT_FEAS_TOL`. Its result
    replaces the primary only if it is a certified ``optimal`` of its own (every route
    check, the #1634 presolve-free cross-solve included), no verified point of either
    solve lies below either bound, and the two bounds agree within the #1640 tolerance
    slack. Anything else -- no budget, no verdict, a disagreement -- returns the
    primary result unchanged: this never withdraws a certificate.
    """
    o_fx = out.stats.get("milp/fixed_int_objective")
    if o_fx is None or out.objective is None or out.bound is None:
        return out
    delta = float(o_fx) - float(out.objective)
    if not delta > 0.0 or _gap_closed(float(o_fx) + delta, float(out.bound), kw):
        return out
    out.stats["milp/feas_artefact_delta"] = delta
    kw2 = dict(kw, feasibility_tolerance=CROSS_TIGHT_FEAS_TOL)
    if kw["time_limit"] is not None:
        kw2["time_limit"] = float(kw["time_limit"]) - (time.monotonic() - t0)
        if kw2["time_limit"] <= 0.0:
            out.stats["milp/feas_artefact_tight_skipped"] = 1.0
            return out
    tight = _certified_milp(sf, kw2)
    out.stats["milp/feas_artefact_tight_ran"] = 1.0
    out.stats["milp/feas_artefact_tight_time"] = float(tight.wall_time)
    why = None
    if not (tight.gap_certified and tight.status == "optimal" and tight.bound is not None):
        why = f"tight re-solve status {tight.status}"
    else:
        tb, ob = float(tight.bound), float(out.bound)
        pts = [float(o_fx)] + [
            p[1] for p in (_verified_mip_point(sf, out.x), _verified_mip_point(sf, tight.x)) if p
        ]
        c_max = float(np.max(np.abs(sf.c))) if sf.c.size else 0.0
        a_max = float(np.max(np.abs(sf.A.data))) if sf.A.nnz else 0.0
        slack = CROSS_TIGHT_SHIFT_FACTOR * CROSS_FEAS_TOL * max(1.0, c_max) * max(1.0, a_max)
        if any(b - p > CERT_ABS + CERT_REL * abs(b) for b in (tb, ob) for p in pts):
            why = "a verified point lies below one of the two bounds"
        elif abs(tb - ob) > slack:
            why = f"bounds {tb:.12g} and {ob:.12g} differ by more than the slack {slack:.3g}"
    if why is not None:
        out.labels["milp/feas_artefact_tight_declined"] = why
        return out
    tight.stats = {**out.stats, **tight.stats, "milp/feas_artefact_tight_adopted": 1.0}
    tight.labels = {
        **out.labels,
        **tight.labels,
        "milp/bound_provenance": f"{tight.labels.get('milp/bound_provenance', 'highs-fp')}, "
        f"mip_feasibility_tolerance={CROSS_TIGHT_FEAS_TOL:g} (#1612)",
    }
    tight.node_count += out.node_count
    tight.iterations += out.iterations
    tight.wall_time = time.monotonic() - t0
    return tight


def _solve_milp_scaled(sf: StdForm, *, presolve: bool = True, **kw: Any) -> HighsOutcome:
    """:func:`_solve_milp_std` with the #1537 logical rescaling and its #1621 cross-check."""
    kw = dict(kw, presolve=presolve)
    f = logical_column_scales(sf, kw["n_struct"])
    if f is None:
        return _solve_milp_std(sf, **kw)
    t0 = time.perf_counter()
    out = _solve_milp_std(_scale_logicals(sf, f), **kw)
    # Back to ``sf``'s variables: x = f x'; a reduced cost is per unit of the column,
    # d = d' / f; a primal ray is a direction in column space. Row duals and a Farkas
    # ray live in row space, and the rows were not touched.
    if out.x is not None:
        out.x = np.asarray(out.x, dtype=np.float64) * f
    if out.col_dual is not None:
        out.col_dual = np.asarray(out.col_dual, dtype=np.float64) / f
    if out.ray is not None and out.status == "unbounded" and np.size(out.ray) == sf.n:
        out.ray = np.asarray(out.ray, dtype=np.float64) * f
    # The rescaling is exact for the MODEL, not for its tolerances: the slack's bound
    # ``s' >= -tol`` is ``s >= -f tol`` in ``sf``'s variables, so a point can be
    # feasible in the scaled form and violate a row of ``sf`` by up to ``f tol``. On
    # ``min -x + 3z, x <= M z`` (M = 1e8, #1414's class) that let x = ub, z = 0 -- a
    # 10-unit violation, 1.5e-7 in the scaled slack -- be certified at -10 against a
    # true -7. So the mapped point is re-verified on ``sf`` itself, with the same
    # checks the route applies to every HiGHS point; if it fails, the rescaled answer
    # is discarded and ``sf`` is solved as given (the pre-#1537 path, whose #1295
    # guard then decides what may be certified).
    if out.x is not None:
        why = readback_problem(out.x, sf) or feasibility_problem(out.x, sf, check_integrality=True)
        if why:
            if kw["time_limit"] is not None:
                kw["time_limit"] = max(0.0, float(kw["time_limit"]) - (time.perf_counter() - t0))
            plain = _solve_milp_std(sf, **kw)
            plain.stats["milp/logicals_rescale_refused"] = 1.0
            plain.labels["milp/logicals_rescale_refused"] = why
            return plain
    out.stats["milp/logicals_rescaled"] = float(np.count_nonzero(f != 1.0))
    if out.gap_certified and out.status in ("optimal", "infeasible"):
        return _cross_check_rescaled(sf, out, kw, t0)
    return out


def _cross_check_rescaled(
    sf: StdForm, out: HighsOutcome, kw: dict[str, Any], t0: float
) -> HighsOutcome:
    """#1621: hold a certificate earned on the rescaled form against ``sf`` itself.

    The incumbent re-verification in :func:`solve_milp_std` covers the POINT of an
    ``optimal`` result; it says nothing about the bound, and an ``infeasible`` has no
    point at all. Both certificates were measured false through the rescaling (#1621:
    ``infeasible`` at M = 1e9 and ``optimal`` 0 at M = 1e10 on a MILP whose optimum is
    473,958.43). So ``sf`` is solved as given and its VERIFIED incumbent, if any, is
    held against the rescaled claim: any verified point refutes ``infeasible``, and a
    verified point below the certified bound by more than the route's own equality
    yardstick (``CERT_ABS + CERT_REL |bound|``, as in the #1509 refutation) refutes
    ``optimal``. A refuted claim is discarded and the plain solve is published -- the
    pre-#1537 path, whose #1295 guard then decides what may be certified.

    This is a falsifier, not a proof, in the #1509 sense: a cross-solve that yields no
    verified point is no evidence either way and the rescaled certificate stands. A
    cross-check that cannot run for want of budget is a safety net that never ran, so
    the certificate is withdrawn (the #1309 rule).
    """
    kw = dict(kw)
    if kw["time_limit"] is not None:
        kw["time_limit"] = max(0.0, float(kw["time_limit"]) - (time.perf_counter() - t0))
        if kw["time_limit"] <= 0.0:
            out.stats["milp/rescale_cross_check_skipped"] = 1.0
            out.labels["milp/certificate"] = "declined"
            out.gap_certified = False
            out.message = (
                "certificate from the rescaled form withdrawn: no time budget left to "
                "cross-check it against the unscaled model (#1621)"
            )
            if out.status == "optimal":
                out.status = "feasible"
                out.bound = out.root_bound
                if out.bound is not None and out.objective is not None:
                    out.bound = min(out.bound, out.objective)
            else:
                out.status, out.bound = "error", None
            return out
    plain = _solve_milp_std(sf, **kw)
    out.stats["milp/rescale_cross_check_ran"] = 1.0
    pt = _verified_mip_point(sf, plain.x)
    refuted = None
    if pt is not None:
        if out.status == "infeasible":
            refuted = f"a verified point of objective {pt[1]:.12g} exists"
        else:
            # A certified optimum's claim is its bound (``objective`` when HiGHS gave
            # none: certified means the incumbent is optimal).
            claim = out.bound if out.bound is not None else out.objective
            if claim is not None and claim - pt[1] > CERT_ABS + CERT_REL * abs(claim):
                refuted = f"bound {claim:.12g} is above a verified point's objective {pt[1]:.12g}"
    if refuted is None:
        return out
    logger.warning(
        "HiGHS MILP route: the rescaled-form %s certificate is refuted on the unscaled "
        "model (%s); publishing the unscaled solve instead (#1621)",
        out.status,
        refuted,
    )
    plain.stats["milp/rescale_certificate_refuted"] = 1.0
    plain.labels["milp/rescale_certificate_refuted"] = f"{out.status}: {refuted}"
    return plain


def _withdraw(out: HighsOutcome, why: str) -> HighsOutcome:
    """#1634: strip a HiGHS MILP certificate the presolve cross-check could not confirm.

    Mirrors ``decertify_root_check`` (#1309/#1320): ``optimal`` keeps its verified
    incumbent as an uncertified ``feasible`` over the NS-safe root bound; ``infeasible``
    has no incumbent and becomes ``error``.
    """
    out.labels["milp/certificate"] = "declined"
    out.gap_certified = False
    out.message = f"HiGHS MILP certificate withdrawn: {why} (#1634)"
    if out.status == "optimal":
        out.status = "feasible"
        out.bound = out.root_bound
        if out.bound is not None and out.objective is not None:
            out.bound = min(out.bound, out.objective)
    else:
        out.status, out.bound = "error", None
    out.labels["milp/bound_provenance"] = "root-ns" if out.bound is not None else "none"
    return out


#: #1655: budget of the #1634 presolve-free cross-solve, as a multiple of the primary
#: solve's wall time (floored at :data:`PRESOLVE_CROSS_MIN_BUDGET` seconds).
PRESOLVE_CROSS_BUDGET_FACTOR = 20.0
PRESOLVE_CROSS_MIN_BUDGET = 2.0

#: #1655: magnitude cap on the integral matrix and cost entries of
#: :func:`presolve_exact_class`.
PRESOLVE_EXACT_MAX_COEF = 1000.0


def presolve_exact_class(sf: StdForm) -> bool:
    """Is ``sf`` in the class where HiGHS's absolute presolve tolerances are exact?

    #1655: the #1634 cross-check exists because HiGHS's MIP presolve decides on
    ABSOLUTE tolerances (1e-7 on costs, 1e-6 on rows) and, on a badly scaled model,
    treats a nonzero difference as a tie -- #1634's witness fixed ``x2`` on a cost
    difference of ``0.0064 / 4.9e5 = 1.3e-8``. When every matrix entry and cost is an
    integer of magnitude at most :data:`PRESOLVE_EXACT_MAX_COEF` = 1e3, and every
    right-hand side and finite bound an integer below 2**52, the quantities that
    mechanism compares are exact: a cost difference per unit of a column,
    ``c_j/a_ij - c_k/a_ik``, is a rational with denominator at most 1e6, so it is 0 or
    at least 1e-6 -- ten times the 1e-7 tie tolerance -- and a row-activity
    difference over integral bounds is 0 or at least 1. Every #1634 witness and
    generator instance has non-integral entries spanning nine decades; the class
    excludes all of them by construction. Every other route check (#1295, #1410,
    #1509, the NS root bound) still runs on such a model.
    """
    if sf.A.nnz == 0:
        return False
    vals = [np.asarray(sf.A.data, dtype=np.float64), np.asarray(sf.c, dtype=np.float64)]
    for v in vals:
        if not (np.all(np.abs(v) <= PRESOLVE_EXACT_MAX_COEF) and np.all(v == np.round(v))):
            return False
    lim = 2.0**52
    rest = [np.asarray(sf.b, dtype=np.float64)]
    fin_l = sf.xl[sf.xl > -READBACK_LIMIT]
    fin_u = sf.xu[sf.xu < READBACK_LIMIT]
    for v in (*rest, fin_l, fin_u):
        a = np.asarray(v, dtype=np.float64)
        if not (np.all(np.abs(a) < lim) and np.all(a == np.round(a))):
            return False
    return True


def _cross_check_presolve(sf: StdForm, out: HighsOutcome, kw: dict[str, Any]) -> HighsOutcome:
    """#1634: hold a HiGHS MILP certificate against a second, presolve-free solve.

    HiGHS's MIP presolve and tree decide on ABSOLUTE tolerances (1e-7 on costs, 1e-6
    on rows). On a badly scaled MILP those stop meaning what they say, and HiGHS then
    prunes a strictly feasible optimum and certifies the wrong value with every route
    guard passing: #1634's witnesses certified 0.02082 against a true 0.00801 (slack
    >= 72). One mechanism is pinned -- the parallel-column rule, now off
    (:data:`MILP_PRESOLVE_RULE_OFF`) -- but measured on the #1634 generator panel the
    class is wider than one rule: with it off, an always-on reduction still cut the
    optimum (6-column panel seed 965, at the first presolve step), and with presolve
    off entirely HiGHS's own tree did (seed 298). Neither configuration is sound on
    its own on this class; on the panel they never failed on the same instance.

    So the primary result is held against the other configuration's VERIFIED
    incumbent, the #1509/#1621 falsifier: a verified point refutes ``infeasible``, and
    a verified point below the certified bound by more than the route's equality
    yardstick (``CERT_ABS + CERT_REL |bound|``) refutes ``optimal``. A refuted
    certificate is declined loudly: the best verified point is published as
    ``feasible`` with the NS-safe root bound, never the presolve-free solve's own
    certificate -- on an instance where two HiGHS configurations disagree neither has
    earned one. An unrefuted certificate must still be CONFIRMED by the cross-solve
    (:func:`_weaker_bound`); one that cannot run for want of budget is withdrawn (#1309).
    The budget already spent is the primary solve's ``wall_time``.
    """
    if presolve_exact_class(sf):
        # #1655: the hazard this check exists for cannot arise here (see
        # :func:`presolve_exact_class`), and on such a model -- a pure-binary
        # scheduling MILP HiGHS proves in 0.07 s -- the presolve-free solve ran for
        # the whole budget and withdrew a correct certificate (260-330 s without a
        # limit). Every other route check still ran on the primary.
        out.stats["milp/presolve_cross_check_exact_class"] = 1.0
        return out
    kw = dict(kw)
    if kw["time_limit"] is not None:
        kw["time_limit"] = max(0.0, float(kw["time_limit"]) - float(out.wall_time))
        if kw["time_limit"] <= 0.0:
            out.stats["milp/presolve_cross_check_skipped"] = 1.0
            return _withdraw(
                out, "no time budget left to cross-check it against a presolve-free solve"
            )
    # #1655: a cross-solve that cannot conclude quickly is given up quickly. Its
    # no-verdict outcome is a withdrawal either way (below), so capping it only
    # changes how long the withdrawal takes, never what is certified on a verdict
    # that arrives within the cap.
    cap = max(PRESOLVE_CROSS_MIN_BUDGET, PRESOLVE_CROSS_BUDGET_FACTOR * float(out.wall_time))
    if kw["time_limit"] is None or kw["time_limit"] > cap:
        kw["time_limit"] = cap
        out.stats["milp/presolve_cross_check_budget"] = cap
    cross = _solve_milp_scaled(sf, presolve=False, **kw)
    out.stats["milp/presolve_cross_check_ran"] = 1.0
    out.stats["milp/presolve_cross_check_time"] = float(cross.wall_time)
    pt = _verified_mip_point(sf, cross.x)
    refuted = None
    if pt is not None:
        if out.status == "infeasible":
            refuted = f"a verified point of objective {pt[1]:.12g} exists"
        else:
            claim = out.bound if out.bound is not None else out.objective
            if claim is not None and claim - pt[1] > CERT_ABS + CERT_REL * abs(claim):
                refuted = f"bound {claim:.12g} is above a verified point's objective {pt[1]:.12g}"
    if refuted is None or pt is None:
        return _weaker_bound(sf, out, cross, kw)
    logger.warning(
        "HiGHS MILP route: the %s certificate is refuted by a presolve-free HiGHS solve "
        "(%s); reporting the verified point uncertified (#1634)",
        out.status,
        refuted,
    )
    x, obj = pt
    mine = _verified_mip_point(sf, out.x)
    if mine is not None and mine[1] < obj:
        x, obj = mine
    # Both root bounds are NS-safe lower bounds of ``sf``; the larger is still valid.
    roots = [b for b in (out.root_bound, cross.root_bound) if b is not None]
    bound = min(max(roots), obj) if roots else None
    stats = {**cross.stats, **out.stats, "milp/presolve_certificate_refuted": 1.0}
    labels = {
        **cross.labels,
        **out.labels,
        "milp/certificate": "declined",
        "milp/bound_provenance": "root-ns" if bound is not None else "none",
        "milp/presolve_certificate_refuted": f"{out.status}: {refuted}",
    }
    return HighsOutcome(
        "feasible",
        x=x,
        objective=obj,
        bound=bound,
        gap_certified=False,
        message=(
            f"HiGHS MILP {out.status} certificate refuted by a presolve-free solve: "
            f"{refuted} (#1634)"
        ),
        highs_status=out.highs_status,
        node_count=out.node_count + cross.node_count,
        iterations=out.iterations + cross.iterations,
        root_bound=bound,
        root_time=out.root_time,
        wall_time=out.wall_time + cross.wall_time,
        stats=stats,
        labels=labels,
    )


def _gap_closed(obj: float, bound: float, kw: dict[str, Any]) -> bool:
    """The route's stop rule (HiGHS's ``mip_abs_gap`` OR ``mip_rel_gap``)."""
    gap = max(0.0, obj - bound)
    abs_tol = kw.get("abs_gap_tolerance")
    return gap <= (1e-6 if abs_tol is None else abs_tol) or (
        gap / max(abs(obj), abs(bound), 1e-10) <= kw["gap_tolerance"]
    )


def _weaker_bound(
    sf: StdForm, out: HighsOutcome, cross: HighsOutcome, kw: dict[str, Any]
) -> HighsOutcome:
    """#1634: a certificate stands only if the presolve-free solve AGREES with it.

    Agreement means the cross-solve reached the same certified verdict (``infeasible``
    for ``infeasible``; a certified ``optimal`` for ``optimal``). A cross-solve with no
    verdict -- an error, a limit, a rejected incumbent -- is the #1309 case: the check
    that exists to catch a false certificate produced nothing, so the certificate is
    withdrawn rather than standing on the absence of evidence. Measured on the panel
    (6-column seed 965) this is not hypothetical: the presolved bound sat 3e-6 above the
    enumerated optimum, and the presolve-free solve's kOptimal incumbent failed the
    route's row re-verification, so the falsifier had no point to hold up.

    When both are certified ``optimal``, the WEAKER of the two bounds is published.

    A point can only refute a bound by more than the equality yardstick; a bound that is
    wrong by less (or one the cross-solve's incumbent, itself stopped at the route's gap
    tolerance, does not reach) survives the falsifier. Measured on the panel (6-column
    seed 965): the presolved bound sat 3e-6 above the enumerated optimum while the
    presolve-free bound was below it. The two configurations failed on disjoint
    instances, so ``min`` of the two bounds is valid whenever either one is -- the
    certificate then rests on the claim of whichever configuration is right. The gap is
    re-tested against the route's own stop rule; a gap the weaker bound reopens is
    declined, not certified -- unless the presolve-free solve re-run at a 100x tighter
    feasibility tolerance confirms the claim (#1640, :func:`_tight_cross_bound`): the
    cross bound sits below the optimum by about ``mip_feasibility_tolerance`` times a
    row coefficient, the same order as the 1e-6 absolute gap, so at 1e-6 it can reopen
    a gap on a correct certificate.
    """
    if not (cross.gap_certified and cross.status == out.status):
        out.stats["milp/presolve_cross_check_no_verdict"] = 1.0
        out.labels["milp/presolve_cross_status"] = cross.status
        return _withdraw(
            out,
            f"the presolve-free cross-solve did not confirm it (status {cross.status}: "
            f"{cross.message or 'no message'})",
        )
    if out.status != "optimal" or out.bound is None or cross.bound is None:
        return out
    out.stats["milp/presolve_cross_bound"] = float(cross.bound)
    if cross.bound >= out.bound:
        return out
    claim = float(out.bound)
    out.bound = min(cross.bound, out.objective) if out.objective is not None else cross.bound
    out.labels["milp/bound_provenance"] = "min(presolved, presolve-free)"
    out.stats["milp/presolve_cross_bound_lowered"] = 1.0
    if out.objective is None:
        return out
    closed = _gap_closed(out.objective, out.bound, kw)
    if not closed:
        closed = _tight_cross_bound(sf, out, cross, kw, claim)
    if not closed:
        out.gap_certified = False
        out.status = "feasible"
        out.labels["milp/certificate"] = "declined"
        out.message = (
            "HiGHS MILP: the presolve-free cross-solve's bound reopens the gap, so the "
            "result is not certified (#1634)"
        )
    return out


#: #1640: ``mip_feasibility_tolerance`` of the re-run presolve-free cross-solve. Its
#: bound is HiGHS's tree bound over node LPs solved to ``mip_feasibility_tolerance``,
#: so it sits below the true optimum by about that tolerance times the row scale --
#: the same order as the route's 1e-6 absolute gap. Measured on the #1494 piecewise
#: ``log``/max case (offset 1e4, h = 1): the cross bound is -1.5e-6 at 1e-6 and
#: -1.5e-8 at 1e-8, while ``mip_abs_gap = mip_rel_gap = 0`` leaves it unchanged.
CROSS_TIGHT_FEAS_TOL = 1e-8

#: ``mip_feasibility_tolerance`` of the first cross-solve (HiGHS's default, which
#: :func:`_solve_milp_std` keeps).
CROSS_FEAS_TOL = 1e-6

#: How far above the cross bound the tight bound may land and still count as the
#: same answer with less tolerance error: ``factor * CROSS_FEAS_TOL * max|c| * max|A|``
#: (each floored at 1). Measured on the #1640 case: shift 1.5e-6 against a slack of
#: 1e-4 x the coefficient scale.
CROSS_TIGHT_SHIFT_FACTOR = 100.0


def _tight_cross_bound(
    sf: StdForm, out: HighsOutcome, cross: HighsOutcome, kw: dict[str, Any], claim: float
) -> bool:
    """#1640: re-run the presolve-free cross-solve at a tighter feasibility tolerance
    when its bound -- and only its bound -- reopened the route's gap.

    The cross bound being weaker than the primary's by about ``mip_feasibility_tolerance``
    times a row coefficient is what that tolerance does to a tree bound, not evidence of
    a false primary certificate; but which of the two it is cannot be told from the
    numbers. So the question is asked again with the tolerance 100x tighter, and the
    tighter solve must earn the confirmation the same way the first one had to: a
    certified ``optimal`` of its own, no verified point refuting the claim (the
    #1509/#1621 falsifier), and ``min(primary bound, tight bound)`` closing the gap
    under the route's stop rule. The published bound is that minimum, so it is
    valid whenever the primary or the tight solve is right; because the cross bound
    is dropped, the tight bound must also land within a tolerance-scaled slack of it
    (:data:`CROSS_TIGHT_SHIFT_FACTOR`) -- a larger jump is two certified solves
    disagreeing, not tolerance error. Anything else --
    no budget, no verdict, a refutation, a gap that stays open -- leaves the
    certificate declined, as before. ``claim`` is the primary's own certified bound. On
    success ``out.bound`` is replaced and True is returned.
    """
    obj = out.objective
    if obj is None:
        return False
    kw = dict(kw)
    if kw["time_limit"] is not None:
        # ``kw`` already had the primary's wall time taken off by the caller
        # (:func:`_cross_check_presolve`); only the cross-solve's is left to charge.
        kw["time_limit"] = float(kw["time_limit"]) - float(cross.wall_time)
        if kw["time_limit"] <= 0.0:
            out.stats["milp/presolve_cross_tight_skipped"] = 1.0
            return False
    tight = _solve_milp_scaled(
        sf, **dict(kw, presolve=False, feasibility_tolerance=CROSS_TIGHT_FEAS_TOL)
    )
    out.stats["milp/presolve_cross_tight_ran"] = 1.0
    out.stats["milp/presolve_cross_tight_time"] = float(tight.wall_time)
    if not (tight.gap_certified and tight.status == "optimal" and tight.bound is not None):
        out.labels["milp/presolve_cross_tight_status"] = tight.status
        return False
    out.stats["milp/presolve_cross_tight_bound"] = float(tight.bound)
    pt = _verified_mip_point(sf, tight.x)
    if pt is not None and claim - pt[1] > CERT_ABS + CERT_REL * abs(claim):
        out.labels["milp/presolve_cross_tight_refuted"] = (
            f"bound {claim:.12g} is above a verified point's objective {pt[1]:.12g}"
        )
        return False
    # The confirmation drops the cross bound, so the tight solve must explain it as a
    # tolerance artefact rather than overrule it: a cross bound below the tight one by
    # more than the tolerance-scaled slack is a disagreement between two certified
    # solves, and the decline stands (review of #1649).
    if cross.bound is None:  # _weaker_bound only calls with a cross bound
        return False
    shift = float(tight.bound) - float(cross.bound)
    c_max = float(np.max(np.abs(sf.c))) if sf.c.size else 0.0
    a_max = float(np.max(np.abs(sf.A.data))) if sf.A.nnz else 0.0
    slack = CROSS_TIGHT_SHIFT_FACTOR * CROSS_FEAS_TOL * max(1.0, c_max) * max(1.0, a_max)
    out.stats["milp/presolve_cross_tight_shift"] = shift
    if shift > slack:
        out.labels["milp/presolve_cross_tight_disagrees"] = (
            f"tight bound {float(tight.bound):.12g} is {shift:.3g} above the cross bound "
            f"{float(cross.bound):.12g}, more than the tolerance slack {slack:.3g}"
        )
        return False
    bound = min(claim, float(tight.bound), obj)
    if not _gap_closed(obj, bound, kw):
        return False
    out.bound = bound
    out.labels["milp/bound_provenance"] = "min(presolved, presolve-free tight)"
    out.stats["milp/presolve_cross_tight_confirmed"] = 1.0
    return True


def _solve_milp_std(
    sf: StdForm,
    *,
    time_limit: Optional[float],
    gap_tolerance: float,
    abs_gap_tolerance: Optional[float] = None,
    max_nodes: int,
    initial_point: Optional[np.ndarray] = None,
    n_struct: Optional[int] = None,
    root_check: bool = True,
    presolve: bool = True,
    feasibility_tolerance: float = 1e-6,
) -> HighsOutcome:
    """Solve the MILP ``sf`` under the §3.2 contract.

    ``presolve=False`` switches HiGHS's presolve off; it is the second solve of the
    #1634 cross-check (:func:`_cross_check_presolve`), not a user option.
    ``feasibility_tolerance`` is HiGHS's ``mip_feasibility_tolerance``; only the #1640
    tightened cross-solve (:func:`_weaker_bound`) changes it.

    ``abs_gap_tolerance`` is ``Model.solve``'s absolute convergence tolerance
    (#1243), mapped onto HiGHS's ``mip_abs_gap``. The mapping is faithful:
    HiGHS stops when EITHER ``mip_rel_gap`` or ``mip_abs_gap`` is met, which is
    the same disjunction ``solver.py::_gap_values_converged`` applies on the
    Python tree, so a model certified here and a model certified there mean the
    same thing by the same numbers.

    ``None`` keeps the 1e-6 this route was written with -- which is also
    ``solver.py::_DEFAULT_ABS_GAP_TOL``, so an omitted argument changes nothing.
    It has to be reachable, though: this route is the DEFAULT for pure MILP, and
    a hardcoded absolute tolerance is exactly what #1243 exists to remove. A
    CALPHAD-style certificate is an absolute test on an optimum near zero, where
    the relative arm carries no information at all.
    """
    highspy = require_highspy()
    t0 = time.perf_counter()
    stats: dict[str, float] = {"route/lp_milp_backend": 1.0, "milp/bound_provenance_highs_fp": 1.0}
    labels = {"milp/bound_provenance": "highs-fp"}

    def remaining() -> Optional[float]:
        return None if time_limit is None else float(time_limit) - (time.perf_counter() - t0)

    # #1295: HiGHS's tree bound and infeasible label are floating-point results with
    # nothing re-derived from them. On a column its scaling cannot equilibrate they were
    # measured wrong (false optimum by up to 20 units), so for that class they are not
    # reported as a certificate: only the NS-safe root bound and a Farkas proof stand.
    open_ratio = open_column_coefficient_ratio(sf)
    stats["milp/open_column_coef_ratio"] = open_ratio
    unscalable = open_ratio < UNSCALABLE_OPEN_RATIO

    def decertify(out: HighsOutcome) -> None:
        stats["milp/decertified_unscalable"] = 1.0
        labels["milp/certificate"] = "declined"
        labels["milp/bound_provenance"] = "root-ns" if out.root_bound is not None else "none"
        why = (
            f"an entry on an unbounded column is {open_ratio:.3g} of its row's largest "
            f"(below {UNSCALABLE_OPEN_RATIO:.3g}); HiGHS's MIP certificate is not "
            "trusted on such models (#1295)"
        )
        if out.status == "infeasible":
            if labels.get("milp/infeasible_provenance") == "farkas-root-lp":
                return
            out.status, out.gap_certified, out.bound = "error", False, None
            out.message = f"HiGHS reported infeasible, unverified: {why}"
            return
        if out.status not in ("optimal", "feasible", "time_limit", "node_limit"):
            return
        if out.status == "optimal":
            out.status = "feasible"
        out.gap_certified = False
        out.bound = out.root_bound
        if out.bound is not None and out.objective is not None:
            out.bound = min(out.bound, out.objective)
        out.message = f"gap left uncertified: {why}"

    def done(out: HighsOutcome) -> HighsOutcome:
        out.wall_time = time.perf_counter() - t0
        if unscalable:
            decertify(out)
        out.stats = {**stats, **out.stats}
        out.labels = {**labels, **out.labels}
        return out

    rem = remaining()
    if rem is not None and rem <= 0.0:
        return done(HighsOutcome("time_limit", message="MILP time budget exhausted"))
    opts: list[tuple[str, Any]] = [
        ("mip_rel_gap", float(gap_tolerance)),
        ("mip_abs_gap", 1e-6 if abs_gap_tolerance is None else float(abs_gap_tolerance)),
        ("mip_max_nodes", int(min(max(int(max_nodes), 0), _MAX_NODES_CAP))),
        ("mip_feasibility_tolerance", float(feasibility_tolerance)),
        ("primal_feasibility_tolerance", 1e-7),
        # #1634: HiGHS's parallel-column presolve fixes integers on an absolute
        # cost-tie test that bad coefficient ratios defeat; see the constant.
        ("presolve_rule_off", int(MILP_PRESOLVE_RULE_OFF)),
    ]
    if not presolve:
        opts.append(("presolve", "off"))
    if rem is not None:
        opts.append(("time_limit", float(rem)))
    h = _new_highs(highspy, opts)
    stats["highs/version"] = _version_number(h)
    # HiGHS MIP on the finite default ±9.999e19 box returns false kInfeasible. Measured
    # on adversarial MILPs, replaying the route's own standard form: 5 of 6 labelled
    # infeasible solve to optimal with that box at ±inf (one 6-column model: 4.0, as
    # with ±1e9), the sixth returns an undecided status instead of a false label.
    # HiGHS is handed the relaxation with those bounds open. That keeps an infeasible
    # label and a dual bound valid for ``sf``, and every incumbent is still verified
    # against the declared box below.
    huge_lo, huge_hi = _huge_box(sf)
    if huge_lo.any() or huge_hi.any():
        stats["milp/huge_box_relaxed"] = 1.0
    # HiGHS gets ``obj_const`` so ``mip_rel_gap`` is measured on the objective this
    # route publishes (#1536); both values read back below therefore include it.
    # #1617: an entry HiGHS would drop on the way in is absorbed into its row's range
    # instead -- a relaxation of ``sf``, so HiGHS's infeasible label and tree bound stay
    # valid for ``sf`` (see :func:`_absorb_tiny_entries`).
    sf_pass, row_lo, row_hi, n_tiny = _absorb_tiny_entries(_relax_huge_box(sf, huge_lo, huge_hi))
    if n_tiny:
        stats["milp/tiny_entries_absorbed"] = float(n_tiny)
    pass_st, pass_why = _pass_model(
        h,
        highspy,
        sf_pass,
        integer=True,
        offset=sf.obj_const,
        row_bounds=(row_lo, row_hi) if n_tiny else None,
    )
    if pass_why:
        # No MILP certificate is re-derivable from ``sf`` (HiGHS's infeasible label and
        # tree bound are trusted as-is), so a model HiGHS changed on the way in -- or
        # refused -- gets no answer from it.
        return done(HighsOutcome("error", message=pass_why))

    if initial_point is not None:
        seed = complete_seed(sf, initial_point, sf.n if n_struct is None else int(n_struct))
        if seed is None:
            stats["milp/seed_unusable"] = 1.0
        else:
            sol = highspy.HighsSolution()
            sol.col_value = seed
            if h.setSolution(sol) != highspy.HighsStatus.kOk:
                logger.warning("HiGHS rejected the MILP start; solving without it")
                stats["milp/seed_rejected"] = 1.0

    run_st = h.run()
    name = _status_name(h)
    info = h.getInfo()
    stats["lp/iters"] = float(info.simplex_iteration_count)
    stats["lp/driver_nodes"] = float(info.mip_node_count)
    nodes = int(info.mip_node_count)
    if run_st == highspy.HighsStatus.kError:
        return done(
            HighsOutcome(
                "error", message=f"HiGHS run() error ({name})", highs_status=name, node_count=nodes
            )
        )

    x = None
    obj = None
    repaired_why: Optional[str] = None
    if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
        x = np.asarray(h.getSolution().col_value, dtype=np.float64)
        why = readback_problem(x, sf) or feasibility_problem(x, sf, check_integrality=True)
        if why:
            # The point is unusable, but the question may still be decidable: a
            # verified Farkas ray of the LP relaxation proves the MILP infeasible
            # (the huge-box case, where HiGHS's own arithmetic cancelled a violation).
            root = solve_lp_std(sf, time_limit=remaining())
            stats["milp/root_check_ran"] = 1.0
            stats["milp/root_lp_iters"] = root.stats.get("lp/iters", 0.0)
            if root.status == "infeasible":
                labels["milp/infeasible_provenance"] = "farkas-root-lp"
                stats["milp/infeasible_provenance_farkas"] = 1.0
                return done(
                    HighsOutcome(
                        "infeasible", gap_certified=True, highs_status=name, node_count=nodes,
                        message=f"HiGHS MILP incumbent ({name}) refused: {why}",
                    )
                )  # fmt: skip
            repaired = _repair_refused_incumbent(sf, x, root, remaining(), stats)
            # #1654: HiGHS's tree accepts an integer column within its
            # ``mip_feasibility_tolerance`` of integral, so on a big-M row ``x <= M z``
            # a binary read back at ``1 - 9e-7`` buys ``9e-7 * M`` of slack, and the
            # integral realisation the route verifies (#1380) violates the row by that
            # much. The point is refused, rightly -- but the integer assignment it
            # carries is still a candidate: the LP over those integers, re-verified on
            # ``sf``, is a genuine feasible point, and it replaces the refused one.
            # HiGHS's tree bound does not rest on the refused point (a node closed on
            # a near-integral LP point closed at that LP's value, a relaxation of the
            # node), so it goes through every check below exactly as an ordinary
            # incumbent's would; the gap is re-tested on the repaired objective.
            if repaired is None:
                return done(
                    HighsOutcome(
                        "error", message=f"HiGHS MILP incumbent ({name}): {why}",
                        highs_status=name, node_count=nodes,
                    )
                )  # fmt: skip
            x, obj = repaired
            repaired_why = why
            labels["milp/incumbent_repaired"] = why
        else:
            obj = float(sf.c @ x) + sf.obj_const
            h_obj = float(info.objective_function_value)  # includes the passed offset
            mismatch = abs(obj - h_obj)
            stats["milp/objective_mismatch"] = mismatch
            if mismatch > 1e-6 * (1.0 + abs(obj)):
                logger.warning(
                    "HiGHS MILP objective %.12g differs from the recomputed %.12g", h_obj, obj
                )

    raw = float(info.mip_dual_bound)  # includes the passed offset
    bound = raw if np.isfinite(raw) else None
    out = HighsOutcome("error", x=x, objective=obj, highs_status=name, node_count=nodes)
    out.iterations = int(info.simplex_iteration_count)

    if name == "kOptimal":
        if x is None:
            out.message = "HiGHS reported optimal without a primal solution"
            return done(out)
        out.status, out.gap_certified = "optimal", True
    elif name == "kInfeasible":
        out.status, out.gap_certified = "infeasible", True
        labels["milp/infeasible_provenance"] = "highs"
        stats["milp/infeasible_provenance_highs"] = 1.0
    elif name in ("kUnbounded", "kUnboundedOrInfeasible"):
        pass  # decided on the LP relaxation below
    elif name in _LIMIT_STATUSES:
        if x is not None:
            out.status = "feasible"
        else:
            out.status = "time_limit" if name == "kTimeLimit" else "node_limit"
    else:
        out.message = f"HiGHS MILP status {name}"
        return done(out)
    if bound is not None and obj is not None:
        bound = min(bound, obj)
    out.bound = bound if out.status in ("optimal", "feasible", "time_limit", "node_limit") else None
    if repaired_why is not None and out.status == "optimal":
        # #1654: kOptimal closed HiGHS's gap on the REFUSED point's objective; the
        # repaired point may sit above it, so the stop rule is asked again.
        gap_kw = {"abs_gap_tolerance": abs_gap_tolerance, "gap_tolerance": gap_tolerance}
        if out.bound is None or not _gap_closed(float(obj), float(out.bound), gap_kw):  # type: ignore[arg-type]
            out.status, out.gap_certified = "feasible", False
            labels["milp/certificate"] = "declined"
        out.message = (
            f"HiGHS MILP incumbent ({name}) refused ({repaired_why}); reporting the "
            "fixed-integer LP over its integers, verified on the model (#1654)"
        )

    def decertify_root_check(why: str) -> None:
        """Strip the certificate from a result whose NS-safe root check did not
        settle — whether it was skipped for want of budget (#1309) or ran and came
        back inconclusive (#1320). Both are the same fact: the safety net that
        exists to catch a tree bound (or an infeasibility claim) already wrong at
        the root (#1295) produced no verdict, so nothing here is certified.

        Every field that reads as a certificate is made consistent (#1320 part 2),
        mirroring the #1295 ``decertify``: kOptimal keeps its incumbent as an
        honest uncertified ``feasible``; kInfeasible has no incumbent to fall back
        on and becomes ``error``; and the unchecked tree bound gives way to the
        NS-safe root bound — ``None`` here, since the check is exactly what did not
        produce one, which in turn leaves ``gap``/``bound_valid`` empty rather than
        a proven-looking zero gap. The label demotes the route's own wording from
        "verified" to "unverified".
        """
        if not out.gap_certified:
            return
        out.message = f"HiGHS MILP {name}: {why}, so this result cannot be certified"
        out.gap_certified = False
        out.status = "feasible" if out.x is not None else "error"
        out.bound = out.root_bound
        if out.bound is not None and out.objective is not None:
            out.bound = min(out.bound, out.objective)
        labels["milp/certificate"] = "declined"
        labels["milp/bound_provenance"] = "root-ns" if out.bound is not None else "none"

    def _refutation_check(out: HighsOutcome, lp: HighsOutcome) -> bool:
        """#1509: hold the reported bound against every VERIFIED point in hand.

        The root cross-check above only compares two numbers, and the #1410 dual
        check only asks whether HiGHS's LP arithmetic certifies an LP. Neither can see
        a tree whose MIP presolve/propagation cut the optimum off: on the #1509
        surrogate the root LP, its duals, and every scaling ratio are clean, HiGHS
        still certifies ``9000012.455`` while ``9e6`` is feasible -- and the root LP
        point itself is that feasible ``9e6`` point.

        A bound above a verified feasible point is false by definition, so the check
        is "no verified point lies below the bound", over the points available at an
        LP's cost: the root LP point (already solved); the NS-safe LP over the
        incumbent's own integer assignment (a ``kOptimal`` incumbent must be optimal
        for its own integers, and that LP never enters HiGHS's MIP presolve); and,
        when the root LP point is fractional, the same LP over its rounding. Every
        candidate passes the incumbent verifier before it counts as evidence.

        This is a falsifier, not a proof: it catches a false bound whenever one of
        those points lies below it, which on the 180-instance #1509 family was every
        one of the 60 false certificates. A false bound with no such point in reach
        is not caught, and the route's standing there is what it was before.

        Returns ``False`` when ``out`` was decertified for want of a verdict and the
        caller should stop; ``True`` otherwise (refuted results are repaired in place).
        """
        cands: list[tuple[np.ndarray, float]] = []
        root_pt = _verified_mip_point(sf, lp.x) if lp.status == "optimal" else None
        if root_pt is not None:
            cands.append(root_pt)
        if out.x is not None and sf.int_idx.size:
            rem_fx = remaining()
            if rem_fx is not None and rem_fx <= 0.0:
                stats["milp/fixed_int_check_skipped"] = 1.0
                decertify_root_check("no time budget left to run the fixed-integer LP check")
                return False
            fx = _fixed_integer_lp(sf, out.x, rem_fx)
            stats["milp/fixed_int_check_ran"] = 1.0
            pt = _verified_mip_point(sf, fx.x) if fx.status == "optimal" else None
            if pt is None:
                # The LP over a verified incumbent's own integers is feasible (the
                # incumbent is a point of it), so a non-verdict here is numerical
                # trouble on this matrix -- the #1320 rule: an unsettled check is not
                # a passed one.
                stats["milp/fixed_int_check_inconclusive"] = 1.0
                decertify_root_check(
                    f"the fixed-integer LP check was inconclusive (LP {fx.status}: {fx.message})"
                )
                return False
            cands.append(pt)
            # #1612: the incumbent's own integers, re-optimised by a separate LP; the
            # distance from HiGHS's objective is what its feasibility tolerance bought
            # (read by :func:`_tighten_feasibility_artefact`).
            stats["milp/fixed_int_objective"] = float(pt[1])
        if root_pt is None and lp.status == "optimal" and lp.x is not None and sf.int_idx.size:
            # The root LP point is not itself integer-feasible: its rounding, with the
            # continuous part re-optimised, is the other point an LP buys. Measured on
            # the #1509 family, HiGHS's own integers can be the wrong ones (a gap-limit
            # stop whose bound is above the optimum) while the root LP sits within
            # 1.2e-5 of the optimal assignment. A failed or infeasible rounding is no
            # evidence either way, so it is simply not a candidate.
            rem_rd = remaining()
            if rem_rd is None or rem_rd > 0.0:
                rd = _fixed_integer_lp(sf, lp.x, rem_rd)
                stats["milp/rounded_root_check_ran"] = 1.0
                pt = _verified_mip_point(sf, rd.x) if rd.status == "optimal" else None
                if pt is not None:
                    cands.append(pt)
        if not cands:
            return True
        best_x, best = min(cands, key=lambda p: p[1])
        assert out.bound is not None
        margin = out.bound - best
        stats["milp/refutation_margin"] = float(margin)
        # The yardstick the route already uses to call an LP objective equal to its
        # NS bound (CERT_ABS + CERT_REL*|v|): a point below the bound by more than
        # that is a contradiction, not round-off.
        if margin <= CERT_ABS + CERT_REL * abs(out.bound):
            return True
        stats["milp/certificate_refuted"] = 1.0
        labels["milp/highs_certificate"] = "refuted"
        was = out.bound
        if out.objective is None or best < out.objective:
            out.x, out.objective = best_x, best
            stats["milp/incumbent_from_refutation"] = 1.0
        # HiGHS's tree bound is now known false, so nothing it derived stands; the
        # NS-safe root bound is the only bound left, and it certifies the gap only if
        # it closes it by the route's own convergence test (the HiGHS stop rule).
        rb = out.root_bound
        out.bound = None if rb is None else min(rb, out.objective)
        labels["milp/bound_provenance"] = "root-ns" if out.bound is not None else "none"
        closed = False
        if out.bound is not None:
            gap = max(0.0, out.objective - out.bound)
            closed = gap <= (1e-6 if abs_gap_tolerance is None else abs_gap_tolerance) or (
                gap / max(abs(out.objective), abs(out.bound), 1e-10) <= gap_tolerance
            )
        out.gap_certified = closed
        out.status = "optimal" if closed else "feasible"
        if not closed:
            labels["milp/certificate"] = "declined"
        out.message = (
            f"HiGHS MILP {name}: bound {was:.12g} refuted by a verified point of "
            f"objective {best:.12g} (#1509); reporting the NS-safe root bound"
        )
        logger.warning("HiGHS MILP route: %s", out.message)
        return True

    # Root LP relaxation, NS-safe: supplies root_bound, decides an unbounded-or-
    # infeasible label, upgrades an infeasible claim to a Farkas proof, and catches a
    # tree bound that is already wrong at the root (§3.2.5).
    need_root = root_check or name in ("kUnbounded", "kUnboundedOrInfeasible")
    rem = remaining()
    if not need_root:
        return done(out)
    if rem is not None and rem <= 0.0:
        stats["milp/root_check_skipped"] = 1.0
        if name in ("kUnbounded", "kUnboundedOrInfeasible"):
            out.status, out.message = "error", f"{name} and no budget left to decide it"
        else:
            decertify_root_check("no time budget left to run the NS-safe root cross-check")
        return done(out)
    t_root = time.perf_counter()
    lp = solve_lp_std(sf, time_limit=rem)
    out.root_time = time.perf_counter() - t_root
    stats["milp/root_check_ran"] = 1.0
    stats["milp/root_lp_iters"] = lp.stats.get("lp/iters", 0.0)

    if name in ("kUnbounded", "kUnboundedOrInfeasible"):
        if lp.status == "infeasible":
            out.status, out.gap_certified, out.bound = "infeasible", True, None
            labels["milp/infeasible_provenance"] = "farkas-root-lp"
            stats["milp/infeasible_provenance_farkas"] = 1.0
            return done(out)
        if lp.status == "unbounded":
            # A verified integer-feasible point plus a verified recession direction
            # of the relaxation (rational data: Meyer) -> the MILP is unbounded.
            #
            # #1339 follow-up: HiGHS reports ``kUnbounded`` / ``kUnboundedOrInfeasible``
            # with NO incumbent, so the point Meyer's theorem needs was simply
            # missing and a genuinely unbounded MILP came back ``error``. It is not
            # missing because it is hard to get -- it is missing because nobody asked
            # for it. Asking is the same move #1337 makes one branch below: the
            # question here is only "is the INTEGER system nonempty?", and the
            # objective is exactly what stops HiGHS answering it, so drop the
            # objective and re-solve. Meyer then applies in full:
            # ``rec(conv(S)) = rec(P)`` for rational data with ``S`` nonempty, so an
            # improving recession direction of the relaxation is one of the integer
            # hull too.
            if x is None:
                x = _integer_feasible_point(sf, remaining(), stats)
            if x is not None:
                out.x = x
                out.status = "unbounded"
                labels["milp/unbounded_provenance"] = "meyer-ray-plus-integer-point"
                return done(out)
            if stats.get("milp/unbounded_feasibility_probe_infeasible"):
                # The probe did not merely fail to find a point -- it PROVED there is
                # none, which decides the ``OrInfeasible`` half outright.
                out.status, out.gap_certified, out.bound = "infeasible", True, None
                labels["milp/infeasible_provenance"] = "integer-feasibility-probe"
                return done(out)
        out.status = "error"
        out.message = f"HiGHS MILP {name}; root LP relaxation {lp.status}: {lp.message}"
        return done(out)

    if lp.status == "infeasible":
        if out.status == "infeasible":
            labels["milp/infeasible_provenance"] = "farkas-root-lp"
            stats["milp/infeasible_provenance_farkas"] = 1.0
            return done(out)
        out.status, out.gap_certified = "error", False
        out.message = "root LP relaxation is Farkas-infeasible but HiGHS returned a MILP point"
        return done(out)
    if lp.status in ("optimal", "feasible") and lp.bound is not None:
        out.root_bound = lp.bound
        if out.status == "infeasible":
            return done(out)
        stats["milp/root_ns_bound"] = float(lp.bound)
        if out.bound is not None and lp.bound > out.bound + 1e-6 * (1.0 + abs(out.bound)):
            out.status, out.gap_certified = "error", False
            out.message = (
                f"tree bound {out.bound:.12g} below the safe root bound {lp.bound:.12g}: "
                "the MILP HiGHS solved is not the model's relaxation"
            )
            return done(out)
        # #1410: the cross-check above compares two NUMBERS and so only catches a tree
        # bound that is already wrong at the root. The failure it misses is the opposite
        # shape: a root bound that is perfectly valid while HiGHS prunes the true optimum
        # deeper in the tree, returning a bound ABOVE it. That cannot be caught by any
        # comparison of the root bound to the tree bound -- but it can be caught by asking
        # whether HiGHS's arithmetic on this matrix produces a certificate at all. If the
        # root LP's own primal/dual pair is not dual-feasible in unscaled doubles, every
        # LP HiGHS solves in the tree is suspect, and the tree's bound is not a
        # certificate. Measured: 7.87e-17 on well-scaled LPs, 1.0 on the #1410 instance,
        # where UNSCALABLE_OPEN_RATIO missed by a factor of 1.43 (1.366e-6 vs 9.537e-7).
        if lp.x is not None and lp.row_dual is not None:
            dual_viol = box_dual_violation(sf, lp.x, lp.row_dual)
            dual_tol = dual_violation_tolerance(sf)
            stats["milp/root_dual_violation"] = dual_viol
            if dual_viol > dual_tol:
                # #1612: the per-column relative test is a sufficient condition, not
                # the question. The question is whether HiGHS's own pair certifies the
                # root LP, and that is decided rigorously by charging every violated
                # reduced cost over the declared box (NS): if the safe bound from
                # HiGHS's duals meets its own primal objective within the route's LP
                # certificate yardstick, the LP is provably solved to that tolerance --
                # in objective units, the units the MILP gap is certified in -- and the
                # violation was round-off. On #1410 the same gap is 0.0221: HiGHS
                # mis-solved the LP, and that is still refused.
                ns_gap = root_pair_ns_gap(sf, lp.x, lp.row_dual)
                lp_obj = float(sf.c @ lp.x) + sf.obj_const
                ns_thr = CERT_ABS + CERT_REL * abs(lp_obj)
                stats["milp/root_dual_ns_gap"] = float("inf") if ns_gap is None else ns_gap
                if ns_gap is not None and ns_gap <= ns_thr:
                    stats["milp/root_dual_certified_by_ns"] = 1.0
                else:
                    stats["milp/root_dual_unverified"] = 1.0
                    gap_txt = "unbounded (-inf NS bound)" if ns_gap is None else f"{ns_gap:.3g}"
                    decertify_root_check(
                        f"the root LP's own primal/dual pair is dual-infeasible by "
                        f"{dual_viol:.3g} relative (round-off bound {dual_tol:.3g}) and "
                        f"its NS-safe duality gap is {gap_txt} (yardstick {ns_thr:.3g}), "
                        f"so HiGHS's arithmetic on this matrix does not certify the LPs "
                        f"it solves in the tree"
                    )
                    return done(out)
        if out.bound is None and out.status != "optimal":
            # No tree bound yet (limit hit before the root finished): the NS root
            # bound is a valid one, so report it rather than nothing.
            out.bound = lp.bound if obj is None else min(lp.bound, obj)
            stats["milp/bound_from_root_ns"] = 1.0
        if out.bound is not None and not _refutation_check(out, lp):
            return done(out)
    else:
        # The check RAN but settled nothing (``solve_lp_std`` hit its own time
        # limit, or errored — a kUnknown / sentinel-magnitude readback refusal).
        # #1309 applied "a certificate whose safety net never ran is not a
        # certificate" only to the skip branch; an inconclusive run is the same
        # fact and gets the same downgrade (#1320 part 1).
        #
        # #1337: unless what defeated the check was the OBJECTIVE rather than
        # anything about feasibility. On a purely INTEGRAL infeasibility the
        # relaxation is perfectly happy — ``2x == 1`` relaxes to ``x = 0.5`` — and
        # if the objective is also unbounded there (``min −y`` over
        # ``y ∈ [0, 1e20]``) the root LP comes back ``unbounded``, which lands here
        # and decertifies a correct ``infeasible`` into ``error``. But the question
        # this branch asks of a kInfeasible claim is only ever "is the relaxation
        # feasible?", and an unbounded objective says nothing about that. So ask it
        # without one: same LP, same box, zero objective.
        #
        # The verdict is held to exactly the standard the ordinary path already
        # uses, not a weaker one — a Farkas-infeasible relaxation proves the MILP
        # empty, and a feasible relaxation leaves the kInfeasible claim standing
        # precisely as the ``lp.status in ("optimal", "feasible")`` branch above
        # already does. Anything else still decertifies.
        _settled = False
        if out.status == "infeasible" and out.gap_certified:
            _rem2 = remaining()
            if _rem2 is None or _rem2 > 0.0:
                stats["milp/root_check_feasibility_only"] = 1.0
                feas = solve_lp_std(
                    dataclasses.replace(sf, c=np.zeros(sf.n, dtype=np.float64), obj_const=0.0),
                    time_limit=_rem2,
                )
                if feas.status == "infeasible":
                    # Stronger than needed: the relaxation itself is empty.
                    labels["milp/infeasible_provenance"] = "farkas-root-lp"
                    stats["milp/infeasible_provenance_farkas"] = 1.0
                    _settled = True
                elif feas.status in ("optimal", "feasible"):
                    labels["milp/infeasible_provenance"] = "highs-root-feasible"
                    _settled = True
        if not _settled:
            stats["milp/root_check_inconclusive"] = 1.0
            decertify_root_check(
                f"the NS-safe root cross-check was inconclusive (root LP {lp.status}: {lp.message})"
            )
    return done(out)


# ─────────────────────────────────────────────────────────────
# Matrix-form LP contract (``lp_simplex.solve_lp`` signature)
# ─────────────────────────────────────────────────────────────


def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, time_limit=None):
    """``min cᵀx s.t. A_ub x <= b_ub, A_eq x = b_eq`` returning an ``LPResult``.

    Used for relaxation-dual recovery. Inequality rows get a nonnegative logical, so
    each row dual is ``∂obj/∂b`` of the row as passed -- the HiGHS convention
    ``_lp_qp_unpack_duals`` expects. ``OPTIMAL`` only for a certified outcome.
    """
    from discopt.solvers import LPResult, SolveStatus
    from discopt.solvers.milp_simplex import _marshal_col_bounds

    c = np.asarray(c, dtype=np.float64).ravel()
    n = c.shape[0]
    xl, xu = _marshal_col_bounds(bounds, n)
    blocks, rhs = [], []
    n_ub = 0
    if A_ub is not None and b_ub is not None:
        a = sp.csr_matrix(A_ub, dtype=np.float64)
        n_ub = a.shape[0]
        blocks.append(sp.hstack([a, sp.identity(n_ub, format="csr")]))
        rhs.append(np.asarray(b_ub, dtype=np.float64).ravel())
    if A_eq is not None and b_eq is not None:
        a = sp.csr_matrix(A_eq, dtype=np.float64)
        blocks.append(sp.hstack([a, sp.csr_matrix((a.shape[0], n_ub))]))
        rhs.append(np.asarray(b_eq, dtype=np.float64).ravel())
    A = sp.vstack(blocks, format="csc") if blocks else sp.csc_matrix((0, n + n_ub))
    b = np.concatenate(rhs) if rhs else np.zeros(0)
    sf = StdForm.from_arrays(
        np.concatenate([c, np.zeros(n_ub)]),
        A,
        b,
        np.concatenate([xl, np.zeros(n_ub)]),
        np.concatenate([xu, np.full(n_ub, INF)]),
    )
    out = solve_lp_std(sf, time_limit=time_limit)
    status = {
        "optimal": SolveStatus.OPTIMAL,
        "infeasible": SolveStatus.INFEASIBLE,
        "unbounded": SolveStatus.UNBOUNDED,
        "time_limit": SolveStatus.TIME_LIMIT,
    }.get(out.status, SolveStatus.ERROR)
    return LPResult(
        status=status,
        x=None if out.x is None else out.x[:n],
        objective=out.objective,
        dual_values=out.row_dual,
        reduced_costs=None if out.col_dual is None else out.col_dual[:n],
        iterations=int(out.stats.get("lp/iters", 0)),
        wall_time=out.wall_time,
    )
