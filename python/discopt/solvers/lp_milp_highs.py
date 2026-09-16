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

import dataclasses
import logging
import time
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
    if check_integrality and sf.int_idx.size:
        seg = x[sf.int_idx]
        frac = np.abs(seg - np.round(seg))
        k = int(np.argmax(frac))
        if frac[k] > INT_TOL:
            return f"integer column {int(sf.int_idx[k])} = {seg[k]:.9g} is fractional"
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


def primal_ray_verified(d: np.ndarray, sf: StdForm) -> bool:
    """True iff ``d`` is a descent direction of the recession cone:
    ``A d = 0``, ``cᵀd < 0``, and ``d`` only moves along open bound sides."""
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
    return cd < -RAY_REL * max(1.0, float(np.abs(sf.c) @ np.abs(d)))


# ─────────────────────────────────────────────────────────────
# Certificate recovery: FBBT box, exact dual correction, phase 1
# ─────────────────────────────────────────────────────────────

#: FBBT sweeps for the certificate box; every sweep keeps a superset of the feasible set.
FBBT_ROUNDS = 20
#: Most columns the exact dual correction zeroes, and most correction rounds. Past
#: either the rational elimination is too slow for a fallback; the bound is then left
#: uncertified (never guessed).
EXACT_MAX_COLUMNS = 256
EXACT_MAX_ROUNDS = 8
#: Work cap for one rational elimination, in bit operations (entries updated times the
#: bit length of the operands). Rational elimination can grow its denominators, and a
#: fallback must not hang a solve; a count, not a clock, so whether a bound is certified
#: does not depend on machine speed (#912). Calibrated on dense 60-bit rational systems
#: at 1.0-2.2e8 units/s (n=60: 5.9e8 units, 6.0 s), so the cap is ~1-2 s of elimination.
EXACT_MAX_WORK = 200_000_000
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


def _exact_solve(M: list, rhs: list, deadline: Optional[float]) -> Optional[list]:
    """``M z = rhs`` over the rationals by Gaussian elimination; ``None`` if singular, past
    ``EXACT_MAX_WORK`` bit operations, or past the caller's ``deadline``."""
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
                work += (n + 1 - col) * (size(f) + row_size)
                if work > EXACT_MAX_WORK:
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
        dy = _exact_solve(M, [rcS[j] for j in indep], deadline)
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


def _relax_huge_box(sf: StdForm, huge_lo: np.ndarray, huge_hi: np.ndarray) -> StdForm:
    """``sf`` with the huge finite bounds opened to infinity, one side at a time: a
    relaxation of ``sf`` that keeps every ordinary declared bound."""
    return dataclasses.replace(
        sf,
        xl=np.where(huge_lo, -INF, sf.xl),
        xu=np.where(huge_hi, INF, sf.xu),
    )


def _pass_model(h, highspy, sf: StdForm, integer: bool) -> tuple[Any, str]:
    """Hand ``sf`` to HiGHS; returns ``(passModel status, reason)``, ``reason`` empty on kOk.

    kWarning means HiGHS changed the model on the way in -- it drops every matrix entry
    with ``|a| <= small_matrix_value`` -- so the caller decides whether its certificates
    survive that. kError means HiGHS holds no model. Neither is raised: both are
    properties of the input, not defects, and the route reports them as ``error``.
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
    lp.row_lower_ = sf.b
    lp.row_upper_ = sf.b
    # obj_const is added by discopt, so every HiGHS objective/bound is constant-free.
    lp.offset_ = 0.0
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


def _new_highs(highspy, opts: list[tuple[str, Any]]):
    h = highspy.Highs()
    # ``threads`` stays at HiGHS's default (0). HiGHS keeps one process-wide scheduler,
    # sized by the first run in the process, and refuses any later run whose nonzero
    # ``threads`` differs from that size (``run()`` -> kError). The OA/GDP paths and user
    # code run highspy with default options, so pinning a value here made every later
    # LP/MILP solve in the same process return ``error``.
    _set_options(h, highspy, [("output_flag", False), ("random_seed", 0)])
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
    """Solve the MILP ``sf`` under the §3.2 contract.

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

    def done(out: HighsOutcome) -> HighsOutcome:
        out.wall_time = time.perf_counter() - t0
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
        ("mip_feasibility_tolerance", 1e-6),
        ("primal_feasibility_tolerance", 1e-7),
    ]
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
    pass_st, pass_why = _pass_model(h, highspy, _relax_huge_box(sf, huge_lo, huge_hi), integer=True)
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
            return done(
                HighsOutcome(
                    "error", message=f"HiGHS MILP incumbent ({name}): {why}",
                    highs_status=name, node_count=nodes,
                )
            )  # fmt: skip
        obj = float(sf.c @ x) + sf.obj_const
        h_obj = float(info.objective_function_value) + sf.obj_const
        mismatch = abs(obj - h_obj)
        stats["milp/objective_mismatch"] = mismatch
        if mismatch > 1e-6 * (1.0 + abs(obj)):
            logger.warning(
                "HiGHS MILP objective %.12g differs from the recomputed %.12g", h_obj, obj
            )

    raw = float(info.mip_dual_bound)
    bound = raw + sf.obj_const if np.isfinite(raw) else None
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
        elif lp.status == "unbounded" and x is not None:
            # A verified integer-feasible point plus a verified recession direction
            # of the relaxation (rational data: Meyer) -> the MILP is unbounded.
            out.status = "unbounded"
        else:
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
        if out.bound is None and out.status != "optimal":
            # No tree bound yet (limit hit before the root finished): the NS root
            # bound is a valid one, so report it rather than nothing.
            out.bound = lp.bound if obj is None else min(lp.bound, obj)
            stats["milp/bound_from_root_ns"] = 1.0
    else:
        stats["milp/root_check_inconclusive"] = 1.0
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
