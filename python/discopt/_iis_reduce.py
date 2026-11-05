"""Exact feasibility-preserving reduction of an IIS engine subsystem: the pre-pass
of the presolve-free confirmation route (``_Engine.confirm_infeasible``).

On general-integer models (MIPcc27's dijle / dingo / dives) presolve-free HiGHS
could not confirm the subsets the deletion filter passes through -- undecided
at 60-300 s -- while presolve-on proved them in under a second: HiGHS's
presolve collapses them to ~70 rows. Re-implementing that reduction ourselves,
exactly, and handing the reduced model to presolve-free HiGHS keeps the second
route independent of HiGHS presolve (the #1634 contract) and decides them:
dijle's path sets in 1-11 s, dingo's in 0.3 s, dives' in 0.0 s
(iis_generalint_strategy.md R1; scratch exp23).

Every rule preserves "the feasible set is nonempty" in BOTH directions with
zero objective, so infeasibility of the reduced model proves infeasibility of
the input:

  P  propagation (FBBT) with integer rounding on held columns; an empty domain
     or an impossible row is a proof by itself;
  R  redundant rows dropped; fixed columns substituted out;
  D  dominated columns: every row prefers the column up (down) -> fix it at
     that bound, or, when that bound is infinite, drop it with its rows;
  T  coefficient tightening of a binary column in a one-sided row;
  A  aggregation u := c0 + c1 x of a column every row but one prefers up
     (down), where that one row caps u by a function of x that stays inside
     u's bounds over x's box (and is integral when u is held).

All arithmetic is :class:`fractions.Fraction` over the exact values of the
input floats. The exported model scales each row by the lcm of its
denominators (integers are exact as floats) and rounds bounds and row sides
OUTWARD, a relaxation: its infeasibility still implies the exact model's.
Only infeasibility is ever concluded from the reduced model; a feasible
reduced model concludes nothing (the caller falls back to its other routes).
"""

from __future__ import annotations

import math
from fractions import Fraction as F
from typing import Optional

import numpy as np


class Infeasible(Exception):
    pass


def _f(v) -> Optional[F]:
    v = float(v)
    if not math.isfinite(v) or abs(v) >= 1e20:
        return None
    return F(v)


class Model:
    def __init__(self, A, rlo, rhi, lb, ub, held):
        A = A.tocsr()
        self.rows: dict[int, dict[int, F]] = {}
        self.rlo: dict[int, Optional[F]] = {}
        self.rhi: dict[int, Optional[F]] = {}
        for r in range(A.shape[0]):
            lo, hi = _f(rlo[r]), _f(rhi[r])
            if lo is None and hi is None:
                continue
            s, e = A.indptr[r], A.indptr[r + 1]
            row = {int(j): F(float(a)) for j, a in zip(A.indices[s:e], A.data[s:e]) if a != 0}
            self.rows[r], self.rlo[r], self.rhi[r] = row, lo, hi
        self.cols: dict[int, set[int]] = {}
        for r, row in self.rows.items():
            for j in row:
                self.cols.setdefault(j, set()).add(r)
        n = A.shape[1]
        self.lb = {j: _f(lb[j]) for j in range(n)}
        self.ub = {j: _f(ub[j]) for j in range(n)}
        self.held = {j for j in range(n) if held[j]}
        self.n_rules = {"P": 0, "R": 0, "F": 0, "D": 0, "T": 0, "A": 0}
        self._n_tight: dict[int, int] = {}
        for j in self.held:  # held integer boxes are integral
            if self.lb[j] is not None:
                self.lb[j] = F(math.ceil(self.lb[j]))
            if self.ub[j] is not None:
                self.ub[j] = F(math.floor(self.ub[j]))
            self._check_col(j)

    # -- helpers -----------------------------------------------------------------
    def _check_col(self, j):
        lo, hi = self.lb[j], self.ub[j]
        if lo is not None and hi is not None and lo > hi:
            raise Infeasible(("col", j))

    def _act(self, r):
        """(minact, n_inf_min, maxact, n_inf_max) of row r."""
        mn = mx = F(0)
        nmn = nmx = 0
        for j, a in self.rows[r].items():
            lo, hi = self.lb[j], self.ub[j]
            if a > 0:
                if lo is None:
                    nmn += 1
                else:
                    mn += a * lo
                if hi is None:
                    nmx += 1
                else:
                    mx += a * hi
            else:
                if hi is None:
                    nmn += 1
                else:
                    mn += a * hi
                if lo is None:
                    nmx += 1
                else:
                    mx += a * lo
        return mn, nmn, mx, nmx

    def _remove_row(self, r):
        for j in self.rows[r]:
            self.cols[j].discard(r)
        del self.rows[r], self.rlo[r], self.rhi[r]

    _MIN_STEP = F(1, 10**6)

    _MAX_TIGHTEN = 64  # per column; propagation on unbounded integer chains need not end

    def _budget(self, j) -> bool:
        k = self._n_tight.get(j, 0)
        if k >= self._MAX_TIGHTEN:
            return False
        self._n_tight[j] = k + 1
        return True

    def _significant(self, j, old, new) -> bool:
        """Held columns: any change (they move by whole units). Continuous ones:
        a relative step of at least 1e-6, so exact propagation cannot creep
        forever. Skipping a valid tightening is always sound."""
        if old is None or j in self.held:
            return True
        return bool(abs(new - old) > self._MIN_STEP * max(F(1), abs(old)))

    def _set_lb(self, j, v):
        if self.held_j(j):
            v = F(math.ceil(v))
        if (self.lb[j] is None or (v > self.lb[j] and self._significant(j, self.lb[j], v))) and (
            self._budget(j)
        ):
            self.lb[j] = v
            self._check_col(j)
            return True
        return False

    def _set_ub(self, j, v):
        if self.held_j(j):
            v = F(math.floor(v))
        if (self.ub[j] is None or (v < self.ub[j] and self._significant(j, self.ub[j], v))) and (
            self._budget(j)
        ):
            self.ub[j] = v
            self._check_col(j)
            return True
        return False

    def held_j(self, j):
        return j in self.held

    # -- rules ---------------------------------------------------------------------
    def propagate_row(self, r) -> set:
        """P on one row; returns columns whose bounds changed. Raises Infeasible."""
        changed = set()
        mn, nmn, mx, nmx = self._act(r)
        lo, hi = self.rlo[r], self.rhi[r]
        if hi is not None and nmn == 0 and mn > hi:
            raise Infeasible(("row", r))
        if lo is not None and nmx == 0 and mx < lo:
            raise Infeasible(("row", r))
        for j, a in list(self.rows[r].items()):
            lbj, ubj = self.lb[j], self.ub[j]
            # contribution of j to min / max activity
            cmin = (
                (a * lbj if lbj is not None else None)
                if a > 0
                else (a * ubj if ubj is not None else None)
            )
            cmax = (
                (a * ubj if ubj is not None else None)
                if a > 0
                else (a * lbj if lbj is not None else None)
            )
            if hi is not None and (nmn == 0 or (nmn == 1 and cmin is None)):
                rest = mn - (cmin if cmin is not None else 0)
                bound = (hi - rest) / a
                if self._set_ub(j, bound) if a > 0 else self._set_lb(j, bound):
                    changed.add(j)
            if lo is not None and (nmx == 0 or (nmx == 1 and cmax is None)):
                rest = mx - (cmax if cmax is not None else 0)
                bound = (lo - rest) / a
                if self._set_lb(j, bound) if a > 0 else self._set_ub(j, bound):
                    changed.add(j)
            if j in changed:
                mn, nmn, mx, nmx = self._act(r)
        if changed:
            self.n_rules["P"] += 1
        return changed

    def redundant_or_empty(self, r) -> bool:
        """R: drop a row implied by the bounds (or check an empty row)."""
        row = self.rows[r]
        lo, hi = self.rlo[r], self.rhi[r]
        if not row:
            if (lo is not None and lo > 0) or (hi is not None and hi < 0):
                raise Infeasible(("empty row", r))
            self._remove_row(r)
            self.n_rules["R"] += 1
            return True
        mn, nmn, mx, nmx = self._act(r)
        lo_ok = lo is None or (nmn == 0 and mn >= lo)
        hi_ok = hi is None or (nmx == 0 and mx <= hi)
        if lo_ok and hi_ok:
            self._remove_row(r)
            self.n_rules["R"] += 1
            return True
        if lo is not None and nmn == 0 and mn >= lo:
            self.rlo[r] = None
        if hi is not None and nmx == 0 and mx <= hi:
            self.rhi[r] = None
        return False

    def fix_and_substitute(self, j) -> set:
        """R: a fixed column is substituted into its rows and removed."""
        v = self.lb[j]
        touched = set(self.cols.get(j, ()))
        for r in touched:
            a = self.rows[r].pop(j)
            if self.rlo[r] is not None:
                self.rlo[r] -= a * v
            if self.rhi[r] is not None:
                self.rhi[r] -= a * v
        self.cols[j] = set()
        self.n_rules["F"] += 1
        return touched

    def direction(self, j, skip=None):
        """+1 if every row (but ``skip``) prefers j up, -1 if down, 0 otherwise."""
        up = down = True
        for r in self.cols.get(j, ()):
            if r == skip:
                continue
            a = self.rows[r][j]
            lo, hi = self.rlo[r], self.rhi[r]
            # raising j by a>0 helps rows with only a lower side
            if a > 0:
                up &= hi is None
                down &= lo is None
            else:
                up &= lo is None
                down &= hi is None
            if not (up or down):
                return 0
        return 1 if up else (-1 if down else 0)

    def dominated(self, j) -> Optional[set]:
        """D: fix a dominated column at its preferred bound (or drop it with its
        rows when that bound is infinite). Returns the rows touched, or None."""
        if not self.cols.get(j):
            return None
        d = self.direction(j)
        if d == 0:
            return None
        target = self.ub[j] if d > 0 else self.lb[j]
        if target is None:
            rows = set(self.cols[j])
            others = set()
            for r in rows:
                others |= set(self.rows[r])
                self._remove_row(r)
            self.n_rules["D"] += 1
            # The rows of the columns that lost a row: those may now propagate.
            return {rr for c in others for rr in self.cols.get(c, ())}
        self.lb[j] = self.ub[j] = target
        self.n_rules["D"] += 1
        return self.fix_and_substitute(j)

    def tighten(self, r) -> bool:
        """T: coefficient tightening of binary columns in a one-sided row."""
        lo, hi = self.rlo[r], self.rhi[r]
        if (lo is None) == (hi is None):
            return False  # equality / ranged rows: not handled here
        # work on sign*row <= b
        if hi is not None:
            sign, b = 1, hi
        else:
            assert lo is not None
            sign, b = -1, -lo
        row = {j: sign * a for j, a in self.rows[r].items()}
        changed = False
        for k, a in list(row.items()):
            if k not in self.held or self.lb[k] != 0 or self.ub[k] != 1:
                continue
            # max activity of the rest
            mx: Optional[F] = F(0)
            for j, c in row.items():
                if j == k:
                    continue
                v = self.ub[j] if c > 0 else self.lb[j]
                if v is None:
                    mx = None
                    break
                mx = mx + c * v if mx is not None else None
            if mx is None:
                continue
            if a < 0 and b < mx < b - a:  # strict: a' = b - mx is a real change
                # x=1 side redundant: a' = b - mx
                row[k] = b - mx
                changed = True
            elif a > 0 and mx < b and b - a < mx:  # strict: a' < a
                # x=0 side redundant: a' = a - (b - mx), b' = mx
                row[k] = a - (b - mx)
                b = mx
                changed = True
        if changed:
            for j, c in row.items():
                if c == 0:
                    self.cols[j].discard(r)
            self.rows[r] = {j: sign * c for j, c in row.items() if c != 0}
            if hi is not None:
                self.rhi[r] = b
            else:
                self.rlo[r] = -b
            self.n_rules["T"] += 1
        return changed

    def aggregate(self, u) -> Optional[set]:
        """A: u := c0 + c1*x through the one row that bounds it against its
        preferred direction. Returns touched rows or None."""
        rows_u = self.cols.get(u)
        if not rows_u:
            return None
        for R in list(rows_u):
            row = self.rows[R]
            if len(row) != 2:
                continue
            d = self.direction(u, skip=R)
            if d == 0:
                continue
            (x,) = [j for j in row if j != u]
            au, ax = row[u], row[x]
            lo, hi = self.rlo[R], self.rhi[R]
            # the side of R that caps u in direction d: up -> au>0 uses hi
            # (au*u <= hi - ax*x), au<0 uses lo; down -> the opposite sides.
            use_hi = (d > 0) == (au > 0)
            cap, other = (hi, lo) if use_hi else (lo, hi)
            if cap is None or other is not None:
                continue  # need exactly the capping side
            c0, c1 = cap / au, -ax / au  # u = c0 + c1*x at the cap
            lx, ux = self.lb[x], self.ub[x]
            if lx is None or ux is None:
                continue
            vals = (c0 + c1 * lx, c0 + c1 * ux)
            vmin, vmax = min(vals), max(vals)
            if (self.lb[u] is not None and vmin < self.lb[u]) or (
                self.ub[u] is not None and vmax > self.ub[u]
            ):
                continue
            if u in self.held and not (
                x in self.held and c0.denominator == 1 and c1.denominator == 1
            ):
                continue
            # substitute u = c0 + c1 x into every other row of u
            touched = set()
            for r in list(rows_u):
                if r == R:
                    continue
                a = self.rows[r].pop(u)
                lo_r, hi_r = self.rlo[r], self.rhi[r]
                if lo_r is not None:
                    self.rlo[r] = lo_r - a * c0
                if hi_r is not None:
                    self.rhi[r] = hi_r - a * c0
                new = self.rows[r].get(x, F(0)) + a * c1
                if new == 0:
                    self.rows[r].pop(x, None)
                    self.cols[x].discard(r)
                else:
                    self.rows[r][x] = new
                    self.cols[x].add(r)
                touched.add(r)
            self._remove_row(R)
            self.cols[u] = set()
            self.n_rules["A"] += 1
            return touched | {r for r in self.cols.get(x, ())}
        return None

    # -- driver ----------------------------------------------------------------------
    def reduce(self, deadline=None, max_rounds=1000):
        import time

        queue = set(self.rows)
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            while queue:
                if deadline is not None and time.perf_counter() > deadline:
                    return "timeout"
                r = queue.pop()
                if r not in self.rows:
                    continue
                changed = self.propagate_row(r)
                for j in changed:
                    queue |= self.cols.get(j, set())
                    if self.lb[j] is not None and self.lb[j] == self.ub[j]:
                        queue |= self.fix_and_substitute(j)
                if r in self.rows and self.tighten(r):
                    queue.add(r)
                    queue |= {rr for j in self.rows.get(r, {}) for rr in self.cols.get(j, ())}
                if r in self.rows:
                    self.redundant_or_empty(r)
            # column rules, then go round again if anything changed
            progress = False
            for j in list(self.cols):
                if not self.cols.get(j):
                    continue
                t = self.dominated(j)
                if t is not None:  # rows to revisit
                    queue |= {rr for rr in t if rr in self.rows}
                    progress = True
                    continue
                t = self.aggregate(j)
                if t is not None:
                    queue |= {rr for rr in t if rr in self.rows}
                    progress = True
            if not progress and not queue:
                return "fixpoint"
        return "maxrounds"

    def export(self):
        """Float arrays of the reduced model (rows that remain, columns in them),
        refusing any datum not exactly representable as a float."""
        import scipy.sparse as sp

        for r in [r for r in self.rows if not self.rows[r]]:
            self.redundant_or_empty(r)  # 0 in [lo, hi]: dropped; else Infeasible
        rows = sorted(self.rows)
        cols = sorted({j for r in rows for j in self.rows[r]})
        ci = {j: k for k, j in enumerate(cols)}
        data, ind, ptr = [], [], [0]

        def ex(v):
            """A coefficient: must be exact (else the reduction is not applied)."""
            f = float(v)
            if F(f) != v:
                raise ValueError("coefficient not exactly representable")
            return f

        def up(v):
            """An upper side / bound, rounded outward (a relaxation: if the
            relaxed model is infeasible, so is the exact one)."""
            if v is None:
                return math.inf
            f = float(v)
            return f if F(f) >= v else math.nextafter(f, math.inf)

        def down(v):
            if v is None:
                return -math.inf
            f = float(v)
            return f if F(f) <= v else math.nextafter(f, -math.inf)

        rlo, rhi = [], []
        for r in rows:
            # Scale the row by the lcm of its coefficients' denominators, so
            # substitution's rational coefficients become integers -- exact as
            # floats while below 2**53. The sides scale with it (then round
            # outward). Scaling a row by a positive constant changes nothing.
            scale = 1
            for a in self.rows[r].values():
                scale = scale * a.denominator // math.gcd(scale, a.denominator)
            for j, a in sorted(self.rows[r].items()):
                data.append(ex(a * scale))
                ind.append(ci[j])
            ptr.append(len(data))
            rlo.append(down(self.rlo[r] * scale if self.rlo[r] is not None else None))
            rhi.append(up(self.rhi[r] * scale if self.rhi[r] is not None else None))
        A = sp.csr_matrix((data, ind, ptr), shape=(len(rows), len(cols)))
        lb = np.array([down(self.lb[j]) for j in cols])
        ub = np.array([up(self.ub[j]) for j in cols])
        held = np.array([j in self.held for j in cols])
        return A, np.array(rlo), np.array(rhi), lb, ub, held


#: Largest finite magnitude handed to HiGHS from a reduced model (coefficients,
#: bounds, row sides); beyond it the route draws no conclusion. MIPcc27's
#: general-integer models stay below 1e4.
_MAX_MAGNITUDE = 1e9

#: The reduced model is solved only if it kept at most this share of the rows.
_MIN_SHRINK = 0.8

#: Cap on the reduction itself, so the pre-pass cannot eat a confirmation's
#: budget (measured 0.3-2.0 s on MIPcc27 sets of 1.4k-4.8k rows).
REDUCE_CAP_S = 10.0


def reduced_infeasible(
    A, rlo, rhi, lb, ub, held, stop_at, guard=None, engine: str = "highs"
) -> bool | None:
    """True iff the exact reduction proves the system infeasible, by itself or
    with presolve-free HiGHS on the reduced model; None otherwise (no
    conclusion: a feasible or undecided reduced model, a reduction that ran out
    of time, or one that cannot be exported exactly).

    ``stop_at`` (``time.perf_counter()``) bounds both steps; None means no
    budget beyond :data:`REDUCE_CAP_S` for the reduction. ``guard(h, stop)``
    installs the caller's wall-clock interrupt on the HiGHS object. ``engine``
    (highs-parity P3.2) picks who solves the reduced model presolve-free:
    ``"highs"`` or ``"rust"`` (the in-house MILP driver, which honours the budget
    itself and takes no guard); the reduction and every check before it are the
    same either way."""
    import time

    now = time.perf_counter()
    cap = now + REDUCE_CAP_S
    model = Model(A, rlo, rhi, lb, ub, held)
    try:
        status = model.reduce(deadline=cap if stop_at is None else min(cap, stop_at))
    except Infeasible:
        return True
    if status != "fixpoint":
        return None
    try:
        Ar, rl, rh, lo, hi, hd = model.export()
    except Infeasible:
        return True  # a row left empty by the reduction cannot hold
    except ValueError:
        return None  # a coefficient is not exactly representable: no conclusion
    if Ar.shape[0] == 0:
        return None  # everything reduced away: the system is feasible
    n_rows = int(np.sum(np.isfinite(np.asarray(rlo, float)) | np.isfinite(np.asarray(rhi, float))))
    if Ar.shape[0] > _MIN_SHRINK * n_rows:
        # The reduction did little: its model is about the input's size, so a
        # presolve-free solve of it is no better than the raw one the caller
        # runs next -- leave that route the whole budget (#2145 review). A
        # blanket half share instead was measured to starve the effective case
        # (dijle 2426 -> 2938 at 600 s: its 66-row reduced models need ~13 s).
        return None
    big = max(
        float(np.max(np.abs(Ar.data), initial=0.0)),
        *(float(np.max(np.abs(v[np.isfinite(v)]), initial=0.0)) for v in (rl, rh, lo, hi)),
    )
    if big > _MAX_MAGNITUDE:
        # Propagation along an unbounded chain can produce bounds like 1e40;
        # HiGHS reads |v| >= 1e20 as infinite and is numerically untrustworthy
        # well below that. No conclusion (the caller's other routes run).
        return None
    left = None if stop_at is None else stop_at - time.perf_counter()
    if left is not None and left <= 0.05:
        return None
    if engine == "rust":
        return _rust_reduced(Ar, rl, rh, lo, hi, hd, left)
    if engine != "highs":
        raise ValueError(f"engine={engine!r}: expected 'rust' or 'highs'")
    return _highs_reduced(Ar, rl, rh, lo, hi, hd, left, stop_at, guard)


def _rust_reduced(Ar, rl, rh, lo, hi, hd, left) -> bool | None:
    """The reduced model, presolve-free, on the in-house MILP driver: True when it
    proves it infeasible, else None."""
    from discopt.solvers.milp_handle import MilpHandle

    n = Ar.shape[1]
    h = MilpHandle(np.zeros(n), Ar, rl, rh, lo, hi, np.asarray(hd, dtype=bool))
    r = h.solve(presolve=False, time_limit=left)
    return True if r.status == "infeasible" else None


def _highs_reduced(Ar, rl, rh, lo, hi, hd, left, stop_at, guard) -> bool | None:
    """The reduced model, presolve-free, on HiGHS (the pre-P3.2 code)."""
    import highspy

    h = highspy.Highs()
    h.silent()
    h.setOptionValue("presolve", "off")
    if left is not None:
        h.setOptionValue("time_limit", left)
    if guard is not None:
        guard(h, stop_at)
    n = Ar.shape[1]
    inf = highspy.kHighsInf
    h.addVars(n, np.where(np.isfinite(lo), lo, -inf), np.where(np.isfinite(hi), hi, inf))
    vt = highspy.HighsVarType
    h.changeColsIntegrality(
        n,
        np.arange(n, dtype=np.int32),
        np.array([vt.kInteger if k else vt.kContinuous for k in hd]),
    )
    Ar = Ar.tocsr()
    for r in range(Ar.shape[0]):
        s, e = Ar.indptr[r], Ar.indptr[r + 1]
        h.addRow(
            rl[r] if np.isfinite(rl[r]) else -inf,
            rh[r] if np.isfinite(rh[r]) else inf,
            e - s,
            Ar.indices[s:e].astype(np.int32),
            Ar.data[s:e],
        )
    run = h.run()
    if run not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        raise RuntimeError(f"HiGHS failed on the reduced model: {run}")
    return True if h.getModelStatus() == highspy.HighsModelStatus.kInfeasible else None
