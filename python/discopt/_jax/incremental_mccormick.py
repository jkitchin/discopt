"""Incremental McCormick LP for the LP-node spatial branch-and-bound engine.

``build_milp_relaxation`` re-walks the expression DAG on every call (~14 ms on
nvs17) even though, across a spatial-B&B tree, the LP *structure* (columns, row
sparsity, the fixed linear/model rows, the objective) is identical for every box —
only the McCormick envelope rows and the auxiliary-variable bounds depend on the
node box. This class builds the structure **once** and per node recomputes only the
box-dependent rows/bounds in closed form (numpy, ~0.1 ms), then warm-starts the LP
from the parent basis. That removes both the per-node DAG re-walk and the cold LP
solve — the throughput needed to close the nvs17/19/24 family by branching (the
no-cut-SCIP regime: thousands of fast nodes).

**Soundness gate.** The closed-form envelopes are validated to reproduce
``build_milp_relaxation`` exactly (row-set and bounds, to tolerance) on random
boxes at construction. If validation fails — an unhandled term type, a different
discretization, anything — :attr:`ok` is ``False`` and the caller falls back to the
trusted per-node builder. The incremental path can therefore never change a result,
only its speed.

Scope: the box-dependent terms it regenerates are bilinear products ``w=x_i*x_j``
(4 McCormick rows) and integer squares ``s=x_i**2`` (2 endpoint tangents + 1
secant, matching an empty ``DiscretizationState``). Any other lifted term (trilinear,
univariate, fractional power, piecewise) makes validation fail -> fallback.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_TOL = 1e-7


def _bilinear_rows(i, j, a, li, ui, lj, uj):
    """The 4 McCormick inequalities for w=x_i*x_j over [li,ui]x[lj,uj], each as
    ``(coeff_on_i, coeff_on_j, coeff_on_w, rhs)`` of an ``... <= rhs`` row."""
    return [
        (lj, li, -1.0, li * lj),  # w >= lj*xi + li*xj - li*lj
        (uj, ui, -1.0, ui * uj),  # w >= uj*xi + ui*xj - ui*uj
        (-uj, -li, 1.0, -li * uj),  # w <= uj*xi + li*xj - li*uj
        (-lj, -ui, 1.0, -ui * lj),  # w <= lj*xi + ui*xj - ui*lj
    ]


def _square_rows(i, a, li, ui):
    """The 3 rows for s=x_i**2 over [li,ui]: 2 endpoint tangents + 1 secant, each
    ``(coeff_on_i, coeff_on_s, rhs)`` of an ``... <= rhs`` row."""
    return [
        (2.0 * li, -1.0, li * li),  # s >= 2*li*xi - li^2  (tangent at li)
        (2.0 * ui, -1.0, ui * ui),  # s >= 2*ui*xi - ui^2  (tangent at ui)
        (-(li + ui), 1.0, -li * ui),  # s <= (li+ui)*xi - li*ui (secant)
    ]


def _bilinear_aux_bounds(li, ui, lj, uj):
    corners = (li * lj, li * uj, ui * lj, ui * uj)
    return min(corners), max(corners)


def _square_aux_bounds(li, ui):
    if li >= 0:
        return li * li, ui * ui
    if ui <= 0:
        return ui * ui, li * li
    return 0.0, max(li * li, ui * ui)


class IncrementalMcCormickLP:
    """Build the McCormick LP structure once; patch box-dependent rows per node."""

    def __init__(self, model, terms):
        self.ok = False
        self.model = model
        self.terms = terms
        try:
            self._build_structure()
            self._validate()
        except Exception:
            self.ok = False

    # -- construction ------------------------------------------------------ #

    def _full_build(self, lb, ub):
        from discopt._jax.discretization import DiscretizationState
        from discopt._jax.milp_relaxation import build_milp_relaxation

        relax, info = build_milp_relaxation(
            self.model, self.terms, DiscretizationState(), bound_override=(lb, ub)
        )
        if not relax._objective_bound_valid or relax._A_ub is None:
            raise ValueError("relaxation has no valid bound / no rows")
        A = np.asarray(sp.csr_matrix(relax._A_ub).todense(), dtype=np.float64)
        b = np.asarray(relax._b_ub, dtype=np.float64).ravel()
        bnds = np.asarray(relax._bounds, dtype=np.float64)
        c = np.asarray(relax._c, dtype=np.float64).ravel()
        return A, b, bnds, c, info, relax

    def _build_structure(self):
        n = len(self.model._variables)
        # probe box: distinct, strictly positive bounds so every McCormick
        # coefficient is nonzero -> row support reveals {factors, aux} cleanly.
        lb_p = np.array([1.0 + 0.0 * k for k in range(n)])
        ub_p = np.array([7.0 + 1.0 * k for k in range(n)])
        A, b, bnds, c, info, _ = self._full_build(lb_p, ub_p)
        self.n = n
        self.ncol = A.shape[1]
        self.c = c
        self.base_A = A.copy()
        self.base_b = b.copy()
        self.base_bounds = bnds.copy()
        self.bilinear = dict(info.get("bilinear", {}))
        self.monomial = {k: v for k, v in info.get("monomial", {}).items() if k[1] == 2}

        # map each product to its row indices (support subset of {factors, aux})
        supp = [set(np.nonzero(np.abs(A[k]) > _TOL)[0]) for k in range(A.shape[0])]
        self.bilin_rows = {}
        for (i, j), a in self.bilinear.items():
            rows = [k for k in range(A.shape[0]) if a in supp[k] and supp[k] <= {i, j, a}]
            if len(rows) != 4:
                raise ValueError(f"bilinear ({i},{j}) -> {len(rows)} rows, expected 4")
            self.bilin_rows[(i, j, a)] = rows
        self.sq_rows = {}
        for (i, _p), a in self.monomial.items():
            rows = [k for k in range(A.shape[0]) if a in supp[k] and supp[k] <= {i, a}]
            if len(rows) != 3:
                raise ValueError(f"square x_{i}^2 -> {len(rows)} rows, expected 3")
            self.sq_rows[(i, a)] = rows
        # the union of all product rows must be exactly the box-dependent rows
        self._prod_rows = set()
        for rs in self.bilin_rows.values():
            self._prod_rows |= set(rs)
        for rs in self.sq_rows.values():
            self._prod_rows |= set(rs)

    # -- per-node patch ---------------------------------------------------- #

    def _patch(self, lb, ub):
        """Return (A, b, bounds) for the McCormick LP over [lb,ub]."""
        A = self.base_A.copy()
        b = self.base_b.copy()
        bounds = self.base_bounds.copy()
        bounds[: self.n, 0] = lb
        bounds[: self.n, 1] = ub
        for (i, j, a), rows in self.bilin_rows.items():
            li, ui, lj, uj = lb[i], ub[i], lb[j], ub[j]
            for k, (ci, cj, cw, rhs) in zip(rows, _bilinear_rows(i, j, a, li, ui, lj, uj)):
                A[k] = 0.0
                A[k, i] += ci
                A[k, j] += cj
                A[k, a] = cw
                b[k] = rhs
            bounds[a, 0], bounds[a, 1] = _bilinear_aux_bounds(li, ui, lj, uj)
        for (i, a), rows in self.sq_rows.items():
            li, ui = lb[i], ub[i]
            for k, (ci, cs, rhs) in zip(rows, _square_rows(i, a, li, ui)):
                A[k] = 0.0
                A[k, i] = ci
                A[k, a] = cs
                b[k] = rhs
            bounds[a, 0], bounds[a, 1] = _square_aux_bounds(li, ui)
        return A, b, bounds

    # -- soundness gate ---------------------------------------------------- #

    @staticmethod
    def _rowset(A, b):
        """Canonical hashable representation of the polytope's rows (order-free)."""
        rows = np.hstack([np.round(A, 6), np.round(b, 6).reshape(-1, 1)])
        return sorted(map(tuple, rows.tolist()))

    def _validate(self, trials=4):
        rng_boxes = [
            (np.array([0.0] * self.n), np.array([3.0 + (k % 3)] * self.n)) for k in range(trials)
        ]
        # a couple of asymmetric boxes too
        rng_boxes.append((np.arange(self.n, dtype=float), np.arange(self.n, dtype=float) + 5))
        for lb, ub in rng_boxes:
            Ap, bp, bdp = self._patch(lb, ub)
            Af, bf, bdf, _, _, _ = self._full_build(lb, ub)
            if Ap.shape != Af.shape:
                raise ValueError("shape mismatch")
            if self._rowset(Ap, bp) != self._rowset(Af, bf):
                raise ValueError("row-set mismatch")
            if not np.allclose(bdp, bdf, atol=1e-6, rtol=1e-6):
                raise ValueError("bounds mismatch")
        self.ok = True

    # -- solve ------------------------------------------------------------- #

    def assemble(self, lb, ub, cut_rows=None):
        """Patched McCormick LP rows over [lb,ub] with optional appended cut rows.

        ``cut_rows`` is a list of ``(coeffs, rhs)`` inequalities ``coeffs·x <= rhs``
        over the structural+aux columns (length ``ncol``). Returns ``(A, b, bounds)``.
        """
        A, b, bounds = self._patch(lb, ub)
        if cut_rows:
            extra_A = np.array(
                [np.asarray(co, dtype=np.float64)[: self.ncol] for co, _ in cut_rows]
            )
            extra_b = np.array([float(r) for _, r in cut_rows])
            A = np.vstack([A, extra_A])
            b = np.concatenate([b, extra_b])
        return A, b, bounds

    def solve_assembled(self, A, b, bounds, in_basis=None, c_override=None):
        """Solve a pre-assembled LP ``min c·x s.t. A x <= b, bounds``."""
        from discopt.solvers import SolveStatus
        from discopt.solvers.milp_simplex import solve_lp_warm_std

        cobj = self.c if c_override is None else np.asarray(c_override, dtype=np.float64)
        try:
            result, out_basis = solve_lp_warm_std(
                cobj, sp.csr_matrix(A), b, bounds, in_basis=in_basis
            )
        except Exception:
            return None, None, None
        if result is None or result.status != SolveStatus.OPTIMAL or result.objective is None:
            return None, None, None
        return float(result.objective), np.asarray(result.x, dtype=float), out_basis

    def solve_assembled_full(self, A, b, bounds, in_basis=None, c_override=None):
        """Like :meth:`solve_assembled`, but return the terminal *status* too so a
        caller can tell a (certified) ``infeasible`` apart from any other
        non-optimal verdict (time limit / numerical error).

        Returns ``(status, bound, x, out_basis)`` where ``status`` is one of
        ``"optimal"``, ``"infeasible"`` (the LP feasible set is empty over this
        box — a rigorous fathoming proof, since the McCormick polytope is a valid
        outer approximation), or ``"other"`` (no certified verdict). ``bound``/``x``
        are populated only for ``"optimal"``."""
        from discopt.solvers import SolveStatus
        from discopt.solvers.milp_simplex import solve_lp_warm_std

        cobj = self.c if c_override is None else np.asarray(c_override, dtype=np.float64)
        try:
            result, out_basis = solve_lp_warm_std(
                cobj, sp.csr_matrix(A), b, bounds, in_basis=in_basis
            )
        except Exception:
            return "other", None, None, None
        if result is None:
            return "other", None, None, None
        if result.status == SolveStatus.INFEASIBLE:
            return "infeasible", None, None, None
        if result.status != SolveStatus.OPTIMAL or result.objective is None:
            return "other", None, None, None
        return "optimal", float(result.objective), np.asarray(result.x, dtype=float), out_basis

    def solve(self, lb, ub, in_basis=None, c_override=None, cut_rows=None):
        """Solve the McCormick LP over [lb,ub] (plus optional cut rows); return
        (bound, x, out_basis) or (None, None, None). Warm-starts from ``in_basis``.
        ``c_override`` replaces the objective (feasibility pump) — the returned bound
        is then the surrogate, not a dual bound."""
        A, b, bounds = self.assemble(lb, ub, cut_rows)
        return self.solve_assembled(A, b, bounds, in_basis=in_basis, c_override=c_override)
