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

EP3 (#632): the closed-form refresh above is byte-identical only for *bare-variable*
bilinear/monomial rows; the uniform factorable engine (the default relaxation) emits
its box-dependent rows from ``evaluate_interval`` over the reconstructed DAG, which no
closed form reproduces bit-for-bit for a folded / affine-argument atom (measured).
:class:`UniformPatchTable` (below) engages the fast path on engine-shaped models
byte-identically by regenerating through the EP1-cached engine build itself — the win
is the per-node separation skip + warm start, not a numpy coefficient refresh.
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


def _monomial_rows(li, ui, p):
    """The 3 envelope rows for ``s = x**p`` over a *sign-definite* box ``[li,ui]``
    (2 endpoint tangents + 1 secant), each ``(coeff_on_x, coeff_on_s, rhs)`` of an
    ``... <= rhs`` row. Generalizes :func:`_square_rows` (p=2) to any integer
    power ``p >= 2``.

    On a sign-definite box ``x**p`` is monotone and single-convexity: convex when
    ``p`` is even or ``li >= 0``; concave when ``p`` is odd and ``ui <= 0``. In the
    convex case the two endpoint tangents underestimate and the secant
    overestimates (exactly the ``x**2`` pattern); in the concave case the roles
    flip. Reproduces ``build_milp_relaxation`` row-for-row (validated).
    """
    fl, fu = li**p, ui**p
    dfl, dfu = p * li ** (p - 1), p * ui ** (p - 1)
    slope = (fu - fl) / (ui - li)
    convex = (p % 2 == 0) or (li >= 0.0)
    if convex:
        return [
            (dfl, -1.0, dfl * li - fl),  # s >= f'(li)*x - (f'(li)*li - f(li))  tangent at li
            (dfu, -1.0, dfu * ui - fu),  # tangent at ui
            (-slope, 1.0, fl - slope * li),  # s <= secant  (overestimator)
        ]
    return [
        (-dfl, 1.0, fl - dfl * li),  # s <= tangent at li  (overestimator)
        (-dfu, 1.0, fu - dfu * ui),  # tangent at ui
        (slope, -1.0, slope * li - fl),  # s >= secant  (underestimator)
    ]


def _monomial_aux_bounds(li, ui, p):
    """min/max of ``x**p`` over a sign-definite ``[li,ui]`` (monotone there)."""
    a, b = li**p, ui**p
    return (a, b) if a <= b else (b, a)


class IncrementalMcCormickLP:
    """Build the McCormick LP structure once; patch box-dependent rows per node."""

    def __init__(self, model, terms):
        self.ok = False
        self.model = model
        self.terms = terms
        self._validated_regimes = frozenset()  # sign regimes _validate exercised
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
        # Per-variable ROOT sign regime (cert:T1.2). A monomial ``x**p`` has a
        # box-*sign*-dependent row structure (3 rows on a sign-definite box, 4/2
        # when the box strictly spans zero), so it can be patched only when the
        # variable's root box is sign-definite — which branching preserves, since
        # it only shrinks boxes. ``+1`` = ``lb>=0``, ``-1`` = ``ub<=0``, ``0`` =
        # spans zero (any monomial on such a var is unmappable below).
        root_lb = np.array([float(np.min(v.lb)) for v in self.model._variables])
        root_ub = np.array([float(np.max(v.ub)) for v in self.model._variables])
        self._root_sign = np.where(root_lb >= 0.0, 1, np.where(root_ub <= 0.0, -1, 0))
        # probe box: distinct, strictly *sign-matched* bounds so every McCormick
        # coefficient is nonzero (row support reveals {factors, aux} cleanly) and
        # the cached convex/concave power rows match the cold build's regime.
        lb_p = np.empty(n)
        ub_p = np.empty(n)
        for k in range(n):
            if self._root_sign[k] < 0:
                lb_p[k], ub_p[k] = -(7.0 + k), -1.0
            else:
                lb_p[k], ub_p[k] = 1.0, 7.0 + k
        A, b, bnds, c, info, _ = self._full_build(lb_p, ub_p)
        self.n = n
        self.ncol = A.shape[1]
        self.c = c
        self.base_A = A.copy()
        self.base_b = b.copy()
        self.base_bounds = bnds.copy()
        self.bilinear = dict(info.get("bilinear", {}))
        self.monomial = dict(info.get("monomial", {}))  # any integer power p >= 2

        # C-44: per-column identity vector for this fixed lifted layout. The
        # incremental structure never rebuilds columns (``_patch`` only rewrites
        # coefficients of existing rows), so this identity vector is stable across
        # every node — a root cut pool (captured on the cold path over its own
        # varmap) is remapped onto these positions by matching identity. Built
        # from the same ``info`` (full varmap) the cold path returns.
        try:
            from discopt._jax.mccormick_lp import column_identities

            self.col_identities = column_identities(info, self.ncol, self.n)
        except Exception:
            self.col_identities = None

        # map each product to its row indices (support subset of {factors, aux})
        supp = [set(np.nonzero(np.abs(A[k]) > _TOL)[0]) for k in range(A.shape[0])]
        self.bilin_rows = {}
        for (i, j), a in self.bilinear.items():
            rows = [k for k in range(A.shape[0]) if a in supp[k] and supp[k] <= {i, j, a}]
            if len(rows) != 4:
                raise ValueError(f"bilinear ({i},{j}) -> {len(rows)} rows, expected 4")
            self.bilin_rows[(i, j, a)] = rows
        # monomial x_i**p, any p >= 2, gated on a sign-definite root box.
        self.mono_rows = {}
        for (i, p), a in self.monomial.items():
            if self._root_sign[i] == 0:
                raise ValueError(f"monomial x_{i}^{p}: root box spans zero (unmappable)")
            rows = [k for k in range(A.shape[0]) if a in supp[k] and supp[k] <= {i, a}]
            if len(rows) != 3:
                raise ValueError(f"monomial x_{i}^{p} -> {len(rows)} rows, expected 3")
            self.mono_rows[(i, a, p)] = rows
        # the union of all product rows must be exactly the box-dependent rows
        self._prod_rows = set()
        for rs in self.bilin_rows.values():
            self._prod_rows |= set(rs)
        for rs in self.mono_rows.values():
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
        for (i, a, p), rows in self.mono_rows.items():
            li, ui = lb[i], ub[i]
            for k, (ci, cs, rhs) in zip(rows, _monomial_rows(li, ui, p)):
                A[k] = 0.0
                A[k, i] = ci
                A[k, a] = cs
                b[k] = rhs
            bounds[a, 0], bounds[a, 1] = _monomial_aux_bounds(li, ui, p)
        return A, b, bounds

    # -- soundness gate ---------------------------------------------------- #

    @staticmethod
    def _rowset(A, b):
        """Canonical hashable representation of the polytope's rows (order-free)."""
        rows = np.hstack([np.round(A, 6), np.round(b, 6).reshape(-1, 1)])
        return sorted(map(tuple, rows.tolist()))

    @staticmethod
    def _box_sign_regime(lo, hi):
        """Classify a single variable's box ``[lo,hi]`` into a sign regime label so
        the validation set can prove it spans several. ``"pos"`` (``lo>0``),
        ``"neg"`` (``hi<0``), ``"span"`` (``lo<0<hi``, strictly crosses zero),
        ``"degen"`` (``lo==hi``), ``"zero_lb"`` (``lo==0<hi``, the boundary)."""
        if lo == hi:
            return "degen"
        if lo == 0.0:
            return "zero_lb"
        if lo > 0.0:
            return "pos"
        if hi <= 0.0:
            return "neg"
        return "span"

    def _validation_boxes(self):
        """The validation boxes fed to :meth:`_validate`, as ``(lo, hi)`` pairs.

        Every box is a *reachable* B&B sub-box of the root: branching only shrinks a
        box, so a var that is sign-definite at the root (``_root_sign != 0``) keeps
        that sign — a positive var never gets ``lb<0``, a negative var never gets
        ``ub>0``. A **spanning** var (``_root_sign==0``), however, carries no
        monomial (gated out in :meth:`_build_structure`) and its real nodes DO carry
        negative / zero-spanning bounds, so the boxes below deliberately drive those
        vars through negative-lb, zero-spanning (``lb<0<ub``), mixed-sign and
        degenerate (``lb==ub``) regimes — exactly the sign regimes that dominate
        real nodes and that the earlier ``lb>=0``-only set never exercised (C-21).
        """
        # Per trial, ``kind`` says how each spanning var sits relative to zero;
        # sign-definite vars follow their root sign with a varying width/offset.
        kinds = ["shift_pos", "zero_lb", "span", "neg", "span_wide", "degen"]
        boxes = []
        for t, kind in enumerate(kinds):
            w = 2.0 + (t % 3)
            lo = np.empty(self.n)
            hi = np.empty(self.n)
            for i in range(self.n):
                if self._root_sign[i] < 0:
                    # Negative-definite root: stay ub<=0 (reachable sub-box).
                    hi[i] = -0.5 - 0.3 * i
                    lo[i] = hi[i] - w
                elif self._root_sign[i] > 0:
                    # Positive-definite root: stay lb>=0 (reachable sub-box). Even
                    # trials touch the lb==0 boundary.
                    lo[i] = (0.5 + 0.3 * i) if (t % 2) else 0.0
                    hi[i] = lo[i] + w
                else:
                    # Spanning root: exercise the negative / zero-spanning regimes
                    # that real nodes reach and the old validation set never did.
                    off = 0.2 * i
                    if kind == "shift_pos":
                        lo[i], hi[i] = 0.5 + off, 0.5 + off + w
                    elif kind == "zero_lb":
                        lo[i], hi[i] = 0.0, w
                    elif kind == "span":
                        lo[i], hi[i] = -(1.0 + off), 1.0 + off
                    elif kind == "neg":
                        hi[i] = -0.5 - off
                        lo[i] = hi[i] - w
                    elif kind == "span_wide":
                        lo[i], hi[i] = -(2.0 + off + w), 1.5 + off
                    else:  # degen
                        lo[i] = hi[i] = -0.5 - off
            boxes.append((lo, hi))
        return boxes

    def _validate(self):
        # Reachable, sign-diverse validation boxes (C-21 / cert:T1.2): each box is a
        # sub-box of the root (so the patched convex/concave power rows are compared
        # against a cold build in the *same* regime), but spanning vars are driven
        # through negative-lb, zero-spanning, mixed-sign and degenerate boxes — the
        # sign regimes real nodes reach. The patched row-set + aux bounds must
        # reproduce the cold ``build_milp_relaxation`` exactly on every one.
        rng_boxes = self._validation_boxes()
        regimes = set()
        for lb, ub in rng_boxes:
            for i in range(self.n):
                regimes.add(self._box_sign_regime(float(lb[i]), float(ub[i])))
            Ap, bp, bdp = self._patch(lb, ub)
            Af, bf, bdf, _, _, _ = self._full_build(lb, ub)
            if Ap.shape != Af.shape:
                raise ValueError("shape mismatch")
            if self._rowset(Ap, bp) != self._rowset(Af, bf):
                raise ValueError("row-set mismatch")
            if not np.allclose(bdp, bdf, atol=1e-6, rtol=1e-6):
                raise ValueError("bounds mismatch")
        self._validated_regimes = frozenset(regimes)
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

    def solve_assembled_full(
        self, A, b, bounds, in_basis=None, c_override=None, *, return_cert=False
    ):
        """Like :meth:`solve_assembled`, but return the terminal *status* too so a
        caller can tell a (certified) ``infeasible`` apart from any other
        non-optimal verdict (time limit / numerical error).

        Returns ``(status, bound, x, out_basis, farkas_certified)`` where
        ``status`` is one of ``"optimal"``, ``"infeasible"`` (the LP feasible set
        is empty over this box — a rigorous fathoming proof, since the McCormick
        polytope is a valid outer approximation), or ``"other"`` (no certified
        verdict). ``bound``/``x`` are populated only for ``"optimal"``.

        ``bound`` is the **Neumaier–Shcherbina safe lower bound** built from the
        simplex's own row duals (issue #356) — sound at any conditioning, so it is
        never above the true LP optimum even when an ill-conditioned lifted basis
        makes the raw vertex objective drift high. ``farkas_certified`` is ``True``
        only when an ``"infeasible"`` verdict was independently proven by a
        verified Farkas dual ray; a caller can then fathom rigorously without any
        second (HiGHS/equilibration) solve.

        When ``return_cert`` is set the tuple is extended to
        ``(..., farkas_certified, cert)`` with the :class:`LpWarmCert` carrying the
        node LP's row duals / column status / safe bound (cert:T2.4a) -- a pure
        side-channel; ``bound``/``x`` are computed identically whether or not it is
        requested."""
        from discopt.solvers import SolveStatus
        from discopt.solvers.milp_simplex import LpWarmCert, solve_lp_warm_std

        cobj = self.c if c_override is None else np.asarray(c_override, dtype=np.float64)
        _empty = LpWarmCert(safe_bound=None, farkas_certified=False)

        def _ret(status, bound, x, out_basis, farkas, cert=_empty):
            if return_cert:
                return status, bound, x, out_basis, farkas, cert
            return status, bound, x, out_basis, farkas

        try:
            result, out_basis, cert = solve_lp_warm_std(
                cobj, sp.csr_matrix(A), b, bounds, in_basis=in_basis, return_cert=True
            )
        except Exception:
            return _ret("other", None, None, None, False)
        if result is None:
            return _ret("other", None, None, None, False)
        if result.status == SolveStatus.INFEASIBLE:
            return _ret("infeasible", None, None, None, bool(cert.farkas_certified), cert)
        if result.status != SolveStatus.OPTIMAL or result.bound is None:
            return _ret("other", None, None, None, False)
        return _ret(
            "optimal",
            float(result.bound),
            np.asarray(result.x, dtype=float),
            out_basis,
            False,
            cert,
        )

    def solve(self, lb, ub, in_basis=None, c_override=None, cut_rows=None):
        """Solve the McCormick LP over [lb,ub] (plus optional cut rows); return
        (bound, x, out_basis) or (None, None, None). Warm-starts from ``in_basis``.
        ``c_override`` replaces the objective (feasibility pump) — the returned bound
        is then the surrogate, not a dual bound."""
        A, b, bounds = self.assemble(lb, ub, cut_rows)
        return self.solve_assembled(A, b, bounds, in_basis=in_basis, c_override=c_override)


# --------------------------------------------------------------------------- #
# EP3 (#632): patch-table node path for the uniform factorable engine
# --------------------------------------------------------------------------- #
class UniformPatchTable(IncrementalMcCormickLP):
    """Engage the incremental fast path on ENGINE-shaped models (issue #632, EP3).

    Why this exists — and why it is *not* the ~0.1 ms numpy coefficient refresh
    :class:`IncrementalMcCormickLP` performs for bare-variable bilinear/monomial
    rows. The uniform factorable engine (``uniform_relax.build_uniform_relaxation``,
    the default relaxation since the #632 cutover) emits its box-dependent rows
    (secant/tangent, McCormick folds, log-space bands) and aux-column bounds from
    ``evaluate_interval`` run over the reconstructed canonical DAG. A closed-form
    recomputation of those coefficients does **not** reproduce ``evaluate_interval``
    bit-for-bit for any multi-term (affine / folded / power-of-affine) argument —
    measured: a closed-form affine interval differs from ``evaluate_interval`` on
    fac2 (3/3 atoms) and alan (1/6). So the cheap-closed-form premise of EP3 is
    **falsified for the engine's dominant families** (see
    ``docs/dev/engine-performance-plan.md`` §3, EP3 row): a byte-identical patch
    cannot be assembled faster than the (EP1-cached) engine build itself, whose
    columns and bounds derive from the interleaved DAG walk.

    What this table *does* deliver, byte-identically and soundly: it engages the
    existing incremental fast path (``MccormickLPRelaxer._try_incremental_node``)
    for engine-shaped models by regenerating the node relaxation through the
    EP1-cached engine build (``_patch`` == ``_full_build``, byte-identical by
    construction), so the per-node solve **skips the per-node separation chain**
    (the inherited root cut pool substitutes) and **warm-starts the Rust simplex**.
    On nvs09 (separation-dominated, ~170 ms/node cold) that is the ~10x lever the
    #632 cutover silently disabled; engaging it is node-count- and objective-neutral
    **to optimality** (verified: nvs09 19 nodes / ex1226 5 nodes, dual-bound and
    incumbent byte-identical with the fast path forced off). The construction-time
    gate is unchanged in spirit: :meth:`_validate` proves the patched relaxation
    reproduces the cold ``build_milp_relaxation`` row-for-row on reachable
    sub-boxes AND that the lifted column layout is box-stable; any model that
    fails either (conditional log-space/composite columns that appear only on a
    child box, an invalid/under-constrained objective bound, a build failure) sets
    ``ok=False`` and the trusted cold path runs unchanged. The fast path therefore
    stays a strict subset of the cold build — never an unsound or non-neutral
    activation.
    """

    def __init__(self, model):  # noqa: D401 — see class docstring
        self.ok = False
        self.model = model
        self._validated_regimes = frozenset()
        # ``terms`` is ignored by the engine delegate (``build_milp_relaxation``
        # routes to ``build_uniform_relaxation``), but ``_full_build`` passes it and
        # ``column_identities`` reads the returned varmap, so classify once.
        try:
            from discopt._jax.term_classifier import classify_nonlinear_terms

            self.terms = classify_nonlinear_terms(model)
        except Exception:
            self.terms = None
        # Box-dependent row bookkeeping for engagement telemetry (see
        # ``n_box_dependent_rows``); does not affect the patch (which regenerates
        # every row through the byte-identical engine build).
        self.n_box_dependent_rows = 0
        self.n_const_rows = 0
        try:
            self._build_structure()
            self._validate()
        except Exception:
            self.ok = False

    def _build_structure(self) -> None:
        n = len(self.model._variables)
        root_lb = np.array([float(np.min(v.lb)) for v in self.model._variables])
        root_ub = np.array([float(np.max(v.ub)) for v in self.model._variables])
        self.n = n
        self._root_lb = root_lb
        self._root_ub = root_ub
        self._root_sign = np.where(root_lb >= 0.0, 1, np.where(root_ub <= 0.0, -1, 0))
        # Probe at the ACTUAL root box (the loosest reachable box), not a synthetic
        # sign-matched one: UniformPatchTable does not reverse-engineer row support
        # from the matrix, so it does not need distinct sign-matched coefficients —
        # it needs the probe's lifted column layout to be the one the reachable
        # sub-boxes carry, which is exactly the root's (branching only shrinks a box;
        # every conditional envelope that fires at a child fired at the wider root,
        # modulo curvature that resolves only on a strictly smaller box — those
        # nodes raise in ``_patch`` and take the cold path, sound). If the root box
        # is unbounded / the objective bound is invalid, ``_full_build`` raises and
        # ``ok`` stays False (the near-unbounded node is declined at runtime anyway).
        A, b, bnds, c, info, _ = self._full_build(root_lb, root_ub)
        self.ncol = A.shape[1]
        self.c = c
        self.base_A = A.copy()
        self.base_b = b.copy()
        self.base_bounds = bnds.copy()
        # Kept for interface parity with IncrementalMcCormickLP (lp_spatial_bb).
        self.bilinear = dict(info.get("bilinear", {}))
        self.monomial = dict(info.get("monomial", {}))
        try:
            from discopt._jax.mccormick_lp import column_identities

            self.col_identities = column_identities(info, self.ncol, self.n)
        except Exception:
            self.col_identities = None

    def _reachable_boxes(self) -> list:
        """Reachable B&B sub-boxes of the root: shrink one finite-width column to its
        lower / upper half per box (the branching pattern real nodes take). These —
        not synthetic sign-regime boxes — are what ``_validate`` must confirm the
        patch reproduces and the lifted layout stays stable on."""
        lb0, ub0 = self._root_lb, self._root_ub
        width = ub0 - lb0
        cols = [c for c in range(self.n) if np.isfinite(width[c]) and width[c] > 1e-9]
        boxes = []
        # Root box itself + a few single-variable bisections (both halves) + an
        # all-vars mild shrink, covering the reachable sub-box regimes.
        boxes.append((lb0.copy(), ub0.copy()))
        for i, col in enumerate(cols[:6]):
            lb = lb0.copy()
            ub = ub0.copy()
            mid = 0.5 * (lb[col] + ub[col])
            if i % 2 == 0:
                ub[col] = mid
            else:
                lb[col] = mid
            boxes.append((lb, ub))
        if cols:
            lb = lb0.copy()
            ub = ub0.copy()
            for col in cols:
                q = 0.25 * (ub[col] - lb[col])
                lb[col] = lb[col] + q
                ub[col] = ub[col] - q
            boxes.append((lb, ub))
        return boxes

    def _patch(self, lb, ub):
        """Regenerate the node relaxation through the EP1-cached engine build.

        Byte-identical to the cold ``build_milp_relaxation`` by construction (it IS
        that call). Raises if the lifted column layout is not box-stable — a child
        box on which a conditional envelope (log-space band / composite lift) fires
        differently than the probe would change ``ncol`` and break the pool/objective
        column contract, so such a node declines the fast path (cold fallback).
        """
        A, b, bnds, c, _info, _relax = self._full_build(lb, ub)
        if A.shape[1] != self.ncol:
            raise ValueError("lifted column layout not box-stable")
        return A, b, bnds

    def _validate(self) -> None:
        # Prove the patch reproduces the cold build row-for-row on reachable,
        # sign-diverse sub-boxes AND that the lifted column layout is box-stable.
        # ``_patch`` regenerates through the same engine build, so equality holds by
        # construction whenever the layout is stable and the build is valid; any
        # box on which the build fails or the layout shifts raises -> ok=False ->
        # cold path unchanged. Column count / objective / bound stay byte-identical.
        regimes = set()
        n_dep = 0
        n_checked = 0
        for lb, ub in self._reachable_boxes():
            for i in range(self.n):
                regimes.add(self._box_sign_regime(float(lb[i]), float(ub[i])))
            Ap, bp, bdp = self._patch(lb, ub)
            Af, bf, bdf, _, _, _ = self._full_build(lb, ub)
            if Ap.shape != Af.shape:
                raise ValueError("shape mismatch")
            if not (np.array_equal(Ap, Af) and np.array_equal(bp, bf)):
                raise ValueError("row-set mismatch")
            if not np.array_equal(bdp, bdf):
                raise ValueError("bounds mismatch")
            n_checked += 1
            # Box-dependent row census (telemetry only): rows that differ from the
            # root probe build.
            if Ap.shape == self.base_A.shape:
                diff = np.any(Ap != self.base_A, axis=1) | (bp != self.base_b)
                n_dep = max(n_dep, int(np.count_nonzero(diff)))
        if n_checked == 0:
            raise ValueError("no reachable validation box")
        self.n_box_dependent_rows = n_dep
        self.n_const_rows = int(self.base_A.shape[0]) - n_dep
        self._validated_regimes = frozenset(regimes)
        self.ok = True
