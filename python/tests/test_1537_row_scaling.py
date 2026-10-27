"""#1537 witnesses under per-row scaling (row equilibration).

Multiplying a constraint row by a positive scalar changes no feasible point, so it
must change no certified answer. The per-row scaling probe (corpus + generated
families, each row scaled by its own seeded log-uniform factor in 10^[-3,3] and
10^[-6,6]) found two witnesses that this file pins:

* **A crossed single-variable pin** (fixed in ``uniform_relax``).
  ``_fix_single_var_equalities`` collapses the box of a variable fixed by
  ``a*x + c == rhs``. It admits a pin up to 1e-9 *outside* the box and then
  intersects, so a pin one ulp above ``ub`` produced ``lb = val > ub`` -- a crossed
  box, on which every ``Interval`` raises and the whole relaxation build (root
  OBBT, the native spatial kernel spec) failed with ``ValueError: Interval lo >
  hi``. A row-scaled copy computes the pin as ``(s*rhs - s*c)/(s*a)``, one ulp off
  the unscaled value, which is how 4stufen with rows scaled in 10^[-3,3] crashed
  ``Model.solve``.
* **A scale-dependent conditioning guard** (``milp_relaxation``; NOT fixed --
  xfail, see #1595). ``sanitize_relaxation_for_conditioning`` drops any relaxation
  row with an entry of magnitude >= 1e10 before the fallback root-bound solve. That
  is a statement about how the row was *written*: ex14_1_9 with its two rows
  multiplied by 8.14 and 3.66 pushes a 2.02e9 coefficient to 1.64e10, the row is
  dropped, the fallback bound falls to -1.07e6 and the certificate the unscaled
  model earns (bound 0.0) is lost. An exact power-of-two "rescue" of over-cap rows
  was tried and withdrawn in the PR #1594 review: it lands the row's largest entry
  in [5e9, 1e10), where the in-house simplex returns false LP bounds (#1595) on
  rows that main drops. A lost certificate is acceptable; a false bound is not.

  Since #1595 (PR #1597) the end-to-end witness certifies again, but *not* through
  the sanitizer: the in-house simplex now solves the scaled root LP soundly, so the
  fallback that drops the row is never reached (measured: zero sanitizer calls,
  bound -1.47e-7 against the ``minlplib.solu`` optimum 0). The sanitizer still drops
  over-cap rows when it does run; that is pinned by the unit tests below.
"""

from __future__ import annotations

import os
import zlib

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from _invariance import _rebuild
from discopt._relax.milp_relaxation import (
    MilpRelaxationModel,
    sanitize_relaxation_for_conditioning,
)
from discopt._relax.uniform_relax import _fix_single_var_equalities, build_uniform_relaxation
from discopt.modeling.core import Constraint


def _pinned_model(scale: float):
    m = dm.Model("pin")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(scale * (3.0 * x - 1.0) == 0.0)
    m.subject_to(x * y >= 0.1)
    m.minimize(y + x * x)
    return m


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_pin_one_ulp_outside_the_box_never_crosses_it(direction):
    """A pin within the 1e-9 admission tolerance but outside the box collapses to
    the hull of the two candidates instead of a crossed ``lb > ub`` box."""
    m = _pinned_model(1.0)
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    lb0, ub0 = _fix_single_var_equalities(m, lb, ub)
    val = lb0[0]
    assert lb0[0] == ub0[0], "the probe's pin did not fire"
    # Put the box edge one ulp on the far side of the pin.
    if direction > 0:
        ub[0] = np.nextafter(val, -np.inf)
        lb[0] = 0.0
    else:
        lb[0] = np.nextafter(val, np.inf)
        ub[0] = 10.0
    out_lb, out_ub = _fix_single_var_equalities(m, lb, ub)
    assert out_lb[0] <= out_ub[0], (out_lb[0], out_ub[0])
    # Hull of the box edge and the pin: both candidates are kept.
    assert out_lb[0] <= val <= out_ub[0]
    assert out_ub[0] - out_lb[0] <= 2.0 * np.spacing(val)


def test_relaxation_builds_when_the_pin_is_one_ulp_past_the_box():
    """Before the fix this raised ``ValueError: Interval lo > hi``."""
    m = _pinned_model(1.0)
    lb0, _ = _fix_single_var_equalities(m, np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    lb = np.array([0.0, 0.0])
    ub = np.array([np.nextafter(lb0[0], -np.inf), 10.0])
    rel = build_uniform_relaxation(m, box=(lb, ub))
    assert rel.model is not None


@pytest.mark.parametrize("scale", [8.14, 1e-3, 3.7e5])
def test_row_scaled_pin_solves_like_the_unscaled_model(scale):
    """Guard, not a before/after witness: this passes on the pre-fix sources too
    (the one-ulp crossing needs the exact 4stufen arithmetic pinned above). It keeps
    the end-to-end answer row-scale invariant for this small pinned model."""
    base = _pinned_model(1.0).solve(time_limit=20)
    scaled = _pinned_model(scale).solve(time_limit=20)
    assert base.status == "optimal" and scaled.status == "optimal"
    assert scaled.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-8)


# --------------------------------------------------------------------------- #
# The fallback-bound sanitizer: over-cap rows are dropped (sound), even though that
# makes the fallback bound depend on row scaling. See the module docstring.
# --------------------------------------------------------------------------- #


def _lp(rows, c, bounds):
    return MilpRelaxationModel(
        c=np.asarray(c, dtype=np.float64),
        A_ub=sp.csr_matrix(np.atleast_2d(np.asarray(rows, dtype=np.float64))),
        b_ub=np.zeros(np.atleast_2d(rows).shape[0]),
        bounds=bounds,
    )


@pytest.mark.parametrize("small", [-9e-10, -1e-6, -1.0])
def test_sanitizer_drops_over_cap_rows_the_simplex_mishandles(small):
    """PR #1594 review witnesses. ``1.5e10*x + small*y <= 0``, x in [0, 1], y >= 0,
    min -x: the true LP minimum is -1. The withdrawn rescue rescaled the row to
    ``[7.5e9, small/2]`` and kept it; on that row the in-house simplex returns the
    false bound 0.0 (#1595). Main and this branch drop it, and the bound is -1."""
    rel = _lp([[1.5e10, small]], [-1.0, 0.0], [(0.0, 1.0), (0.0, np.inf)])
    out = sanitize_relaxation_for_conditioning(rel)
    assert out._A_ub is None or out._A_ub.shape[0] == 0, "over-cap row kept"
    res = out.solve(backend="simplex")
    assert res.status == "optimal"
    assert res.objective <= -1.0 + 1e-9  # a valid lower bound for the min


def test_sanitizer_leaves_rows_under_the_cap_untouched():
    rows = np.array([[3.0, -1e9, 1e-31], [1.0, 2.0, 0.0]])
    rel = MilpRelaxationModel(
        c=np.zeros(3),
        A_ub=sp.csr_matrix(rows),
        b_ub=np.array([5.0, 7.0]),
        bounds=[(-1.0, 1.0)] * 3,
    )
    out = sanitize_relaxation_for_conditioning(rel)
    assert np.array_equal(np.asarray(out._A_ub.todense()), rows)
    assert np.array_equal(out._b_ub, [5.0, 7.0])


def test_row_scaled_ex14_1_9_keeps_its_certificate():
    """The end-to-end witness: ex14_1_9 with its two rows scaled by 8.14 and 3.66
    (the probe's seeded per-row factors). Before #1595 it reported ``feasible`` with
    bound -1.07e6 (the root LP failed, and the fallback root bound lost the scaled
    row); the unscaled model certifies 0.0. It must certify, with a bound that does
    not cross the known optimum 0 (``minlplib.solu``)."""
    path = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "ex14_1_9.nl")
    base_model = dm.from_nl(path)
    scaled = _rebuild(base_model, lambda v: np.zeros(v.lb.shape), 1.0, "ex14_rows")
    factors = [8.14, 3.66]
    assert len(scaled._constraints) == len(factors)
    scaled._constraints = [
        Constraint(body=s * c.body, sense=c.sense, rhs=0.0, name=c.name)
        for s, c in zip(factors, scaled._constraints)
    ]
    base = base_model.solve(time_limit=30)
    other = scaled.solve(time_limit=30)
    assert base.gap_certified
    assert other.gap_certified, (other.status, other.bound)
    assert other.objective == pytest.approx(base.objective, abs=1e-6)
    assert other.bound <= 0.0 + 1e-6, other.bound


# --------------------------------------------------------------------------- #
# The routed OA master under per-row scaling (#1537).
# --------------------------------------------------------------------------- #


def _per_row_scaled(name: str, span: float):
    """The probe's model: every row of ``name`` multiplied by its own seeded
    log-uniform factor in ``10^[-span, span]`` (seed = crc32 of the file name)."""
    path = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", name)
    base = dm.from_nl(path)
    new = _rebuild(base, lambda v: np.zeros(v.lb.shape), 1.0, f"{name}_pr{span:g}")
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    new._constraints = [
        Constraint(
            body=float(10.0 ** rng.uniform(-span, span)) * c.body,
            sense=c.sense,
            rhs=0.0,
            name=c.name,
        )
        for c in new._constraints
    ]
    return new


def test_row_scaled_m3_certifies_on_the_highs_master(monkeypatch):
    """m3 with rows scaled in 10^[-6,6]. An OA-master column for a variable in a row
    scaled by ~1e5 and one scaled by ~1e-6 holds a 1e-11 ratio. HiGHS (the routed
    master) fits each master row to its own coefficient window
    (``milp_highs._fit_rows_to_window``), so the route certifies, with no bound
    above the optimum. The retired in-house master lost this certificate
    (performance-plan §25.13)."""
    monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_OA_MASTER", raising=False)
    res = _per_row_scaled("m3.nl", 6.0).solve(time_limit=5)
    assert "master=highs" in (res.algorithm_route or ""), res.algorithm_route
    assert res.gap_certified, (res.status, res.bound, res.algorithm_route)
    # minlplib.solu: m3 =opt= 37.8 (minimize). Certified, never above it.
    assert res.bound <= 37.8 + 1e-6, res.bound
    assert res.objective == pytest.approx(37.8, rel=1e-4)


def test_row_scaled_flay03m_admits_no_infeasible_incumbent(monkeypatch):
    """flay03m with rows scaled in 10^[-6,6], on the UNequilibrated fixed NLP
    (``DISCOPT_OA_NLP_ROW_EQUILIBRATE=0``). OA's fixed-NLP subproblem returned
    iteration-limited points at objectives 39.10 and 27.16 (optimum 48.99) that
    violate a row scaled by ~5e-6 by ~3e-5 -- inside the old absolute ``1e-4``
    screen, about 6.5 in the row's own units. Admitted as the incumbent, the 27.16
    point ended OA with the master bound above it.

    Candidates are judged by the exit gate's verifier (``verify_point``, keyed on
    each row's scale). Equilibration (default-ON) makes every fixed NLP here
    converge, so the gate has nothing to refuse; the flag is switched off to keep
    feeding it the unconverged points it exists for. Anti-vacuity (CLAUDE.md §6):
    the trace must show the gate refusing candidates.

    This pins soundness, not a certificate: without equilibration the outcome
    depended on the platform (macOS certified after 7 refusals, Linux CI fell back
    uncertified), which is what the equilibration test below fixes."""
    monkeypatch.setenv("DISCOPT_OA_NLP_ROW_EQUILIBRATE", "0")
    res = _per_row_scaled("flay03m.nl", 6.0).solve(time_limit=30, solver="mip-nlp")
    summary = (res.mip_nlp_trace or {}).get("summary", {})
    why = (res.status, res.gap_certified, res.bound, res.objective, summary)
    assert summary.get("rejected_incumbent_count", 0) > 0, why
    # minlplib.solu: flay03m =opt= 48.98979486 (minimize). No super-optimal
    # incumbent and no bound above the optimum.
    assert res.objective is not None, why
    assert res.objective >= 48.98979486 - 1e-4, why
    assert res.bound is None or res.bound <= 48.98979486 + 1e-6, why


def _fixed_nlp_statuses(monkeypatch):
    """Record the status of every fixed-integer NLP (``scale_tol=True``) OA solves."""
    import discopt.solvers.oa as oa

    seen: list[str] = []
    real = oa._solve_nlp_attempt

    def spy(*args, **kwargs):
        attempt = real(*args, **kwargs)
        if kwargs.get("scale_tol"):
            seen.append(str(attempt.status).rsplit(".", 1)[-1])
        return attempt

    monkeypatch.setattr(oa, "_solve_nlp_attempt", spy)
    return seen


def test_row_equilibrated_fixed_nlp_converges_and_certifies_flay03m(monkeypatch):
    """The fixed-integer NLP is solved on row-equilibrated constraints
    (``_RowEquilibratedEvaluator``). Before: on flay03m at 10^[-6,6] 11 of 18 fixed
    NLPs on the default route stopped at ``ITERATION_LIMIT`` (133 of 139 on the
    explicit mip-nlp path, and 12 of 19 with ``max_iter=3000``), the exit gate
    refused 7 candidates, and the route certified on macOS but fell back uncertified
    on Linux CI. After: every fixed NLP converges and the route certifies, with a
    bound within 1e-6 relative of the optimum (48.98507 before).

    Pinned to the bare HiGHS master. Since #2013 the master also separates at its
    LP relaxation by default (``DISCOPT_HIGHS_MASTER_LP_CUTS``). That tightens the
    tree enough for the driver's early exit to fire at a claim inside the 1e-4
    certificate tolerance: certified, bound 48.98724 (gap 5.2e-5), confirmed by the
    presolve-free cross-solve. The 1e-6 pin below measures the bare master's tree
    running to its own end, so that is the path it reads."""
    monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_OA_MASTER", raising=False)
    monkeypatch.setenv("DISCOPT_HIGHS_MASTER_LP_CUTS", "0")
    monkeypatch.setenv("DISCOPT_OA_NLP_ROW_EQUILIBRATE", "1")
    seen = _fixed_nlp_statuses(monkeypatch)
    res = _per_row_scaled("flay03m.nl", 6.0).solve(time_limit=30)
    route = res.algorithm_route or ""
    why = (res.status, res.gap_certified, res.bound, res.objective, route, seen)
    assert seen, "no fixed-integer NLP was solved"  # anti-vacuity (CLAUDE.md §6)
    assert all(s == "OPTIMAL" for s in seen), why
    assert "fell back" not in route, why
    assert res.gap_certified, why
    # minlplib.solu: flay03m =opt= 48.98979486 (minimize). Never above it.
    assert res.bound <= 48.98979486 + 1e-6, why
    assert res.bound >= 48.98979486 * (1 - 1e-6), why
    assert res.objective == pytest.approx(48.98979486, rel=1e-6)


# --------------------------------------------------------------------------- #
# The row-equilibrated evaluator itself.
# --------------------------------------------------------------------------- #


class _ToyEvaluator:
    """g(x) = [1e-6 * x0**2 + 1e-3 * x1 - 1e-3, 1e5 * (x0 - x1)], body-rhs form."""

    n_variables = 2
    n_constraints = 2

    def __init__(self, bounds=((-1e20, 0.0), (0.0, 0.0))):
        self._cl = np.array([b[0] for b in bounds])
        self._cu = np.array([b[1] for b in bounds])

    def evaluate_constraints(self, x):
        return np.array([1e-6 * x[0] ** 2 + 1e-3 * x[1] - 1e-3, 1e5 * (x[0] - x[1])])

    def evaluate_jacobian(self, x):
        return np.array([[2e-6 * x[0], 1e-3], [1e5, -1e5]])

    def jacobian_structure(self):
        return np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])

    def evaluate_jacobian_values(self, x):
        return self.evaluate_jacobian(x).ravel()

    def evaluate_lagrangian_hessian(self, x, obj_factor, lagrange):
        return np.array([[2e-6 * lagrange[0], 0.0], [0.0, 0.0]])

    def evaluate_hessian_values(self, x, obj_factor, lagrange):
        return np.array([2e-6 * lagrange[0]])


def _bounds_of(ev):
    return ev._cl, ev._cu


def test_row_equilibration_is_exactly_a_row_scaling(monkeypatch):
    import discopt.solvers.nlp_ipopt as nlp_ipopt
    from discopt.solvers.oa import _RowEquilibratedEvaluator

    monkeypatch.setattr(nlp_ipopt, "_infer_constraint_bounds", _bounds_of)
    ev = _ToyEvaluator()
    x0 = np.array([2.0, 0.5])
    w = _RowEquilibratedEvaluator.wrap(ev, x0)
    assert w is not None
    # 1 / max_j |J_ij(x0)| is (1e3, 1e-5); only ever scaled up, so row 1 keeps 1.
    s = np.array([1e3, 1.0])
    x = np.array([0.3, -0.7])
    lam = np.array([2.0, -3.0])
    assert np.allclose(w.evaluate_constraints(x), s * ev.evaluate_constraints(x), rtol=1e-15)
    assert np.allclose(w.evaluate_jacobian(x), s[:, None] * ev.evaluate_jacobian(x))
    assert np.allclose(
        w.evaluate_jacobian_values(x), (s[:, None] * ev.evaluate_jacobian(x)).ravel()
    )
    # Lagrangian of the scaled rows with lam == Lagrangian of the originals with s*lam.
    assert np.allclose(w.evaluate_hessian_values(x, 1.0, lam), [2e-6 * s[0] * lam[0]])
    assert np.allclose(w.unscale_multipliers(lam), s * lam)
    # Everything else is forwarded untouched.
    assert w.n_variables == 2 and w.n_constraints == 2


def test_row_equilibration_refuses_a_nonzero_row_bound(monkeypatch):
    """Scaling a row without its bound would change the feasible set."""
    import discopt.solvers.nlp_ipopt as nlp_ipopt
    from discopt.solvers.oa import _RowEquilibratedEvaluator

    monkeypatch.setattr(nlp_ipopt, "_infer_constraint_bounds", _bounds_of)
    ev = _ToyEvaluator(bounds=((-1e20, 0.0), (0.0, 2.5)))
    assert _RowEquilibratedEvaluator.wrap(ev, np.array([2.0, 0.5])) is None
    # The 1e20 sentinel is an infinite bound, not a finite non-zero one.
    assert _RowEquilibratedEvaluator.wrap(_ToyEvaluator(), np.array([2.0, 0.5])) is not None


def test_row_equilibration_clamps_and_skips_degenerate_rows(monkeypatch):
    import discopt.solvers.nlp_ipopt as nlp_ipopt
    from discopt.solvers.oa import _NLP_ROW_SCALE_CLAMP, _RowEquilibratedEvaluator

    monkeypatch.setattr(nlp_ipopt, "_infer_constraint_bounds", _bounds_of)
    # At x0 = (0, 0) row 0's gradient is (0, 1e-3) and row 1's is 1e5: s = (1e3, 1).
    w = _RowEquilibratedEvaluator.wrap(_ToyEvaluator(), np.zeros(2))
    assert np.allclose(w._s, [1e3, 1.0])

    class Big(_ToyEvaluator):
        def evaluate_jacobian(self, x):
            return np.array([[1e3, 0.0], [0.0, 1e5]])

    # Every row already at or above unit norm: nothing to scale, no wrapper.
    assert _RowEquilibratedEvaluator.wrap(Big(), np.zeros(2)) is None

    class Tiny(_ToyEvaluator):
        def evaluate_jacobian(self, x):
            return np.array([[1e-12, 0.0], [0.0, 0.0]])

    w = _RowEquilibratedEvaluator.wrap(Tiny(), np.zeros(2))
    assert w._s[0] == _NLP_ROW_SCALE_CLAMP  # clamped, not 1e12
    assert w._s[1] == 1.0  # zero gradient: left alone


@pytest.mark.parametrize("span", [0.0, 3.0])
def test_tls2_route_certifies_over_nlp_unresolved_configurations(monkeypatch, span):
    """tls2 on the OA route: the master closes ``LB = UB = 5.3`` in well under a
    second, but most fixed NLPs end in ``error`` on infeasible assignments, and
    C-35 withheld the certificate whenever any configuration was unresolved. The
    route returned ``feasible`` and the spatial fallback spent ~25 s re-proving
    the instance unscaled, and ran out of budget with rows scaled in 10^[-3,3].

    The bound covers those configurations (they are cut only by valid
    linearizations; see ``_unresolved_configs_block_certificate``), so it now
    certifies. Anti-vacuity (CLAUDE.md §6): the run must still contain an
    unresolved configuration, or it no longer exercises this path."""
    monkeypatch.delenv("DISCOPT_OA_UNRESOLVED_BLOCKS_CERT", raising=False)
    res = _per_row_scaled("tls2.nl", span).solve(time_limit=30, solver="mip-nlp")
    trace = res.mip_nlp_trace or {}
    assert trace["summary"]["unresolved_integer_config_count"] > 0, trace["summary"]
    assert res.gap_certified, (res.status, res.bound, trace.get("termination_reason"))
    # minlplib.solu: tls2 =opt= 5.3 (minimize). Never above it.
    assert res.bound <= 5.3 + 1e-6, res.bound
    assert res.objective == pytest.approx(5.3, rel=1e-4)
