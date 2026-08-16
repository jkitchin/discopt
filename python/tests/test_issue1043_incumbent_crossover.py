"""#1043(b): the reported incumbent must be stationary, not a barrier-interior point.

The terminal incumbent polish in ``solve_model`` re-solves the continuous part of
the incumbent with a KKT-accurate NLP solve.  Two defects made that polish return
a point the examiner's ``stationarity`` check rejects, on a solve that is
otherwise fully converged and certified ``optimal``:

* **The polish ran in the reduced box.**  ``lb``/``ub`` at that point carry
  FBBT/OBBT/root reductions.  A reduction can place a bound exactly ON the
  optimum, and an interior-point polish then stops a barrier distance inside it.
  Measured on ``nlp_008_010``: stationarity 1.331e-06 (fail, tol 1e-6) in the
  reduced box, 7.84e-10 in the declared box.
* **A weakly active bound is approached slowly.**  An IPM leaves the iterate
  ``d ~ mu / lambda`` inside an active bound; where the objective gradient
  vanishes on the bound, ``lambda -> 0`` and ``d`` is large while the
  complementarity ``lambda * d`` the solver tests stays tiny.  The periodicity
  reduction (``_relax/nonlinear_bound_tightening.py``) maps a doubly-infinite
  periodic-only variable to exactly ``[-pi, pi]``, and a periodic function
  attains its extrema at the period boundary -- so it *systematically* creates
  this case.  Measured on ``nlp_001_010`` (``min ... + cos(y)``): the polish
  stopped at ``y = -3.14138899``, ``lambda = 2.037e-04``, complementarity
  4.15e-08 (converged), stationarity 2.037e-04.  POUNCE is right about the box
  it was given; the point is simply not the vertex.

The repairs are (i) polish over the declared box for free columns, and (ii) a
crossover that puts the point on a weakly active bound, guarded by feasibility,
objective-not-worse, and the dual bound.

The unit tests below pin each of the three conditions of the crossover rule with
the measured numbers.  All three are load-bearing: (1) the IPM's complementarity
certificate is nearly vacuous at convergence (an inactive finite bound carries
``lambda ~ mu / d``), (2) rejects the gross moves (1) nominates, and (3) keeps a
variable absent from the objective -- which passes (1) and (2) at every finite
bound -- from being relocated for no reason.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _weakly_active_crossover
from discopt.validation.examiner import assert_examined, examine

# ── measured inputs, from the polish subsolves of the two instances ──────────

# nlp_001_010: min x*exp(x) + cos(y) + z^3 - z^2, z >= 1.  ``y`` is reduced to
# [-pi, pi] by the periodicity pass, and cos attains its minimum at -pi.
NLP001_X = np.array([-1.0, -3.1413889875324084, 1.0000000090909087])
NLP001_LB = np.array([-9.999e19, -np.pi, 1.0])
NLP001_UB = np.array([9.999e19, np.pi, 9.999e19])
NLP001_MULT_L = np.array([0.0, 2.03667508e-04, 1.00000004])
NLP001_MULT_U = np.array([0.0, 1.44691002e-09, 0.0])
NLP001_GRAD = np.array([0.0, 2.0366605597687062e-04, 1.0000000363636348])

# nlp_008_010: z sits 0.54 from its lower bound with a tiny multiplier, so its
# complementarity passes condition (1).  Snapping it would be a wholesale
# relocation; condition (2) must reject it.
NLP008_X = np.array([-0.59315858, 0.24404795, 0.54062715])
NLP008_LB = np.array([-9.999e19, -9.999e19, 0.0])
NLP008_UB = np.array([9.999e19, 9.999e19, 1.0])
NLP008_MULT_L = np.array([0.0, 0.0, 4.532566e-09])
NLP008_MULT_U = np.array([0.0, 0.0, 5.638838e-09])
NLP008_GRAD = np.array([1.0, 0.4880959, 0.8768331459513673])


class TestCrossoverRule:
    """The three-condition rule, on the measurements that motivated it."""

    def test_selects_the_weakly_active_periodic_bound(self):
        cand = _weakly_active_crossover(
            NLP001_X, NLP001_LB, NLP001_UB, NLP001_MULT_L, NLP001_MULT_U, NLP001_GRAD
        )
        assert cand == {1: pytest.approx(-np.pi)}, (
            f"the weakly active y >= -pi bound must be nominated; got {cand}"
        )

    def test_snapping_recovers_stationarity(self):
        """The whole point: on the bound, the residual is machine zero."""
        cand = _weakly_active_crossover(
            NLP001_X, NLP001_LB, NLP001_UB, NLP001_MULT_L, NLP001_MULT_U, NLP001_GRAD
        )
        assert cand
        y = cand[1]
        # d(cos y)/dy = -sin(y); 2.037e-04 at the returned point, ~1e-16 at -pi.
        assert abs(-np.sin(NLP001_X[1])) > 1e-6
        assert abs(-np.sin(y)) < 1e-12

    def test_rejects_a_distant_bound_that_passes_complementarity_alone(self):
        """Condition (1) is not sufficient -- condition (2) is load-bearing."""
        d = NLP008_X[2] - NLP008_LB[2]
        assert NLP008_MULT_L[2] * d < 1e-6, "premise: complementarity does pass"
        assert abs(NLP008_GRAD[2]) * d > 1e-2, "premise: the objective effect is gross"
        cand = _weakly_active_crossover(
            NLP008_X, NLP008_LB, NLP008_UB, NLP008_MULT_L, NLP008_MULT_U, NLP008_GRAD
        )
        assert cand == {}, f"no coordinate should be nominated here; got {cand}"

    def test_rejects_the_far_side_of_the_same_variable(self):
        """``y <= pi`` passes condition (1) too -- only (2) rules it out."""
        d_up = NLP001_UB[1] - NLP001_X[1]
        assert NLP001_MULT_U[1] * d_up < 1e-6, "premise: complementarity does pass"
        assert abs(NLP001_GRAD[1]) * d_up > 1e-4, "premise: the objective effect is not negligible"
        cand = _weakly_active_crossover(
            NLP001_X, NLP001_LB, NLP001_UB, NLP001_MULT_L, NLP001_MULT_U, NLP001_GRAD
        )
        assert 1 in cand and cand[1] < 0.0, f"the upper bound must not win; got {cand}"

    def test_rejects_a_strongly_active_bound(self):
        """Condition (2) is not sufficient either -- condition (1) is load-bearing.

        A coordinate the barrier is still pushing away from a strongly priced
        bound is an unconverged iterate, not a weakly-active vertex; relocating
        it is not the solver's business.
        """
        cand = _weakly_active_crossover(
            np.array([0.5]),
            np.array([0.0]),
            np.array([1e20]),
            np.array([10.0]),  # lam * d = 5.0, condition (1) fails
            np.array([0.0]),
            np.array([2e-6]),  # |g| * d = 1e-6, conditions (2) and (3) pass
        )
        assert cand == {}

    def test_rejects_a_variable_absent_from_the_objective(self):
        """Condition (3) is load-bearing.

        A variable with no objective gradient satisfies (1) and (2) at EVERY
        finite bound, however distant -- at convergence an inactive bound carries
        ``lambda ~ mu / d``, so ``lambda * d ~ mu`` is always tiny. Without (3)
        such a variable is relocated to a bound for no reason at all.
        """
        x = np.array([0.5])
        lam = np.array([2e-9])  # mu / d with mu ~ 1e-9: an ordinary inactive bound
        assert float(lam[0]) * 0.5 < 1e-6, "premise: complementarity passes at both bounds"
        cand = _weakly_active_crossover(
            x, np.array([0.0]), np.array([1.0]), lam, lam, np.array([0.0])
        )
        assert cand == {}, f"a variable not in the objective must not be moved; got {cand}"

    def test_infinite_bounds_and_missing_multipliers_are_inert(self):
        free = np.array([0.0])
        g = np.array([1.0])
        lam = np.array([1e-9])
        assert (
            _weakly_active_crossover(free, np.array([-1e20]), np.array([1e20]), lam, lam, g) == {}
        )
        assert _weakly_active_crossover(free, np.array([-1.0]), np.array([1.0]), None, lam, g) == {}
        assert (
            _weakly_active_crossover(free, np.array([-1.0]), np.array([1.0]), lam, lam, None) == {}
        )

    def test_a_point_already_on_its_bound_is_not_moved(self):
        lam = np.array([1e-9])
        cand = _weakly_active_crossover(
            np.array([1.0]),
            np.array([1.0]),
            np.array([2.0]),
            lam,
            lam,
            np.array([1.0]),  # a real gradient, so (3) passes
        )
        # The lower bound has d = 0 (nothing to do); the upper bound is 1.0 away
        # with |g| * d = 1.0, which (2) rejects.
        assert cand == {}

    def test_mismatched_shapes_are_inert(self):
        assert (
            _weakly_active_crossover(
                np.zeros(2), np.zeros(2), np.ones(2), np.zeros(1), np.zeros(2), np.ones(2)
            )
            == {}
        )


class TestReportedIncumbentIsStationary:
    """End-to-end: both instances failed the examiner before this fix."""

    def _solve(self, model):
        return model.solve(time_limit=120.0, gap_tolerance=1e-6)

    def test_periodic_reduction_instance(self):
        """nlp_001_010 -- the weakly active bound the periodicity pass creates."""
        m = dm.Model("issue1043b_periodic")
        x = m.continuous("x")
        y = m.continuous("y")
        z = m.continuous("z", lb=1.0)
        m.minimize(x * dm.exp(x) + dm.cos(y) + z**3 - z**2)
        result = self._solve(m)
        assert result.status in ("optimal", "feasible")
        assert_examined(result, m, "issue1043b_periodic")
        # The reported point is the vertex, not a barrier-interior neighbour.
        y_val = float(np.ravel(result.x["y"])[0])
        assert abs(abs(y_val) - np.pi) < 1e-9, f"y = {y_val!r} is not at the cos minimum"

    def test_reduced_box_instance(self):
        """nlp_008_010 -- the polish must run in the declared box."""
        m = dm.Model("issue1043b_reduced_box")
        x = m.continuous("x")
        y = m.continuous("y")
        z = m.continuous("z", lb=0.0, ub=1.0)
        m.minimize(x + y**2 + z**3)
        m.subject_to(y >= dm.exp(-x - 2) + dm.exp(-z - 2) - 2)
        m.subject_to(x**2 <= y**2 + z**2)
        m.subject_to(y >= x / 2 + z)
        result = self._solve(m)
        assert result.status in ("optimal", "feasible")
        report = examine(result, m)
        stat = next((c for c in report.checks if c.name == "stationarity"), None)
        assert stat is not None, "the stationarity check must have run"
        assert stat.passed, f"stationarity {stat.max_violation:.3e} > {stat.tolerance:.3e}"
        assert result.objective == pytest.approx(-0.3755859312158738, abs=1e-6)

    def test_certificate_invariant_holds(self):
        """The polish must never report a point below the rigorous dual bound."""
        m = dm.Model("issue1043b_cert")
        x = m.continuous("x")
        y = m.continuous("y")
        z = m.continuous("z", lb=1.0)
        m.minimize(x * dm.exp(x) + dm.cos(y) + z**3 - z**2)
        result = self._solve(m)
        assert result.bound is not None
        assert result.objective is not None
        assert result.bound <= result.objective + 1e-9 * (1.0 + abs(result.objective)), (
            f"bound {result.bound!r} above incumbent {result.objective!r}"
        )
