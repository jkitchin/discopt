"""#1447 -- a claimed leaf must actually contain the returned point.

The per-leaf MILP encoding separated only the RIGHT branch from its threshold,
and by exactly 1e-6 -- equal to the solver's absolute feasibility tolerance, so
that protection was fully consumed. The LEFT branch (``x <= thr``) had none.

A MILP solver may satisfy a constraint to within its feasibility tolerance, and
a decision tree is **discontinuous** at its thresholds, so tolerance-sized slack
becomes an O(1) error in the prediction: the optimizer returned a point 3.1e-14
on the wrong side of a threshold while claiming the leaf on the near side, and
reported that leaf's value. Feeding its own ``x`` back through the ensemble gave
a different number.

The fix separates both branches. The invariant these tests pin is the one the
embedding exists to provide and that nothing asserted before:

    ensemble.predict(x_optimal) == reported objective
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import Model
from discopt.ml.formulations.tree_ensemble import TreeEnsembleFormulation
from discopt.ml.predictor import add_predictor
from discopt.ml.tree import DecisionTree, TreeEnsembleDefinition

#: The tree the defect was found on. Its optimum sits on a threshold, which is
#: where a piecewise-constant function's optima live, so the encoding is pushed
#: straight onto the boundary it was mishandling.
_THE_TREE = dict(
    n_features=2,
    feature=np.array([1, 1, 0, -1, -1, 0, -1, -1, 1, 1, -1, -1, 1, -1, -1]),
    threshold=np.array(
        [
            0.2703512755100843,
            -0.59118292326463,
            1.326232,
            0,
            0,
            1.1489502676272676,
            0,
            0,
            -0.602728,
            0.361965,
            0,
            0,
            1.388176,
            0,
            0,
        ]
    ),
    left_child=np.array([1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1]),
    right_child=np.array([8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1]),
    value=np.array(
        [
            0,
            0,
            0,
            0.964717,
            2.902632,
            0,
            1.881544,
            -3.100085,
            0,
            0,
            0.810365,
            0.999123,
            0,
            0.019461,
            2.710226,
        ]
    ),
)


def _the_ensemble():
    lo, hi = np.full(2, -2.0), np.full(2, 2.0)
    return TreeEnsembleDefinition(
        trees=[DecisionTree(**_THE_TREE)],
        n_features=2,
        base_score=0.25,
        input_bounds=(lo, hi),
    )


class TestTheOptimalPointProducesTheReportedValue:
    @pytest.mark.parametrize("sense", ["max", "min"])
    def test_predict_at_x_optimal_matches_the_objective(self, sense):
        ens = _the_ensemble()
        m = Model("t")
        x = m.continuous("x", shape=(2,), lb=-2.0, ub=2.0)
        out, _form = add_predictor(m, x, ens)
        (m.maximize if sense == "max" else m.minimize)(out[0])
        r = m.solve(time_limit=180.0, gap_tolerance=1e-8)

        assert r.status in ("optimal", "feasible"), r.status
        xs = np.asarray(r.x["x"], float).ravel()
        assert float(ens.predict(xs)) == pytest.approx(r.objective, abs=1e-6), (
            f"MIP reports {r.objective!r} at x={xs}, but the ensemble predicts "
            f"{float(ens.predict(xs))!r} there"
        )

    @pytest.mark.parametrize("sense", ["max", "min"])
    def test_the_optimum_is_still_reachable(self, sense):
        """Separating the branches must not cost an attainable leaf value."""
        ens = _the_ensemble()
        g = np.linspace(-2.0, 2.0, 401)
        grid = np.array(np.meshgrid(g, g)).T.reshape(-1, 2)
        vals = np.array([ens.predict(p) for p in grid])
        target = float(vals.max() if sense == "max" else vals.min())

        m = Model("t")
        x = m.continuous("x", shape=(2,), lb=-2.0, ub=2.0)
        out, _form = add_predictor(m, x, ens)
        (m.maximize if sense == "max" else m.minimize)(out[0])
        r = m.solve(time_limit=180.0, gap_tolerance=1e-8)

        if sense == "max":
            assert r.objective >= target - 1e-6, (r.objective, target)
        else:
            assert r.objective <= target + 1e-6, (r.objective, target)


def _one_node_ensemble():
    """One split at 0.5 on [0, 1]: left leaf 1.0, right leaf 3.0."""
    t = DecisionTree(
        n_features=1,
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, 1.0, 3.0]),
    )
    return TreeEnsembleDefinition(
        trees=[t],
        n_features=1,
        base_score=0.0,
        input_bounds=(np.array([0.0]), np.array([1.0])),
    )


def _pinned_value(ens, point, eps):
    m = Model("pin")
    form = TreeEnsembleFormulation(m, ens, prefix="p", split_eps=eps)
    inp, out = form.build()
    inp.lb = np.asarray(point, float).copy()
    inp.ub = np.asarray(point, float).copy()
    m.minimize(out[0])
    r = m.solve(time_limit=60.0)
    return r.objective


class TestBothSidesOfAThresholdAreAssignedCorrectly:
    @pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-4])
    @pytest.mark.parametrize("delta", [-1e-3, -1e-4, 1e-4, 1e-3])
    def test_a_point_off_the_threshold_gets_its_own_leaf(self, eps, delta):
        ens = _one_node_ensemble()
        p = np.array([0.5 + delta])
        assert _pinned_value(ens, p, eps) == pytest.approx(float(ens.predict(p)), abs=1e-9)

    @pytest.mark.parametrize("eps", [1e-6, 1e-5])
    def test_thr_plus_and_minus_eps_both_solve(self, eps):
        """The excluded set is the threshold POINT, not a 2*eps band."""
        ens = _one_node_ensemble()
        assert _pinned_value(ens, np.array([0.5 - eps]), eps) == pytest.approx(1.0)
        assert _pinned_value(ens, np.array([0.5 + eps]), eps) == pytest.approx(3.0)

    @pytest.mark.parametrize("eps", [1e-6, 1e-5])
    def test_exactly_on_the_threshold_is_refused_not_misassigned(self, eps):
        """The documented trade, pinned so a future change cannot flip it silently.

        A discontinuous function forces a choice at the breakpoint. Refusing it
        is loud; assigning it to whichever leaf scores better is silent and is
        what produced the O(1) discrepancy this issue is about.
        """
        ens = _one_node_ensemble()
        assert _pinned_value(ens, np.array([0.5]), eps) is None


class TestTheDefaultSeparationClearsTheTolerance:
    def test_default_split_eps_exceeds_the_feasibility_tolerance(self):
        """split_eps == tol leaves zero margin; that was half the defect."""
        import inspect

        sig = inspect.signature(TreeEnsembleFormulation.__init__)
        default = sig.parameters["split_eps"].default
        assert default > 1e-6, (
            f"split_eps default {default} does not exceed the 1e-6 absolute "
            f"feasibility tolerance, so a leaf's rows can be satisfied by "
            f"tolerance slack alone"
        )
