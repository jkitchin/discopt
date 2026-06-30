import numpy as np
import pytest
from discopt.solvers.mip_nlp_rootsearch import (
    MIPNLPInteriorPointStore,
    MIPNLPRootSearchStatus,
    rootsearch_between_points,
    rootsearch_from_store,
)


class _CircleEvaluator:
    n_variables = 2
    n_constraints = 1

    def evaluate_constraints(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] ** 2 + x[1] ** 2 - 1.0], dtype=float)


def test_mip_nlp_rootsearch_bisection_converges_on_known_circle_crossing():
    evaluator = _CircleEvaluator()
    store = MIPNLPInteriorPointStore(2)
    record = store.add(
        [0.0, 0.0],
        source="initial_poa",
        metadata={"node_count": 3},
        evaluator=evaluator,
        constraint_senses=["<="],
        require_feasible=True,
    )

    result = rootsearch_from_store(
        evaluator,
        [2.0, 0.0],
        store,
        constraint_senses=["<="],
        residual_tol=1e-10,
        x_tol=1e-10,
    )

    assert record is not None
    assert record.source == "initial_poa"
    assert record.metadata["node_count"] == 3
    assert result.status is MIPNLPRootSearchStatus.CONVERGED
    assert result.interior_source == "initial_poa"
    assert result.t == pytest.approx(0.5, abs=1e-6)
    assert result.point == pytest.approx(np.array([1.0, 0.0]), abs=1e-6)
    assert abs(result.residual) <= 1e-6


def test_mip_nlp_rootsearch_reports_infeasible_interior_endpoint():
    result = rootsearch_between_points(
        _CircleEvaluator(),
        [1.2, 0.0],
        [2.0, 0.0],
        constraint_senses=["<="],
    )

    assert result.status is MIPNLPRootSearchStatus.INTERIOR_INFEASIBLE
    assert result.t == 0.0
    assert result.residual > 0.0


def test_mip_nlp_rootsearch_fixed_discrete_keeps_candidate_value():
    evaluator = _CircleEvaluator()
    store = MIPNLPInteriorPointStore(
        2,
        int_indices=[1],
        lb=[0.0, 0.0],
        ub=[3.0, 1.0],
    )
    store.add(
        [0.0, 1.0],
        source="incumbent",
        evaluator=evaluator,
        constraint_senses=["<="],
        require_feasible=True,
    )

    result = rootsearch_from_store(
        evaluator,
        [2.0, 1.0],
        store,
        fixed_discrete=True,
        constraint_senses=["<="],
    )

    assert result.status is MIPNLPRootSearchStatus.CONVERGED
    assert result.point[1] == pytest.approx(1.0)


def test_mip_nlp_rootsearch_rejects_incompatible_fixed_discrete_reference():
    store = MIPNLPInteriorPointStore(
        2,
        int_indices=[1],
        lb=[0.0, 0.0],
        ub=[3.0, 1.0],
    )
    store.add([0.0, 0.0], source="incumbent")

    result = rootsearch_from_store(
        _CircleEvaluator(),
        [2.0, 1.0],
        store,
        fixed_discrete=True,
        constraint_senses=["<="],
    )

    assert result.status is MIPNLPRootSearchStatus.INCOMPATIBLE_DISCRETE
    assert result.point is None


def test_mip_nlp_rootsearch_missing_interior_point_fallback():
    result = rootsearch_from_store(
        _CircleEvaluator(),
        [2.0, 0.0],
        MIPNLPInteriorPointStore(2),
        constraint_senses=["<="],
    )

    assert result.status is MIPNLPRootSearchStatus.MISSING_INTERIOR_POINT
    assert result.point is None


def test_mip_nlp_rootsearch_time_limit_status():
    result = rootsearch_between_points(
        _CircleEvaluator(),
        [0.0, 0.0],
        [2.0, 0.0],
        constraint_senses=["<="],
        time_limit=0.0,
    )

    assert result.status is MIPNLPRootSearchStatus.TIME_LIMIT
