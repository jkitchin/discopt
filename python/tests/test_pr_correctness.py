"""PR-fast correctness subset (issue #68, Phase 5).

Five representative known-optima instances drawn from
:mod:`test_correctness`. These run on every PR (~30 s budget) so a
regression in the optimal-value check fails fast — without the full
136-instance correctness suite, which stays nightly via the
``correctness`` (and ``slow``) marker.

Each instance is chosen to cover a distinct solver code path:

* ``constrained_quadratic``    — pure continuous convex QP
* ``simple_minlp``             — convex MINLP (textbook)
* ``binary_knapsack``          — pure 0-1 MILP
* ``circle_minlp``             — nonconvex MINLP (``x^2 + y^2 >= 1``)
* ``exp_binary_minlp``         — OA-friendly convex MINLP with ``exp``

The file deliberately does **not** import or re-apply the
``slow`` mark used at module level in :mod:`test_correctness`; it
re-uses only the build functions and the optimal-value helper.
"""

from __future__ import annotations

import math

import pytest

from test_correctness import (
    ProblemInstance,
    _build_binary_knapsack,
    _build_circle_minlp,
    _build_constrained_quadratic,
    _build_exp_binary_minlp,
    _build_simple_minlp,
    assert_optimal_value,
)

pytestmark = pytest.mark.pr_correctness


PR_INSTANCES: list[ProblemInstance] = [
    ProblemInstance(
        name="constrained_quadratic",
        build_fn=_build_constrained_quadratic,
        expected_obj=0.5,
        integer_vars=[],
        bounds={"x": (-5, 5), "y": (-5, 5)},
        description="Pure continuous convex QP",
    ),
    ProblemInstance(
        name="simple_minlp",
        build_fn=_build_simple_minlp,
        expected_obj=0.5,
        integer_vars=["x3"],
        bounds={"x1": (0, 5), "x2": (0, 5), "x3": (0, 1)},
        description="Convex MINLP textbook",
    ),
    ProblemInstance(
        name="binary_knapsack",
        build_fn=_build_binary_knapsack,
        expected_obj=-6.0,
        integer_vars=["x1", "x2", "x3"],
        bounds={"x1": (0, 1), "x2": (0, 1), "x3": (0, 1)},
        description="Pure 0-1 MILP",
    ),
    ProblemInstance(
        name="circle_minlp",
        build_fn=_build_circle_minlp,
        expected_obj=1.0,
        integer_vars=["z"],
        bounds={"x": (0, 3), "y": (0, 3), "z": (0, 1)},
        description="Nonconvex MINLP",
    ),
    ProblemInstance(
        name="exp_binary_minlp",
        build_fn=_build_exp_binary_minlp,
        expected_obj=math.e,
        integer_vars=["y"],
        bounds={"x": (0, 3), "y": (0, 1)},
        description="OA-friendly convex MINLP",
    ),
]


@pytest.mark.parametrize(
    "instance",
    PR_INSTANCES,
    ids=[inst.name for inst in PR_INSTANCES],
)
def test_pr_optimal_value(instance: ProblemInstance) -> None:
    """Curated PR subset: solver finds the known optimum within tolerance."""
    model = instance.build_fn()
    result = model.solve(time_limit=30.0, gap_tolerance=1e-6, max_nodes=10_000)
    assert_optimal_value(result, instance.expected_obj, instance.name)
