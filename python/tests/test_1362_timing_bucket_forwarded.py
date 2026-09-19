"""Issue #1362: two in-tree evaluators did not declare a `timing_bucket`.

`_IpoptCallbacks.__init__` warns when an evaluator declares no `timing_bucket`,
because its derivative-callback time is then left with the enclosing solver
region and the layer profile over-reports that layer. The comment there said the
warning "fires only for a duck-typed evaluator from outside the package"; that
was not true. Three notebooks carried the warning in their committed output:

    Evaluator FeasibilityPhaseEvaluator declares no `timing_bucket`; ...
    Evaluator _BoundsProxy declares no `timing_bucket`; ...

`benders._feasibility.FeasibilityPhaseEvaluator` and
`solvers.gdpopt_loa._BoundsProxy` both implement the backend callback surface
*explicitly* rather than delegating through `__getattr__`, so the attribute did
not reach the adapter. (`solvers.oa._BoundsProxy` forwards everything via
`__getattr__` and was already fine -- which is why this went unnoticed.)

This test enumerates every wrapper in the package that is handed to a backend and
asserts the attribute reaches it, so the next such wrapper cannot be written
without one.
"""

from __future__ import annotations

import numpy as np
import pytest

_SENTINEL = "rust"


class _Inner:
    """Minimal stand-in for an evaluator that declares its layer."""

    timing_bucket = _SENTINEL
    n_variables = 2
    n_constraints = 1

    @property
    def variable_bounds(self):
        return np.zeros(2), np.ones(2)

    @property
    def _model(self):
        return None

    @property
    def _obj_fn(self):
        return None

    def objective(self, x):
        return 0.0

    def gradient(self, x):
        return np.zeros(2)

    def constraints(self, x):
        return np.zeros(1)

    def jacobian(self, x):
        return np.zeros((1, 2))


def test_gdpopt_bounds_proxy_forwards_the_bucket():
    from discopt.solvers.gdpopt_loa import _BoundsProxy

    proxy = _BoundsProxy(_Inner(), np.zeros(2), np.ones(2))
    assert proxy.timing_bucket == _SENTINEL


def test_oa_bounds_proxy_forwards_the_bucket():
    """Already correct via ``__getattr__``; pinned so a refactor cannot lose it."""
    from discopt.solvers.oa import _BoundsProxy

    proxy = _BoundsProxy(_Inner(), np.zeros(2), np.ones(2))
    assert proxy.timing_bucket == _SENTINEL


def test_benders_feasibility_evaluator_forwards_the_bucket():
    from discopt.decomposition.benders._feasibility import FeasibilityPhaseEvaluator

    evaluator = FeasibilityPhaseEvaluator(
        _Inner(),
        cl=np.array([-np.inf]),
        cu=np.array([1.0]),
        lb=np.zeros(2),
        ub=np.ones(2),
    )
    assert evaluator.timing_bucket == _SENTINEL


def test_the_adapter_does_not_warn_for_an_evaluator_that_declares_one(caplog):
    """The other half: a declared bucket must produce no warning at all."""
    import logging

    pytest.importorskip("cyipopt")
    from discopt.solvers import nlp_ipopt

    with caplog.at_level(logging.WARNING, logger=nlp_ipopt.__name__):
        nlp_ipopt._IpoptCallbacks(_Inner())
    assert not [r for r in caplog.records if "timing-bucket-unknown" in r.getMessage()], [
        r.getMessage() for r in caplog.records
    ]


def test_the_adapter_still_warns_for_an_evaluator_that_declares_none(caplog):
    """The warning must keep working -- this is the probe-fired check."""
    import logging

    pytest.importorskip("cyipopt")
    from discopt.solvers import nlp_ipopt

    class _Undeclared(_Inner):
        timing_bucket = None

    with caplog.at_level(logging.WARNING, logger=nlp_ipopt.__name__):
        nlp_ipopt._IpoptCallbacks(_Undeclared())
    assert [r for r in caplog.records if "timing-bucket-unknown" in r.getMessage()]
