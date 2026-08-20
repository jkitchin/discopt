"""#1061: a free column must never be certified optimal at its parked value.

The primal simplex has no ``FREE`` nonbasic status -- ``run()`` parks a free
column under ``AT_LOWER`` and ``nb_value`` special-cases it to sit at 0.  Before
the fix, pricing read that label literally ("may only increase"), so a free
column with ``dj > tol`` -- improving by *decreasing* -- was invisible and the
loop exited ``Optimal`` at the parked point.

The Rust unit tests in ``lp/simplex/primal.rs`` guard the pricing rule itself.
This guards the consequence on real corpus data, which is where it actually bit:
``bootstrap_finite_bounds`` finitizes an open bound by solving ``min x_i`` and
trusting ``SolveStatus.OPTIMAL``, so a false ``Optimal`` becomes a variable bound
the LP never proved, and from there a ``gap_certified=True`` root bound.

Probe instance: ``wallfix`` (MINLPLib, 6 vars, 0 discrete).  Its objective is
``x0`` and ``x0`` is free in both directions; nothing in the root relaxation
bounds it below, so ``min x0`` is genuinely UNBOUNDED.  Measured 2026-08-20:
without the fix that LP returned ``OPTIMAL objective=0.0`` -- exactly the parked
value -- which finitized ``x0 >= -1e-06`` and certified the root.  With the fix
it returns ``UNBOUNDED`` and the solver correctly declines to certify.

If a future relaxation legitimately bounds ``x0`` below on this instance, this
probe must be re-derived rather than relaxed: the invariant under test is "no
``Optimal`` at the parked value", not "wallfix is uncertifiable".
"""

import os

import numpy as np
import pytest
from discopt.modeling.core import from_nl

CORPUS = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
INSTANCE = os.path.join(CORPUS, "wallfix.nl")


@pytest.mark.skipif(not os.path.exists(INSTANCE), reason="MINLPLib corpus not present")
def test_free_column_bootstrap_lp_is_not_certified_optimal_at_its_parked_value():
    import discopt._relax.obbt as obbt
    from discopt.solvers import SolveStatus

    model = from_nl(INSTANCE)
    x0 = model._variables[0]
    lb0 = float(np.asarray(x0.lb).reshape(-1)[0])
    ub0 = float(np.asarray(x0.ub).reshape(-1)[0])
    assert not np.isfinite(lb0) and not np.isfinite(ub0), (
        f"probe precondition broken: {x0.name} is no longer free ({lb0}, {ub0})"
    )

    seen = []
    real_get = obbt.get_exact_lp_solver

    def instrumented(*args, **kwargs):
        inner = real_get(*args, **kwargs)
        if inner is None:
            return None

        def call(**kw):
            result = inner(**kw)
            costs = np.asarray(kw["c"], dtype=float).reshape(-1)
            nonzero = np.nonzero(costs)[0]
            # Only the probe that MINIMIZES the free column over a box that is
            # still open in both directions. Anything else is a different LP.
            if nonzero.size == 1 and nonzero[0] == 0 and costs[0] > 0:
                lo, hi = kw["bounds"][0]
                if not np.isfinite(lo) and not np.isfinite(hi):
                    seen.append((result.status, result.objective))
            return result

        return call

    obbt.get_exact_lp_solver = instrumented
    try:
        model.solve(time_limit=30)
    finally:
        obbt.get_exact_lp_solver = real_get

    # SS6: a silent zero-interception run would "pass" while measuring nothing.
    assert seen, "intercepted no `min x0` LP over the open box -- probe measured nothing"
    for status, objective in seen:
        assert status is not SolveStatus.OPTIMAL, (
            f"free column certified {status} objective={objective!r} on an LP that is "
            "unbounded below -- the #1061 defect is back"
        )
