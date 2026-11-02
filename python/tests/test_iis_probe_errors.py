"""A feasibility probe that RAISES is a bug, not an inconclusive solve.

``compute_iis`` used to catch every exception from a probe's ``Model.solve`` and
record the probe as inconclusive. That turned real defects into plausible
output: a tape-lowering bug (a shape-(1,) variable falling to the uninstalled
JAX evaluator, ``MissingJaxError``) surfaced as "could not prove the model
infeasible within the time limit" on a model HiGHS proves infeasible instantly,
and mid-filter it would silently keep every element it touched and report a
non-minimal set. Probes that *return* a non-proof status (time limit, error
status) stay inconclusive; an exception now propagates, with the model restored.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.infeasibility import compute_iis
from discopt.modeling.core import Model


def _model():
    m = dm.Model("c")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(x >= 5, name="a")
    m.subject_to(x <= 2, name="b")
    m.subject_to(x <= 9, name="loose")
    m.minimize(x)
    return m, x


@pytest.mark.smoke
@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_probe_exception_propagates_and_model_is_restored(monkeypatch, fail_on_call):
    m, x = _model()
    cons_before = list(m._constraints)
    obj_before = m._objective
    lb_before, ub_before = np.asarray(x.lb).copy(), np.asarray(x.ub).copy()

    real_solve = Model.solve
    calls = {"n": 0}

    def flaky_solve(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            raise RuntimeError("probe blew up")
        return real_solve(self, *a, **k)

    monkeypatch.setattr(Model, "solve", flaky_solve)
    with pytest.raises(RuntimeError, match="probe blew up"):
        compute_iis(m)

    assert calls["n"] == fail_on_call
    assert m._constraints == cons_before
    assert m._objective is obj_before
    np.testing.assert_array_equal(np.asarray(x.lb), lb_before)
    np.testing.assert_array_equal(np.asarray(x.ub), ub_before)


@pytest.mark.smoke
def test_returned_non_proof_status_is_still_inconclusive(monkeypatch):
    """A probe that RETURNS e.g. ``time_limit`` keeps the element (unchanged contract)."""
    m, _ = _model()
    real_solve = Model.solve
    calls = {"n": 0}

    class _R:
        status = "time_limit"

    def slow_solve(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:  # first deletion probe
            return _R()
        return real_solve(self, *a, **k)

    monkeypatch.setattr(Model, "solve", slow_solve)
    iis = compute_iis(m, include_bounds=False)
    assert not iis.proven_irreducible
    assert {"a", "b"} <= {c.name for c in iis.constraints}
