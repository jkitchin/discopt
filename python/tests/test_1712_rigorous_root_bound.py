"""The root-relaxation fallback publishes only a certified bound.

Found while investigating #1712: under CPU load, ``bchoco08`` reached the #654
"time_limit spent before spatial search build" fallback, whose contract is a
*rigorous* root-relaxation bound. Its separated candidate came from a node LP with
no Neumaier-Shcherbina certificate, and ``_certify`` accepted the vertex on trust
(``trusted_vertex``). That failed ``test_1278_bound_provenance`` under load on main
too. ``solve_at_node(rigorous_only=True)`` declines both trust arms, and the
fallback asks for it.

The LP is stubbed to return no certificate (``safe_bound=None``), which is the
state the timing-dependent path produced, so the test does not depend on load.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from discopt import Model
from discopt._relax import milp_relaxation as mr
from discopt._relax.mccormick_lp import (
    BOUND_PROVENANCE,
    MccormickLPRelaxer,
    reset_bound_provenance,
)
from discopt._relax.model_utils import flat_variable_bounds


@pytest.fixture
def uncertified_lp(monkeypatch):
    """Every relaxation LP comes back optimal with no safe bound attached."""
    monkeypatch.setenv("DISCOPT_INCREMENTAL_MC", "0")  # the generic, per-solve path
    real = mr.MilpRelaxationModel.solve

    def no_certificate(self, *a, **k):
        return dataclasses.replace(real(self, *a, **k), safe_bound=None)

    monkeypatch.setattr(mr.MilpRelaxationModel, "solve", no_certificate)


def _bilinear() -> Model:
    m = Model("bil")
    x = m.continuous("x", lb=-1, ub=2)
    y = m.continuous("y", lb=-1, ub=3)
    m.minimize(x * y + x)
    m.subject_to(x + y >= 0.5)
    return m


@pytest.mark.smoke
def test_rigorous_only_declines_the_trusted_vertex(uncertified_lp):
    m = _bilinear()
    lb, ub = (b.astype(float) for b in flat_variable_bounds(m))

    reset_bound_provenance()
    default = MccormickLPRelaxer(m).solve_at_node(lb, ub)
    assert dict(BOUND_PROVENANCE) == {"trusted_vertex": 1}  # the probe reached the arm
    assert default.lower_bound is not None

    reset_bound_provenance()
    strict = MccormickLPRelaxer(m).solve_at_node(lb, ub, rigorous_only=True)
    assert dict(BOUND_PROVENANCE) == {"declined": 1}
    assert strict.lower_bound is None


@pytest.mark.smoke
def test_root_relaxation_fallback_asks_for_a_rigorous_bound(uncertified_lp, monkeypatch):
    from discopt.solver import _root_relaxation_lower_bound

    seen = []
    real = MccormickLPRelaxer.solve_at_node

    def spy(self, lb, ub, time_limit=None, **kw):
        seen.append(kw.get("rigorous_only", False))
        return real(self, lb, ub, time_limit, **kw)

    monkeypatch.setattr(MccormickLPRelaxer, "solve_at_node", spy)
    m = _bilinear()
    lb, ub = (b.astype(float) for b in flat_variable_bounds(m))
    reset_bound_provenance()
    bound = _root_relaxation_lower_bound(m, lb, ub, 30.0)
    assert seen == [True]
    assert not {"trusted_vertex", "trusted_backend"} & set(BOUND_PROVENANCE)
    # Whatever it reports is still a lower bound on the true minimum over the
    # feasible part of the box, found by dense sampling.
    xs, ys = np.meshgrid(np.linspace(-1, 2, 301), np.linspace(-1, 3, 401))
    feasible = xs + ys >= 0.5
    true_min = float(np.min((xs * ys + xs)[feasible]))
    if bound is not None:
        assert bound <= true_min + 1e-9
