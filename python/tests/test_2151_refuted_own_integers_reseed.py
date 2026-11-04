"""#2151: a HiGHS ``kOptimal`` refuted by the LP over its OWN integers is re-solved.

On a 2-32-32-1 ``relu_bigm`` network, HiGHS reported ``kOptimal`` with objective and
``mip_dual_bound`` both ``-46.6583412166`` (``mip_gap`` 0). The #1509 refutation guard
re-optimised the continuous part over HiGHS's own integer assignment and found
``-46.6583423116``, an exactly feasible point (row residual 1.5e-16). So the tree bound
really was 1.09e-6 above a feasible point. The guard was right to refuse it, but the
result fell back to the NS root bound (-510.17) and was published ``feasible`` with gap
0.91, although the incumbent was optimal.

Measured causes that were ruled out: HiGHS's stop rule (``mip_rel_gap = 0``,
``mip_abs_gap = 1e-9`` return the same false bound) and ``mip_feasibility_tolerance``
at 1e-7 / 1e-8. The fix is one certified re-solve seeded with the refuting point
(:func:`lp_milp_highs._reseed_refuted`). It is adopted only if it certifies by every
route check and its bound lies below every verified point in hand.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ml import NNFormulation
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition
from discopt.solvers import lp_milp_highs
from discopt.solvers.lp_milp_highs import HighsOutcome, StdForm

DATA = Path(__file__).parent / "data" / "issue_2151" / "net.json"

#: The LP over HiGHS's own integers, measured on the issue's network.
TRUE_OBJ = -46.65834231157295


def _model() -> dm.Model:
    d = json.loads(DATA.read_text())
    layers = [(np.array(lay["W"]), np.array(lay["b"])) for lay in d["layers"]]
    net = NetworkDefinition(
        [DenseLayer(W, b, Activation.RELU) for W, b in layers[:-1]]
        + [DenseLayer(*layers[-1], Activation.LINEAR)],
        input_bounds=(np.array(d["lo"]), np.array(d["hi"])),
    )
    m = dm.Model("i2151")
    nn = NNFormulation(m, net, strategy="relu_bigm")
    nn.formulate()
    m.minimize(dm.sum(nn.outputs))
    return m


def test_issue_2151_network_certifies(monkeypatch):
    """The issue's reproducer: ``optimal`` with a bound at or below the true point.
    Before the fix: ``feasible``, bound -510.17, gap 0.9085."""
    seen: list[HighsOutcome] = []
    real = lp_milp_highs._reseed_refuted

    def spy(sf, out, kw, t0, n_struct):
        res = real(sf, out, kw, t0, n_struct)
        seen.append(res)
        return res

    monkeypatch.setattr(lp_milp_highs, "_reseed_refuted", spy)
    r = _model().solve(time_limit=120, threads=1, deterministic=True)
    assert r.status == "optimal", (r.status, r.gap)
    assert r.objective == pytest.approx(TRUE_OBJ, abs=1e-6)
    assert r.bound is not None and r.bound <= r.objective + 1e-12
    # No bound above the verified point the #1509 guard found.
    assert r.bound <= TRUE_OBJ + 1e-6 + 1e-9 * abs(TRUE_OBJ)
    assert r.gap is not None and r.gap <= 1e-4
    # This HiGHS build may no longer reproduce the refutation. If it does, the reseed
    # must be the thing that certified the result.
    if seen:
        assert seen[0].stats.get("milp/refuted_reseed_adopted") == 1.0
        assert seen[0].labels.get("milp/primary_refuted_by") == "own-integers"
        assert "milp/certificate" not in seen[0].labels


def _sf() -> StdForm:
    return StdForm.from_arrays(
        c=np.array([-1.0, -2.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )


def _refuted(sf: StdForm) -> HighsOutcome:
    out = HighsOutcome(
        "feasible", x=np.array([0.0, 10.0]), objective=-20.0, bound=-30.0, node_count=3
    )
    out.labels = {
        "milp/certificate": "declined",
        "milp/highs_certificate": "refuted",
        "milp/refuted_by": "own-integers",
    }
    return out


KW = {"time_limit": 30.0, "gap_tolerance": 1e-4, "abs_gap_tolerance": None, "max_nodes": 100}


@pytest.mark.parametrize(
    "fake, reason",
    [
        (HighsOutcome("feasible", x=np.array([0.0, 10.0]), objective=-20.0, bound=-30.0), "status"),
        # Certified, but the bound lies above the refuting point (-20) we hold.
        (
            HighsOutcome(
                "optimal", x=np.array([10.0, 0.0]), objective=-10.0, bound=-10.0, gap_certified=True
            ),
            "above a verified point",
        ),
    ],
)
def test_reseed_declines_keep_the_refuted_result(monkeypatch, fake, reason):
    sf = _sf()
    calls = []

    def fake_certified(sf_, kw):
        calls.append(kw)
        return fake

    monkeypatch.setattr(lp_milp_highs, "_certified_milp", fake_certified)
    out = _refuted(sf)
    res = lp_milp_highs._reseed_refuted(sf, out, dict(KW), time.monotonic(), 2)
    assert calls, "the re-solve never ran, so this asserts nothing"
    np.testing.assert_array_equal(calls[0]["initial_point"], [0.0, 10.0])
    assert res is out
    assert res.status == "feasible" and not res.gap_certified and res.bound == -30.0
    assert reason in res.labels["milp/refuted_reseed_declined"]


def test_reseed_not_run_for_a_different_integer_refutation(monkeypatch):
    """The #1509 class itself (refuted by a DIFFERENT integer assignment) keeps the NS
    root bound, unchanged. ``max x + 3y  s.t.  x + 2y + s = 21`` with HiGHS's answer
    replaced by ``(10, 0)`` "certified" at -10: the rounded root LP point ``(0, 10)``
    refutes it at -30, and the root bound (-31.5) leaves the gap open."""
    import highspy

    calls = []
    monkeypatch.setattr(lp_milp_highs, "_reseed_refuted", lambda *a, **k: calls.append(a) or a[1])
    real_solution = highspy.Highs.getSolution
    real_info = highspy.Highs.getInfo
    faked = []

    def is_mip(h) -> bool:
        return len(h.getLp().integrality_) > 0

    def false_solution(self):
        sol = real_solution(self)
        if is_mip(self):
            # The row form folds the slack into a row bound (#1733): 2 or 3 columns.
            sol.col_value = [10.0, 0.0, 11.0][: len(sol.col_value)]
            faked.append("x")
        return sol

    def false_info(self):
        info = real_info(self)
        if is_mip(self):
            info.objective_function_value = -10.0
            info.mip_dual_bound = -10.0
            faked.append("bound")
        return info

    monkeypatch.setattr(highspy.Highs, "getSolution", false_solution)
    monkeypatch.setattr(highspy.Highs, "getInfo", false_info)
    sf = StdForm.from_arrays(
        c=np.array([-1.0, -3.0, 0.0]),
        A=np.array([[1.0, 2.0, 1.0]]),
        b=np.array([21.0]),
        xl=np.array([0.0, 0.0, 0.0]),
        xu=np.array([1e20, 1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )
    out = lp_milp_highs.solve_milp_std(
        sf, time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000, n_struct=2
    )
    assert "x" in faked and "bound" in faked, "the false HiGHS answer was never injected"
    assert out.labels.get("milp/highs_certificate") == "refuted"
    assert out.labels.get("milp/refuted_by") in ("root-lp", "rounded-root")
    assert out.status == "feasible" and not out.gap_certified
    assert not calls
