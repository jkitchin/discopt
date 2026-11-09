"""#2151: a HiGHS ``kOptimal`` refuted by the LP over its OWN integers is re-solved
at a tight feasibility tolerance (``DISCOPT_MILP_REFUTED_TIGHT_RESOLVE``, default ON;
``=0`` opts out). A verified point better than HiGHS's incumbent by less than the
refutation yardstick is published, with the bound held at or below it.

On a 2-32-32-1 ``relu_bigm`` network, HiGHS reported ``kOptimal`` with objective and
``mip_dual_bound`` both ``-46.6583412166`` (``mip_gap`` 0). The #1509 refutation guard
re-optimised the continuous part over HiGHS's own integer assignment and found
``-46.6583423116``, an exactly feasible point (row residual 1.5e-16). So the tree bound
really was 1.09e-6 above a feasible point. The guard was right to refuse it, but the
result fell back to the NS root bound (-510.17) and was published ``feasible`` with gap
0.91, although the incumbent was optimal.

Measured: ``mip_rel_gap = 0`` / ``mip_abs_gap = 1e-9`` return the same false bound, as
does ``mip_feasibility_tolerance`` at 1e-7 and 1e-8. At 1e-9 the certified bound sits
1.4e-14 below the point. :func:`lp_milp_highs._tight_resolve_refuted` runs that tight
solve cold and adopts it only under a 1e-9 yardstick. The 1e-6 refutation yardstick is
as wide as the defect, so it is not used for adoption.
"""

from __future__ import annotations

import json
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ml import NNFormulation
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition
from discopt.solvers import lp_milp_highs
from discopt.solvers.lp_milp_highs import HighsOutcome, StdForm


@pytest.fixture(autouse=True)
def _highs_milp_route(monkeypatch):
    """These tests pin the HiGHS MILP route's own mechanics. Since 2026-11-09 the
    pure-MILP default is the Rust engine on every install (PR #2196), so they name
    HiGHS explicitly, the optional ``discopt[highs]`` extra."""
    pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")


DATA = Path(__file__).parent / "data" / "issue_2151" / "net.json"
FLAG = "DISCOPT_MILP_REFUTED_TIGHT_RESOLVE"

#: The LP over HiGHS's own integers, measured on the issue's network.
TRUE_OBJ = -46.65834231157295


def _model(perm: str | None = None) -> dm.Model:
    d = json.loads(DATA.read_text())
    layers = [(np.array(lay["W"]), np.array(lay["b"])) for lay in d["layers"]]
    if perm is not None:
        pq = json.loads((DATA.parent / "perms.json").read_text())[perm]
        p, q = np.array(pq["p"]), np.array(pq["q"])
        (W1, b1), (W2, b2), (W3, b3) = layers
        layers = [(W1[:, p], b1[p]), (W2[p][:, q], b2[q]), (W3[q], b3)]
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


def _spy(monkeypatch) -> list[HighsOutcome]:
    seen: list[HighsOutcome] = []
    real = lp_milp_highs._tight_resolve_refuted

    def spy(sf, out, kw):
        assert out.refuted_by == "own-integers"
        res = real(sf, out, kw)
        seen.append(res)
        return res

    monkeypatch.setattr(lp_milp_highs, "_tight_resolve_refuted", spy)
    return seen


def test_issue_2151_network_certifies_by_default(monkeypatch):
    """The issue's reproducer: ``optimal``, bound at or below the true point.
    Without the fix: ``feasible``, bound -510.17, gap 0.9085."""
    monkeypatch.delenv(FLAG, raising=False)
    seen = _spy(monkeypatch)
    r = _model().solve(time_limit=120, threads=1, deterministic=True)
    # The probe fired: the refutation reproduced and the tight re-solve ran.
    assert seen, "the tight re-solve never ran, so this asserts nothing"
    assert seen[0].stats.get("milp/refuted_tight_resolve_adopted") == 1.0
    assert r.status == "optimal", (r.status, r.gap)
    assert r.objective == pytest.approx(TRUE_OBJ, abs=1e-9)
    assert r.bound is not None and r.bound <= r.objective + 1e-12
    # The adopted bound is held to the tight yardstick, not to CERT_ABS = 1e-6.
    assert r.bound <= TRUE_OBJ + lp_milp_highs.REFUTED_ADOPT_ABS
    assert r.gap is not None and r.gap <= 1e-4


def test_issue_2151_network_opt_out_restores_the_old_route(monkeypatch):
    """``=0`` (the CLAUDE.md §5 opt-out): the refuted result is published as before."""
    monkeypatch.setenv(FLAG, "0")
    seen = _spy(monkeypatch)
    r = _model().solve(time_limit=120, threads=1, deterministic=True)
    assert not seen
    assert r.status == "feasible"
    assert r.objective == pytest.approx(TRUE_OBJ, abs=1e-9)
    assert r.bound is not None and r.bound <= TRUE_OBJ


@pytest.mark.parametrize("perm", ["r1", "r3"])
def test_permuted_network_bound_never_above_the_objective(perm):
    """Neuron permutations of the issue's net (the same function). Before #2151, HiGHS
    certified r1 at -46.65834203738 and r3 at -46.65834223610, while the LP over its
    own integers gave -46.6583423116. That is better by 2.7e-7 (7.5e-8), inside the
    refutation yardstick, so the point was dropped. The #1537 E repair then published
    it under the old bound: on r1 the bound sat 2.7e-7 ABOVE the objective."""
    r = _model(perm).solve(time_limit=120, threads=1, deterministic=True)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(TRUE_OBJ, abs=1e-9)
    assert r.bound is not None and r.bound <= r.objective
    assert r.bound <= TRUE_OBJ + 1e-9


def _sf() -> StdForm:
    return StdForm.from_arrays(
        c=np.array([-1.0, -2.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )


def _refuted() -> HighsOutcome:
    """A refuted primary publishing the verified point (0, 10) at -20."""
    out = HighsOutcome(
        "feasible",
        x=np.array([0.0, 10.0]),
        objective=-20.0,
        bound=-30.0,
        node_count=3,
        iterations=7,
        wall_time=0.5,
        refuted_by="own-integers",
    )
    out.stats = {"milp/certificate_refuted": 1.0, "milp/refutation_margin": 1e-6, "lp/iters": 7.0}
    out.labels = {
        "milp/certificate": "declined",
        "milp/highs_certificate": "refuted",
        "milp/refuted_by": "own-integers",
        "milp/bound_provenance": "root-ns",
    }
    return out


KW = {"time_limit": 30.0, "gap_tolerance": 1e-4, "abs_gap_tolerance": None, "max_nodes": 100}


def _fake_certified(monkeypatch, fake: HighsOutcome) -> list[dict]:
    calls: list[dict] = []

    def fake_certified(sf_, kw):
        calls.append(kw)
        return fake

    monkeypatch.setattr(lp_milp_highs, "_certified_milp", fake_certified)
    return calls


def _check_resolve_kw(kw: dict) -> None:
    # Cold, tight, and capped like the #1634 cross-solve.
    assert kw["initial_point"] is None
    assert kw["feasibility_tolerance"] == lp_milp_highs.REFUTED_TIGHT_FEAS_TOL
    cap = max(
        lp_milp_highs.PRESOLVE_CROSS_MIN_BUDGET, lp_milp_highs.PRESOLVE_CROSS_BUDGET_FACTOR * 0.5
    )
    assert kw["time_limit"] == pytest.approx(min(cap, 30.0 - 0.5))


@pytest.mark.parametrize(
    "fake, reason",
    [
        (
            HighsOutcome("feasible", x=np.array([0.0, 10.0]), objective=-20.0, bound=-30.0),
            "status",
        ),
        # Certified, but its bound (-10) is far above the refuting point (-20).
        (
            HighsOutcome(
                "optimal",
                x=np.array([10.0, 0.0]),
                objective=-10.0,
                bound=-10.0,
                gap_certified=True,
            ),
            "above a verified point",
        ),
        # The review's case: 5e-7 above the refuting point. The 1e-6 yardstick would let
        # this through; the tight one does not.
        (
            HighsOutcome(
                "optimal",
                x=np.array([0.0, 10.0]),
                objective=-20.0,
                bound=-20.0 + 5e-7,
                gap_certified=True,
            ),
            "above a verified point",
        ),
    ],
)
def test_tight_resolve_declines_keep_the_refuted_result(monkeypatch, fake, reason):
    sf = _sf()
    calls = _fake_certified(monkeypatch, fake)
    out = _refuted()
    res = lp_milp_highs._tight_resolve_refuted(sf, out, dict(KW))
    assert calls, "the re-solve never ran, so this asserts nothing"
    _check_resolve_kw(calls[0])
    assert res is out
    assert res.status == "feasible" and not res.gap_certified and res.bound == -30.0
    assert reason in res.labels["milp/refuted_tight_resolve_declined"]


def test_tight_resolve_adopts_and_keeps_the_better_incumbent(monkeypatch):
    """Adoption: the re-solve certifies a bound below every verified point, but its own
    incumbent (10, 0) at -10 is WORSE than the refuting point (0, 10) at -20. The
    better point is published, the primary's refutation moves to history, and work
    counts accumulate."""
    sf = _sf()
    fake = HighsOutcome(
        "optimal",
        x=np.array([10.0, 0.0]),
        objective=-10.0,
        bound=-20.0 - 1e-12,
        gap_certified=True,
        node_count=5,
        iterations=11,
        wall_time=0.25,
    )
    fake.labels = {"milp/bound_provenance": "highs-fp"}
    calls = _fake_certified(monkeypatch, fake)
    res = lp_milp_highs._tight_resolve_refuted(sf, _refuted(), dict(KW))
    assert calls, "the re-solve never ran, so this asserts nothing"
    _check_resolve_kw(calls[0])
    assert res.status == "optimal" and res.gap_certified
    np.testing.assert_array_equal(res.x, [0.0, 10.0])
    assert res.objective == -20.0
    assert res.bound == pytest.approx(-20.0 - 1e-12) and res.bound <= res.objective
    assert res.node_count == 8 and res.iterations == 18
    assert res.wall_time == pytest.approx(0.75)
    # Nothing on the adopted result reads as a refuted or declined certificate.
    assert "milp/certificate" not in res.labels
    assert "milp/highs_certificate" not in res.labels
    assert "milp/refuted_by" not in res.labels
    assert not [
        k
        for k in res.stats
        if "refut" in k and "before_tight_resolve" not in k and "refuted_tight_resolve" not in k
    ]
    assert res.stats["milp/before_tight_resolve/certificate_refuted"] == 1.0
    assert res.labels["milp/before_tight_resolve/refuted_by"] == "own-integers"
    assert res.labels["milp/before_tight_resolve/certificate"] == "declined"
    assert res.stats["milp/refuted_tight_resolve_adopted"] == 1.0
    assert res.stats["lp/iters"] == 7.0
    assert "mip_feasibility_tolerance=1e-09" in res.labels["milp/bound_provenance"]


def test_tight_resolve_budget_exhausted_is_skipped(monkeypatch):
    calls = _fake_certified(monkeypatch, HighsOutcome("error"))
    out = _refuted()
    res = lp_milp_highs._tight_resolve_refuted(_sf(), out, dict(KW, time_limit=0.4))
    assert not calls
    assert res is out and res.stats["milp/refuted_tight_resolve_skipped"] == 1.0


def test_cross_solve_refuted_by_label_does_not_leak():
    """The trigger is a field set by THIS solve's own #1509 check. A #1634 refutation
    builds a fresh outcome, so the cross-solve's provenance cannot trigger the
    re-solve, and its label is not merged either."""
    sf = _sf()
    out = HighsOutcome(
        "optimal",
        x=np.array([10.0, 0.0]),
        objective=-10.0,
        bound=-10.0,
        gap_certified=True,
        root_bound=-20.0,
    )
    cross = HighsOutcome("feasible", x=np.array([0.0, 10.0]), objective=-20.0, bound=-20.0)
    cross.labels = {"milp/refuted_by": "own-integers"}
    cross.refuted_by = "own-integers"
    kw = dict(KW, time_limit=30.0)
    res = lp_milp_highs._judge_presolve_cross(sf, out, cross, kw)
    assert res.labels.get("milp/presolve_certificate_refuted"), res.labels
    assert res.refuted_by == ""
    assert "milp/refuted_by" not in res.labels


def test_tight_resolve_not_run_for_a_different_integer_refutation(monkeypatch):
    """The #1509 class itself (refuted by a DIFFERENT integer assignment) keeps the NS
    root bound, unchanged. ``max x + 3y  s.t.  x + 2y + s = 21`` with HiGHS's answer
    replaced by ``(10, 0)`` "certified" at -10: the rounded root LP point ``(0, 10)``
    refutes it at -30, and the root bound (-31.5) leaves the gap open."""
    import highspy

    monkeypatch.delenv(FLAG, raising=False)
    calls = []
    monkeypatch.setattr(
        lp_milp_highs, "_tight_resolve_refuted", lambda *a, **k: calls.append(a) or a[1]
    )
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
