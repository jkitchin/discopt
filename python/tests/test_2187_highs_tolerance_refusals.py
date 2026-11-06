"""#2187: round-off-level refusals on the HiGHS MILP route for trained ReLU networks.

Three 2-32-32-1 nets (``data/issue_2187``) where ``relu_bigm`` certifies, but the
same network under another formulation was refused. Each refusal was sound (nothing
false was certified), and each had its own cause:

* ``s413_p1`` + ``relu_extended``: ``feasible``, gap 0.915. The #1509 refutation
  (1.097e-6 over a 1.05e-6 yardstick) was by the LP over HiGHS's OWN integers. The
  #2151 tight re-solve was refuted the same way, because the cause is HiGHS's PRESOLVE:
  the false bound comes back at every ``mip_feasibility_tolerance`` (1e-6, 1e-9) and
  stop rule while presolve is on, and presolve off at 1e-9 gives 1.7e-13. Fix: a
  presolve-free stage of the tight re-solve (``DISCOPT_MILP_REFUTED_PRESOLVE_FREE``).
* ``s406_p0`` + ideal cuts: ``feasible``, gap 0.940. Here (Linux) the LP over the
  incumbent's own integers hit a HiGHS ``run()`` error: postsolve handed dual simplex an
  objective of -3.3e15 ("excessive dual values"), so the #1509 check was inconclusive.
  The same LP solves without presolve. Fix: ``solve_lp_std``'s one presolve-free
  re-solve also covers an error. The issue reports a #1509 refutation for this case on
  macOS; both paths end at the fixes here.
* ``s404_p0`` + ``relu_extended``: ``error``. HiGHS stops at its root, and its bound is
  its own root LP solved within ``mip_feasibility_tolerance`` 1e-6, 5.4e-5 below the
  exact NS root bound. A weaker lower bound is not a false one. Fix: drop the tree
  bound and certify on the NS root bound alone (``DISCOPT_MILP_TREE_BELOW_ROOT_REPAIR``).
  An incumbent below that bound is a row-tolerance artefact and is never certified.

Every end-to-end assertion compares against an exact numpy forward pass of the network.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ml import NNFormulation
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition
from discopt.solvers import lp_milp_highs
from discopt.solvers.lp_milp_highs import HighsOutcome, StdForm

DATA = Path(__file__).parent / "data" / "issue_2187"
TREE_FLAG = "DISCOPT_MILP_TREE_BELOW_ROOT_REPAIR"
FREE_FLAG = "DISCOPT_MILP_REFUTED_PRESOLVE_FREE"

#: ``relu_bigm`` certified optima of the same networks (the issue's table, reproduced).
BIGM_OPT = {
    "rosenbrock_Bh_lam1.0_s413_p1": -54.5248663510,
    "rosenbrock_Bh_lam1.0_s406_p0": -38.1894278908,
    "rosenbrock_Bm_lam3.0_s404_p0": 20.5561384254,
    # A hidden-neuron permutation of s406_p0 (the same function), from the #2187 panel.
    "rosenbrock_Bh_lam1.0_s406_p0_perm4": -38.1894278908,
}


def _net(name: str):
    z = np.load(DATA / f"{name}.npz")
    Ws = [z[f"W{i}"] for i in range(3)]
    bs = [z[f"b{i}"] for i in range(3)]
    return Ws, bs, z["lo"], z["hi"]


def _forward(Ws, bs, x) -> float:
    h = np.asarray(x, dtype=np.float64)
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W + b
        if i < len(Ws) - 1:
            h = np.maximum(h, 0.0)
    return float(np.sum(h))


def _solve(name: str, **kw):
    Ws, bs, lo, hi = _net(name)
    layers = [DenseLayer(W, b, Activation.RELU) for W, b in zip(Ws[:-1], bs[:-1])]
    layers.append(DenseLayer(Ws[-1], bs[-1], Activation.LINEAR))
    m = dm.Model("i2187")
    nn = NNFormulation(m, NetworkDefinition(layers, input_bounds=(lo, hi)), **kw)
    nn.formulate()
    m.minimize(dm.sum(nn.outputs))
    extra = dict(cut_callback=nn.cut_callback) if kw.get("ideal_cuts") else {}
    r = m.solve(time_limit=120, threads=1, deterministic=True, **extra)
    f = None if r.objective is None else _forward(Ws, bs, r.value(nn.inputs))
    return r, f


def _assert_certified_optimum(name: str, r, f) -> None:
    assert r.status == "optimal", (r.status, r.algorithm_route)
    # The published input really attains the published objective ...
    assert f is not None and f == pytest.approx(r.objective, abs=1e-6 * (1 + abs(f)))
    # ... which is the bigm optimum of the same function ...
    assert f == pytest.approx(BIGM_OPT[name], abs=1e-6)
    # ... and the bound is a lower bound of an exactly evaluated point.
    assert r.bound is not None and r.bound <= f + 1e-9 * (1 + abs(f))
    assert r.bound <= r.objective


# ── end to end: the issue's three refusals ───────────────────────────────────


def test_s413_extended_certifies_via_presolve_free_tight_resolve(monkeypatch):
    """Before #2187: ``feasible``, bound -638.36, gap 0.915."""
    monkeypatch.delenv(FREE_FLAG, raising=False)
    name = "rosenbrock_Bh_lam1.0_s413_p1"
    r, f = _solve(name, strategy="relu_extended")
    _assert_certified_optimum(name, r, f)
    assert r.solver_stats["milp/refuted_presolve_free_adopted"] == 1.0
    assert "presolve-free" in r.algorithm_route


def test_s413_extended_opt_out_restores_the_refusal(monkeypatch):
    monkeypatch.setenv(FREE_FLAG, "0")
    r, _ = _solve("rosenbrock_Bh_lam1.0_s413_p1", strategy="relu_extended")
    assert r.status == "feasible" and r.gap > 0.5
    assert "milp/refuted_presolve_free_ran" not in r.solver_stats


def test_s406_ideal_cuts_certifies():
    """Before #2187: ``feasible``, bound -640.87, gap 0.940 (fixed-integer LP check
    inconclusive on a HiGHS ``run()`` error here; a #1509 refutation on macOS)."""
    name = "rosenbrock_Bh_lam1.0_s406_p0"
    r, f = _solve(name, strategy="relu_bigm", ideal_cuts=True)
    _assert_certified_optimum(name, r, f)
    assert "milp/fixed_int_check_inconclusive" not in r.solver_stats


def test_s404_extended_certifies_on_the_root_bound(monkeypatch):
    """Before #2187: ``error`` with no objective and no bound."""
    monkeypatch.delenv(TREE_FLAG, raising=False)
    name = "rosenbrock_Bm_lam3.0_s404_p0"
    r, f = _solve(name, strategy="relu_extended")
    _assert_certified_optimum(name, r, f)
    assert r.solver_stats["milp/tree_bound_below_root"] > 0.0
    assert r.solver_stats["milp/presolve_cross_moot_root_bound"] == 1.0
    # Not the presolve-free arm's 20.5560762, which sits 5.1e-5 below the root bound.
    assert r.objective >= r.bound


def test_s404_extended_opt_out_restores_the_error(monkeypatch):
    monkeypatch.setenv(TREE_FLAG, "0")
    r, _ = _solve("rosenbrock_Bm_lam3.0_s404_p0", strategy="relu_extended")
    assert r.status == "error"
    assert r.objective is None and r.bound is None


# ── solve_lp_std: a HiGHS run() error gets the presolve-free re-solve ────────


def _lp() -> StdForm:
    # min -x - 2y  s.t.  x + y + s = 4, 0 <= x, y <= 3: optimum -7 at (1, 3).
    return StdForm.from_arrays(
        c=np.array([-1.0, -2.0, 0.0]),
        A=np.array([[1.0, 1.0, 1.0]]),
        b=np.array([4.0]),
        xl=np.zeros(3),
        xu=np.array([3.0, 3.0, 1e20]),
        int_idx=np.array([], dtype=int),
    )


class _ErrorWhenPresolved:
    """A HiGHS whose ``run()`` returns ``kError`` unless presolve was switched off."""

    def __init__(self, h, highspy, presolve_off: bool, log: list[bool]):
        self._h, self._hs, self._off, self._log = h, highspy, presolve_off, log

    def run(self):
        self._log.append(self._off)
        return self._h.run() if self._off else self._hs.HighsStatus.kError

    def __getattr__(self, name):
        return getattr(self._h, name)


def _error_when_presolved(monkeypatch) -> list[bool]:
    log: list[bool] = []
    real = lp_milp_highs._new_highs

    def fake(highspy, opts):
        off = ("presolve", "off") in list(opts)
        return _ErrorWhenPresolved(real(highspy, opts), highspy, off, log)

    monkeypatch.setattr(lp_milp_highs, "_new_highs", fake)
    return log


def test_lp_run_error_is_resolved_without_presolve(monkeypatch):
    log = _error_when_presolved(monkeypatch)
    out = lp_milp_highs.solve_lp_std(_lp(), time_limit=30.0)
    assert log == [False, True], log
    assert out.status == "optimal" and out.gap_certified
    assert out.objective == pytest.approx(-7.0, abs=1e-9)
    assert out.bound is not None and out.bound <= -7.0 + 1e-9
    assert out.stats["lp/presolve_off_error_resolve"] == 1.0


def test_lp_run_error_without_presolve_stays_an_error(monkeypatch):
    log: list[bool] = []
    real = lp_milp_highs._new_highs

    def fake(highspy, opts):  # errors on every run, presolve or not
        h = _ErrorWhenPresolved(real(highspy, opts), highspy, False, log)
        return h

    monkeypatch.setattr(lp_milp_highs, "_new_highs", fake)
    out = lp_milp_highs.solve_lp_std(_lp(), time_limit=30.0)
    assert len(log) == 2, log  # one re-solve, then the error stands
    assert out.status == "error" and "run() error" in out.message


# ── _repair_tree_below_root ──────────────────────────────────────────────────

GAP_KW = {"abs_gap_tolerance": None, "gap_tolerance": 1e-4}


def _tree_out(objective: float | None, bound: float, status: str = "optimal") -> HighsOutcome:
    return HighsOutcome(
        status,
        x=None if objective is None else np.zeros(1),
        objective=objective,
        bound=bound,
        gap_certified=status == "optimal",
    )


def test_repair_certifies_when_the_root_bound_closes_the_gap():
    out, labels, stats = _tree_out(20.5561384254, 20.5560726712), {}, {}
    lp_milp_highs._repair_tree_below_root(out, 20.5561266974, labels, stats, GAP_KW)
    assert out.status == "optimal" and out.gap_certified
    assert out.bound == 20.5561266974
    assert labels["milp/bound_provenance"] == "root-ns"
    assert stats["milp/tree_bound_below_root"] == pytest.approx(5.40262e-5, rel=1e-4)


def test_repair_declines_when_the_root_bound_leaves_the_gap_open():
    out, labels, stats = _tree_out(25.0, 19.0), {}, {}
    lp_milp_highs._repair_tree_below_root(out, 20.0, labels, stats, GAP_KW)
    assert out.status == "feasible" and not out.gap_certified
    assert out.bound == 20.0 and labels["milp/certificate"] == "declined"


def test_repair_never_certifies_an_incumbent_below_the_root_bound():
    """The presolve-free arm's point on the s404 net: 20.5560762, 5.1e-5 under the NS
    root bound. No exactly feasible point beats a valid lower bound."""
    out, labels, stats = _tree_out(20.5560761659, 20.5560726712), {}, {}
    lp_milp_highs._repair_tree_below_root(out, 20.5561266974, labels, stats, GAP_KW)
    assert out.status == "feasible" and not out.gap_certified
    assert out.bound <= out.objective
    assert labels["milp/certificate"] == "declined"
    assert stats["milp/incumbent_below_root"] == pytest.approx(5.05315e-5, rel=1e-4)


def test_repair_on_a_limit_exit_only_raises_the_bound():
    out, labels, stats = _tree_out(30.0, 10.0, status="time_limit"), {}, {}
    lp_milp_highs._repair_tree_below_root(out, 12.0, labels, stats, GAP_KW)
    assert out.status == "time_limit" and not out.gap_certified and out.bound == 12.0


# ── #1634 is moot for a root-bound certificate ───────────────────────────────


def _sf2() -> StdForm:
    return StdForm.from_arrays(
        c=np.array([-1.0, -2.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )


def test_weaker_bound_keeps_a_root_bound_certificate_and_ignores_the_cross_point():
    out = HighsOutcome(
        "optimal", x=np.array([0.0, 10.0]), objective=-20.0, bound=-20.0, gap_certified=True
    )
    out.stats = {"milp/tree_bound_below_root": 1e-5}
    out.labels = {"milp/bound_provenance": "root-ns"}
    cross = HighsOutcome("feasible", x=np.array([10.0, 0.0]), objective=-10.0, bound=-25.0)
    res = lp_milp_highs._weaker_bound(_sf2(), out, cross, dict(GAP_KW, time_limit=10.0))
    assert res is out and res.status == "optimal" and res.gap_certified
    assert res.bound == -20.0
    np.testing.assert_array_equal(res.x, [0.0, 10.0])
    assert res.stats["milp/presolve_cross_moot_root_bound"] == 1.0


def test_weaker_bound_still_withdraws_a_tree_bound_certificate():
    """The same pair without the #2187 marker is the #1634 rule, unchanged."""
    out = HighsOutcome(
        "optimal", x=np.array([0.0, 10.0]), objective=-20.0, bound=-20.0, gap_certified=True
    )
    out.labels = {"milp/bound_provenance": "highs-fp"}
    cross = HighsOutcome("feasible", x=np.array([10.0, 0.0]), objective=-10.0, bound=-25.0)
    res = lp_milp_highs._weaker_bound(_sf2(), out, cross, dict(GAP_KW, time_limit=10.0))
    assert not res.gap_certified
    assert "milp/presolve_cross_moot_root_bound" not in res.stats


# ── the presolve-free stage of the #2151 tight re-solve ──────────────────────

KW = {"time_limit": 30.0, "gap_tolerance": 1e-4, "abs_gap_tolerance": None, "max_nodes": 100}


def _refuted() -> HighsOutcome:
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
    out.stats = {"milp/certificate_refuted": 1.0}
    out.labels = {
        "milp/certificate": "declined",
        "milp/highs_certificate": "refuted",
        "milp/refuted_by": "own-integers",
        "milp/bound_provenance": "root-ns",
    }
    return out


def _fakes(monkeypatch, tight: HighsOutcome, free: HighsOutcome) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(lp_milp_highs, "_certified_milp", lambda sf, kw: tight)

    def fake_scaled(sf, *, presolve=True, **kw):
        calls.append(dict(kw, presolve=presolve))
        return free

    monkeypatch.setattr(lp_milp_highs, "_solve_milp_scaled", fake_scaled)
    return calls


def _tight_refuted_again() -> HighsOutcome:
    return HighsOutcome(
        "feasible",
        x=np.array([0.0, 10.0]),
        objective=-20.0,
        bound=-30.0,
        wall_time=0.25,
        refuted_by="own-integers",
    )


def test_presolve_free_stage_adopts_a_certified_presolve_free_solve(monkeypatch):
    monkeypatch.delenv(FREE_FLAG, raising=False)
    free = HighsOutcome(
        "optimal",
        x=np.array([0.0, 10.0]),
        objective=-20.0,
        bound=-20.0 - 1e-13,
        gap_certified=True,
        node_count=9,
        wall_time=0.1,
    )
    free.labels = {"milp/bound_provenance": "highs-fp"}
    calls = _fakes(monkeypatch, _tight_refuted_again(), free)
    res = lp_milp_highs._tight_resolve_refuted(_sf2(), _refuted(), dict(KW))
    assert len(calls) == 1, "the presolve-free stage never ran"
    kw = calls[0]
    assert kw["presolve"] is False and kw["initial_point"] is None
    assert kw["feasibility_tolerance"] == lp_milp_highs.REFUTED_TIGHT_FEAS_TOL
    cap = max(
        lp_milp_highs.PRESOLVE_CROSS_MIN_BUDGET, lp_milp_highs.PRESOLVE_CROSS_BUDGET_FACTOR * 0.5
    )
    assert kw["time_limit"] == pytest.approx(min(cap, 30.0 - 0.5) - 0.25)
    assert res.status == "optimal" and res.gap_certified
    assert res.bound == pytest.approx(-20.0 - 1e-13) and res.bound <= res.objective
    assert res.stats["milp/refuted_presolve_free_adopted"] == 1.0
    assert "presolve-free" in res.labels["milp/bound_provenance"]
    assert "milp/certificate" not in res.labels


def test_presolve_free_stage_declines_a_bound_above_the_point(monkeypatch):
    monkeypatch.delenv(FREE_FLAG, raising=False)
    free = HighsOutcome(
        "optimal", x=np.array([0.0, 10.0]), objective=-20.0, bound=-20.0 + 5e-7, gap_certified=True
    )
    calls = _fakes(monkeypatch, _tight_refuted_again(), free)
    out = _refuted()
    res = lp_milp_highs._tight_resolve_refuted(_sf2(), out, dict(KW))
    assert calls, "the presolve-free stage never ran, so this asserts nothing"
    assert res is out and res.status == "feasible" and res.bound == -30.0
    why = res.labels["milp/refuted_tight_resolve_declined"]
    assert "presolve-free re-solve" in why and "above a verified point" in why


@pytest.mark.parametrize("refuted_by", ["", "rounded-root"])
def test_presolve_free_stage_needs_an_own_integers_refutation_again(monkeypatch, refuted_by):
    monkeypatch.delenv(FREE_FLAG, raising=False)
    tight = _tight_refuted_again()
    tight.refuted_by = refuted_by
    calls = _fakes(monkeypatch, tight, HighsOutcome("error"))
    res = lp_milp_highs._tight_resolve_refuted(_sf2(), _refuted(), dict(KW))
    assert not calls and res.status == "feasible"


def test_presolve_free_stage_opt_out(monkeypatch):
    monkeypatch.setenv(FREE_FLAG, "0")
    calls = _fakes(monkeypatch, _tight_refuted_again(), HighsOutcome("error"))
    res = lp_milp_highs._tight_resolve_refuted(_sf2(), _refuted(), dict(KW))
    assert not calls and res.status == "feasible"


# ── an artefactual incumbent is replaced by the LP over its own integers ─────


def _pending(objective: float, root: float) -> tuple[HighsOutcome, dict, dict]:
    out = HighsOutcome(
        "feasible", x=np.zeros(1), objective=objective, bound=objective, root_bound=root
    )
    labels = {"milp/certificate": "declined", "milp/bound_provenance": "root-ns"}
    stats = {
        "milp/incumbent_below_root": root - objective,
        "milp/incumbent_below_root_pending": 1.0,
    }
    return out, labels, stats


def test_rederive_below_root_publishes_the_lp_point_and_certifies():
    """highspy 1.11.0 on the s404 net: HiGHS's own incumbent 20.5560762 is 5.05e-5 under
    the root bound; the LP over its integers gives the bigm optimum 20.5561384254."""
    out, labels, stats = _pending(20.5560761659, 20.5561266974)
    pt = (np.ones(1), 20.5561384254)
    lp_milp_highs._rederive_below_root(out, pt, labels, stats, GAP_KW)
    assert out.objective == 20.5561384254 and out.x[0] == 1.0
    assert out.bound == 20.5561266974 and out.bound <= out.objective
    assert out.status == "optimal" and out.gap_certified
    assert "milp/certificate" not in labels
    assert "milp/incumbent_below_root_pending" not in stats


def test_rederive_below_root_keeps_the_gap_test():
    out, labels, stats = _pending(15.0, 20.0)
    lp_milp_highs._rederive_below_root(out, (np.ones(1), 25.0), labels, stats, GAP_KW)
    assert out.objective == 25.0 and out.bound == 20.0
    assert out.status == "feasible" and not out.gap_certified
    assert labels["milp/certificate"] == "declined"


def test_rederive_below_root_refuses_an_lp_point_also_below_the_bound():
    out, labels, stats = _pending(15.0, 20.0)
    lp_milp_highs._rederive_below_root(out, (np.ones(1), 19.0), labels, stats, GAP_KW)
    assert out.objective == 15.0 and out.status == "feasible" and not out.gap_certified


# ── #1678 adoption takes the RE-DERIVED cross point, never the raw one ───────


def test_s406_perm4_extended_publishes_the_value_of_its_point():
    """Before: ``optimal`` at -38.1894713, the raw presolve-free incumbent, while the
    network's exact value at the published inputs is -38.1894279 (4.3e-5 apart), on
    the merge base as well. The raw point was adopted as 4.34e-5 "better" than the
    primary's; the LP over its own integers gives the exact value."""
    name = "rosenbrock_Bh_lam1.0_s406_p0_perm4"
    r, f = _solve(name, strategy="relu_extended")
    _assert_certified_optimum(name, r, f)
    assert abs(r.objective - f) <= 1e-9 * (1 + abs(f))
    assert "milp/presolve_cross_incumbent_adopted" not in r.solver_stats


def _two_point_sf() -> StdForm:
    # min -x - 2y  s.t.  x + y + s = 10, x, y integer >= 0: optimum -20 at (0, 10).
    return StdForm.from_arrays(
        c=np.array([-1.0, -2.0, 0.0]),
        A=np.array([[1.0, 1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.zeros(3),
        xu=np.array([1e20, 1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )


def _verifier_accepting(monkeypatch, raw_obj: float):
    """The route's verifier, except that the cross point ``(0, 10)`` is reported at
    ``raw_obj``: a raw HiGHS point whose objective the row tolerance flattered."""
    real = lp_milp_highs._verified_mip_point
    flagged = {"raw": 0}

    def fake(sf, x):
        if x is not None and np.asarray(x).size == 3 and np.isnan(np.asarray(x)[2]):
            flagged["raw"] += 1
            return (np.array([0.0, 10.0, 0.0]), raw_obj)
        return real(sf, x)

    monkeypatch.setattr(lp_milp_highs, "_verified_mip_point", fake)
    return flagged


def test_adopt_uses_the_lp_point_not_the_raw_cross_point(monkeypatch):
    sf = _two_point_sf()
    flagged = _verifier_accepting(monkeypatch, -20.5)  # raw: 0.5 "better" than truth
    out = HighsOutcome("optimal", x=np.array([1.0, 9.0, 0.0]), objective=-19.0, gap_certified=True)
    cross = HighsOutcome("optimal", x=np.array([0.0, 10.0, np.nan]))
    lp_milp_highs._adopt_better_cross_incumbent(sf, out, cross, 10.0)
    assert flagged["raw"] == 1, "the raw cross point was never offered"
    assert out.stats["milp/presolve_cross_incumbent_adopted"] == 1.0
    assert out.objective == pytest.approx(-20.0, abs=1e-9)  # the LP point, not -20.5
    assert out.stats["milp/presolve_cross_incumbent_rederive_shift"] == pytest.approx(0.5)


def test_adopt_declines_when_the_lp_point_is_no_better(monkeypatch):
    sf = _two_point_sf()
    flagged = _verifier_accepting(monkeypatch, -20.5)
    out = HighsOutcome("optimal", x=np.array([0.0, 10.0, 0.0]), objective=-20.0, gap_certified=True)
    cross = HighsOutcome("optimal", x=np.array([0.0, 10.0, np.nan]))
    lp_milp_highs._adopt_better_cross_incumbent(sf, out, cross, 10.0)
    assert flagged["raw"] == 1, "the raw cross point was never offered"
    assert out.objective == -20.0
    assert "milp/presolve_cross_incumbent_adopted" not in out.stats


def test_adopt_without_budget_adopts_nothing(monkeypatch):
    sf = _two_point_sf()
    _verifier_accepting(monkeypatch, -20.5)
    out = HighsOutcome("optimal", x=np.array([1.0, 9.0, 0.0]), objective=-19.0, gap_certified=True)
    cross = HighsOutcome("optimal", x=np.array([0.0, 10.0, np.nan]))
    lp_milp_highs._adopt_better_cross_incumbent(sf, out, cross, 0.0)
    assert out.objective == -19.0
    assert out.stats["milp/presolve_cross_incumbent_not_rederived"] == 1.0
