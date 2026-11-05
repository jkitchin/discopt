"""#2143: ``DISCOPT_ROOT_PUMP_SKIP_WITH_INCUMBENT`` -- no root pumps once an incumbent exists.

At the end of the root batch the LP-vertex feasibility pump and the NLP-relaxation
pump (a root NLP multistart seeding a second pump) run whether or not an incumbent
already exists. Both are primal only. Measured on GDPlib ``syngas`` under hull (the
other #2143 flags ON, 120 s): they cost 15.0 s plus a 23.9 s multistart after LOA had
already found the optimum, and the tree reached 7 nodes. With them skipped it reached
29.

ON skips both pumps when the tree holds an incumbent whose gap to the tree's dual
bound is already closed (#2170: an incumbent with an open gap no longer suppresses
them); with none they still run. Default ON; the legacy path is pinned under ``=0``.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import pytest
from discopt import solver as S
from discopt._relax import primal_heuristics as PH

pytestmark = pytest.mark.smoke

_FLAG = "DISCOPT_ROOT_PUMP_SKIP_WITH_INCUMBENT"
_SKIP_LOG = "Feasibility pump (root): skipped, "


def _model():
    m = dm.Model("pump_skip")
    x = m.continuous("x", lb=-3, ub=3)
    z = m.continuous("z", lb=-3, ub=3)
    y = m.binary("y")
    m.subject_to(x * x + z * z <= 4 + 3 * y)
    m.subject_to(x * z >= 0.5)
    m.minimize(-x - z + 1.5 * y + 0.2 * x * x * z)
    return m, {x: 1.2, z: 1.1, y: 1.0}


def _solve(monkeypatch, caplog, flag: str, *, start: bool):
    """Root pump calls, whether the skip was logged, and the result.

    ``start=True`` poses the caller's start first (``DISCOPT_ROOT_WARMSTART_FIRST``),
    so the tree holds an incumbent before the pumps; ``start=False`` gives none.
    """
    monkeypatch.setenv(_FLAG, flag)
    monkeypatch.setenv("DISCOPT_ROOT_WARMSTART_FIRST", "1" if start else "0")
    calls: list[int] = []
    real_pump = PH.feasibility_pump

    def pump(*a, **k):
        calls.append(1)
        return real_pump(*a, **k)

    monkeypatch.setattr(PH, "feasibility_pump", pump)
    m, x0 = _model()
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r = m.solve(time_limit=30, initial_solution=x0 if start else None)
    skipped = any(_SKIP_LOG in rec.getMessage() for rec in caplog.records)
    caplog.clear()
    return len(calls), skipped, r


def test_flag_is_default_on(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert S._root_pump_skip_enabled() is True  # graduated 2026-11-05 (#2143)
    monkeypatch.setenv(_FLAG, "0")
    assert S._root_pump_skip_enabled() is False
    monkeypatch.setenv(_FLAG, "1")
    assert S._root_pump_skip_enabled() is True


def test_legacy_runs_the_pumps_with_an_incumbent(monkeypatch, caplog):
    n_pump, skipped, r = _solve(monkeypatch, caplog, "0", start=True)
    assert not skipped
    assert n_pump >= 1
    assert r.status == "optimal"


def _closed_model():
    """A start that is optimal AND matched by the root relaxation bound (both 2)."""
    m = dm.Model("pump_skip_closed")
    x = m.continuous("x", lb=1, ub=3)
    z = m.continuous("z", lb=1, ub=3)
    y = m.binary("y")
    m.subject_to(x * z >= 1 + y)
    m.subject_to(x * x + z * z <= 10)
    m.minimize(x + z + 3 * y)
    return m, {x: 1.0, z: 1.0, y: 0.0}


def test_on_skips_the_pumps_when_the_gap_is_closed(monkeypatch, caplog):
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setenv("DISCOPT_ROOT_WARMSTART_FIRST", "1")
    calls: list[int] = []
    real_pump = PH.feasibility_pump

    def pump(*a, **k):
        calls.append(1)
        return real_pump(*a, **k)

    monkeypatch.setattr(PH, "feasibility_pump", pump)
    m, x0 = _closed_model()
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r = m.solve(time_limit=30, initial_solution=x0)
    assert any(_SKIP_LOG in rec.getMessage() for rec in caplog.records)
    assert calls == []
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0, abs=1e-6)


def test_on_runs_the_pumps_when_the_incumbent_gap_is_open(monkeypatch, caplog):
    """#2170: an incumbent far from the bound no longer suppresses the pumps.

    The gap is open by construction, not by when the root batch reaches the tree:
    the start's objective (-1.502) is worse than ``_model``'s optimum (-2.399), so
    no valid bound can close it (the optimum itself leaves a 37 % gap). ON runs the
    pumps exactly as ``=0`` does, and says why.
    """
    n_off, _, r_off = _solve(monkeypatch, caplog, "0", start=True)
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setenv("DISCOPT_ROOT_WARMSTART_FIRST", "1")
    calls: list[int] = []
    real_pump = PH.feasibility_pump

    def pump(*a, **k):
        calls.append(1)
        return real_pump(*a, **k)

    monkeypatch.setattr(PH, "feasibility_pump", pump)
    m, x0 = _model()
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r_on = m.solve(time_limit=30, initial_solution=x0)
    gate = [
        rec.getMessage() for rec in caplog.records if "Feasibility pump (root):" in rec.getMessage()
    ]
    assert gate and "is open against the bound" in gate[0], gate
    assert r_on.objective < -2.39  # the start (-1.502) was not optimal
    assert n_off >= 1 and len(calls) == n_off
    assert r_on.status == r_off.status == "optimal"
    assert r_on.objective == pytest.approx(r_off.objective, abs=1e-6)


def test_the_gate_logs_a_finite_closed_bound_at_the_root(monkeypatch, caplog):
    """At the pump site the root batch is not yet imported into the tree, so the gate
    also takes the batch's own relaxation bound (``root_lb``; review on
    jkitchin/discopt#2175).

    Measured: on this model the tree already holds a finite root bound there (2.0,
    seeded before the batch), so this end-to-end check passes with or without
    ``root_lb``. It pins the observable outcome, a finite closed bound and a skip.
    ``test_skip_decision_takes_the_larger_of_tree_and_root_bound`` pins ``root_lb``
    itself (it fails when the decision ignores it).
    """
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setenv("DISCOPT_ROOT_WARMSTART_FIRST", "1")
    m, x0 = _closed_model()
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        m.solve(time_limit=30, initial_solution=x0)
    gate = [
        rec.getMessage() for rec in caplog.records if "Feasibility pump (root):" in rec.getMessage()
    ]
    assert gate == [
        "Feasibility pump (root): skipped, the incumbent 2 meets the gap test against the bound 2"
    ], gate


def test_the_gate_reads_a_seeded_bound_in_a_real_tree():
    """The LOA-seeded case on the Rust tree, in the state the pump site sees.

    Root exported, not yet imported; an incumbent injected; a bound installed with
    ``seed_root_bound``, the proposer's hook. The syngas hull numbers: the hull's root
    relaxation (4347.80) leaves the gap open, LOA's bound (4668.809) closes it.
    """
    import numpy as np
    from discopt._rust import PyTreeManager

    t = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
    t.initialize()
    t.export_batch(1)
    t.inject_incumbent(np.array([1.0, 0.5]), 4669.023485)
    skip, why = S._root_pump_skip_decision(t, 1e-4, 1e-6, root_lb=4347.80)
    assert not skip and "open" in why
    t.seed_root_bound(4668.808968)
    skip, why = S._root_pump_skip_decision(t, 1e-4, 1e-6, root_lb=4347.80)
    assert skip and "4668.808968" in why


class _Tree:
    def __init__(self, inc, glb):
        self._inc, self._glb = inc, glb

    def incumbent(self):
        return None if self._inc is None else (None, self._inc)

    def stats(self):
        return {"global_lower_bound": self._glb}


@pytest.mark.parametrize(
    "inc, glb, skip",
    [
        (None, -1.0, False),  # no incumbent
        (645.51, -4699.67, False),  # methanol hull (#2170): gap ~830 %
        (4669.0235, 4347.80, False),  # syngas hull before LOA's bound: 6.9 %
        (4669.0235, 4668.809, True),  # syngas hull with LOA's bound: 4.6e-5
        (1e-7, 0.0, True),  # absolute gap under abs_gap_tol
        (5.0, float("-inf"), False),  # no finite bound
    ],
)
def test_skip_decision_is_the_gap_test(inc, glb, skip):
    got, why = S._root_pump_skip_decision(_Tree(inc, glb), 1e-4, 1e-6)
    assert got is skip
    assert why


def test_skip_decision_takes_the_larger_of_tree_and_root_bound():
    tree = _Tree(4669.0235, float("-inf"))  # root batch not yet imported
    assert S._root_pump_skip_decision(tree, 1e-4, 1e-6, root_lb=4668.809)[0] is True
    assert S._root_pump_skip_decision(tree, 1e-4, 1e-6, root_lb=4347.80)[0] is False
    seeded = _Tree(4669.0235, 4668.809)  # a seeded bound above the root batch wins
    assert S._root_pump_skip_decision(seeded, 1e-4, 1e-6, root_lb=4347.80)[0] is True


def test_on_runs_the_pumps_without_an_incumbent(monkeypatch, caplog):
    """MINLPLib ``ex1221`` reaches both root pumps with no incumbent: ON runs them."""
    from pathlib import Path

    from discopt.modeling.core import from_nl

    monkeypatch.setenv(_FLAG, "1")
    calls: list[int] = []
    real_pump = PH.feasibility_pump

    def pump(*a, **k):
        calls.append(1)
        return real_pump(*a, **k)

    monkeypatch.setattr(PH, "feasibility_pump", pump)
    m = from_nl(str(Path(__file__).parent / "data" / "minlplib_nl" / "ex1221.nl"))
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r = m.solve(time_limit=30)
    assert not any(_SKIP_LOG in rec.getMessage() for rec in caplog.records)
    assert len(calls) == 2  # the LP-vertex pump and the NLP-relaxation pump
    assert r.status == "optimal"


def test_a_loa_seeded_closed_gap_skips_the_pumps_end_to_end(monkeypatch, caplog):
    """The syngas path, end to end, on the #2143 proposer toy (optimum 2.0).

    The root wave is forced to fail, so the LOA proposer fires and its point (2.0)
    is injected. Its bound is then installed through the real
    ``PyTreeManager.seed_root_bound``. On this toy LOA stops at its first incumbent
    with bound 1.017, so the value is pinned to the toy's proved optimum 2.0, a valid
    floor. The gate must read that seed through the tree and skip the pumps.
    """
    pytest.importorskip("pounce")
    from test_2143_gdp_loa_proposer_bound import TOY_OPT, _toy

    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setenv("DISCOPT_GDP_LOA_PROPOSER", "1")
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    monkeypatch.setattr(PH, "one_hot_config_subnlp", lambda *a, **k: [])
    seeded: list[float] = []

    def seed(tree, work, loa, found_obj):
        b = min(float(found_obj), TOY_OPT)
        tree.seed_root_bound(b)
        seeded.append(b)
        return True

    monkeypatch.setattr(S, "_seed_loa_proposer_bound", seed)
    calls: list[int] = []
    real_pump = PH.feasibility_pump

    def pump(*a, **k):
        calls.append(1)
        return real_pump(*a, **k)

    monkeypatch.setattr(PH, "feasibility_pump", pump)
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r = _toy().solve(gdp_method="hull", time_limit=60)
    gate = [m.getMessage() for m in caplog.records if "Feasibility pump (root):" in m.getMessage()]
    assert seeded == [pytest.approx(TOY_OPT)]
    assert gate and gate[0].startswith("Feasibility pump (root): skipped, "), gate
    assert calls == []
    assert r.objective == pytest.approx(TOY_OPT, abs=1e-6)
