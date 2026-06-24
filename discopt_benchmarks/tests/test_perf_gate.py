"""Unit tests for the perf-gate logic (perf plan Stage 0).

These are fast and deterministic — they exercise ``soundness_violation`` and
``check_gate`` on synthetic records, with no solving. The full panel run is the
nightly ``make perf-gate``; this guards the gate's *decision logic* so a broken
gate (one that passes a regression, or fails a sound run) is caught in CI.
"""

from __future__ import annotations

from discopt_benchmarks.perf import gate as gate_mod
from discopt_benchmarks.perf.gate import check_gate, soundness_violation
from discopt_benchmarks.perf.measure import PerfRecord
from discopt_benchmarks.perf.panel import PanelInstance


def _rec(name, status="optimal", obj=1.0, bound=1.0, nodes=100, compiles=10, wall=10.0):
    return PerfRecord(
        instance=name,
        status=status,
        objective=obj,
        bound=bound,
        wall_time=wall,
        node_count=nodes,
        jax_time=1.0,
        rust_time=0.0,
        python_time=1.0,
        subnlp_calls=0,
        xla_compile_count=compiles,
        xla_compile_seconds=0.1,
        time_to_first_incumbent=1.0,
    )


_GEAR4 = PanelInstance("gear4", 60, 1.6434, "min", "test")


def _base(rec: PerfRecord) -> dict:
    return {rec.instance: rec.to_json()}


# ───────────────────────── soundness_violation ─────────────────────────
class TestSoundness:
    def test_sound_record_passes(self):
        assert soundness_violation(_rec("gear4", obj=1.6434, bound=1.6434), _GEAR4) is None

    def test_false_feasible_caught(self):
        # min: incumbent below the optimum is impossible for a real feasible point
        v = soundness_violation(_rec("gear4", status="feasible", obj=1.0, bound=0.5), _GEAR4)
        assert v and "FALSE-FEASIBLE" in v

    def test_false_optimal_caught(self):
        v = soundness_violation(_rec("gear4", status="optimal", obj=2.5, bound=2.5), _GEAR4)
        assert v and "FALSE-OPTIMAL" in v

    def test_invalid_bound_caught(self):
        # min: a valid lower bound can never exceed the true optimum
        v = soundness_violation(_rec("gear4", status="feasible", obj=3.0, bound=2.0), _GEAR4)
        assert v and "INVALID BOUND" in v

    def test_false_infeasible_caught(self):
        v = soundness_violation(_rec("gear4", status="infeasible", obj=None, bound=None), _GEAR4)
        assert v and "FALSE-INFEASIBLE" in v

    def test_suboptimal_feasible_is_sound(self):
        # min: incumbent ABOVE the optimum (correct direction), open bound — sound
        assert (
            soundness_violation(_rec("gear4", status="feasible", obj=5.0, bound=0.0), _GEAR4)
            is None
        )


# ───────────────────────── check_gate ─────────────────────────
class TestGate:
    def test_identical_passes(self, monkeypatch):
        monkeypatch.setattr(gate_mod, "PANEL", [_GEAR4])
        rec = _rec("gear4", obj=1.6434, bound=1.6434, nodes=5921, compiles=5)
        corr, regr, warn = check_gate([rec], _base(rec))
        assert corr == [] and regr == []

    def test_node_count_regression_caught(self, monkeypatch):
        monkeypatch.setattr(gate_mod, "PANEL", [_GEAR4])
        base = _rec("gear4", obj=1.6434, bound=1.6434, nodes=5921, compiles=5)
        cur = _rec("gear4", obj=1.6434, bound=1.6434, nodes=8000, compiles=5)  # +35 %
        corr, regr, warn = check_gate([cur], _base(base))
        assert corr == []
        assert any("node_count" in r for r in regr)

    def test_compiles_per_node_regression_caught(self, monkeypatch):
        monkeypatch.setattr(gate_mod, "PANEL", [_GEAR4])
        base = _rec("gear4", obj=1.6434, bound=1.6434, nodes=100, compiles=10)  # cpn 0.1
        cur = _rec("gear4", obj=1.6434, bound=1.6434, nodes=100, compiles=20)  # cpn 0.2 (+100 %)
        corr, regr, warn = check_gate([cur], _base(base))
        assert any("compiles/node" in r for r in regr)

    def test_correctness_failure_is_hard(self, monkeypatch):
        monkeypatch.setattr(gate_mod, "PANEL", [_GEAR4])
        base = _rec("gear4", obj=1.6434, bound=1.6434, nodes=5921, compiles=5)
        cur = _rec("gear4", status="optimal", obj=0.5, bound=0.5, nodes=5921, compiles=5)
        corr, regr, warn = check_gate([cur], _base(base))
        assert any("gear4" in c for c in corr)

    def test_wall_regression_is_only_a_warning(self, monkeypatch):
        monkeypatch.setattr(gate_mod, "PANEL", [_GEAR4])
        base = _rec("gear4", obj=1.6434, bound=1.6434, nodes=5921, compiles=5, wall=10.0)
        cur = _rec("gear4", obj=1.6434, bound=1.6434, nodes=5921, compiles=5, wall=20.0)
        corr, regr, warn = check_gate([cur], _base(base))
        assert regr == [] and any("wall" in w for w in warn)


def test_panel_instances_are_vendored():
    """Every panel instance's .nl must be vendored in-repo (self-contained gate)."""
    import os

    from discopt_benchmarks.perf.panel import PANEL

    missing = [i.name for i in PANEL if not os.path.exists(i.path)]
    assert missing == [], f"panel instances not vendored: {missing}"
