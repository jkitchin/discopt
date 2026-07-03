"""Phase 0 certification-observability instrumentation (cert:T0.1, cert:T0.2).

Pins the producers added to make the certification loop measurable:

* T0.1 — ``root_gap`` / ``root_time`` are populated on every discopt benchmark
  row, and ``evaluate_phase_gate`` computes ``root_gap_ratio_vs_baron`` without
  a KeyError (with and without a reference solver).
* T0.2 — opt-in bound-trajectory recording produces a monotone-in-time,
  bound-non-decreasing (min sense) trajectory, downsampled to the configured
  cap, and is *off* by default so the standard solve path is unchanged.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from benchmarks.metrics import (  # noqa: E402
    BenchmarkResults,
    SolveResult,
    SolveStatus,
    evaluate_phase_gate,
    root_gap_populated_fraction,
    root_gap_ratio,
)
from benchmarks.runner import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
    SolverConfig,
    _downsample_trajectory,
)

# A small spatial-B&B instance that opens a tree (so the root snapshot fires and
# a multi-point trajectory is produced).
_SPATIAL_INSTANCE = "gear"


def _make_runner(**cfg_kwargs) -> tuple[BenchmarkRunner, SolverConfig]:
    solver = SolverConfig(name="discopt", command="", solver_type="internal")
    cfg = BenchmarkConfig(
        suite_name="cert-test",
        time_limit=60,
        num_runs=1,
        solvers=[solver],
        **cfg_kwargs,
    )
    return BenchmarkRunner(cfg), solver


def _run(instance: str, **cfg_kwargs) -> SolveResult:
    runner, solver = _make_runner(**cfg_kwargs)
    if runner._find_nl_file(instance) is None:
        pytest.skip(f"{instance}.nl not vendored")
    return runner._run_discopt(solver, instance, 0)


# ─────────────────────────── T0.1 ───────────────────────────


@pytest.mark.smoke
def test_root_gap_and_time_populated():
    """A discopt row carries a non-null, sane root_gap/root_time."""
    res = _run(_SPATIAL_INSTANCE)
    assert res.status == SolveStatus.OPTIMAL
    assert res.root_time is not None and res.root_time >= 0.0
    assert res.root_gap is not None
    # A gap is a non-negative relative quantity; the root bound is on the
    # correct side of the incumbent.
    assert res.root_gap >= -1e-12
    # The root cannot have taken longer than the whole solve.
    assert res.root_time <= res.wall_time + 1e-6


@pytest.mark.smoke
def test_root_gap_ratio_gate_no_keyerror():
    """evaluate_phase_gate computes root_gap_ratio_vs_baron with and without a
    baron reference — never a KeyError, per T0.1's test clause."""
    res = _run(_SPATIAL_INSTANCE)
    baron = SolveResult(
        instance=res.instance,
        solver="baron",
        status=res.status,
        objective=res.objective,
        bound=res.bound,
        root_gap=(res.root_gap or 0.0) * 0.9 + 1e-6,
    )

    class _Bench:
        instance_info: dict = {}

        def get_results(self, name):
            return [res] if name == "discopt" else []

    gate_cfg = {
        "criteria": {
            "rg": {"max": 1.3, "suite": "global50", "metric": "root_gap_ratio_vs_baron"}
        }
    }
    # With a baron reference: a finite ratio, no exception.
    ok, crits = evaluate_phase_gate(
        "cert", _Bench(), gate_cfg, reference_solvers={"baron": [baron]}
    )
    assert len(crits) == 1
    import math

    assert not math.isnan(crits[0].actual)
    # Without any reference solver: still no KeyError (actual is NaN → fails).
    ok2, crits2 = evaluate_phase_gate("cert", _Bench(), gate_cfg, reference_solvers=None)
    assert len(crits2) == 1


def test_root_gap_ratio_skips_nulls():
    """The ratio metric skips rows with a null root_gap on either side."""
    a = [SolveResult(instance="i1", solver="discopt", status=SolveStatus.OPTIMAL, root_gap=0.2)]
    b = [SolveResult(instance="i1", solver="baron", status=SolveStatus.OPTIMAL, root_gap=None)]
    import math

    assert math.isnan(root_gap_ratio(a, b))


# ─────────────────────────── T0.2 ───────────────────────────


def test_downsample_trajectory_preserves_endpoints_and_cap():
    pts = [[float(i), i, -100.0 + i, None if i < 3 else 5.0] for i in range(1200)]
    ds = _downsample_trajectory(pts, 500)
    assert len(ds) <= 500
    assert ds[0] == pts[0] and ds[-1] == pts[-1]
    assert all(ds[i][0] <= ds[i + 1][0] for i in range(len(ds) - 1))
    # Small inputs pass through unchanged.
    small = [[0.0, 0, -1.0, None], [1.0, 1, -0.5, 2.0]]
    assert _downsample_trajectory(small, 500) == small


@pytest.mark.smoke
def test_trajectory_recorded_when_opted_in():
    res = _run(_SPATIAL_INSTANCE, record_trajectory=True, trajectory_max_points=500)
    traj = res.trajectory
    assert traj is not None and len(traj) >= 1
    assert len(traj) <= 500
    times = [p[0] for p in traj]
    bounds = [p[2] for p in traj]
    # Monotone in t; bound non-decreasing in the internal minimization sense.
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))
    assert all(bounds[i] <= bounds[i + 1] + 1e-9 for i in range(len(bounds) - 1))


@pytest.mark.smoke
def test_trajectory_off_by_default():
    """Default config records no trajectory (keeps the default path bound-neutral)."""
    res = _run(_SPATIAL_INSTANCE)
    assert res.trajectory is None


# ─────────────────────────── T0.3 ───────────────────────────


@pytest.mark.smoke
def test_reduction_separation_timers_present_and_bounded():
    """On a spatial instance the per-family reduction/separation timers are
    populated, non-negative, individually non-zero somewhere, and sum to no
    more than the wall time (cert:T0.3)."""
    import discopt.modeling as dm

    runner, _ = _make_runner()
    nl = runner._find_nl_file(_SPATIAL_INSTANCE)
    if nl is None:
        pytest.skip(f"{_SPATIAL_INSTANCE}.nl not vendored")
    result = dm.from_nl(nl).solve(time_limit=30, gap_tolerance=1e-4, max_nodes=100_000)

    stats = result.solver_stats
    assert stats is not None and len(stats) > 0
    # Every entry is a non-negative float keyed under a reduce/ or separate/ family.
    assert all(isinstance(v, float) and v >= 0.0 for v in stats.values())
    assert all(k.startswith(("reduce/", "separate/")) for k in stats)
    # At least one timer is strictly positive (the solve did some separation).
    assert any(v > 0.0 for v in stats.values())
    # The instrumented phases are a subset of the wall clock.
    assert sum(stats.values()) <= result.wall_time + 1e-6


# ─────────────────────────── T0.5 ───────────────────────────


def test_root_gap_populated_fraction():
    rows = [
        SolveResult(instance="a", solver="discopt", status=SolveStatus.OPTIMAL, root_gap=0.1),
        SolveResult(instance="b", solver="discopt", status=SolveStatus.OPTIMAL, root_gap=0.0),
        SolveResult(instance="c", solver="discopt", status=SolveStatus.TIME_LIMIT, root_gap=None),
    ]
    assert root_gap_populated_fraction(rows) == pytest.approx(2 / 3)
    assert root_gap_populated_fraction([]) == 0.0


def _load_cert0_gate_config():
    import tomllib
    from pathlib import Path

    toml = Path(__file__).resolve().parents[1] / "config" / "benchmarks.toml"
    with open(toml, "rb") as fh:
        return tomllib.load(fh)["gates"]["cert0"]


def test_cert0_gate_config_present_and_evaluates():
    """The [gates.cert0] criteria wire the T0.5 metric + the zero-slack
    correctness guard, and evaluate green on a synthetic all-covered, all-correct
    panel."""
    gate = _load_cert0_gate_config()
    crit = gate["criteria"]
    assert crit["root_gap_coverage"]["metric"] == "root_gap_populated_fraction"
    assert crit["root_gap_coverage"]["min"] == 0.9
    # Correctness stays a zero-slack gate.
    assert crit["zero_incorrect"]["metric"] == "incorrect_count"
    assert crit["zero_incorrect"]["max"] == 0

    rows = [
        SolveResult(
            instance=f"i{i}", solver="discopt", status=SolveStatus.OPTIMAL,
            objective=float(i), root_gap=0.01 * i,
        )
        for i in range(10)
    ]
    bench = BenchmarkResults(suite="cert0", timestamp="t")
    for r in rows:
        bench.add_result(r)
    optima = {f"i{i}": float(i) for i in range(10)}

    ok, crits = evaluate_phase_gate("cert0", bench, gate, known_optima=optima)
    by = {c.name: c for c in crits}
    assert by["root_gap_coverage"].actual == pytest.approx(1.0)
    assert by["root_gap_coverage"].passed
    assert by["zero_incorrect"].actual == 0
    assert by["zero_incorrect"].passed
    assert ok


def test_cert0_gate_fails_on_incorrect():
    """zero_incorrect must fail when an OPTIMAL row disagrees with the oracle —
    the check is never weakened."""
    gate = _load_cert0_gate_config()
    rows = [
        SolveResult(
            instance="i0", solver="discopt", status=SolveStatus.OPTIMAL,
            objective=99.0, root_gap=0.0,
        )
    ]
    bench = BenchmarkResults(suite="cert0", timestamp="t")
    for r in rows:
        bench.add_result(r)
    ok, crits = evaluate_phase_gate("cert0", bench, gate, known_optima={"i0": 0.0})
    by = {c.name: c for c in crits}
    assert by["zero_incorrect"].actual >= 1
    assert not by["zero_incorrect"].passed
    assert not ok
