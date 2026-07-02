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
    SolveResult,
    SolveStatus,
    evaluate_phase_gate,
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
