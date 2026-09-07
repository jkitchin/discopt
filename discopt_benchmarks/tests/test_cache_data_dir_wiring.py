"""Regression tests for the ``--use-cache`` panel that measured nothing.

A 150-instance MIQP panel reported "12/150" for BOTH discopt and SCIP, with the
other 138 rows finishing in 0.02 s at ``nodes=0``. That is not a solver result:
``--use-cache`` selected instance *names* from the ~1700-file MINLPLib cache while
``BenchmarkRunner._find_nl_file`` resolved *paths* against the 61-file vendored
corpus, so only the 12 names present in both ever opened a model. The lookup
consulted ``config.data_dir`` via ``getattr(..., None)``, but ``BenchmarkConfig``
had no such field -- the branch was unreachable on every code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from benchmarks.metrics import InstanceInfo  # noqa: E402
from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig  # noqa: E402


def _runner(**cfg_kwargs) -> BenchmarkRunner:
    return BenchmarkRunner(BenchmarkConfig(suite_name="t", solvers=[], **cfg_kwargs))


def test_data_dir_is_a_real_field_and_is_used(tmp_path):
    """``data_dir`` must exist AND drive resolution.

    Before the fix ``BenchmarkConfig(data_dir=...)`` raised TypeError, so this
    whole resolution route was dead.
    """
    (tmp_path / "made_up_instance.nl").write_text("g3 0 1 0\n")
    r = _runner(data_dir=tmp_path)
    assert r.config.data_dir == tmp_path
    assert r._find_nl_file("made_up_instance") == str(tmp_path / "made_up_instance.nl")


def test_unknown_instance_still_resolves_to_none(tmp_path):
    """The fallback must not invent a path -- ``None`` is what the caller checks."""
    assert _runner(data_dir=tmp_path)._find_nl_file("no_such_instance_xyz") is None


def test_vendored_fallback_survives_when_no_data_dir():
    """Suites that do not pass ``--use-cache`` keep the in-repo corpus."""
    r = _runner()
    assert r.config.data_dir is None
    root = Path(__file__).resolve().parent.parent.parent
    vendored = sorted((root / "python/tests/data/minlplib_nl").glob("*.nl"))
    if not vendored:
        pytest.skip("vendored corpus not present")
    assert r._find_nl_file(vendored[0].stem) is not None


def test_run_all_is_instance_major_so_solvers_are_interleaved(tmp_path, monkeypatch):
    """Every solver must see an instance in the same wall-clock window (§9).

    Solver-major order ran all of solver A then all of solver B, charging any
    drift in machine load to whichever solver ran second. Asserts the *observed*
    call order, not the source.
    """
    from benchmarks.runner import SolveResult, SolveStatus

    for name in ("i1", "i2", "i3"):
        (tmp_path / f"{name}.nl").write_text("g3 0 1 0\n")

    solvers = [
        SolverConfig(name="a", command="a", solver_type="internal"),
        SolverConfig(name="b", command="b", solver_type="internal"),
    ]
    cfg = BenchmarkConfig(suite_name="t", solvers=solvers, num_runs=1, data_dir=tmp_path)
    r = BenchmarkRunner(cfg)
    r.load_instances([InstanceInfo(name=n, num_variables=1, num_constraints=1) for n in
                      ("i1", "i2", "i3")])

    seen: list[tuple[str, str]] = []

    def _fake(solver, instance, run_idx):
        seen.append((instance, solver.name))
        return SolveResult(
            instance=instance, solver=solver.name, status=SolveStatus.OPTIMAL,
            objective=0.0, bound=0.0, wall_time=0.01, node_count=1,
        )

    monkeypatch.setattr(r, "_run_single", _fake)
    r.run_all()

    assert seen == [
        ("i1", "a"), ("i1", "b"),
        ("i2", "a"), ("i2", "b"),
        ("i3", "a"), ("i3", "b"),
    ], f"expected instance-major interleaving, got {seen}"
