"""#1917: ``heatexch_gen2`` under ``deterministic=True`` -- the re-measurement.

#1917 listed, as a correctness item, that ``heatexch_gen2`` "finds no incumbent
in 30 s in most runs and finds one in some (4/4 empty on one tree at load 1.4,
1/4 found on another), so some budget still reads a wall clock". Those runs all
exited on ``time_limit``, and a run whose role-1 budget binds is outside the
``deterministic`` contract (``solve_model``'s docstring; #1236): the contract
covers a run that stops on **work**. So the item mixed two questions, and the
re-measurement on main ``6fb6b7a`` (2026-10-27, 4 cores, fresh process per
solve, idle vs 3 busy-loop burners interleaved per repetition) answers both:

* **At the issue's 30 s budget the symptom is gone.** 8/8 runs (4 idle at load
  0.5-3.6, 4 loaded at 3.1-5.2) return 635838.846 -- the MINLPLib best -- with
  bound 555767.790 at 3 nodes. The objectives differ only at about 1e-11
  relative (whether the #1537 repair fires on the published point), which a
  run cut by the role-1 clock is allowed to do.
* **Work-bound, the run is bit-identical.** ``max_nodes=3, time_limit=600``:
  6/6 runs (3 idle, 3 loaded at 5.6-6.1) return objective 635838.8464309899,
  bound 555767.7902857299, 3 nodes and the same repair, in 55-65 s. Nothing on
  that path reads a wall clock to decide how much work to do. Deeper, at
  ``max_nodes=10, time_limit=1200``, 4/4 runs (2 idle, 2 loaded at 5.3-5.5)
  return the same objective, bound 566237.0760006903 and 15 nodes.

This test pins the second half, which is the contract. It does not assert the
30 s result: that is a statement about the clock (#1236), and the suite must not
claim determinism for a budgeted exit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from discopt.modeling.core import from_nl

pytest.importorskip("pounce", reason="the default NLP route is pounce")

_PATH = Path(__file__).resolve().parent / "data" / "minlplib_nl" / "heatexch_gen2.nl"


def _solve():
    return from_nl(str(_PATH)).solve(
        time_limit=600.0, max_nodes=3, gap_tolerance=1e-4, deterministic=True
    )


@pytest.mark.slow
@pytest.mark.correctness
def test_work_bound_heatexch_gen2_is_bit_identical_under_deterministic():
    if not _PATH.exists():  # pragma: no cover - corpus is checked in
        pytest.skip(f"corpus instance missing: {_PATH}")

    runs = [_solve() for _ in range(2)]

    # Prove the probe is in the regime it claims (CLAUDE.md §6): a run that hit
    # the role-1 clock would make the equality below a statement about timing.
    for r in runs:
        assert r.node_count == 3, f"not node-bound: {r.node_count} nodes, status {r.status!r}"
        assert r.wall_time < 300.0, f"role-1 clock came close to binding: {r.wall_time:.1f} s"
        assert r.objective is not None, "no incumbent on a work-bound run"
        assert r.bound is not None and r.bound <= r.objective + 1e-6 * abs(r.objective)

    a, b = runs
    assert (a.objective, a.bound, a.node_count, a.status) == (
        b.objective,
        b.bound,
        b.node_count,
        b.status,
    )
    # The same point, not just the same value.
    assert {k: v.tolist() for k, v in a.x.items()} == {k: v.tolist() for k, v in b.x.items()}
