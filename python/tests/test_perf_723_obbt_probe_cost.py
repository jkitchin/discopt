"""Entry-experiment pins for #723 lever 3 — "cap per-node LP solves".

The issue's lever 3 proposed capping nvs05's ~59 per-node LP solves. The entry
experiment (``discopt_benchmarks/scripts/nvs05_obbt_probe_cost_measurement.py``,
recorded in ``docs/dev/performance-plan.md`` §6) FALSIFIED that framing and
located the real per-probe cost. These tests pin the two structural facts the
writeup rests on, so a future engine change that invalidates either flags that
the #723 §6 note needs revisiting:

1. **Per-node OBBT rounds are load-bearing** (NOT diminishing) — the later
   sweeps still tighten, so "cap rounds/probes" trades bound quality for wall
   time, it is not a free cut.
2. **The per-probe cost is a pivot-heavy tail, not a uniform per-call overhead.**
   The per-probe wall is *bimodal* — most probes are cheap warm re-solves, but a
   ~quarter are near-cold re-solves (warm start rejected by simultaneous
   box-tightening + objective-flip) doing ~220 simplex pivots, and that tail is
   the majority of probe wall. This is why the factorization-reuse experiment
   (``DISCOPT_OBBT_FACTOR_REUSE``, since reverted) was net-neutral: reusing a
   ~0.4 ms sparse factorization cannot move a ~220-pivot cost.

   NOTE the ``iters`` field returned by ``solve_lp_warm_csc_py`` reports 0 for the
   near-cold tail (the cold/primal-fallback path does not surface its pivot
   count), so this test pins the tail via the *wall distribution*, not ``iters``.

Instrumentation-only: the patched wrappers delegate to the real implementations,
so the solve itself is unchanged. nvs05 is the vendored probe for the OBBT-probe
class (functionally-dependent continuous intermediates → per-node OBBT engages).
"""

from __future__ import annotations

import time
from pathlib import Path

import discopt._jax.obbt as obbtmod
import numpy as np
import pytest
from discopt.modeling.core import from_nl

pytestmark = pytest.mark.slow

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def _instrumented_solve(inst: str, time_limit: float):
    """Solve ``inst`` while recording OBBT sweep economics + per-probe wall times.

    Returns ``(by_round, probe_walls_ms)`` where ``by_round[i] = [probes,
    tightened, calls]`` for sweep index ``i`` and ``probe_walls_ms`` is the list
    of per-probe LP wall times in milliseconds.
    """
    sweep_log: list[tuple[int, int]] = []
    calls: list[list[tuple[int, int]]] = []
    probe_walls: list[float] = []

    _orig_run = obbtmod.run_obbt_on_relaxation
    _orig_tighten = obbtmod.obbt_tighten_root
    _OrigProbe = obbtmod._PersistentProbeLP

    def _run(*a, **k):
        res = _orig_run(*a, **k)
        sweep_log.append((res.n_lp_solves, res.n_tightened))
        return res

    def _tighten(model, lb, ub, **k):
        start = len(sweep_log)
        res = _orig_tighten(model, lb, ub, **k)
        calls.append(sweep_log[start:])
        return res

    class _Probe(_OrigProbe):
        def solve(self, c, lb_arr, ub_arr, warm_basis):
            from discopt._rust import solve_lp_warm_csc_py

            n, m = self._n, self._m
            c_std = np.concatenate([np.asarray(c, dtype=np.float64).ravel(), np.zeros(m)])
            lb_std = np.concatenate([np.asarray(lb_arr, dtype=np.float64), np.zeros(m)])
            ub_std = np.concatenate([np.asarray(ub_arr, dtype=np.float64), np.full(m, 1e20)])
            warm = warm_basis is not None
            cs0 = np.ascontiguousarray(warm_basis[0], dtype=np.int8) if warm else None
            bv0 = np.ascontiguousarray(warm_basis[1], dtype=np.int64) if warm else None
            t0 = time.perf_counter()
            solve_lp_warm_csc_py(
                np.ascontiguousarray(c_std),
                m,
                n + m,
                self._indptr,
                self._indices,
                self._data,
                self._b,
                np.ascontiguousarray(lb_std),
                np.ascontiguousarray(ub_std),
                cs0,
                bv0,
                1e-9,
                100_000,
            )
            probe_walls.append((time.perf_counter() - t0) * 1e3)
            return super().solve(c, lb_arr, ub_arr, warm_basis)

    obbtmod.run_obbt_on_relaxation = _run
    obbtmod.obbt_tighten_root = _tighten
    obbtmod._PersistentProbeLP = _Probe
    try:
        model = from_nl(str(_NL_DIR / f"{inst}.nl"))
        model.solve(time_limit=time_limit)
    finally:
        obbtmod.run_obbt_on_relaxation = _orig_run
        obbtmod.obbt_tighten_root = _orig_tighten
        obbtmod._PersistentProbeLP = _OrigProbe

    by_round: dict[int, list[int]] = {}
    for sweeps in calls:
        for i, (probes, tight) in enumerate(sweeps):
            row = by_round.setdefault(i, [0, 0, 0])
            row[0] += probes
            row[1] += tight
            row[2] += 1
    return by_round, probe_walls


def test_per_node_obbt_iterates_beyond_the_first_sweep():
    """Per-node OBBT does not fixpoint after one sweep — capping rounds discards work.

    Pins the #723 §6 falsification of "cap per-node OBBT rounds/probes". The
    engine runs up to ``_PER_NODE_OBBT_ROUNDS`` sweeps and breaks early only at a
    ``sweep_tight == 0`` fixpoint; if round 1 captured all the tightening, calls
    would fixpoint immediately and never reach a second sweep. They do reach it —
    a non-trivial share of per-node OBBT calls run >= 2 sweeps — so capping the
    round budget to 1 would discard tightening the engine is actively producing
    (looser node bounds -> more B&B nodes), not shave free overhead.

    Deliberately weaker than the per-sweep *rate* (which is sensitive to how many
    deep nodes the time budget reaches — see the script for full-solve numbers):
    this pins only the robust, timing-insensitive structural fact.
    """
    by_round, _ = _instrumented_solve("nvs05", time_limit=15.0)
    n_calls_sweep0 = by_round.get(0, [0, 0, 0])[2]
    n_calls_sweep1 = by_round.get(1, [0, 0, 0])[2]
    if n_calls_sweep0 < 3:
        pytest.skip("too few per-node OBBT calls in the time budget to characterize")
    # Round 0 must itself be productive (OBBT is engaging, not a no-op) and a
    # meaningful fraction of calls must continue past it (round 1 exists).
    assert by_round[0][1] > 0, "sweep 0 tightened nothing — OBBT not engaging on nvs05"
    assert n_calls_sweep1 >= max(2, n_calls_sweep0 // 3), (
        f"only {n_calls_sweep1}/{n_calls_sweep0} per-node OBBT calls ran a 2nd sweep — "
        "if round 1 now fixpoints immediately, the #723 §6 'capping rounds is not "
        "free' note is stale and should be re-measured"
    )


def test_per_probe_cost_has_a_pivot_heavy_tail():
    """Per-probe wall is bimodal — a heavy tail dominates, so the cost is pivoting.

    Pins the #723 §6 (corrected) root cause: the per-probe cost is NOT a uniform
    per-call overhead (which a factorization/scaling/marshaling story would
    predict, and which factorization reuse would have cut) but a *heavy-tailed*
    distribution — a minority of probes whose warm start is rejected do near-cold
    ~220-pivot re-solves and dominate the wall. If this tail ever flattened
    (p90 ≈ p50), the cost would be uniform and the §6 "cost is the warm-start
    rejection tail, not factorization" note would be stale.

    Uses a *ratio* (p90 / p50), which is invariant to absolute machine speed, so
    it is robust across CI hosts; the measured ratio is ~3–4×.
    """
    _, walls = _instrumented_solve("nvs05", time_limit=15.0)
    if len(walls) < 30:
        pytest.skip("too few OBBT probes recorded to characterize the distribution")
    w = np.array(walls)
    p50 = float(np.percentile(w, 50))
    p90 = float(np.percentile(w, 90))
    assert p50 > 0.0, "degenerate timing (p50==0)"
    # Heavy tail: the slow (warm-rejection, pivot-heavy) probes are well above the
    # cheap-warm median. Lenient 1.8× floor (measured ~3–4×) absorbs host noise.
    assert p90 >= 1.8 * p50, (
        f"per-probe wall not heavy-tailed (p90={p90:.2f}ms vs p50={p50:.2f}ms) — if the "
        "distribution is now uniform, the #723 §6 'cost is the warm-start-rejection "
        "pivot tail, not factorization' root-cause should be re-measured"
    )
