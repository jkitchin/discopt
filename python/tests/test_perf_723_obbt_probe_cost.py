"""Entry-experiment pins for #723 lever 3 — "cap per-node LP solves".

The issue's lever 3 proposed capping nvs05's ~59 per-node LP solves. The entry
experiment (``discopt_benchmarks/scripts/nvs05_obbt_probe_refactor_measurement.py``,
recorded in ``docs/dev/performance-plan.md`` §6) FALSIFIED that framing and
root-caused the residual cost instead. These tests pin the two structural facts
the writeup rests on, so a future engine change that invalidates either flags
that the #723 §6 note needs revisiting:

1. **Per-node OBBT rounds are load-bearing** (NOT diminishing) — the later
   sweeps still tighten a substantial share of what the first sweep does, so
   "cap rounds/probes" trades bound quality for wall time, it is not a free cut.
2. **Warm probes do ~0 simplex pivots** on a non-trivial (m ≫ 1) LP — proving the
   per-probe cost is basis re-factorization in the stateless
   ``solve_lp_warm_csc_py`` binding, not pivoting or "too many LPs". This is the
   evidence for the scoped bound-neutral fix (factorization reuse across probes).

Instrumentation-only: the patched wrappers delegate to the real implementations,
so the solve itself is unchanged. nvs05 is the vendored probe for the OBBT-probe
class (functionally-dependent continuous intermediates → per-node OBBT engages).
"""

from __future__ import annotations

from pathlib import Path

import discopt._jax.obbt as obbtmod
import numpy as np
import pytest
from discopt.modeling.core import from_nl

pytestmark = pytest.mark.slow

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def _instrumented_solve(inst: str, time_limit: float):
    """Solve ``inst`` while recording OBBT sweep economics + per-probe pivots.

    Returns ``(by_round, probe_recs)`` where ``by_round[i] = [probes, tightened,
    calls]`` for sweep index ``i`` and ``probe_recs`` is a list of
    ``(iters, warm, m, n)`` per probe LP.
    """
    sweep_log: list[tuple[int, int]] = []
    calls: list[list[tuple[int, int]]] = []
    probe_recs: list[tuple[int, bool, int, int]] = []

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
            _r = solve_lp_warm_csc_py(
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
            probe_recs.append((int(_r[3]), bool(warm), m, n))
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
    return by_round, probe_recs


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


def test_warm_obbt_probes_are_refactorization_bound():
    """Warm probes do ~0 pivots on a non-trivial LP → cost is re-factorization.

    Pins the #723 §6 root cause: the residual per-node LP cost is NOT pivoting
    (median 0 pivots) and NOT a small toy LP (m is hundreds of rows), so the
    multi-ms/probe wall is the stateless binding re-factorizing the basis every
    probe. This is the evidence for the scoped bound-neutral fix (share the
    factorization across probes on the fixed sweep matrix).
    """
    _, probe_recs = _instrumented_solve("nvs05", time_limit=15.0)
    warm = [r for r in probe_recs if r[1]]
    if len(warm) < 20:
        pytest.skip("too few warm OBBT probes recorded to characterize the cost")

    iters = np.array([r[0] for r in warm])
    m_rows = np.array([r[2] for r in warm])
    # Warm probes are essentially pivot-free (previous optimal basis stays
    # optimal for the next objective on this degenerate relaxation).
    assert np.median(iters) <= 2, (
        f"warm probes median pivots={np.median(iters):.0f} — the 0-pivot premise "
        "of the #723 §6 refactorization root-cause no longer holds"
    )
    # And the LP is large enough that a per-probe basis factorization is the real
    # cost (a handful-of-rows LP would refactor in microseconds).
    assert np.median(m_rows) >= 50, (
        f"OBBT LP has only ~{np.median(m_rows):.0f} rows — re-factorization would "
        "be negligible; the #723 §6 cost attribution should be re-measured"
    )
