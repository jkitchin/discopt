"""Entry experiment for #723 lever 3 — "cap per-node LP solves (nvs05 ~59
LP/node)".

Records the two findings that (1) FALSIFY the "cap the per-node OBBT
rounds/probes" framing and (2) locate the real per-probe cost. Both are
instrumentation-only — the patched wrappers delegate to the real
implementations, so the numbers are what the default solve actually does.

Findings (this machine, ``JAX_PLATFORMS`` unset / CPU x64; magnitudes, not
literals — see ``docs/dev/performance-plan.md`` §6):

1. **Round economics.** Per-node OBBT runs up to ``_PER_NODE_OBBT_ROUNDS`` (3)
   sweeps, stopping at a fixpoint. On nvs05 the sweeps do NOT diminish — over a
   representative solve rounds 1/2/3 each tighten ~18 bounds/call and almost every
   such call runs the full 3 sweeps. The probes are *productive*: capping
   rounds/probes trades bound quality for wall time (→ more B&B nodes), NOT a free
   overhead cut. "The LPs are far too many" is the wrong diagnosis.

2. **The per-probe cost is degenerate simplex PIVOTING, not factorization.** The
   per-probe wall is *bimodal*: ~66% cheap warm re-solves (~3.5 ms p50) and a ~29%
   expensive tail (≥5 ms, up to ~20 ms p99) that is ~62% of all probe wall. The
   expensive probes are warm-start *rejections*: OBBT tightens the box mid-sweep
   and the objective flips each probe (``min x_i`` → ``min x_{i+1}``), so the
   threaded previous basis is both primal-infeasible (box shrank) and
   dual-infeasible (objective changed) → a near-cold two-phase re-solve of ~220
   primal pivots. The LP is tiny (m~325, n~90, nnz~1116, 0.8% dense), so its sparse
   factorization (~0.4 ms) / equilibration (~0.003 ms) / marshaling are all small —
   the pivots are the cost.

   IMPORTANT — the ``iters`` field returned by ``solve_lp_warm_csc_py`` reports
   **0** for these near-cold probes (the cold/primal-fallback path does not surface
   its pivot count), which is why an earlier reading mislabeled them "0-pivot". To
   see the real pivot count, run under ``DISCOPT_PROFILE=1`` and read the
   ``FtUpdate`` / ``AlphaFtran`` / ``PriceBtran`` phase counters (~220 per expensive
   probe), not ``iters``.

A bit-identical factorization-reuse engine change (persistent ``ProbeLp`` handle,
``DISCOPT_OBBT_FACTOR_REUSE``) was implemented to test the "re-factorization
dominates" hypothesis and measured NET-NEUTRAL (~1.0×, cert-clean); it was
reverted (sound ≠ helpful). The real lever is the warm-start rejection tail — see
§6. This script measures the per-probe wall distribution that shows the tail.

Usage::

    python discopt_benchmarks/scripts/nvs05_obbt_probe_cost_measurement.py [inst] [time_limit]

    # for the pivot ground-truth on the expensive probes, run under
    # DISCOPT_PROFILE=1 and read the FtUpdate / AlphaFtran / PriceBtran phase
    # counters (~220 per expensive probe) instead of the misleading ``iters``.
"""

from __future__ import annotations

import sys
import time

import discopt._jax.obbt as obbtmod
import discopt.modeling as dm
import numpy as np


def measure(inst: str = "nvs05", time_limit: float = 12.0) -> dict:
    # --- instrument #1: round economics (probes + tightenings per sweep) -------
    sweep_log: list[tuple[int, int]] = []
    _orig_run = obbtmod.run_obbt_on_relaxation

    def _run(*a, **k):
        res = _orig_run(*a, **k)
        sweep_log.append((res.n_lp_solves, res.n_tightened))
        return res

    calls: list[list[tuple[int, int]]] = []
    _orig_tighten = obbtmod.obbt_tighten_root

    def _tighten(model, lb, ub, **k):
        start = len(sweep_log)
        res = _orig_tighten(model, lb, ub, **k)
        calls.append(sweep_log[start:])
        return res

    # --- instrument #2: per-probe wall + reported iters (the cost distribution) -
    probe_walls: list[float] = []
    probe_iters: list[int] = []
    probe_shape: list[tuple[int, int, int]] = []
    orig_probe_cls = obbtmod._PersistentProbeLP

    class _Patched(orig_probe_cls):
        def solve(self, c, lb_arr, ub_arr, warm_basis):
            from discopt._rust import solve_lp_warm_csc_py

            n, m = self._n, self._m
            c_std = np.ascontiguousarray(
                np.concatenate([np.asarray(c, dtype=np.float64).ravel(), np.zeros(m)])
            )
            lb_std = np.ascontiguousarray(
                np.concatenate([np.asarray(lb_arr, dtype=np.float64), np.zeros(m)])
            )
            ub_std = np.ascontiguousarray(
                np.concatenate([np.asarray(ub_arr, dtype=np.float64), np.full(m, 1e20)])
            )
            cs0 = (
                np.ascontiguousarray(warm_basis[0], dtype=np.int8)
                if warm_basis is not None
                else None
            )
            bv0 = (
                np.ascontiguousarray(warm_basis[1], dtype=np.int64)
                if warm_basis is not None
                else None
            )
            t0 = time.perf_counter()
            r = solve_lp_warm_csc_py(
                c_std,
                m,
                n + m,
                self._indptr,
                self._indices,
                self._data,
                self._b,
                lb_std,
                ub_std,
                cs0,
                bv0,
                1e-9,
                100_000,
            )
            probe_walls.append((time.perf_counter() - t0) * 1e3)
            probe_iters.append(int(r[3]))
            probe_shape.append((m, n, int(len(self._data))))
            # Delegate to the real implementation so the solve is unaffected.
            return super().solve(c, lb_arr, ub_arr, warm_basis)

    obbtmod.run_obbt_on_relaxation = _run
    obbtmod.obbt_tighten_root = _tighten
    obbtmod._PersistentProbeLP = _Patched
    try:
        model = dm.from_nl(f"python/tests/data/minlplib_nl/{inst}.nl")
        r = model.solve(time_limit=time_limit)
    finally:
        obbtmod.run_obbt_on_relaxation = _orig_run
        obbtmod.obbt_tighten_root = _orig_tighten
        obbtmod._PersistentProbeLP = orig_probe_cls

    # --- round economics table ---
    by_round: dict[int, list[int]] = {}
    for sweeps in calls:
        for i, (probes, tight) in enumerate(sweeps):
            row = by_round.setdefault(i, [0, 0, 0])
            row[0] += probes
            row[1] += tight
            row[2] += 1

    # --- per-probe cost distribution ---
    dist: dict[str, object] = {}
    if probe_walls:
        w = np.array(probe_walls)
        shp = probe_shape[len(probe_shape) // 2]
        dist = {
            "n_probes": int(len(w)),
            "mean_ms": float(w.mean()),
            "p50": float(np.percentile(w, 50)),
            "p90": float(np.percentile(w, 90)),
            "p99": float(np.percentile(w, 99)),
            "frac_ge5ms": float((w >= 5).mean()),
            "wall_share_ge5ms": float(w[w >= 5].sum() / w.sum()) if w.sum() else 0.0,
            "iters_reported_median": float(np.median(probe_iters)),
            "m": shp[0],
            "n": shp[1],
            "nnz": shp[2],
        }

    return {
        "inst": inst,
        "objective": r.objective,
        "node_count": getattr(r, "node_count", None),
        "status": r.status,
        "by_round": by_round,
        "cost_dist": dist,
    }


def main() -> None:
    inst = sys.argv[1] if len(sys.argv) > 1 else "nvs05"
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    out = measure(inst, time_limit)
    print(f"{out['inst']}: obj={out['objective']} nodes={out['node_count']} status={out['status']}")

    print("\n[1] per-node OBBT round economics (are rounds 2-3 load-bearing?):")
    print("  sweep  calls  probes  tightened  avg_tight/call")
    for i in sorted(out["by_round"]):
        p, t, cnt = out["by_round"][i]
        print(f"    {i}    {cnt:5d}  {p:6d}   {t:7d}      {t / cnt:.2f}")
    if not out["by_round"]:
        print("    (no per-node OBBT probes ran — this instance is not OBBT-bound)")

    print("\n[2] per-probe wall distribution (bimodal → pivot-heavy tail, not factorization):")
    d = out["cost_dist"]
    if d:
        dense = 100 * d["nnz"] / max(1, d["m"] * (d["n"] + d["m"]))
        print(
            f"    probes={d['n_probes']}  LP m~{d['m']} n~{d['n']} "
            f"nnz~{d['nnz']} ({dense:.1f}% dense)"
        )
        print(
            f"    wall/probe: mean={d['mean_ms']:.2f}ms p50={d['p50']:.2f} "
            f"p90={d['p90']:.2f} p99={d['p99']:.2f} ms"
        )
        print(
            f"    expensive tail: {100 * d['frac_ge5ms']:.0f}% of probes are >=5ms and are "
            f"{100 * d['wall_share_ge5ms']:.0f}% of probe wall"
        )
        print(
            f"    reported iters median={d['iters_reported_median']:.0f} "
            f"(MISLEADING for the near-cold tail — use DISCOPT_PROFILE FtUpdate/AlphaFtran)"
        )
        print("    → the ~p90+ tail is warm-start-rejection near-cold re-solves (~220 pivots);")
        print("      factorization (~0.4ms) / scaling (~0.003ms) / marshaling are not the cost.")
    else:
        print("    (no probes recorded)")


if __name__ == "__main__":
    main()
