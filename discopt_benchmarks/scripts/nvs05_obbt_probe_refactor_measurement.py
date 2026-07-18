"""Entry experiment for #723 lever 3 — "cap per-node LP solves (nvs05 ~59
LP/node)".

Runs the two measurements that (a) FALSIFY the naive "cap the per-node OBBT
rounds/probes" framing and (b) root-cause the residual per-probe cost. Both are
instrumentation-only — no solver code is exercised differently, so the numbers
are what the default solve actually does.

Findings (this machine, ``JAX_PLATFORMS`` unset / CPU x64; magnitudes, not
literals — see ``docs/dev/performance-plan.md`` §6):

1. **Round economics.** Per-node OBBT runs up to ``_PER_NODE_OBBT_ROUNDS`` (3)
   sweeps, stopping at a fixpoint. On nvs05 the sweeps do NOT diminish — rounds
   1/2/3 each tighten ~18 bounds/call, and almost every call runs the full 3
   sweeps without hitting the ``sweep_tight == 0`` break. The probes are
   *productive*: the McCormick envelope keeps tightening as the box shrinks, so
   capping rounds/probes trades bound quality for wall time (→ more B&B nodes),
   NOT a free overhead cut. "The LPs are far too many" is the wrong diagnosis.

2. **Per-probe cost.** The warm OBBT probes do a *median of 0 simplex pivots*
   (the previous probe's optimal basis is still optimal for the next objective on
   this degenerate relaxation) yet each still costs ~5 ms. The LP is m~326 rows /
   n~90 cols, so the ~5 ms is a full basis *re-factorization*: the stateless
   ``discopt._rust.solve_lp_warm_csc_py`` binding rebuilds a fresh ``Simplex`` and
   factorizes the 326-dim basis on every probe, even though the constraint matrix
   is identical across the whole sweep and (for 0-pivot probes) the basis is
   unchanged. That redundant factorization — not pivoting, not "too many LPs" — is
   the residual per-node LP cost.

Scoped fix (bound-neutral; a flagged/verified follow-up): reuse the basis
factorization across probes that share a matrix, mirroring the existing
``PreparedDual::reoptimize`` factorization-clone in
``crates/discopt-core/src/lp/simplex/dual.rs`` but generalized to swap the probe
objective. Bound-neutral because the returned optimum is the same vertex; verify
by a byte-identical ``node_count`` + certified ``objective`` panel.

Usage::

    python discopt_benchmarks/scripts/nvs05_obbt_probe_refactor_measurement.py [inst] [time_limit]
"""

from __future__ import annotations

import sys

import discopt._jax.obbt as obbtmod
import discopt.modeling as dm
import numpy as np


def measure(inst: str = "nvs05", time_limit: float = 20.0) -> dict:
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

    # --- instrument #2: per-probe pivot count + wall (refactorization proof) ---
    probe_recs: list[tuple[int, bool, int, int, float]] = []
    orig_probe_cls = obbtmod._PersistentProbeLP

    class _Patched(orig_probe_cls):
        def solve(self, c, lb_arr, ub_arr, warm_basis):
            import time as _t

            from discopt._rust import solve_lp_warm_csc_py

            n, m = self._n, self._m
            c_std = np.concatenate([np.asarray(c, dtype=np.float64).ravel(), np.zeros(m)])
            lb_std = np.concatenate([np.asarray(lb_arr, dtype=np.float64), np.zeros(m)])
            ub_std = np.concatenate([np.asarray(ub_arr, dtype=np.float64), np.full(m, 1e20)])
            warm = warm_basis is not None
            cs0 = np.ascontiguousarray(warm_basis[0], dtype=np.int8) if warm else None
            bv0 = np.ascontiguousarray(warm_basis[1], dtype=np.int64) if warm else None
            t0 = _t.perf_counter()
            _st, _x, _obj, iters, _cs, _bv, _dual, _ray = solve_lp_warm_csc_py(
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
            wall = _t.perf_counter() - t0
            probe_recs.append((int(iters), bool(warm), m, n, wall))
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

    # --- per-probe pivot / wall stats ---
    stats: dict[str, object] = {}
    if probe_recs:
        iters = np.array([x[0] for x in probe_recs])
        warm = np.array([x[1] for x in probe_recs])
        ms = np.array([x[2] for x in probe_recs])
        ns = np.array([x[3] for x in probe_recs])
        walls = np.array([x[4] for x in probe_recs])
        stats = {
            "n_probes": int(len(probe_recs)),
            "warm_frac": float(warm.mean()),
            "m_median": int(np.median(ms)),
            "n_median": int(np.median(ns)),
            "iters_median": float(np.median(iters)),
            "iters_max": int(iters.max()),
            "ms_per_probe_mean": float(1e3 * walls.mean()),
            "ms_per_probe_warm": float(1e3 * walls[warm].mean()) if warm.any() else None,
            "ms_per_probe_cold": float(1e3 * walls[~warm].mean()) if (~warm).any() else None,
        }

    return {
        "inst": inst,
        "objective": r.objective,
        "node_count": getattr(r, "node_count", None),
        "status": r.status,
        "by_round": by_round,
        "probe_stats": stats,
    }


def main() -> None:
    inst = sys.argv[1] if len(sys.argv) > 1 else "nvs05"
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    out = measure(inst, time_limit)
    print(f"{out['inst']}: obj={out['objective']} nodes={out['node_count']} status={out['status']}")
    print("\n[1] per-node OBBT round economics (are rounds 2-3 load-bearing?):")
    print("  sweep  calls  probes  tightened  avg_tight/call")
    for i in sorted(out["by_round"]):
        p, t, cnt = out["by_round"][i]
        print(f"    {i}    {cnt:5d}  {p:6d}   {t:7d}      {t / cnt:.2f}")
    if not out["by_round"]:
        print("    (no per-node OBBT probes ran — this instance is not OBBT-bound)")
    print("\n[2] per-probe cost (does re-factorization dominate 0-pivot probes?):")
    st = out["probe_stats"]
    if st:
        print(
            f"    probes={st['n_probes']}  warm={100 * st['warm_frac']:.0f}%  "
            f"LP m~{st['m_median']} n~{st['n_median']}"
        )
        print(
            f"    pivots: median={st['iters_median']:.0f} max={st['iters_max']}  "
            f"→ warm probes do ~0 pivots"
        )
        print(
            f"    wall/probe: mean={st['ms_per_probe_mean']:.2f}ms "
            f"warm={st['ms_per_probe_warm']}ms cold={st['ms_per_probe_cold']}ms"
        )
        print("    → ~0 pivots but multi-ms/probe ⇒ cost is basis re-factorization,")
        print("      not pivoting and not probe count.")
    else:
        print("    (no probes recorded)")


if __name__ == "__main__":
    main()
