"""Entry experiment for the #723 lever-3 re-scope — OBBT warm-start strategies.

The §6 record (``docs/dev/performance-plan.md``) located nvs05's per-probe OBBT
cost in a heavy tail of warm-start rejections doing ~220 near-cold pivots. The
re-scoped lever was "cut that tail with a better warm-start strategy". This
script runs that entry experiment: capture every real OBBT probe from an nvs05
solve (per-sweep matrix, per-probe objective/box/threaded basis), then REPLAY
the captured stream offline under each candidate strategy and compare wall time
and results against the baseline (today's probe→probe threading).

Strategies measured (results on this machine; magnitudes, not literals):

  two-stage   Stage 1: dual re-opt of the box change under the PREVIOUS
              objective (basis dual-feasible). Stage 2: swap to the new
              objective from stage-1's primal-feasible basis.
              → FALSIFIED: 1.10× *slower* on the tail. Stage 1 is cheap
              (~0.85 ms) — the expensive half is the objective flip itself:
              walking from ``min x_i``'s vertex to ``min x_j``'s vertex costs
              the ~220 pivots no matter how fresh the starting basis.

  self-warm   Diagnostic floor: re-solve each probe from its OWN optimal basis.
              → p50 ~0.86 ms even on the tail (100% optimal): the LP is EASY
              from the right basis — degeneracy-stall is ruled out; basis
              quality is the lever, but the right basis isn't known in advance.

  memory      Per-objective basis memory: warm ``min/max x_j`` from the basis
              that solved the SAME objective previously (process-lifetime).
              → tail-hit p50 drops 7.0 → 1.3 ms (the predicted win), BUT ~1/3
              of hits fail to transfer (matrix coefficients drift across
              rounds → basis rejected → cold fallback) and carry the wall:
              aggregate 1.02–1.07×.

  hybrid      Upper bound of memory + threaded fallback on rejection (what an
              in-engine "try basis A, else basis B" would achieve; +0.3 ms
              detection overhead per hit): min(memory, baseline).
              → 1.34× probe wall ≈ ~8% of nvs05 solve wall. This is the
              CEILING of the warm-start-strategy family.

  filter      Solution-based probe filtering (Gleixner/Berthold/Müller 2017):
              skip a probe when a previously returned optimal point sits at the
              probe's bound (a feasibility witness that no tightening exists).
              → UNSOUND as naively implemented: the solver vertex is feasible
              only to ~1e-6·scale, and on nvs05's ~1e4-scale boxes that slack
              fakes witnesses — the audit found ~44% of "filterable" probes
              would actually have tightened (a silent bound loosening). Even
              optimistically it caps at ~11% of probe wall.

Verdict: the warm-start lever family is measured and capped at ~8% solve wall
(hybrid upper bound) — below the ≥50%-tail-reduction kill criterion set before
the experiment, and below the bar for a correctness-critical engine change.
Recorded as the third lever-3 falsification in performance-plan §6.

Usage::

    python discopt_benchmarks/scripts/nvs05_obbt_warmstart_replay.py [inst] [time_limit]
"""

from __future__ import annotations

import sys
import time

import discopt._jax.obbt as obbtmod
import discopt.modeling as dm
import numpy as np
from discopt._rust import solve_lp_warm_csc_py

REPS = 3  # min-wall over REPS repetitions per replayed solve (noise control)
TAIL_MS = 5.0  # a baseline probe at/above this is "tail" (near-cold re-solve)
DETECT_MS = 0.3  # assumed in-engine cost of trying + rejecting a basis
FILTER_EPS = 1e-7  # matches OBBT's tightening eps


def capture(inst: str, time_limit: float) -> list[dict]:
    """Solve ``inst`` capturing every OBBT probe (matrix, objective, box, warm)."""
    sweeps: list[dict] = []
    call_id = [0]
    orig_tighten = obbtmod.obbt_tighten_root

    def tighten_wrap(*a, **k):
        call_id[0] += 1
        return orig_tighten(*a, **k)

    orig_cls = obbtmod._PersistentProbeLP

    class Cap(orig_cls):
        def __init__(self, A_ub, b_ub, n_total):  # noqa: N803 (parent signature)
            super().__init__(A_ub, b_ub, n_total)
            self._rec = {
                "call": call_id[0],
                "indptr": self._indptr,
                "indices": self._indices,
                "data": self._data,
                "b": self._b,
                "m": self._m,
                "n": self._n,
                "probes": [],
            }
            sweeps.append(self._rec)

        def solve(self, c, lb_arr, ub_arr, warm_basis):
            m = self._m
            if m > 0:
                c_std = np.ascontiguousarray(
                    np.concatenate([np.asarray(c, dtype=np.float64).ravel(), np.zeros(m)])
                )
                lb_std = np.ascontiguousarray(
                    np.concatenate([np.asarray(lb_arr, dtype=np.float64), np.zeros(m)])
                )
                ub_std = np.ascontiguousarray(
                    np.concatenate([np.asarray(ub_arr, dtype=np.float64), np.full(m, 1e20)])
                )
                wb = None
                if warm_basis is not None:
                    wb = (
                        np.ascontiguousarray(warm_basis[0], dtype=np.int8),
                        np.ascontiguousarray(warm_basis[1], dtype=np.int64),
                    )
                self._rec["probes"].append({"c": c_std, "lb": lb_std, "ub": ub_std, "warm": wb})
            return super().solve(c, lb_arr, ub_arr, warm_basis)

    obbtmod.obbt_tighten_root = tighten_wrap
    obbtmod._PersistentProbeLP = Cap
    try:
        dm.from_nl(f"python/tests/data/minlplib_nl/{inst}.nl").solve(time_limit=time_limit)
    finally:
        obbtmod.obbt_tighten_root = orig_tighten
        obbtmod._PersistentProbeLP = orig_cls
    return sweeps


def lp_call(rec: dict, c, lb, ub, warm):
    cs0, bv0 = warm if warm is not None else (None, None)
    t0 = time.perf_counter()
    out = solve_lp_warm_csc_py(
        c,
        rec["m"],
        rec["n"] + rec["m"],
        rec["indptr"],
        rec["indices"],
        rec["data"],
        rec["b"],
        lb,
        ub,
        cs0,
        bv0,
        1e-9,
        100_000,
    )
    return out, (time.perf_counter() - t0) * 1e3


def lp_best(rec: dict, c, lb, ub, warm):
    """Min-wall over REPS repetitions (returns the fastest rep's output)."""
    w, o = np.inf, None
    for _ in range(REPS):
        out, ms = lp_call(rec, c, lb, ub, warm)
        if ms < w:
            w, o = ms, out
    return o, w


def obj_key(c_std, n: int):
    j = int(np.argmax(np.abs(c_std[:n])))
    return (j, 1 if c_std[j] > 0 else -1)


def basis_of(out):
    return (
        np.ascontiguousarray(out[4], dtype=np.int8),
        np.ascontiguousarray(out[5], dtype=np.int64),
    )


def replay(sweeps: list[dict]) -> None:
    memory: dict = {}  # obj_key -> (basis, m): process-lifetime per-objective memory
    rows = []
    n_filt = n_filt_unsound = 0
    filt_wall = 0.0

    for rec in sweeps:
        prev_c = None
        witnesses: list[np.ndarray] = []  # per-sweep feasible points (filter study)
        for p in rec["probes"]:
            if p["warm"] is None or prev_c is None:
                # First probes have no threaded basis / no prev objective — every
                # strategy degenerates to the baseline there; skip for comparison.
                base_out, base_w = lp_best(rec, p["c"], p["lb"], p["ub"], p["warm"])
                if base_out[0] == "optimal":
                    witnesses.append(np.asarray(base_out[1], dtype=np.float64))
                    memory[obj_key(p["c"], rec["n"])] = (basis_of(base_out), rec["m"])
                prev_c = p["c"]
                continue

            # ---- baseline: today's probe->probe threading -----------------------
            base_out, base_w = lp_best(rec, p["c"], p["lb"], p["ub"], p["warm"])

            # ---- two-stage: box change under prev objective, then swap ----------
            s1_out, s1_w = lp_best(rec, prev_c, p["lb"], p["ub"], p["warm"])
            if s1_out[0] == "optimal":
                s2_out, s2_w = lp_best(rec, p["c"], p["lb"], p["ub"], basis_of(s1_out))
                two_w = s1_w + s2_w
            else:
                two_w = s1_w + base_w  # stage-1 failure degrades to baseline

            # ---- self-warm floor: probe's own optimal basis ---------------------
            self_w = np.nan
            if base_out[0] == "optimal":
                _, self_w = lp_best(rec, p["c"], p["lb"], p["ub"], basis_of(base_out))

            # ---- per-objective memory (+ hybrid upper bound) --------------------
            k = obj_key(p["c"], rec["n"])
            mem = memory.get(k)
            hit = mem is not None and mem[1] == rec["m"]
            if hit:
                mem_out, mem_w = lp_best(rec, p["c"], p["lb"], p["ub"], mem[0])
                hyb_w = min(mem_w, base_w + DETECT_MS) + DETECT_MS
                upd = mem_out if mem_w <= base_w else base_out
            else:
                mem_w, hyb_w, upd = base_w, base_w, base_out
            if upd[0] == "optimal":
                memory[k] = (basis_of(upd), rec["m"])

            # ---- filter study: would a witness have (unsoundly) skipped this? ---
            j, sense = k
            filtered = False
            for x in witnesses:
                if np.any(x < p["lb"] - 1e-9) or np.any(x > p["ub"] + 1e-9):
                    continue  # witness left the tightened box — no longer valid
                if sense == 1 and x[j] <= p["lb"][j] + FILTER_EPS:
                    filtered = True
                    break
                if sense == -1 and x[j] >= p["ub"][j] - FILTER_EPS:
                    filtered = True
                    break
            if filtered:
                n_filt += 1
                filt_wall += base_w
                if base_out[0] == "optimal":
                    opt = float(base_out[2])
                    would_tighten = (sense == 1 and opt > p["lb"][j] + FILTER_EPS + 1e-9) or (
                        sense == -1 and opt < p["ub"][j] - FILTER_EPS - 1e-9
                    )
                    n_filt_unsound += bool(would_tighten)
            if base_out[0] == "optimal":
                witnesses.append(np.asarray(base_out[1], dtype=np.float64))

            rows.append(
                {
                    "base": base_w,
                    "two": two_w,
                    "selfw": self_w,
                    "mem": mem_w,
                    "hyb": hyb_w,
                    "hit": hit,
                }
            )
            prev_c = p["c"]

    b = np.array([r["base"] for r in rows])
    tail = b >= TAIL_MS

    def agg(name, arr):
        arr = np.asarray(arr)
        print(
            f"  {name:10s} all {arr.sum():7.0f} ms ({b.sum() / arr.sum():4.2f}x) | "
            f"tail {arr[tail].sum():7.0f} ms ({b[tail].sum() / max(arr[tail].sum(), 1e-9):4.2f}x)"
        )

    print(
        f"\nreplayed={len(rows)} probes, tail(base>={TAIL_MS:.0f}ms)={tail.sum()} "
        f"({100 * tail.mean():.0f}%), memory hits={100 * np.mean([r['hit'] for r in rows]):.0f}%"
    )
    print(f"  {'baseline':10s} all {b.sum():7.0f} ms          | tail {b[tail].sum():7.0f} ms")
    agg("two-stage", [r["two"] for r in rows])
    agg("memory", [r["mem"] for r in rows])
    agg("hybrid-ub", [r["hyb"] for r in rows])
    sw = np.array([r["selfw"] for r in rows])
    ok = ~np.isnan(sw)
    print(
        f"  {'self-warm':10s} p50={np.percentile(sw[ok], 50):.2f} ms "
        f"(tail p50={np.percentile(sw[ok & tail], 50):.2f}) — the unreachable floor"
    )
    print(
        f"\nfilter study: {n_filt} probes 'filterable' "
        f"({100 * filt_wall / b.sum():.0f}% of baseline wall) but {n_filt_unsound} of them "
        f"({100 * n_filt_unsound / max(n_filt, 1):.0f}%) WOULD have tightened — naive "
        f"filtering is unsound in floating point"
    )
    print(
        "\nverdict: best implementable = hybrid upper bound "
        f"({b.sum() / np.array([r['hyb'] for r in rows]).sum():.2f}x probe wall); "
        "below the >=50%-tail kill criterion → warm-start lever falsified."
    )


def main() -> None:
    inst = sys.argv[1] if len(sys.argv) > 1 else "nvs05"
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    sweeps = capture(inst, tl)
    print(f"capture: {sum(len(s['probes']) for s in sweeps)} probes in {len(sweeps)} sweeps")
    replay(sweeps)


if __name__ == "__main__":
    main()
