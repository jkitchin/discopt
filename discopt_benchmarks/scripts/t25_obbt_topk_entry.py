"""cert:T2.5 ENTRY EXPERIMENT — scored top-k per-node OBBT de-gate ON vs OFF.

Measures, per instance (casctanks, tanksize, nvs05 + two mid-size spatial NLPs):
whether per-node OBBT now RUNS on n>100 models with the ``DISCOPT_OBBT_TOPK``
flag, and whether the certified dual bound climbs / node count drops. Distinguishes
three outcomes the task asks us to separate:

  * OBBT-ran?     -> solver_stats["reduce/obbt"] > 0 (per-node OBBT consumed wall)
  * bound moved?  -> certified bound / root bound ON vs OFF
  * net wall win? -> wall ON vs OFF (noisy on a shared box; reported, not gated)

Run in the isolated worktree venv:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      python discopt_benchmarks/scripts/t25_obbt_topk_entry.py
"""

from __future__ import annotations

import os
import os.path as osp
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm

SNAP = osp.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
TL = float(os.environ.get("T25_TL", "60"))

# (stem, oracle) — oracle is =opt= where known, else =best= (primal) for a
# sanity floor; the soundness check that matters is "certified bound <= oracle".
INSTANCES = [
    ("casctanks", 9.1634793880),
    ("tanksize", 1.2686437540),
    ("nvs05", 5.4709341080),
    ("ex8_3_7", -1.2326191910),
    ("ex8_3_13", -43.0894781500),
]


def _solve(stem: str, topk_on: bool):
    if topk_on:
        os.environ["DISCOPT_OBBT_TOPK"] = "1"
    else:
        os.environ.pop("DISCOPT_OBBT_TOPK", None)
    m = dm.from_nl(osp.join(SNAP, f"{stem}.nl"))
    nvars = sum(v.size for v in m._variables)
    t0 = time.perf_counter()
    r = m.solve(time_limit=TL)
    wall = time.perf_counter() - t0
    stats = getattr(r, "solver_stats", None) or {}
    obbt_wall = float(stats.get("reduce/obbt", 0.0))
    return {
        "nvars": nvars,
        "status": str(getattr(r, "status", "?")),
        "obj": getattr(r, "objective", None),
        "bound": getattr(r, "bound", None),
        "root_bound": getattr(r, "root_bound", None),
        "nodes": getattr(r, "node_count", None),
        "obbt_ran": obbt_wall > 0.0,
        "obbt_wall": obbt_wall,
        "wall": wall,
    }


def _fmt(v, w=12, p=6):
    if v is None:
        return f"{'—':>{w}}"
    if isinstance(v, bool):
        return f"{('YES' if v else 'no'):>{w}}"
    if isinstance(v, float):
        return f"{v:>{w}.{p}f}"
    return f"{str(v):>{w}}"


def main() -> int:
    print(f"T2.5 entry experiment — TL={TL}s, top_k per-node OBBT ON vs OFF\n")
    rows = []
    for stem, oracle in INSTANCES:
        if not osp.isfile(osp.join(SNAP, f"{stem}.nl")):
            print(f"[skip] {stem}: not in snapshot")
            continue
        off = _solve(stem, topk_on=False)
        on = _solve(stem, topk_on=True)
        rows.append((stem, oracle, off, on))
        print(f"=== {stem} (n={off['nvars']}, oracle={oracle}) ===")
        print(f"  {'':16s}{'OFF':>14s}{'ON':>14s}")
        print(f"  {'obbt_ran':16s}{_fmt(off['obbt_ran'], 14)}{_fmt(on['obbt_ran'], 14)}")
        print(
            f"  {'obbt_wall(s)':16s}{_fmt(off['obbt_wall'], 14, 3)}{_fmt(on['obbt_wall'], 14, 3)}"
        )
        print(f"  {'cert_bound':16s}{_fmt(off['bound'], 14)}{_fmt(on['bound'], 14)}")
        print(f"  {'root_bound':16s}{_fmt(off['root_bound'], 14)}{_fmt(on['root_bound'], 14)}")
        print(f"  {'obj':16s}{_fmt(off['obj'], 14)}{_fmt(on['obj'], 14)}")
        print(f"  {'nodes':16s}{_fmt(off['nodes'], 14)}{_fmt(on['nodes'], 14)}")
        print(f"  {'status':16s}{_fmt(off['status'], 14)}{_fmt(on['status'], 14)}")
        print(f"  {'wall(s)':16s}{_fmt(off['wall'], 14, 2)}{_fmt(on['wall'], 14, 2)}")
        print()

    # Soundness: certified bound must never cross the oracle (min sense: bound <= oracle+tol).
    print("=== SOUNDNESS (certified bound must be <= oracle for min sense) ===")
    bad = 0
    for stem, oracle, off, on in rows:
        for label, res in (("OFF", off), ("ON", on)):
            b = res["bound"]
            if b is not None and b > oracle + 1e-4 * (1 + abs(oracle)):
                print(f"  VIOLATION {stem} {label}: bound {b} > oracle {oracle}")
                bad += 1
    print(f"  bound-crossing violations: {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
