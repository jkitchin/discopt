"""#928 re-panel: the three coupled budget flags with the round-cut-short floor.

Same three arms and the same metric set as
``issue966_coupled_graduation_panel.py`` (whose ``soundness``/``compare``/
``run_one`` this imports rather than re-deriving) — ``base`` / ``seam`` / ``cand``,
interleaved per instance, every flag named explicitly in every arm — with two
differences that the environment and the last panel's failure force:

1. **Multi-rep in one run.** §14b's verdict turned on a delta that flipped sign
   between reps, so a single run cannot settle the net-positive bar. Reps are
   interleaved at the instance level with the arms, and the per-rep deltas are
   reported with their spread (CLAUDE.md §9).

2. **The oracle is stated at its real strength, not assumed.** ``minlplib.solu``
   (the library-wide oracle §14b-qual switched to) is NOT reachable from this
   container — the network policy denies minlplib.org — so the only oracle here is
   ``python/tests/data/known_optima.toml``, which covers 1 of the 19 binding
   instances. Rather than report a soundness result that rests on one instance,
   this panel adds a **cross-arm primal ceiling**: for a MINIMIZE instance every
   arm's verified incumbent is an upper bound on the optimum, so the best
   incumbent found by ANY arm in ANY rep is a valid ceiling for every other arm's
   bound. Coverage of both oracles is counted and printed; a run that makes zero
   bound-vs-ceiling comparisons exits non-zero (§6).

Usage:
    python -u discopt_benchmarks/scripts/issue928_round_floor_panel.py \
        --budget 20 --reps 3 --out discopt_benchmarks/results/issue928_floor20.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from issue966_coupled_graduation_panel import (  # noqa: E402
    ARMS,
    CORPUS,
    compare,
    loadavg,
    run_one,
    soundness,
)

BINDING = (
    "4stufen,bchoco06,bchoco07,bchoco08,beuster,casctanks,clay0303hfsg,contvar,hda,"
    "heatexch_gen1,heatexch_gen2,heatexch_gen3,nvs05,syn05hfsg,tls2,tspn05,tspn08,"
    "tspn10,tspn12"
)


def curated_optimum(name: str):
    """``known_optima.toml`` only — the .solu oracle is unreachable here.

    Only ``KeyError`` (the genuine "not in the registry" outcome) is swallowed;
    a broken registry crashes, per CLAUDE.md §7.
    """
    sys.path.insert(0, str(ROOT / "python/tests"))
    from _optima import known_optimum  # type: ignore

    try:
        return known_optimum(name)
    except KeyError:
        return None


def primal_ceilings(all_cells: list[dict]) -> dict[str, float]:
    """Best verified incumbent per instance over every arm and rep.

    A feasible point's objective bounds the optimum from above (min sense), so
    this is a sound ceiling for every arm's dual bound on that instance — and it
    is measured in this very panel rather than taken on faith. Cells whose
    incumbent failed verification are excluded.
    """
    best: dict[str, float] = {}
    for c in all_cells:
        for key, _l, _e in ARMS:
            rec = c[key]
            if rec["objective"] is None or rec["incumbent_verification_failed"]:
                continue
            v = float(rec["objective"])
            cur = best.get(c["instance"])
            better = v < cur if cur is not None else True
            if rec["sense"] == "max":
                better = v > cur if cur is not None else True
            best[c["instance"]] = v if better else cur
    return best


def ceiling_violations(cells: list[dict], ceilings: dict[str, float]) -> tuple[int, list[str]]:
    """Counted bound-vs-ceiling control. Returns (comparisons, violations)."""
    cmps, bad = 0, []
    for c in cells:
        ceil = ceilings.get(c["instance"])
        if ceil is None:
            continue
        for key, _l, _e in ARMS:
            rec = c[key]
            if rec["bound"] is None:
                continue
            cmps += 1
            over = (
                rec["bound"] < ceil - 1e-4
                if rec["sense"] == "max"
                else rec["bound"] > ceil + 1e-4
            )
            if over:
                bad.append(f"rep{c['rep']} {c['instance']}:{key}: bound {rec['bound']} vs {ceil}")
    return cmps, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--instances", default=BINDING)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [s for s in args.instances.split(",") if s]
    all_cells: list[dict] = []
    t0 = time.perf_counter()
    load_start = loadavg()

    for rep in range(1, args.reps + 1):
        for idx, name in enumerate(names, 1):
            cell = {
                "rep": rep,
                "instance": name,
                "budget": args.budget,
                "reference_optimum": curated_optimum(name),
            }
            for key, _label, env in ARMS:
                cell[key] = run_one(CORPUS / f"{name}.nl", args.budget, env)
            all_cells.append(cell)
            parts = " | ".join(
                f"{key} {cell[key]['wall']:5.1f}s cert={int(bool(cell[key]['gap_certified']))} "
                f"b={cell[key]['bound']}"
                for key, _l, _e in ARMS
            )
            print(f"[rep{rep} {idx}/{len(names)}] {name:16s} {parts}", flush=True)

    ceilings = primal_ceilings(all_cells)
    per_rep = []
    for rep in range(1, args.reps + 1):
        cells = [c for c in all_cells if c["rep"] == rep]
        snd = soundness(cells)
        pairs = {
            "cand_vs_base": compare(cells, "base", "cand"),
            "cand_vs_seam": compare(cells, "seam", "cand"),
            "seam_vs_base": compare(cells, "base", "seam"),
        }
        cmps, bad = ceiling_violations(cells, ceilings)
        grad = pairs["cand_vs_base"]
        cert_clean = not (
            snd["unsound"]
            or bad
            or snd["incumbent_verification_failed"]
            or grad["cert_regressions"]
            or grad["lost_incumbents"]
            or grad["lost_bound"]
        )
        per_rep.append(
            {
                "rep": rep,
                **snd,
                "ceiling_comparisons_executed": cmps,
                "ceiling_violations": bad,
                "pairs": pairs,
                "cert_clean": cert_clean,
            }
        )
        print(
            f"\nrep{rep}: CERT_CLEAN={cert_clean} "
            f"cand-base overrun {grad['overrun_delta_s']}s  "
            f"lost_bound={grad['lost_bound']} cert_regressions={grad['cert_regressions']} "
            f"unsound={snd['unsound']} ceiling_violations={bad}",
            flush=True,
        )

    deltas = [r["pairs"]["cand_vs_base"]["overrun_delta_s"] for r in per_rep]
    seam_deltas = [r["pairs"]["seam_vs_base"]["overrun_delta_s"] for r in per_rep]
    total_cmps = sum(r["ceiling_comparisons_executed"] for r in per_rep)
    summary = {
        "budget": args.budget,
        "reps": args.reps,
        "instances": names,
        "oracle": "known_optima.toml + cross-arm primal ceiling (minlplib.solu unreachable)",
        "cand_vs_base_overrun_deltas_s": deltas,
        "seam_vs_base_overrun_deltas_s": seam_deltas,
        "cert_clean_all_reps": all(r["cert_clean"] for r in per_rep),
        "ceiling_comparisons_executed": total_cmps,
        "loadavg_start": [round(x, 2) for x in load_start],
        "loadavg_end": [round(x, 2) for x in loadavg()],
        "panel_wall_s": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "reps": per_rep, "cells": all_cells}, indent=2))

    print()
    print(json.dumps(summary, indent=2))
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    print(f"\nCAND_VS_BASE_OVERRUN_DELTA_S={mean:.1f} +/- {sd:.1f}  (per rep: {deltas})")
    print(f"CERT_CLEAN_ALL_REPS={summary['cert_clean_all_reps']}")
    print(f"CEILING_COMPARISONS_EXECUTED={total_cmps}")
    print(f"COMPARISONS_EXECUTED={len(all_cells)}")
    return 0 if (total_cmps and all_cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
