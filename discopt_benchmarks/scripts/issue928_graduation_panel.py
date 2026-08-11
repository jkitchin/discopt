"""#928 graduation panel on the MERGED tree, with the compile gate as its own arm.

§14d (the round-cut-short floor) and §14e (a short-granted round yields instead of
skipping) were each panelled against ``aca13dd`` — separately. The merged tree has
never been panelled, and §14e's closing paragraph says so explicitly: *"the
graduation panel owed to #966 item 3 must now be run once, on the merged tree"*.
This is that run, with one change of shape forced by a measurement.

**Why four arms, not three.** Both prior panels failed ``CERT_CLEAN`` on the same
class of item — an incumbent the ``base`` arm reports and the all-flags arm does
not (tspn12; sporadically tspn10/tls2). The three-arm layout could not attribute it,
because ``seam`` sets ``DISCOPT_NODE_ROUND_BUDGET`` and ``DISCOPT_HESS_COMPILE_GATE``
together. Attribution has now been measured directly
(``scratchpad/issue928_incumbent_attribution.py``): the round budget keeps the
incumbent, the **compile gate** is what drops it. So the arms here separate the two,
and the graduation question for THIS issue — ``DISCOPT_LP_WARM_DEADLINE`` — is asked
of an arm that does not carry the unrelated compile gate:

    A  base   W0 R0 H0   today's default, the control
    B  warm   W1 R0 H0   #928's flag ALONE — the issue's own graduation question
    C  wr     W1 R1 H0   #928 + the round budget (the seam that made #928 pay off)
    D  cand   W1 R1 H1   all three — what §14b/§14d/§14e scored

``warm``/``wr`` vs ``base`` are the #928 verdict; ``cand`` vs ``wr`` isolates the
compile gate's own effect (#966's to graduate or not).

Everything else is the §14d panel's: the same 19 binding instances, the same
worker, the same ``soundness``/``compare``, multi-rep in one run with the arms and
reps interleaved per instance (CLAUDE.md §9), and the cross-arm primal ceiling
(``minlplib.solu`` is unreachable from this container, so the curated oracle alone
would be 1/19 — §14b-qual's rule). Counted per §6; a run that makes zero
bound-vs-ceiling comparisons exits non-zero.

    python -u discopt_benchmarks/scripts/issue928_graduation_panel.py \
        --budget 20 --reps 3 --out discopt_benchmarks/results/issue928_grad20.json
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

import issue966_coupled_graduation_panel as _coupled  # noqa: E402
from issue928_round_floor_panel import (  # noqa: E402
    BINDING,
    ceiling_violations,
    curated_optimum,
    primal_ceilings,
)
from issue966_coupled_graduation_panel import (  # noqa: E402
    CORPUS,
    compare,
    loadavg,
    run_one,
    soundness,
)

WARM = "DISCOPT_LP_WARM_DEADLINE"
ROUND = "DISCOPT_NODE_ROUND_BUDGET"
HESS = "DISCOPT_HESS_COMPILE_GATE"

# (key, label, env). Every flag named in every arm: an arm must never inherit.
ARMS = (
    ("base", "all OFF (today's default)", {WARM: "0", ROUND: "0", HESS: "0"}),
    ("warm", "#928 alone", {WARM: "1", ROUND: "0", HESS: "0"}),
    ("wr", "#928 + round budget", {WARM: "1", ROUND: "1", HESS: "0"}),
    ("cand", "all three", {WARM: "1", ROUND: "1", HESS: "1"}),
)

# ``soundness`` / ``primal_ceilings`` / ``ceiling_violations`` iterate the ARMS of
# the module they were defined in. Rebind it there so the imported scorers see THIS
# panel's four arms instead of the three-arm layout — an unbound rebind would leave
# them scoring a ``seam`` key that does not exist here (KeyError, not a silent skip,
# but the rebind is what makes the reuse honest).
_coupled.ARMS = ARMS
sys.modules["issue928_round_floor_panel"].ARMS = ARMS  # type: ignore[attr-defined]


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
                f"{key} {cell[key]['wall']:5.1f}s inc={cell[key]['objective']} "
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
            "warm_vs_base": compare(cells, "base", "warm"),
            "wr_vs_base": compare(cells, "base", "wr"),
            "cand_vs_base": compare(cells, "base", "cand"),
            "cand_vs_wr": compare(cells, "wr", "cand"),
        }
        cmps, bad = ceiling_violations(cells, ceilings)

        def _clean(pair: dict, _snd=snd, _bad=bad) -> bool:
            return not (
                _snd["unsound"]
                or _bad
                or _snd["incumbent_verification_failed"]
                or pair["cert_regressions"]
                or pair["lost_incumbents"]
                or pair["lost_bound"]
            )

        per_rep.append(
            {
                "rep": rep,
                **snd,
                "ceiling_comparisons_executed": cmps,
                "ceiling_violations": bad,
                "pairs": pairs,
                "cert_clean": {k: _clean(v) for k, v in pairs.items()},
            }
        )
        for k, v in pairs.items():
            print(
                f"rep{rep} {k:14s} CERT_CLEAN={_clean(v)} overrun_delta={v['overrun_delta_s']}s "
                f"lost_bound={v['lost_bound']} lost_inc={v['lost_incumbents']} "
                f"cert_reg={v['cert_regressions']}",
                flush=True,
            )
        print(f"rep{rep} unsound={snd['unsound']} ceiling_violations={bad}", flush=True)

    def _deltas(pair_key: str) -> list[float]:
        return [r["pairs"][pair_key]["overrun_delta_s"] for r in per_rep]

    total_cmps = sum(r["ceiling_comparisons_executed"] for r in per_rep)
    summary = {
        "budget": args.budget,
        "reps": args.reps,
        "instances": names,
        "arms": {k: env for k, _l, env in ARMS},
        "oracle": "known_optima.toml + cross-arm primal ceiling (minlplib.solu unreachable)",
        "overrun_deltas_s": {k: _deltas(k) for k in per_rep[0]["pairs"]},
        "cert_clean_all_reps": {
            k: all(r["cert_clean"][k] for r in per_rep) for k in per_rep[0]["pairs"]
        },
        "ceiling_comparisons_executed": total_cmps,
        "loadavg_start": [round(x, 2) for x in load_start],
        "loadavg_end": [round(x, 2) for x in loadavg()],
        "panel_wall_s": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "reps": per_rep, "cells": all_cells}, indent=2)
    )

    print()
    print(json.dumps(summary, indent=2))
    for k in per_rep[0]["pairs"]:
        d = _deltas(k)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        print(f"{k.upper()}_OVERRUN_DELTA_S={statistics.fmean(d):.1f} +/- {sd:.1f}  per rep: {d}")
    print(f"CERT_CLEAN_ALL_REPS={summary['cert_clean_all_reps']}")
    print(f"CEILING_COMPARISONS_EXECUTED={total_cmps}")
    print(f"COMPARISONS_EXECUTED={len(all_cells)}")
    return 0 if (total_cmps and all_cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
