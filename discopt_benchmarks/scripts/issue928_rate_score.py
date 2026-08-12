"""Score a multi-rep #928 panel by PER-CELL RATE instead of per-rep one-shot.

Why this exists. The §5 bar-1 failures that kept ``DISCOPT_LP_WARM_DEADLINE``
default-OFF across four panels were, on inspection, cells that *neither* arm
holds reliably: nvs05, tls2, syn05hfsg and casctanks each certify (or find an
incumbent) in some reps and not others, in the base arm as much as in the
candidate. One-shot scoring turns such a cell into a ``cert_regression`` whenever
the coin lands base-heads/candidate-tails, and into a ``cert_gain`` the other way
-- the metric cannot distinguish a flag effect from a race. Two panels recorded
one direction, a repeat probe recorded the other (tls2: base 1/5 vs warm 4/5).

So: run N reps and compare RATES. A regression must clear a pre-registered
margin to count. The margin is stated in ``--margin`` (default 3 of 5) and is
fixed before the run, not chosen after seeing the results.

What is NOT rate-scored, because soundness has no slack (CLAUDE.md §1): a bound
past the oracle ceiling, a bound crossing the arm's own incumbent, or an
incumbent that failed verification. One occurrence in one rep disqualifies.

Ends with the executed comparison count and exits non-zero if it compared
nothing (§6): "no regressions" over zero comparisons is not a result.

    python -u discopt_benchmarks/scripts/issue928_rate_score.py \
        --control base --candidate warm \
        discopt_benchmarks/results/issue928_warmalone20_rep*.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (m, math.sqrt(var))


def _tighter(sense: str, a: float, b: float) -> bool:
    """Is bound ``b`` tighter than bound ``a`` for this sense?"""
    return (b > a) if sense == "min" else (b < a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+")
    ap.add_argument("--control", default="base")
    ap.add_argument("--candidate", default="warm")
    ap.add_argument(
        "--margin",
        type=int,
        default=3,
        help="reps by which the control must beat the candidate on a cell before it "
        "counts as a regression. Pre-registered; do not tune after seeing results.",
    )
    args = ap.parse_args()
    ctrl, cand = args.control, args.candidate

    runs = [json.loads(Path(p).read_text()) for p in sorted(args.artifacts)]
    n_reps = len(runs)

    # --- hard soundness, any single rep disqualifies -------------------------
    unsound: list[str] = []
    verif_failed: list[str] = []
    oracle_cmps = 0
    for i, r in enumerate(runs, 1):
        s = r["summary"]
        unsound += [f"rep{i}:{x}" for x in s.get("unsound", [])]
        verif_failed += [f"rep{i}:{x}" for x in s.get("incumbent_verification_failed", [])]
        oracle_cmps += s.get("oracle_comparisons_executed") or 0

    # --- per-cell tallies ----------------------------------------------------
    names = [c["instance"] for c in runs[0]["cells"]]
    tally: dict[str, dict] = {
        n: {
            "cert": {ctrl: 0, cand: 0},
            "incumbent": {ctrl: 0, cand: 0},
            "bound": {ctrl: 0, cand: 0},
            "tighter": 0,
            "looser": 0,
            "reps": 0,
        }
        for n in names
    }
    cells_compared = 0
    for r in runs:
        for c in r["cells"]:
            n = c["instance"]
            if n not in tally:  # an artifact with a different instance set
                raise SystemExit(f"artifact instance sets differ: {n} not in rep 1")
            a, b = c[ctrl], c[cand]
            t = tally[n]
            t["reps"] += 1
            cells_compared += 1
            for key, rec in ((ctrl, a), (cand, b)):
                t["cert"][key] += int(bool(rec["gap_certified"]))
                t["incumbent"][key] += int(rec["objective"] is not None)
                t["bound"][key] += int(rec["bound"] is not None)
            if a["bound"] is not None and b["bound"] is not None:
                rel = abs(b["bound"] - a["bound"]) / max(1.0, abs(a["bound"]))
                if rel > 1e-9:
                    if _tighter(a["sense"], a["bound"], b["bound"]):
                        t["tighter"] += 1
                    else:
                        t["looser"] += 1

    def _regressions(field: str) -> list[str]:
        out = []
        for n, t in tally.items():
            d = t[field][ctrl] - t[field][cand]
            if d >= args.margin:
                out.append(f"{n} ({t[field][ctrl]}/{t['reps']} -> {t[field][cand]}/{t['reps']})")
        return out

    def _gains(field: str) -> list[str]:
        out = []
        for n, t in tally.items():
            d = t[field][cand] - t[field][ctrl]
            if d >= args.margin:
                out.append(f"{n} ({t[field][ctrl]}/{t['reps']} -> {t[field][cand]}/{t['reps']})")
        return out

    def _noise(field: str) -> list[str]:
        out = []
        for n, t in tally.items():
            d = abs(t[field][ctrl] - t[field][cand])
            if 0 < d < args.margin:
                out.append(f"{n} ({t[field][ctrl]}/{t['reps']} vs {t[field][cand]}/{t['reps']})")
        return out

    cert_reg, inc_reg, bound_reg = (_regressions(f) for f in ("cert", "incumbent", "bound"))
    cert_gain, inc_gain, bound_gain = (_gains(f) for f in ("cert", "incumbent", "bound"))

    # --- bar 2 axes ----------------------------------------------------------
    deltas, over_ctrl, over_cand = [], [], []
    nodes = {ctrl: 0, cand: 0}
    for r in runs:
        pair = next(p for p in r["summary"]["pairs"] if p["pair"] == f"{cand} vs {ctrl}")
        deltas.append(pair["overrun_delta_s"])
        over_ctrl.append(pair["total_overrun_s"][ctrl])
        over_cand.append(pair["total_overrun_s"][cand])
        nodes[ctrl] += pair["node_count_total"][ctrl]
        nodes[cand] += pair["node_count_total"][cand]
    d_mean, d_sd = _mean_sd(deltas)

    tighter_cells = [n for n, t in tally.items() if t["tighter"] >= args.margin]
    looser_cells = [n for n, t in tally.items() if t["looser"] >= args.margin]

    bar1 = not (unsound or verif_failed or cert_reg or inc_reg or bound_reg)

    report = {
        "reps": n_reps,
        "instances": len(names),
        "control": ctrl,
        "candidate": cand,
        "margin_reps": args.margin,
        "hard_unsound": unsound,
        "hard_incumbent_verification_failed": verif_failed,
        "cert_regressions": cert_reg,
        "cert_gains": cert_gain,
        "cert_noise_cells": _noise("cert"),
        "incumbent_regressions": inc_reg,
        "incumbent_gains": inc_gain,
        "incumbent_noise_cells": _noise("incumbent"),
        "bound_lost": bound_reg,
        "bound_gained": bound_gain,
        "bound_noise_cells": _noise("bound"),
        "overrun_delta_s_per_rep": deltas,
        "overrun_delta_s_mean": round(d_mean, 2),
        "overrun_delta_s_sd": round(d_sd, 2),
        "total_overrun_s": {ctrl: over_ctrl, cand: over_cand},
        "node_count_total": nodes,
        "bound_tighter_cells": sorted(tighter_cells),
        "bound_looser_cells": sorted(looser_cells),
    }
    print(json.dumps(report, indent=2))
    print(f"\nBAR1_CERT_CLEAN={bar1}")
    print(f"OVERRUN_DELTA_S={d_mean:.2f} +/- {d_sd:.2f}")
    print(f"BOUND_LEDGER tighter={len(tighter_cells)} looser={len(looser_cells)}")
    print(f"CELLS_COMPARED={cells_compared}")
    print(f"ORACLE_COMPARISONS_EXECUTED={oracle_cmps}")
    if cells_compared == 0:
        print("SCORER COMPARED NOTHING", flush=True)
        return 1
    if oracle_cmps == 0:
        # Same rule the panel itself enforces: no oracle reached means the
        # soundness line is a formatting artifact, not a measurement.
        print("NO ORACLE COMPARISONS -- soundness unproven", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
