"""Graduation panel for ``DISCOPT_LP_WARM_DEADLINE`` (warm pure-LP node deadline).

``MilpRelaxationModel.solve`` takes a ``time_limit`` and its default
``backend="simplex"`` pure-LP fast path dropped it: the warm attempts took no
deadline and ``lp_bindings.rs`` hardcoded ``SimplexOptions { deadline: None }``. The
flag threads one shared budget through that path. Cutting an LP short changes the
bound it returns, so this is bound-CHANGING and needs the §5 differential panel.

The headline metric is **budget compliance** (that is what the change is for), but
the gate is soundness first and bound quality second: on nvs24 the flag halves the
overrun at small budgets for an identical bound, yet at 60 s returns a bound ~2.3x
looser. The panel decides whether that trade generalises.

Runs OFF and ON back-to-back per instance (interleaved, CLAUDE.md §9) in isolated
subprocesses. Ends with an executed-comparison count; exits non-zero if it compared
nothing (§6).

    python discopt_benchmarks/scripts/issue917_lp_warm_deadline_panel.py \
        --budget 15 --out discopt_benchmarks/results/issue917_lp_warm_deadline.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "python/tests/data/minlplib_nl"
WORKER = Path(__file__).with_name("issue917_reserve_extension_worker.py")
FLAG = "DISCOPT_LP_WARM_DEADLINE"
TOL = 1e-6


def reference_optimum(name: str):
    sys.path.insert(0, str(ROOT / "python/tests"))
    try:
        from _optima import known_optimum  # type: ignore

        return known_optimum(name)
    except Exception:
        return None


def run_one(nl: Path, budget: float, flag: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-u", str(WORKER), str(nl), str(budget)],
        capture_output=True,
        text=True,
        env={**os.environ, FLAG: flag},
        timeout=40 * budget + 900,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-6000:] + "\n")
        raise SystemExit(f"worker failed on {nl.stem} flag={flag}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--instances", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = (
        [s for s in args.instances.split(",") if s]
        if args.instances
        else sorted(p.stem for p in CORPUS.glob("*.nl"))
    )
    budget = args.budget

    cells, compared = [], 0
    t_panel = time.perf_counter()
    for name in names:
        nl = CORPUS / f"{name}.nl"
        off = run_one(nl, budget, "0")
        on = run_one(nl, budget, "1")
        compared += 1
        cells.append(
            {
                "instance": name,
                "budget": budget,
                "reference_optimum": reference_optimum(name),
                "off": off,
                "on": on,
            }
        )
        print(
            f"{name:22s} OFF wall={off['wall']:7.1f} ({off['wall'] / budget:5.2f}x) "
            f"cert={int(bool(off['gap_certified']))} obj={off['objective']} bound={off['bound']} | "
            f"ON wall={on['wall']:7.1f} ({on['wall'] / budget:5.2f}x) "
            f"cert={int(bool(on['gap_certified']))} obj={on['objective']} bound={on['bound']}",
            flush=True,
        )

    cert_gains, cert_regressions = [], []
    lost_incumbents, gained_incumbents = [], []
    worse_obj, better_obj = [], []
    looser_bound, tighter_bound = [], []
    lost_bound, gained_bound = [], []
    unsound, verification_failed = [], []
    over_off = over_on = 0
    overrun_s_off = overrun_s_on = 0.0

    for c in cells:
        off, on, name, b = c["off"], c["on"], c["instance"], c["budget"]
        sense = on["sense"]
        for arm, rec in (("off", off), ("on", on)):
            if rec["wall"] > b * 1.05:
                if arm == "off":
                    over_off += 1
                else:
                    over_on += 1
            ov = max(0.0, rec["wall"] - b)
            if arm == "off":
                overrun_s_off += ov
            else:
                overrun_s_on += ov
            if rec["incumbent_verification_failed"]:
                verification_failed.append(f"{name}:{arm}")
            if rec["bound"] is not None and rec["objective"] is not None:
                bad = (
                    rec["bound"] < rec["objective"] - 1e-4
                    if sense == "max"
                    else rec["bound"] > rec["objective"] + 1e-4
                )
                if bad:
                    unsound.append(f"{name}:{arm}:bound-crosses-incumbent")
            if rec["bound"] is not None and c["reference_optimum"] is not None:
                ref = float(c["reference_optimum"])
                bad = rec["bound"] < ref - 1e-4 if sense == "max" else rec["bound"] > ref + 1e-4
                if bad:
                    unsound.append(f"{name}:{arm}:bound-past-oracle")

        if bool(on["gap_certified"]) and not bool(off["gap_certified"]):
            cert_gains.append(name)
        if bool(off["gap_certified"]) and not bool(on["gap_certified"]):
            cert_regressions.append(name)
        if off["objective"] is not None and on["objective"] is None:
            lost_incumbents.append(name)
        if off["objective"] is None and on["objective"] is not None:
            gained_incumbents.append(name)

        ob, nb = off["bound"], on["bound"]
        # A finite bound going to None is the most severe bound outcome there is --
        # the solve stops claiming anything at all -- and comparing only cells where
        # BOTH arms are finite silently skips it. The first cut of this panel did
        # exactly that and scored two real regressions (bchoco08 1.0 -> None, contvar
        # 171244.81 -> None) as clean.
        if ob is not None and nb is None:
            lost_bound.append(f"{name} ({ob} -> None)")
        if ob is None and nb is not None:
            gained_bound.append(f"{name} (None -> {nb})")
        if ob is not None and nb is not None:
            rel = abs(nb - ob) / max(1.0, abs(ob))
            if rel > 1e-9:
                if (nb < ob) if sense == "min" else (nb > ob):
                    looser_bound.append(f"{name} ({ob} -> {nb})")
                else:
                    tighter_bound.append(f"{name} ({ob} -> {nb})")

        oo, no = off["objective"], on["objective"]
        if oo is not None and no is not None and abs(no - oo) > TOL * max(1.0, abs(oo)):
            if (no > oo) if sense == "min" else (no < oo):
                worse_obj.append(f"{name} ({oo} -> {no})")
            else:
                better_obj.append(f"{name} ({oo} -> {no})")

    summary = {
        "instances": len(cells),
        "budget": budget,
        "cells_over_budget_off": over_off,
        "cells_over_budget_on": over_on,
        "total_overrun_s_off": round(overrun_s_off, 1),
        "total_overrun_s_on": round(overrun_s_on, 1),
        "cert_gains": cert_gains,
        "cert_regressions": cert_regressions,
        "gained_incumbents": gained_incumbents,
        "lost_incumbents": lost_incumbents,
        "better_objective": better_obj,
        "worse_objective": worse_obj,
        "tighter_bound": tighter_bound,
        "looser_bound": looser_bound,
        "gained_bound": gained_bound,
        "lost_bound": lost_bound,
        "unsound": unsound,
        "incumbent_verification_failed": verification_failed,
        "panel_wall_s": round(time.perf_counter() - t_panel, 1),
    }
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")
    cert_clean = not (
        cert_regressions or lost_incumbents or unsound or verification_failed or lost_bound
    )
    print(f"\nCERT_CLEAN={cert_clean}")
    print(f"COMPARISONS_EXECUTED={compared}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "cells": cells}, indent=2))
        print(f"wrote {out}")

    if compared == 0:
        print("PANEL FIRED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
