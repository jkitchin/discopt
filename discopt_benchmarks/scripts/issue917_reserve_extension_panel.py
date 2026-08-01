"""#917 graduation panel: the incumbent-conditional reserve extension, OFF vs ON.

``Model.solve`` deducts a 35% reserve from the caller's ``time_limit`` for every
model the #844 no-incumbent fallback could serve, and spends it only when the
primary returns nothing. ``DISCOPT_LP_SPATIAL_RESERVE_EXTENSION=1`` lets the
primary reclaim that slice at its reduced deadline, and only while it already
holds an incumbent — the one state in which the fallback provably has nothing to
contribute, so the #844 path is untouched.

The panel sweeps a UNIFORM budget grid over every in-scope instance in the two
in-repo corpora, running the flag OFF and ON back-to-back per cell (interleaved,
not sequentially, per CLAUDE.md §9) in isolated subprocesses.

Both CLAUDE.md §5 bars are scored:

*cert-clean*  — no certification regression (``gap_certified`` True -> False), no
                lost incumbent, no bound above the instance's reference optimum,
                no bound above its own incumbent, no incumbent-verification
                failure, no wall past the caller's stated limit that OFF did not
                also blow.
*net-positive* — certification gains, bound/incumbent improvements, and how much
                of the caller's stated budget each arm actually spends.

Usage::

    python discopt_benchmarks/scripts/issue917_reserve_extension_panel.py \
        [--budgets 6,9,13,20,30,45,60] [--reps 1] [--instances nvs17,nvs19] \
        --out results/issue917_panel.json

Ends with an executed-comparison count and exits non-zero if it compared nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPORA = [ROOT / "python/tests/data/minlplib_nl", ROOT / "python/tests/data/minlplib"]
WORKER = Path(__file__).with_name("issue917_reserve_extension_worker.py")
FLAG = "DISCOPT_LP_SPATIAL_RESERVE_EXTENSION"

# Every instance the #844 reserve is deducted for under the shipped default gate
# (``_is_in_scope(mixed=False)`` plus at least one constraint row), across both
# in-repo corpora. Produced by ``scratchpad/issue917_scope_scan.py``.
IN_SCOPE = [
    "nvs03", "nvs07", "nvs10", "nvs11", "nvs12", "nvs13", "nvs15",
    "nvs17", "nvs18", "nvs19", "nvs23", "nvs24",
    "prob02", "prob03",
    "st_miqp1", "st_miqp2", "st_miqp3", "st_test1", "st_testgr3",
]  # fmt: skip

TOL = 1e-6


def find(name: str) -> Path:
    for d in CORPORA:
        p = d / f"{name}.nl"
        if p.exists():
            return p
    raise SystemExit(f"instance not found in either corpus: {name}")


def reference_optimum(name: str):
    """Published global optimum from the in-repo oracle, or None if absent."""
    sys.path.insert(0, str(ROOT / "python/tests"))
    try:
        from _optima import known_optimum  # type: ignore
    except Exception:
        return None
    try:
        return known_optimum(name)
    except Exception:
        return None


def run_one(nl: Path, budget: float, flag: str) -> dict:
    """One isolated solve. Any worker failure is fatal — never swallowed (§7)."""
    proc = subprocess.run(
        [sys.executable, "-u", str(WORKER), str(nl), str(budget)],
        capture_output=True,
        text=True,
        env={**_env(), FLAG: flag},
        timeout=20 * budget + 900,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-6000:] + "\n")
        raise SystemExit(f"worker failed on {nl.stem} @ {budget}s flag={flag}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _env() -> dict:
    import os

    return dict(os.environ)


def better(sense: str, a, b) -> bool:
    """True if objective ``a`` is strictly better than ``b`` in ``sense``."""
    if a is None:
        return False
    if b is None:
        return True
    return a > b + TOL if sense == "max" else a < b - TOL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="6,9,13,20,30,45,60")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--instances", default=",".join(IN_SCOPE))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    budgets = [float(b) for b in args.budgets.split(",") if b]
    instances = [s for s in args.instances.split(",") if s]

    cells: list[dict] = []
    compared = 0
    t_panel = time.perf_counter()

    for name in instances:
        nl = find(name)
        opt = reference_optimum(name)
        for budget in budgets:
            for rep in range(args.reps):
                # Interleaved, adjacent in time: OFF then ON for the same cell.
                off = run_one(nl, budget, "0")
                on = run_one(nl, budget, "1")
                compared += 1
                cell = {
                    "instance": name,
                    "budget": budget,
                    "rep": rep,
                    "reference_optimum": opt,
                    "off": off,
                    "on": on,
                }
                cells.append(cell)
                for arm, rec in (("OFF", off), ("ON ", on)):
                    print(
                        f"{name:11s} T={budget:5.1f} rep={rep} {arm} "
                        f"wall={rec['wall']:6.1f} {rec['status']:10s} "
                        f"cert={int(bool(rec['gap_certified']))} nodes={rec['node_count']:7d} "
                        f"obj={rec['objective']} bound={rec['bound']} "
                        f"ext={rec['extension_s']}",
                        flush=True,
                    )

    # ---- scoring -------------------------------------------------------
    cert_gains, cert_regressions = [], []
    lost_incumbents, gained_incumbents = [], []
    better_obj, worse_obj = [], []
    tighter_bound, looser_bound = [], []
    lost_bound, gained_bound = [], []
    unsound, verification_failed = [], []
    overshoot_new = []
    extension_fired = 0
    wall_used_off = wall_used_on = 0.0
    wall_budget_total = 0.0

    for c in cells:
        off, on, name, budget = c["off"], c["on"], c["instance"], c["budget"]
        sense = on["sense"]
        tag = f"{name}@{budget}"

        if on["extension_s"]:
            extension_fired += 1
        wall_budget_total += budget
        wall_used_off += min(off["wall"], budget)
        wall_used_on += min(on["wall"], budget)

        if bool(on["gap_certified"]) and not bool(off["gap_certified"]):
            cert_gains.append(tag)
        if bool(off["gap_certified"]) and not bool(on["gap_certified"]):
            cert_regressions.append(tag)

        if off["objective"] is not None and on["objective"] is None:
            lost_incumbents.append(tag)
        if off["objective"] is None and on["objective"] is not None:
            gained_incumbents.append(tag)
        if better(sense, on["objective"], off["objective"]) and off["objective"] is not None:
            better_obj.append(tag)
        if better(sense, off["objective"], on["objective"]) and on["objective"] is not None:
            worse_obj.append(tag)

        ob, nb = off["bound"], on["bound"]
        # A finite bound going to None is the most severe bound outcome there is, and
        # comparing only cells where BOTH arms are finite silently skips it. The sibling
        # lp-warm-deadline panel scored two real regressions (bchoco08 1.0 -> None,
        # contvar 171244.81 -> None) as clean before this check existed.
        if ob is not None and nb is None:
            lost_bound.append(f"{tag} ({ob} -> None)")
        if ob is None and nb is not None:
            gained_bound.append(f"{tag} (None -> {nb})")
        if ob is not None and nb is not None:
            if (nb < ob - TOL) if sense == "max" else (nb > ob + TOL):
                tighter_bound.append(tag)
            elif (nb > ob + TOL) if sense == "max" else (nb < ob - TOL):
                looser_bound.append(tag)

        # Soundness, scored on the ON arm only where the OFF arm is clean.
        for arm_name, arm in (("off", off), ("on", on)):
            if arm["incumbent_verification_failed"]:
                verification_failed.append(f"{tag}:{arm_name}")
            if arm["bound"] is not None and arm["objective"] is not None:
                crossed = (
                    arm["bound"] < arm["objective"] - 1e-4
                    if sense == "max"
                    else arm["bound"] > arm["objective"] + 1e-4
                )
                if crossed:
                    unsound.append(f"{tag}:{arm_name}:bound-crosses-incumbent")
            if arm["bound"] is not None and c["reference_optimum"] is not None:
                ref = float(c["reference_optimum"])
                crossed = arm["bound"] < ref - 1e-4 if sense == "max" else arm["bound"] > ref + 1e-4
                if crossed:
                    unsound.append(f"{tag}:{arm_name}:bound-past-oracle")

        # A wall the ON arm blows and the OFF arm did not.
        if on["wall"] > budget * 1.05 and off["wall"] <= budget * 1.05:
            overshoot_new.append(f"{tag} ({on['wall']:.1f}s of {budget:.0f}s)")

    summary = {
        "cells": len(cells),
        "extension_fired": extension_fired,
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
        "new_overshoots": overshoot_new,
        "budget_utilisation_off": wall_used_off / wall_budget_total if wall_budget_total else 0.0,
        "budget_utilisation_on": wall_used_on / wall_budget_total if wall_budget_total else 0.0,
        "panel_wall_s": time.perf_counter() - t_panel,
    }

    print()
    for k, v in summary.items():
        print(f"{k}: {v}")
    cert_clean = not (
        cert_regressions or lost_incumbents or unsound or verification_failed or lost_bound
    )
    net_positive = bool(cert_gains or gained_incumbents or better_obj or tighter_bound)
    print(f"\nCERT_CLEAN={cert_clean}  NET_POSITIVE={net_positive}")
    print(f"COMPARISONS_EXECUTED={compared}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "cells": cells}, indent=2))
        print(f"wrote {out}")

    if compared == 0:
        print("PANEL FIRED NOTHING: zero comparisons executed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
