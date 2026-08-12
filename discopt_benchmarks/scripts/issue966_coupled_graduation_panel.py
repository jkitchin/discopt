"""Coupled §5 graduation panel for the three budget flags (#928 + #966 item 3).

``DISCOPT_LP_WARM_DEADLINE`` (#917/#928) failed the net-positive bar for one
*measured* reason: the LP layer honours its grant, but the enclosing
separated-relaxation round does not, so the budget-honouring LPs simply let the
loop fit more (unclamped) rounds and the ON arm's wall went UP at a 20 s budget
(ON-OFF +325.4 / +68.5 / +12.7 s over three reps). #966 fixed that seam behind
``DISCOPT_NODE_ROUND_BUDGET`` and ``DISCOPT_HESS_COMPILE_GATE``. The three are
therefore ONE change for graduation purposes, and this panel scores them as one.

Three arms, run back-to-back per instance (interleaved, CLAUDE.md §9) in isolated
subprocesses, every flag set explicitly on every arm:

    A  base       all three OFF  -- today's default, the control
    B  seam       #966 ON, #928 OFF -- isolates the round/compile budget fix
    C  candidate  all three ON   -- what graduation would make the default

C vs A is the graduation question. C vs B isolates #928's marginal effect now
that the seam is closed, and B vs A isolates #966's own effect, so a failure can
be attributed instead of guessed at.

Ends with an executed-comparison count; exits non-zero if it compared nothing
(§6). Every arm's raw record is written to ``--out`` for re-analysis.

    python -u discopt_benchmarks/scripts/issue966_coupled_graduation_panel.py \
        --budget 20 --instances 4stufen,bchoco06,... \
        --out discopt_benchmarks/results/issue966_coupled_binding20_rep1.json
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
WORKER = Path(__file__).with_name("issue966_coupled_worker.py")
TOL = 1e-6

WARM = "DISCOPT_LP_WARM_DEADLINE"
ROUND = "DISCOPT_NODE_ROUND_BUDGET"
HESS = "DISCOPT_HESS_COMPILE_GATE"

# (key, label, env). Every flag is named in every arm: an arm must never inherit.
ALL_ARMS = {
    "base": ("all OFF (today's default)", {WARM: "0", ROUND: "0", HESS: "0"}),
    "seam": ("#966 ON, #928 OFF", {WARM: "0", ROUND: "1", HESS: "1"}),
    "cand": ("all three ON", {WARM: "1", ROUND: "1", HESS: "1"}),
    # #928's flag ALONE. The three-arm default cannot answer "does
    # DISCOPT_LP_WARM_DEADLINE graduate", because every arm carrying it also
    # carries #966's two, and #966's own panel (issue #966, f2565241) kept those
    # OFF. ``cand - seam`` isolates it only under the counterfactual that the
    # other two are ON, which is not the state of the tree.
    "warm": ("#928 ON alone", {WARM: "1", ROUND: "0", HESS: "0"}),
}
ARMS = tuple((k, ALL_ARMS[k][0], ALL_ARMS[k][1]) for k in ("base", "seam", "cand"))


def loadavg() -> list[float]:
    """1/5/15-minute load, recorded into the artifact at panel start and end.

    CLAUDE.md §9 makes a load gate part of any timing claim; recording it beside
    the numbers keeps the gate auditable instead of a sentence in a PR body. The
    arms are interleaved per instance precisely so a load excursion hits all
    three within seconds of each other rather than biasing one.
    """
    return list(os.getloadavg())


def reference_optimum(name: str):
    """The largest value a valid dual bound may take, or ``None`` if unknown.

    This used to read ``python/tests/_optima.py`` behind a bare ``except``, which
    is CLAUDE.md §7's failure exactly: the module covers 27 curated instances, so
    for the other 18 of this panel's 19 the lookup returned ``None`` and the arm
    was skipped -- and the panel still printed ``unsound: []``, which reads as a
    soundness result over 19 instances and was one comparison. The oracle is now
    ``minlplib.solu``, which covers the library; ``_optima`` remains the fallback
    for anything the .solu file does not name, and a missing/unreadable .solu is
    reported rather than swallowed (the caller counts coverage, see
    ``soundness``).
    """
    ceiling = _solu_ceiling(name)
    if ceiling is not None:
        return ceiling
    sys.path.insert(0, str(ROOT / "python/tests"))
    from _optima import known_optimum  # type: ignore

    try:
        return known_optimum(name)
    except KeyError:
        # The one genuine "no oracle" outcome, and the only one that may be
        # swallowed: everything else (missing .solu, bad import) must crash.
        return None


def _solu_ceiling(name: str):
    """``minlplib.solu``'s ceiling for ``name``, or ``None`` when unnamed there.

    ``DISCOPT_MINLPLIB_SOLU=none`` declares the library oracle **unavailable in
    this environment** (the benchmark snapshot is not mounted and minlplib.org is
    unreachable). It is an explicit, recorded declaration -- ``solu_oracle`` in
    the panel summary -- not a swallowed error: a .solu path that is *set* but
    unreadable still crashes, which is the §7 state this keeps distinguishable.
    Under it every instance falls through to the narrow ``_optima`` fallback, so
    the run's oracle coverage collapses (1/19 on this panel's names) and the
    printed ``oracle_comparisons_executed`` / ``instances_without_oracle`` counts
    are what the soundness claim may rest on -- exactly the §14b-qual lesson.
    """
    if solu_oracle_state() == "none":
        return None
    sys.path.insert(0, str(Path(__file__).parent))
    from minlplib_solu import load, primal_ceiling  # type: ignore

    global _SOLU
    if _SOLU is None:
        _SOLU = load()
    return primal_ceiling(name, _SOLU)


def solu_oracle_state() -> str:
    """``"none"`` when the library oracle is declared unavailable, else its path."""
    return "none" if os.environ.get("DISCOPT_MINLPLIB_SOLU", "").strip() == "none" else "solu"


_SOLU: dict | None = None


def run_one(nl: Path, budget: float, env: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, "-u", str(WORKER), str(nl), str(budget)],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=40 * budget + 900,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-6000:] + "\n")
        raise SystemExit(f"worker failed on {nl.stem} env={env}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def soundness(cells: list[dict]) -> dict:
    """Per-arm soundness scoring: a bound past the oracle or across the incumbent,
    or an incumbent that failed verification, is disqualifying on its own.

    Returns the number of oracle comparisons actually executed and the instances
    with no oracle at all (§6). ``unsound: []`` over zero comparisons is not a
    soundness result, and the caller must be able to tell the two apart.
    """
    unsound, verification_failed, uncovered = [], [], []
    oracle_cmps = 0
    for c in cells:
        ref = c["reference_optimum"]
        if ref is None:
            uncovered.append(c["instance"])
        for key, _label, _env in ARMS:
            rec = c[key]
            sense = rec["sense"]
            if rec["incumbent_verification_failed"]:
                verification_failed.append(f"{c['instance']}:{key}")
            if rec["bound"] is not None and rec["objective"] is not None:
                bad = (
                    rec["bound"] < rec["objective"] - 1e-4
                    if sense == "max"
                    else rec["bound"] > rec["objective"] + 1e-4
                )
                if bad:
                    unsound.append(f"{c['instance']}:{key}:bound-crosses-incumbent")
            if rec["bound"] is not None and ref is not None:
                oracle_cmps += 1
                bad = (
                    rec["bound"] < float(ref) - 1e-4
                    if sense == "max"
                    else rec["bound"] > float(ref) + 1e-4
                )
                if bad:
                    unsound.append(f"{c['instance']}:{key}:bound-past-oracle")
    return {
        "unsound": unsound,
        "incumbent_verification_failed": verification_failed,
        "oracle_comparisons_executed": oracle_cmps,
        "instances_without_oracle": sorted(set(uncovered)),
    }


def compare(cells: list[dict], a_key: str, b_key: str) -> dict:
    """Score arm ``b`` against arm ``a`` with the #917 panel's metric set."""
    cert_gains, cert_regressions = [], []
    lost_incumbents, gained_incumbents = [], []
    worse_obj, better_obj = [], []
    looser_bound, tighter_bound = [], []
    lost_bound, gained_bound = [], []
    over_a = over_b = 0
    overrun_a = overrun_b = 0.0
    nodes_a = nodes_b = 0

    for c in cells:
        a, b, name, budget = c[a_key], c[b_key], c["instance"], c["budget"]
        sense = a["sense"]
        for rec, is_a in ((a, True), (b, False)):
            if rec["wall"] > budget * 1.05:
                if is_a:
                    over_a += 1
                else:
                    over_b += 1
            ov = max(0.0, rec["wall"] - budget)
            if is_a:
                overrun_a += ov
                nodes_a += rec["node_count"]
            else:
                overrun_b += ov
                nodes_b += rec["node_count"]

        if bool(b["gap_certified"]) and not bool(a["gap_certified"]):
            cert_gains.append(name)
        if bool(a["gap_certified"]) and not bool(b["gap_certified"]):
            cert_regressions.append(name)
        if a["objective"] is not None and b["objective"] is None:
            lost_incumbents.append(name)
        if a["objective"] is None and b["objective"] is not None:
            gained_incumbents.append(name)

        ab, bb = a["bound"], b["bound"]
        # A finite bound going to None is the most severe bound outcome there is --
        # the solve stops claiming anything at all -- and comparing only cells where
        # BOTH arms are finite silently skips it (the #917 panel's original bug).
        if ab is not None and bb is None:
            lost_bound.append(f"{name} ({ab} -> None)")
        if ab is None and bb is not None:
            gained_bound.append(f"{name} (None -> {bb})")
        if ab is not None and bb is not None and abs(bb - ab) / max(1.0, abs(ab)) > 1e-9:
            if (bb < ab) if sense == "min" else (bb > ab):
                looser_bound.append(f"{name} ({ab} -> {bb})")
            else:
                tighter_bound.append(f"{name} ({ab} -> {bb})")

        ao, bo = a["objective"], b["objective"]
        if ao is not None and bo is not None and abs(bo - ao) > TOL * max(1.0, abs(ao)):
            if (bo > ao) if sense == "min" else (bo < ao):
                worse_obj.append(f"{name} ({ao} -> {bo})")
            else:
                better_obj.append(f"{name} ({ao} -> {bo})")

    return {
        "pair": f"{b_key} vs {a_key}",
        "cells_over_budget": {a_key: over_a, b_key: over_b},
        "total_overrun_s": {a_key: round(overrun_a, 1), b_key: round(overrun_b, 1)},
        "overrun_delta_s": round(overrun_b - overrun_a, 1),
        "node_count_total": {a_key: nodes_a, b_key: nodes_b},
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
    }


def main() -> int:
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--instances", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--arms",
        default="base,seam,cand",
        help=f"comma-separated arm keys to run, from {sorted(ALL_ARMS)}. Each arm sets "
        "EVERY flag explicitly, so a two-arm run is as controlled as the three-arm one.",
    )
    ap.add_argument(
        "--pair",
        default=None,
        help="graduation pair as 'control,candidate' (default: first,last of --arms). "
        "CERT_CLEAN is scored on this pair.",
    )
    ap.add_argument(
        "--rescore",
        default=None,
        help="Re-run soundness() over a saved artifact's cells instead of solving. "
        "The cells already carry every arm's bound, so a panel whose oracle was "
        "broken can be re-scored without spending the wall time again.",
    )
    args = ap.parse_args()

    keys = [k for k in args.arms.split(",") if k]
    unknown = [k for k in keys if k not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(ALL_ARMS)}")
    if len(keys) < 2:
        raise SystemExit("need at least two arms to compare")
    ARMS = tuple((k, ALL_ARMS[k][0], ALL_ARMS[k][1]) for k in keys)
    ctrl, cand_key = args.pair.split(",") if args.pair else (keys[0], keys[-1])
    if ctrl not in keys or cand_key not in keys:
        raise SystemExit(f"--pair {ctrl},{cand_key} names an arm not in --arms {keys}")

    if args.rescore:
        saved = json.loads(Path(args.rescore).read_text())["cells"]
        # Score the arms the artifact actually holds, not the ones --arms defaults
        # to: rescoring a two-arm panel under the three-arm default would KeyError,
        # and rescoring a three-arm panel under a two-arm run would silently skip
        # an arm's soundness.
        saved_keys = [k for k in ALL_ARMS if saved and k in saved[0]]
        ARMS = tuple((k, ALL_ARMS[k][0], ALL_ARMS[k][1]) for k in saved_keys)
        for c in saved:
            c["reference_optimum"] = reference_optimum(c["instance"])
        snd = soundness(saved)
        print(json.dumps(snd, indent=2))
        print(f"ORACLE_COMPARISONS_EXECUTED={snd['oracle_comparisons_executed']}")
        return 0 if snd["oracle_comparisons_executed"] else 1

    names = (
        [s for s in args.instances.split(",") if s]
        if args.instances
        else sorted(p.stem for p in CORPUS.glob("*.nl"))
    )
    budget = args.budget

    cells, compared = [], 0
    t_panel = time.perf_counter()
    load_start = loadavg()
    for idx, name in enumerate(names, 1):
        nl = CORPUS / f"{name}.nl"
        cell = {"instance": name, "budget": budget, "reference_optimum": reference_optimum(name)}
        for key, _label, env in ARMS:
            cell[key] = run_one(nl, budget, env)
        cells.append(cell)
        compared += 1
        parts = " | ".join(
            f"{key} wall={cell[key]['wall']:6.1f} ({cell[key]['wall'] / budget:4.2f}x) "
            f"cert={int(bool(cell[key]['gap_certified']))} bound={cell[key]['bound']}"
            for key, _l, _e in ARMS
        )
        print(f"[{idx}/{len(names)}] {name:22s} {parts}", flush=True)

    snd = soundness(cells)
    others = [
        (a, b)
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
        if (a, b) != (ctrl, cand_key)
    ]
    pairs = [compare(cells, ctrl, cand_key)] + [compare(cells, a, b) for a, b in others]
    grad = pairs[0]
    cert_clean = not (
        snd["unsound"]
        or snd["incumbent_verification_failed"]
        or grad["cert_regressions"]
        or grad["lost_incumbents"]
        or grad["lost_bound"]
    )
    summary = {
        "instances": len(cells),
        "budget": budget,
        "arms": {k: lbl for k, lbl, _e in ARMS},
        **snd,
        "pairs": pairs,
        "solu_oracle": solu_oracle_state(),
        "panel_wall_s": round(time.perf_counter() - t_panel, 1),
        "loadavg_start": [round(x, 2) for x in load_start],
        "loadavg_end": [round(x, 2) for x in loadavg()],
    }

    print()
    print(json.dumps(summary, indent=2))
    print(f"\nCERT_CLEAN={cert_clean}  (graduation pair: {cand_key} vs {ctrl})")
    print(f"OVERRUN_DELTA_S_{cand_key.upper()}_VS_{ctrl.upper()}={grad['overrun_delta_s']}")
    print(f"COMPARISONS_EXECUTED={compared}")
    print(f"ORACLE_COMPARISONS_EXECUTED={snd['oracle_comparisons_executed']}")
    print(f"INSTANCES_WITHOUT_ORACLE={snd['instances_without_oracle']}")

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
    if snd["oracle_comparisons_executed"] == 0:
        # A panel that never reached an oracle cannot report on soundness, and
        # ``unsound: []`` from it is a formatting artifact, not a measurement.
        print("PANEL MADE NO ORACLE COMPARISONS", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
