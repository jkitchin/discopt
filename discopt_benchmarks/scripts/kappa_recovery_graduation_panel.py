#!/usr/bin/env python3
"""Graduation panel -- ``DISCOPT_LP_KAPPA_RECOVERY`` (the dual optimality gate's
condition-number trigger).

The CLAUDE.md §5 differential panel for a bound-changing gate: flag ON vs OFF over
the vendored corpus, requiring BOTH

  1. *cert-clean* -- no incumbent past its reference optimum, no dual bound
     crossing that optimum or its own incumbent, no certification regression
     (``gap_certified=True`` OFF must not become False ON), no objective drift on
     instances both arms solve, and every incumbent independently
     feasibility-verified against the pristine model; AND
  2. the applicable benefit bar.

**What the benefit bar is here, and why it is not "bound or nodes".** The existing
trigger reads element growth ``‖U‖∞/‖U₀‖∞``, which measures what *elimination* did
to the factor. It is one-sided by construction: on a basis elimination never has to
fight -- a near-diagonal one -- growth is exactly 1 however singular the basis is
(``linsolve::growth_is_blind_to_a_class_kappa_sees``). κ₁ covers that blind spot.
So this is a **soundness guard**, governed by §1, not a bound-tightening feature:
the win is a certificate not issued from a basis whose ``x_B`` has no correct
digits, and the cost is wall and nodes. The panel therefore scores

  * SOUNDNESS: any instance where the two arms disagree on status, objective or
    certification -- each such row is inspected against the oracle, because the
    arm that *changed* is the one that stopped trusting a bad basis;
  * COST: node count and wall, ON vs OFF.

**Proving the mechanism fired (CLAUDE.md §6).** A panel of a trigger that never
trips is not a pass, it is a no-op that reads as one. Each arm is bracketed with
``profile_reset_py``/``profile_counters_py`` under ``DISCOPT_PROFILE=1``, and the
ON-minus-OFF difference in ``RefinedRecoveryAttemptsDual`` is the number of
recoveries κ₁ caused that growth did not. The panel FAILS if that difference is
zero over the whole corpus -- there would be nothing to have measured.

Arms are interleaved within each instance and the arm order alternates by index,
so machine-load drift cannot systematically favour one arm (measurement rule 9).

Usage::

    python -u discopt_benchmarks/scripts/kappa_recovery_graduation_panel.py \
        [--time-limit 20] [--threshold 1] [--out-dir reports]

``--threshold`` is passed through as the ON arm's flag value, so the panel can
sweep the κ₁ line (``1`` selects the compiled-in ``KAPPA_REFINE_TRIGGER``).

Exit codes: 0 = PASS, 2 = cert-clean but the trigger never fired / no effect,
1 = any violation or zero executed assertions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent.parent
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_OPTIMA = _REPO / "docs" / "dev" / "data" / "cert-optima.json"

_TOL_REL = 1e-4
_TOL_ABS = 1e-6

_FLAG = "DISCOPT_LP_KAPPA_RECOVERY"
_ATTEMPTS = "RefinedRecoveryAttemptsDual"
_RESCUES = "RefinedRecoveryRescuesDual"


_RUNNER = """
import json, os, sys, time

import numpy as np
import discopt.modeling as dm
from discopt._rust import profile_counters_py, profile_reset_py
from discopt.modeling.core import ObjectiveSense

path, tl = sys.argv[1], float(sys.argv[2])
model = dm.from_nl(path)
profile_reset_py()
t0 = time.perf_counter()
res = model.solve(time_limit=tl, deterministic=True)
wall = time.perf_counter() - t0
ctrs = dict(profile_counters_py())

verified = None
if res.x:
    from discopt._relax.primal_heuristics import passes_false_primal_screen
    from discopt._tape_nlp_evaluator import build_evaluator

    def _jax_evaluator():
        from discopt._relax.nlp_evaluator import cached_evaluator
        return cached_evaluator(model)

    ev = build_evaluator(model, _jax_evaluator)
    flat = []
    for v in model._variables:
        flat.extend(np.asarray(res.x[v.name], dtype=float).ravel().tolist())
    verified = bool(passes_false_primal_screen(ev, np.asarray(flat, dtype=float)))

minimize = model._objective is None or model._objective.sense == ObjectiveSense.MINIMIZE
print("PANELJSON " + json.dumps({
    "status": str(res.status),
    "objective": res.objective,
    "bound": res.bound,
    "gap_certified": bool(res.gap_certified),
    "nodes": int(res.node_count or 0),
    "wall": wall,
    "verified": verified,
    "minimize": bool(minimize),
    "attempts": int(ctrs.get("RefinedRecoveryAttemptsDual", 0)),
    "rescues": int(ctrs.get("RefinedRecoveryRescuesDual", 0)),
}), flush=True)
"""


def _solve_once(path: Path, flag: str, time_limit: float) -> dict:
    """One solve with ``DISCOPT_LP_KAPPA_RECOVERY=flag``, **in its own process**.

    This must be a subprocess, and the reason is the whole point of the panel.
    ``kappa_refine_trigger()`` caches its value in a Rust ``OnceLock``, read once
    per process. Running both arms in one interpreter pins whichever arm ran first
    and silently re-runs it as the second arm -- the harness then reports two
    identical arms as a clean "no effect". That is not hypothetical: an in-process
    first draft of this panel reported 0 kappa-caused recoveries at threshold
    1.001, where kappa_1 > 1.001 holds for *every* basis.

    Never swallows an exception (measurement rule 7): a non-zero exit or missing
    PANELJSON line raises rather than reading as a miss.
    """
    import subprocess
    import sys

    env = {**os.environ, _FLAG: flag, "DISCOPT_PROFILE": "1"}
    proc = subprocess.run(
        [sys.executable, "-u", "-c", _RUNNER, str(path), str(time_limit)],
        capture_output=True,
        text=True,
        timeout=time_limit * 8 + 300,
        env=env,
    )
    for ln in proc.stdout.splitlines():
        if ln.startswith("PANELJSON "):
            return json.loads(ln[len("PANELJSON ") :])
    raise RuntimeError(
        f"{path.stem} [{_FLAG}={flag}] produced no result "
        f"(rc={proc.returncode}): {proc.stderr.strip()[-400:]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--threshold", default="1", help="ON-arm value for " + _FLAG)
    ap.add_argument("--out-dir", default=str(_REPO / "reports"))
    ap.add_argument("--instances", default="")
    args = ap.parse_args()

    # Counters only accumulate while DISCOPT_PROFILE is set; without it the
    # fired-count check below would read 0 for both arms and the panel would
    # report "never fired" no matter what happened.
    os.environ.setdefault("DISCOPT_PROFILE", "1")

    optima = json.loads(_OPTIMA.read_text()) if _OPTIMA.exists() else {}
    names = (
        [n.strip() for n in args.instances.split(",") if n.strip()]
        if args.instances
        else sorted(p.stem for p in _NL_DIR.glob("*.nl"))
    )

    rows: list[dict] = []
    checks = 0
    violations: list[str] = []
    # Uncertified incumbent changes: legitimate, but the whole point of the panel,
    # so they are collected and scored against the oracle rather than discarded.
    incumbent_moves: list[dict] = []

    for i, name in enumerate(names):
        path = _NL_DIR / f"{name}.nl"
        order = ("0", args.threshold) if i % 2 == 0 else (args.threshold, "0")
        arms: dict[str, dict] = {}
        for flag in order:
            arms["0" if flag == "0" else "1"] = _solve_once(path, flag, args.time_limit)
        off, on = arms["0"], arms["1"]
        rows.append({"instance": name, "off": off, "on": on})

        ref = optima.get(name)
        for tag, a in (("off", off), ("on", on)):
            if a["verified"] is False:
                violations.append(f"{name}[{tag}]: incumbent failed the false-primal screen")
            if a["verified"] is not None:
                checks += 1
            if a["objective"] is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(a["objective"])) + _TOL_ABS
                if a["minimize"] and a["bound"] > a["objective"] + slack:
                    violations.append(
                        f"{name}[{tag}]: bound {a['bound']} above incumbent {a['objective']}"
                    )
                if not a["minimize"] and a["bound"] < a["objective"] - slack:
                    violations.append(
                        f"{name}[{tag}]: bound {a['bound']} below incumbent {a['objective']}"
                    )
            if ref is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(ref)) + _TOL_ABS
                if a["minimize"] and a["bound"] > ref + slack:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} above optimum {ref}")
                if not a["minimize"] and a["bound"] < ref - slack:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} below optimum {ref}")
            if ref is not None and a["objective"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(ref)) + _TOL_ABS
                if a["minimize"] and a["objective"] < ref - slack:
                    violations.append(
                        f"{name}[{tag}]: incumbent {a['objective']} below optimum {ref}"
                    )
                if not a["minimize"] and a["objective"] > ref + slack:
                    violations.append(
                        f"{name}[{tag}]: incumbent {a['objective']} above optimum {ref}"
                    )

        checks += 1
        if off["gap_certified"] and not on["gap_certified"]:
            violations.append(f"{name}: certification lost with the flag ON")
        # Objective drift is a violation only between two arms that both CERTIFIED
        # optimality -- two certified optima of the same model must agree. On an
        # instance neither arm proved (a time limit), a different incumbent is the
        # expected outcome of changing when the engine distrusts a basis, and it is
        # not evidence of unsoundness: the soundness question there is "is the
        # incumbent past the reference optimum", which is checked above against
        # `cert-optima.json`, and independently by the false-primal screen.
        #
        # This was wrong in the first version, which flagged any drift. It would
        # have reported `nvs05` as a violation -- where the OFF arm sits at
        # 5.8873550 (7.6% above optimal) and the ON arm returns 5.4709341, the true
        # optimum to 2.5e-9. Calling that a violation would have inverted the
        # finding. Narrowing the rule does not weaken the panel (CLAUDE.md §1): the
        # oracle, bound-vs-incumbent, certification and false-primal checks are
        # untouched, and uncertified drift is still REPORTED below, just not
        # counted as a defect.
        both_certified = off["gap_certified"] and on["gap_certified"]
        if off["objective"] is not None and on["objective"] is not None:
            checks += 1
            drift = abs(on["objective"] - off["objective"])
            if drift > _TOL_REL * max(1.0, abs(off["objective"])) + _TOL_ABS:
                if both_certified:
                    violations.append(
                        f"{name}: objective drift between two CERTIFIED arms "
                        f"{off['objective']} -> {on['objective']}"
                    )
                else:
                    incumbent_moves.append(
                        {
                            "instance": name,
                            "off": off["objective"],
                            "on": on["objective"],
                            "off_status": off["status"],
                            "on_status": on["status"],
                            "reference": ref,
                        }
                    )

        extra = on["attempts"] - off["attempts"]
        print(
            f"{name:24s} OFF {str(off['status']):11s} obj={off['objective']} "
            f"nodes={off['nodes']} att={off['attempts']} | "
            f"ON {str(on['status']):11s} obj={on['objective']} "
            f"nodes={on['nodes']} att={on['attempts']} (kappa-extra {extra:+d})",
            flush=True,
        )

    fired = sum(max(0, r["on"]["attempts"] - r["off"]["attempts"]) for r in rows)
    rescued = sum(max(0, r["on"]["rescues"] - r["off"]["rescues"]) for r in rows)
    fired_on = [r["instance"] for r in rows if r["on"]["attempts"] > r["off"]["attempts"]]
    disagree = [
        r["instance"]
        for r in rows
        if r["off"]["status"] != r["on"]["status"]
        or r["off"]["gap_certified"] != r["on"]["gap_certified"]
    ]
    # A looser dual bound with the certificate intact is not a violation -- for a
    # soundness guard it is exactly the cost being bought -- but it is the COST
    # column §5 asks for, so it is reported rather than left implicit.
    loosened = []
    for r in rows:
        bo, bn = r["off"]["bound"], r["on"]["bound"]
        if bo is None or bn is None:
            continue
        tol = _TOL_REL * max(1.0, abs(bo)) + _TOL_ABS
        if (r["off"]["minimize"] and bn < bo - tol) or (not r["off"]["minimize"] and bn > bo + tol):
            loosened.append({"instance": r["instance"], "off": bo, "on": bn})

    nodes_off = sum(r["off"]["nodes"] for r in rows)
    nodes_on = sum(r["on"]["nodes"] for r in rows)
    wall_off = sum(r["off"]["wall"] for r in rows)
    wall_on = sum(r["on"]["wall"] for r in rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "kappa_recovery_graduation_panel.json"
    out.write_text(
        json.dumps(
            {
                "time_limit": args.time_limit,
                "threshold": args.threshold,
                "instances": len(rows),
                "executed_checks": checks,
                "violations": violations,
                "kappa_extra_attempts": fired,
                "kappa_extra_rescues": rescued,
                "instances_where_kappa_fired": fired_on,
                "instances_disagreeing": disagree,
                "bounds_loosened": loosened,
                "incumbent_moves": incumbent_moves,
                "nodes_off": nodes_off,
                "nodes_on": nodes_on,
                "wall_off": wall_off,
                "wall_on": wall_on,
                "rows": rows,
            },
            indent=2,
        )
    )

    print(f"\ninstances                 : {len(rows)}")
    print(f"dual bounds loosened      : {len(loosened)} {[d['instance'] for d in loosened]}")

    better = worse = unknown = 0
    for mv in incumbent_moves:
        r = mv["reference"]
        if r is None:
            unknown += 1
            continue
        # distance to the oracle, sense-agnostic: an incumbent is valid on the
        # far side of the optimum, so "better" means strictly closer to it.
        if abs(mv["on"] - r) < abs(mv["off"] - r):
            better += 1
        else:
            worse += 1
    print(
        f"uncertified incumbent moves: {len(incumbent_moves)} "
        f"(closer to optimum {better}, further {worse}, no oracle {unknown})"
    )
    for mv in incumbent_moves:
        print(
            f"    {mv['instance']:20s} {mv['off']} [{mv['off_status']}] -> "
            f"{mv['on']} [{mv['on_status']}]  ref={mv['reference']}"
        )
    print(f"executed checks           : {checks}")
    print(f"violations                : {len(violations)}")
    for v in violations:
        print("  !", v)
    print(f"kappa-caused recoveries   : {fired} on {len(fired_on)} instance(s) {fired_on}")
    print(f"  of which rescued a node : {rescued}")
    print(f"status/cert disagreements : {len(disagree)} {disagree}")
    print(f"nodes   OFF {nodes_off} -> ON {nodes_on}")
    print(f"wall    OFF {wall_off:.1f}s -> ON {wall_on:.1f}s")
    print(f"report                    : {out}")

    if checks == 0:
        print("\nVERDICT: FAIL - the panel executed zero checks (it measured nothing).")
        return 1
    if violations:
        print("\nVERDICT: FAIL - gate 1 (cert-clean) violated.")
        return 1
    if fired == 0:
        print(
            "\nVERDICT: INCONCLUSIVE - cert-clean, but the kappa trigger never "
            "fired on this corpus at this threshold, so nothing was measured. "
            "Lower --threshold or widen the corpus before reading this as a pass."
        )
        return 2
    print(
        f"\nVERDICT: PASS - cert-clean, and the trigger fired {fired} time(s) "
        f"that growth did not catch."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
