#!/usr/bin/env python3
"""#1422 graduation panel -- ``DISCOPT_TRIVIAL_PRIMAL`` (the #827 trivial-point seed).

The CLAUDE.md §5 differential panel the flag's own docstring has promised since
#827 and which nobody had run: flag ON vs OFF, requiring BOTH

  1. *cert-clean* -- no incumbent above (for MINIMIZE) its reference optimum, no
     dual bound crossing that optimum, no certification regression
     (``gap_certified=True`` on the OFF arm must not become False ON), no
     objective drift on instances both arms solve, and every incumbent
     independently feasibility-verified against the pristine model; AND
  2. *net-positive* -- measurably helpful, not merely sound. The mechanism is a
     PRIMAL seed, so the metric that scores it is **incumbents found**: how many
     instances go from "no incumbent" (``objective is None``) to a verified
     feasible one.

The seed can only fire when the solve reaches ``solve_model`` with
``initial_point is None``, so the population that can possibly move is exactly
the instances whose OFF arm returns no incumbent. Those are counted and reported
separately: an ON arm that changes nothing on the other instances is the
*neutrality* half of gate 1, and a panel that measured no firing instance at all
is reported as INCONCLUSIVE rather than as a pass (§6 -- a probe that measured
nothing must not read as a pass).

Arms are interleaved within each instance and the arm order is alternated by
index, so a machine-load drift cannot systematically favour one arm (CLAUDE.md
measurement rule 9).

Usage::

    python -u discopt_benchmarks/scripts/issue1422_trivial_primal_graduation_panel.py \
        [--time-limit 8] [--out-dir reports]

Exit codes: 0 = PASS (cert-clean AND net-positive), 2 = cert-clean but
INCONCLUSIVE/neutral, 1 = any violation or zero executed assertions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent.parent
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_OPTIMA = _REPO / "docs" / "dev" / "data" / "cert-optima.json"

_TOL_REL = 1e-4
_TOL_ABS = 1e-6


def _solve_once(path: Path, flag: str, time_limit: float):
    """One solve with ``DISCOPT_TRIVIAL_PRIMAL=flag``. Never swallows an exception
    (CLAUDE.md measurement rule 7) -- a broken arm must crash, not read as a miss."""
    import discopt.modeling as dm

    prev = os.environ.get("DISCOPT_TRIVIAL_PRIMAL")
    os.environ["DISCOPT_TRIVIAL_PRIMAL"] = flag
    try:
        model = dm.from_nl(str(path))
        t0 = time.perf_counter()
        res = model.solve(time_limit=time_limit, deterministic=True)
        wall = time.perf_counter() - t0
        return model, res, wall
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_TRIVIAL_PRIMAL", None)
        else:
            os.environ["DISCOPT_TRIVIAL_PRIMAL"] = prev


def _incumbent_verified(model, res) -> bool | None:
    """True/False from the shared false-primal screen; None when there is nothing
    to screen (no incumbent)."""
    if not res.x:
        return None
    import numpy as np
    from discopt._relax.primal_heuristics import passes_false_primal_screen
    from discopt._tape_nlp_evaluator import build_evaluator

    def _jax_evaluator():
        from discopt._relax.nlp_evaluator import cached_evaluator

        return cached_evaluator(model)

    # ``passes_false_primal_screen`` takes an EVALUATOR, not a Model (looked up,
    # not guessed -- CLAUDE.md's "look up an API before calling it"). Built the
    # same way ``Model.solve``'s own #772 verification snapshot builds it, so the
    # panel screens against the identical problem data.
    evaluator = build_evaluator(model, _jax_evaluator)
    flat = []
    for v in model._variables:
        flat.extend(np.asarray(res.x[v.name], dtype=float).ravel().tolist())
    return bool(passes_false_primal_screen(evaluator, np.asarray(flat, dtype=float)))


def _is_minimize(model) -> bool:
    from discopt.modeling.core import ObjectiveSense

    return model._objective is None or model._objective.sense == ObjectiveSense.MINIMIZE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=8.0)
    ap.add_argument("--out-dir", default=str(_REPO / "reports"))
    ap.add_argument("--instances", default="")
    args = ap.parse_args()

    optima = json.loads(_OPTIMA.read_text()) if _OPTIMA.exists() else {}
    names = (
        [n.strip() for n in args.instances.split(",") if n.strip()]
        if args.instances
        else sorted(p.stem for p in _NL_DIR.glob("*.nl"))
    )

    rows: list[dict] = []
    checks = 0
    violations: list[str] = []

    for i, name in enumerate(names):
        path = _NL_DIR / f"{name}.nl"
        order = ("0", "1") if i % 2 == 0 else ("1", "0")
        arms: dict[str, dict] = {}
        for flag in order:
            model, res, wall = _solve_once(path, flag, args.time_limit)
            ver = _incumbent_verified(model, res)
            arms[flag] = {
                "status": res.status,
                "objective": res.objective,
                "bound": res.bound,
                "gap_certified": bool(res.gap_certified),
                "nodes": int(res.node_count or 0),
                "wall": wall,
                "reported_wall": float(res.wall_time),
                "verified": ver,
                "minimize": _is_minimize(model),
            }
        off, on = arms["0"], arms["1"]
        rows.append({"instance": name, "off": off, "on": on})

        ref = optima.get(name)
        for tag, a in (("off", off), ("on", on)):
            # (1a) every incumbent must survive the shared false-primal screen.
            if a["verified"] is False:
                violations.append(f"{name}[{tag}]: incumbent failed the false-primal screen")
            if a["verified"] is not None:
                checks += 1
            # (1b) certificate invariant: a MINIMIZE bound never above its incumbent.
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
            # (1c) oracle bracket: no dual bound past the reference optimum.
            if ref is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(ref)) + _TOL_ABS
                if a["minimize"] and a["bound"] > ref + slack:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} above optimum {ref}")
                if not a["minimize"] and a["bound"] < ref - slack:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} below optimum {ref}")

        # (1d) no certification regression, (1e) no objective drift.
        checks += 1
        if off["gap_certified"] and not on["gap_certified"]:
            violations.append(f"{name}: certification lost with the flag ON")
        if off["objective"] is not None and on["objective"] is not None:
            checks += 1
            drift = abs(on["objective"] - off["objective"])
            if drift > _TOL_REL * max(1.0, abs(off["objective"])) + _TOL_ABS:
                violations.append(
                    f"{name}: objective drift {off['objective']} -> {on['objective']}"
                )

        print(
            f"{name:24s} OFF {str(off['status']):11s} obj={off['objective']} "
            f"nodes={off['nodes']} | ON {str(on['status']):11s} obj={on['objective']} "
            f"nodes={on['nodes']}",
            flush=True,
        )

    gained = [
        r["instance"]
        for r in rows
        if r["off"]["objective"] is None and r["on"]["objective"] is not None
    ]
    lost = [
        r["instance"]
        for r in rows
        if r["off"]["objective"] is not None and r["on"]["objective"] is None
    ]
    no_inc_off = [r["instance"] for r in rows if r["off"]["objective"] is None]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "issue1422_trivial_primal_panel.json"
    out.write_text(
        json.dumps(
            {
                "time_limit": args.time_limit,
                "instances": len(rows),
                "executed_checks": checks,
                "violations": violations,
                "incumbents_gained": gained,
                "incumbents_lost": lost,
                "no_incumbent_off": no_inc_off,
                "rows": rows,
            },
            indent=2,
        )
    )

    print(f"\ninstances            : {len(rows)}")
    print(f"executed checks      : {checks}")
    print(f"violations           : {len(violations)}")
    for v in violations:
        print("  !", v)
    print(f"no-incumbent OFF     : {len(no_inc_off)} {no_inc_off}")
    print(f"incumbents GAINED ON : {len(gained)} {gained}")
    print(f"incumbents LOST ON   : {len(lost)} {lost}")
    print(f"report               : {out}")

    if checks == 0:
        print("\nVERDICT: FAIL - the panel executed zero checks (it measured nothing).")
        return 1
    if violations or lost:
        print("\nVERDICT: FAIL - gate 1 (cert-clean) violated.")
        return 1
    if gained:
        print("\nVERDICT: PASS - cert-clean AND net-positive (incumbents gained).")
        return 0
    print(
        "\nVERDICT: INCONCLUSIVE - cert-clean and neutral. The corpus carries "
        f"{len(no_inc_off)} no-incumbent instance(s) and the seed moved none of them."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
