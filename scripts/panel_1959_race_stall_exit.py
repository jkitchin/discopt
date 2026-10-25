"""#1959 panel: end the lift race's lift slice on the MILP engine's stall, ON vs OFF.

``DISCOPT_IPX_RACE_STALL_EXIT`` (see ``solver._ipx_race_stall_exit_enabled``).
OFF is the #1940 race as merged: when the native MILP engine's #291 stall guard
defers with no incumbent inside the 5 s lift slice, the rest of the slice runs the
Python MILP tree on the lifted model. ON ends the slice there, keeping the engine's
bound, and moves on to the un-lifted slice.

Instances and cert-clean checks are ``panel_1940_lift_race.py``'s (imported, not
copied): the 12 in-repo instances the lift is adopted on, the dense / sparse /
dense+linear integer QPs, the constraint-product families, integer x continuous
objectives, knapsack, the #1937 PWL case and the array-variable models. Every
incumbent is re-verified on the original model, every bound is checked against the
best certified objective, every certified objective against the others, and no
instance certified by OFF may be uncertified by ON.

The hypothesis is also tested directly. In the OFF arm, every Python-tree run that
follows a stall inside a lift slice is recorded, with whether it returned an
incumbent or a certificate. The change rests on that tree never doing either; any
such run is printed as ``KILL`` and fails the panel.

``--slice S`` shrinks the race's slices to ``S`` seconds (and its minimum limit to
``3 S``) and the time limit in proportion, a proxy for a machine ``5 / S`` times
slower (issue #1959: CI runs about 3x slower, so ``--slice 1.67``).

Usage: ``python -u scripts/panel_1959_race_stall_exit.py [--time-limit 30] [--reps 2]
[--slice 5] [--only a,b] [--nl a.nl,b.nl]``. Prints per-solve progress and an executed-check count;
exits non-zero on any failure or when nothing was checked (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))

import discopt.solver as solver_mod  # noqa: E402
import panel_1940_lift_race as P  # noqa: E402

# CLAUDE.md §8: the version under test carries the stall exit.
assert hasattr(solver_mod, "_ipx_race_stall_exit_enabled"), solver_mod.__file__
print("solver loaded from", solver_mod.__file__, flush=True)

FLAG = "DISCOPT_IPX_RACE_STALL_EXIT"
STALL_ROUTE = "ended on the engine's stall"

CHECKS = 0
FAILS: list[str] = []
#: per solve: Python-tree runs after a stall in a lift slice (OFF) / stall exits (ON)
EVENTS: list[str] = []


def check(cond: bool, msg: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)
        print("  FAIL:", msg, flush=True)


def _in_lift_slice() -> bool:
    return solver_mod._IPX_RACE_ARM.get() == "lift" and solver_mod._IPX_RACE_IN_SLICE.get()


_real_bb = solver_mod._solve_milp_bb
_real_solve = solver_mod.solve_model


def _bb(*a, **k):
    r = _real_bb(*a, **k)
    if _in_lift_slice():
        inc = r.objective is not None
        cert = solver_mod._result_certifies(r)
        EVENTS.append(f"pytree inc={inc} cert={cert}")
        # The hypothesis: this run never yields an incumbent or a certificate.
        check(not inc and not cert, f"KILL: Python tree after a stall gave inc={inc} cert={cert}")
    return r


def _solve(*a, **k):
    r = _real_solve(*a, **k)
    if _in_lift_slice() and STALL_ROUTE in (getattr(r, "algorithm_route", None) or ""):
        EVENTS.append("stall-exit")
    return r


solver_mod._solve_milp_bb = _bb
solver_mod.solve_model = _solve


def solve_arm(name, on, tl):
    os.environ[FLAG] = "1" if on else "0"
    EVENTS.clear()
    rec = P.solve_arm(name, True, tl)  # race ON in both arms; FLAG is the variable
    rec["stall_exit"] = on
    rec["events"] = list(EVENTS)
    if on:
        check("pytree" not in " ".join(EVENTS), f"{name} ON: Python tree ran inside a lift slice")
    else:
        check("stall-exit" not in EVENTS, f"{name} OFF: the stall exit fired")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--slice", type=float, default=None, help="race slice length, seconds")
    ap.add_argument("--only", default=None, help="comma-separated instance names")
    ap.add_argument("--out", default=None, help="JSON-lines file, one line per solve")
    ap.add_argument(
        "--nl",
        default=None,
        help="comma-separated extra .nl files (e.g. MINLPLib tln6/tln7) added to the panel",
    )
    args = ap.parse_args()
    for path in args.nl.split(",") if args.nl else []:
        nm = os.path.splitext(os.path.basename(path))[0]
        P.INSTANCES[nm] = ((lambda path=path: (P.from_nl(path),) * 2), False)
    tl = args.time_limit
    if args.slice is not None:
        scale = args.slice / solver_mod._IPX_RACE_SLICE_S
        solver_mod._IPX_RACE_SLICE_S = args.slice
        solver_mod._IPX_RACE_MIN_LIMIT_S = 3.0 * args.slice
        tl = args.time_limit * scale
    print(
        f"load: {os.getloadavg()} slice {solver_mod._IPX_RACE_SLICE_S} s, limit {tl:.2f} s",
        flush=True,
    )
    names = args.only.split(",") if args.only else list(P.INSTANCES)
    out = open(args.out, "a") if args.out else None
    rows = []
    n_stall = 0
    for name in names:
        runs = {True: [], False: []}
        order = [on for _ in range(args.reps // 2 or 1) for on in (True, False, False, True)]
        for on in order[: 2 * args.reps]:
            rec = solve_arm(name, on, tl)
            n_stall += rec["events"].count("stall-exit")
            runs[on].append(rec)
            print(json.dumps(rec, default=str), flush=True)
            if out:
                out.write(json.dumps(rec, default=str) + "\n")
                out.flush()
        allr = runs[True] + runs[False]
        sgn = -1.0 if allr[0]["maximize"] else 1.0
        cert_objs = [sgn * a["obj"] for a in allr if a["cert"]]
        incs = [sgn * a["obj"] for a in allr if a["obj"] is not None]
        ref = min(cert_objs) if cert_objs else (min(incs) if incs else None)
        for a in allr:
            tag = f"{name} {'ON' if a['stall_exit'] else 'OFF'}"
            if a["viol"] is not None:
                check(a["viol"] <= 1e-5, f"{tag}: incumbent violates the model by {a['viol']:.3g}")
            if ref is not None and a["bound"] is not None:
                b = sgn * a["bound"]
                check(b <= ref + 1e-6 * max(1.0, abs(ref)), f"{tag}: bound {b} above {ref}")
            if a["cert"] and cert_objs:
                o = sgn * a["obj"]
                check(
                    abs(o - min(cert_objs)) <= 1e-4 * max(1.0, abs(o)) + 1e-6,
                    f"{tag}: certified {o} disagrees with {min(cert_objs)}",
                )
        if any(a["cert"] for a in runs[False]):
            check(all(a["cert"] for a in runs[True]), f"{name}: certification regression")
        row = dict(
            name=name,
            on_wall=statistics.mean(a["wall"] for a in runs[True]),
            off_wall=statistics.mean(a["wall"] for a in runs[False]),
            on_sd=statistics.pstdev(a["wall"] for a in runs[True]),
            off_sd=statistics.pstdev(a["wall"] for a in runs[False]),
            on_cert=sum(a["cert"] for a in runs[True]),
            off_cert=sum(a["cert"] for a in runs[False]),
        )
        rows.append(row)
        print(
            f"  {name:16s} OFF {row['off_wall']:6.2f}+-{row['off_sd']:.2f} cert {row['off_cert']}"
            f" | ON {row['on_wall']:6.2f}+-{row['on_sd']:.2f} cert {row['on_cert']}",
            flush=True,
        )
    tot_on = sum(r["on_wall"] for r in rows)
    tot_off = sum(r["off_wall"] for r in rows)
    cert_on = sum(r["on_cert"] > 0 for r in rows)
    cert_off = sum(r["off_cert"] > 0 for r in rows)

    def _beyond(r, faster):
        a, b = (r["on_wall"], r["off_wall"]) if faster else (r["off_wall"], r["on_wall"])
        return a < b - 2 * max(r["on_sd"], r["off_sd"], 0.05) and a < b / 1.2

    faster = [r["name"] for r in rows if _beyond(r, True)]
    slower = [r["name"] for r in rows if _beyond(r, False)]
    print(
        f"\nSUMMARY instances={len(rows)} total wall OFF {tot_off:.1f}s ON {tot_on:.1f}s; "
        f"certified OFF {cert_off} ON {cert_on}; stall exits {n_stall}; "
        f"faster {len(faster)} {faster}; slower {len(slower)} {slower}",
        flush=True,
    )
    print(f"executed checks: {CHECKS}; failures: {len(FAILS)}", flush=True)
    for f in FAILS:
        print("  -", f)
    if CHECKS == 0 or FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
