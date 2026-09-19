#!/usr/bin/env python
"""#1346 — CLAUDE.md §5 Regime-2 graduation panel for ``DISCOPT_CONVEX_KERNEL``.

Flag ON vs OFF over the **in-repo** MINLPLib corpus
(``python/tests/data/minlplib_nl``), arms **interleaved within each instance**
and the arm order alternated by instance index so a systematic
first-arm-is-colder effect cannot masquerade as a result (CLAUDE.md §9).
``deterministic=True`` renders the role-2 wall sub-budgets inert (#912/#1116),
so a status/node difference between the arms is attributable to the flag rather
than to the clock.

Scored on the two §5 bars:

1. **cert-clean** — ``incorrect_count = 0``: no bound past its reference optimum,
   no ``bound``/incumbent inversion (checked against each model's own sense), no
   ``gap_certified=True`` instance regressing to uncertified, objective drift
   within ``rel=1e-4``.
2. **net-positive** — measurably helpful on certification / nodes / wall, not
   merely sound (the ``DISCOPT_CUT_INHERIT`` lesson).

**Instrument discipline.** The script asserts which ``discopt`` it loaded (§8),
prints per-instance as it goes (§10), counts executed comparisons and exits
non-zero when that count is zero (§6), and never swallows an exception in the
instrument — a raise is recorded as an ``ERROR`` cell and counted, never hidden
(§7).

**Known coverage limit, stated so the verdict is not over-read.** Only 16 of the
66 in-repo instances carry a reference optimum in ``known_optima.toml``; the
``.solu`` oracle lives in the out-of-repo MINLPLib snapshot. For the other 50 the
cert-clean bar is enforced through the sense-correct bound/incumbent invariant
and the certification-regression check, which do not need an oracle. The in-repo
corpus also contains no member of the ``watercontamination`` counter-case class
(a model that classifies convex and then spends the attempt without producing a
bound), so **this panel cannot exercise that risk** — it is the reason
graduation ships with the root-bound guard rather than on the panel alone.

Usage::

    python -u discopt_benchmarks/scripts/issue1346_convex_kernel_graduation_panel.py \
        [--time-limit 60] [--out results.json] [--only name1,name2]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import discopt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CORPUS = REPO / "python" / "tests" / "data" / "minlplib_nl"
ORACLE = REPO / "python" / "tests" / "data" / "known_optima.toml"

REL_TOL = 1e-4
ABS_TOL = 1e-6


def _load_oracle() -> dict[str, float]:
    import tomllib

    data = tomllib.loads(ORACLE.read_text())
    return {
        k: float(v["optimum"]) for k, v in data.items() if isinstance(v, dict) and "optimum" in v
    }


def _sense_is_min(model) -> bool:
    """True for a MINIMIZE objective. Read from the model, never assumed.

    The bound/incumbent invariant flips with the sense, so guessing it turns the
    soundness check into noise -- ``syn05hfsg`` is a MAXIMIZE model whose
    ``bound > incumbent`` is correct, and a min-only check would report it as a
    false certificate.
    """
    obj = model._objective
    return "MINIMIZE" in str(obj.sense).upper()


def _solve_arm(path: pathlib.Path, flag: str, time_limit: float) -> dict:
    """One arm. An exception is recorded, not swallowed (§7)."""
    from discopt.modeling.core import from_nl
    from discopt.solvers._convex_kernel import last_guard_decision

    os.environ["DISCOPT_CONVEX_KERNEL"] = flag
    model = from_nl(str(path))
    is_min = _sense_is_min(model)
    t0 = time.perf_counter()
    try:
        r = model.solve(time_limit=time_limit, deterministic=True)
        cell = {
            "status": r.status,
            "objective": r.objective,
            "bound": r.bound,
            "gap_certified": bool(getattr(r, "gap_certified", False)),
            "node_count": getattr(r, "node_count", None),
        }
    except Exception as exc:  # noqa: BLE001 -- recorded, never hidden
        cell = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "objective": None,
            "bound": None,
            "gap_certified": False,
            "node_count": None,
        }
    cell["wall"] = time.perf_counter() - t0
    cell["is_min"] = is_min
    # §6: read the guard's own verdict rather than inferring it from a wall reading.
    reason, probe_bound, probe_nodes = last_guard_decision()
    cell["guard"] = {"reason": reason, "probe_bound": probe_bound, "probe_nodes": probe_nodes}
    return cell


def _invariant_violations(cell: dict, optimum: float | None) -> list[str]:
    """Sense-correct soundness checks on a single arm."""
    bad: list[str] = []
    obj, bnd, is_min = cell["objective"], cell["bound"], cell["is_min"]
    slack = ABS_TOL + REL_TOL * max(1.0, abs(obj) if obj is not None else 1.0)
    if obj is not None and bnd is not None:
        if is_min and bnd > obj + slack:
            bad.append(f"bound {bnd!r} above incumbent {obj!r} (min)")
        if not is_min and bnd < obj - slack:
            bad.append(f"bound {bnd!r} below incumbent {obj!r} (max)")
    if bnd is not None and optimum is not None:
        oslack = ABS_TOL + REL_TOL * max(1.0, abs(optimum))
        if is_min and bnd > optimum + oslack:
            bad.append(f"bound {bnd!r} above reference optimum {optimum!r} (min)")
        if not is_min and bnd < optimum - oslack:
            bad.append(f"bound {bnd!r} below reference optimum {optimum!r} (max)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument(
        "--guard",
        choices=("on", "off"),
        default="on",
        help=(
            "state of the #1346 root-bound guard during the ON arm. 'off' reproduces "
            "the pre-#1346 single-shot attempt, which is the arm the §5 graduation "
            "gate scores; 'on' confirms the shipped default costs nothing."
        ),
    )
    args = ap.parse_args()

    # §8 -- name the code under test, not just the path. Both runs use ONE build and
    # differ by one env var, so a build difference cannot masquerade as a result; the
    # marker below is what distinguishes the two arms in the record.
    print(f"marker: discopt.__file__ = {discopt.__file__}", flush=True)
    assert (REPO / "python" / "discopt" / "__init__.py").samefile(discopt.__file__), (
        f"loaded the wrong discopt: {discopt.__file__}"
    )

    from discopt.modeling.core import from_nl
    from discopt.solvers._convex_kernel import (  # noqa: PLC0415
        build_convex_spec,
        convex_kernel_guard_enabled,
    )

    os.environ["DISCOPT_CONVEX_KERNEL_GUARD"] = "1" if args.guard == "on" else "0"
    assert convex_kernel_guard_enabled() is (args.guard == "on"), (
        "marker mismatch: the loaded _convex_kernel does not honour "
        "DISCOPT_CONVEX_KERNEL_GUARD -- this is a pre-#1346 module"
    )
    print(
        f"marker: #1346 guard present, requested={args.guard}, "
        f"effective={convex_kernel_guard_enabled()}",
        flush=True,
    )

    oracle = _load_oracle()
    names = sorted(p.stem for p in CORPUS.glob("*.nl"))
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        names = [n for n in names if n in wanted]
    print(
        f"corpus: {len(names)} instances, time_limit={args.time_limit}s, "
        f"oracle-backed: {sum(n in oracle for n in names)}",
        flush=True,
    )

    comparisons = 0
    rows: list[dict] = []
    for idx, name in enumerate(names):
        path = CORPUS / f"{name}.nl"
        os.environ["DISCOPT_CONVEX_KERNEL"] = "1"
        try:
            eligible = build_convex_spec(from_nl(str(path))) is not None
            elig_err = ""
        except Exception as exc:  # noqa: BLE001 -- §7
            eligible, elig_err = False, f"{type(exc).__name__}: {exc}"[:120]

        # Alternate arm order by index so a first-arm effect cannot alias the flag.
        order = ("off", "on") if idx % 2 == 0 else ("on", "off")
        arms: dict[str, dict] = {}
        for arm in order:
            arms[arm] = _solve_arm(path, "1" if arm == "on" else "0", args.time_limit)
            comparisons += 1

        off, on = arms["off"], arms["on"]
        opt = oracle.get(name)
        viol = {
            "off": _invariant_violations(off, opt),
            "on": _invariant_violations(on, opt),
        }
        cert_regression = off["gap_certified"] and not on["gap_certified"]
        cert_gain = on["gap_certified"] and not off["gap_certified"]
        # Objective drift is a SOUNDNESS check between two certified optima, so it is
        # only meaningful when BOTH arms certified. Comparing a certified optimum
        # against the other arm's uncertified incumbent measures the gain, not a
        # defect: on `clay0303hfsg` the OFF arm stops 12% above the optimum, and
        # scoring that as "drift 3.6e-01" reported this panel's clearest win as a
        # cert-clean FAILURE. Improvement over an uncertified arm is recorded
        # separately, as a gain, and never gates.
        drift = None
        both_certified = off["gap_certified"] and on["gap_certified"]
        if both_certified and off["objective"] is not None and on["objective"] is not None:
            denom = max(1.0, abs(off["objective"]))
            drift = abs(on["objective"] - off["objective"]) / denom

        tag = "GAIN " if cert_gain else ("REGR " if cert_regression else "     ")
        print(
            f"[{idx + 1:2}/{len(names)}] {tag}{name:22} elig={str(eligible):5} "
            f"off={off['status']}/{off['gap_certified']}/{off['wall']:.1f}s/"
            f"n={off['node_count']} "
            f"on={on['status']}/{on['gap_certified']}/{on['wall']:.1f}s/"
            f"n={on['node_count']}"
            + (f"  drift={drift:.2e}" if drift is not None else "")
            + (f"  UNSOUND_OFF={viol['off']}" if viol["off"] else "")
            + (f"  UNSOUND_ON={viol['on']}" if viol["on"] else ""),
            flush=True,
        )
        rows.append(
            {
                "name": name,
                "eligible": eligible,
                "eligibility_error": elig_err,
                "off": off,
                "on": on,
                "optimum": opt,
                "violations": viol,
                "cert_regression": cert_regression,
                "cert_gain": cert_gain,
                "objective_drift": drift,
                "arm_order": list(order),
            }
        )

    print(f"\nexecuted solves: {comparisons}", flush=True)
    if comparisons == 0:
        print("PROBE FIRED NOTHING -- no comparison was executed", flush=True)
        return 2

    unsound_on = [r["name"] for r in rows if r["violations"]["on"]]
    unsound_off = [r["name"] for r in rows if r["violations"]["off"]]
    regressions = [r["name"] for r in rows if r["cert_regression"]]
    gains = [r["name"] for r in rows if r["cert_gain"]]
    drifted = [
        r["name"]
        for r in rows
        if r["objective_drift"] is not None and r["objective_drift"] > REL_TOL
    ]
    errors = [r["name"] for r in rows if "ERROR" in (r["off"]["status"], r["on"]["status"])]
    eligible = [r["name"] for r in rows if r["eligible"]]
    wall_off = sum(r["off"]["wall"] for r in rows)
    wall_on = sum(r["on"]["wall"] for r in rows)

    print("\n=== GATE 1: CERT-CLEAN ===", flush=True)
    print(f"  unsound (ON arm)          : {len(unsound_on)} {unsound_on}", flush=True)
    print(f"  unsound (OFF arm, context): {len(unsound_off)} {unsound_off}", flush=True)
    print(f"  certification regressions : {len(regressions)} {regressions}", flush=True)
    print(
        f"  objective drift > {REL_TOL:g}   : {len(drifted)} {drifted}"
        f"   (scored on the {sum(1 for r in rows if r['objective_drift'] is not None)}"
        f" instances where BOTH arms certified)",
        flush=True,
    )
    print(f"  solver errors             : {len(errors)} {errors}", flush=True)
    clean = not (unsound_on or regressions or drifted or errors)
    print(f"  --> {'PASS' if clean else 'FAIL'}", flush=True)

    print("\n=== GATE 2: NET-POSITIVE ===", flush=True)
    print(f"  kernel-eligible instances : {len(eligible)}/{len(rows)}", flush=True)
    print(f"  certification GAINS       : {len(gains)} {gains}", flush=True)
    print(f"  total wall OFF / ON       : {wall_off:.1f}s / {wall_on:.1f}s", flush=True)
    for r in rows:
        if r["eligible"]:
            print(
                f"    {r['name']:22} off {r['off']['wall']:7.2f}s "
                f"n={r['off']['node_count']}  ->  on {r['on']['wall']:7.2f}s "
                f"n={r['on']['node_count']}",
                flush=True,
            )
    net_positive = len(gains) > 0 and not regressions
    print(f"  --> {'PASS' if net_positive else 'INCONCLUSIVE/FAIL'}", flush=True)

    # §6: the guard must be observed firing, not assumed. A run whose ON arm never
    # records a guard decision measured an unguarded kernel and says nothing about
    # the guard, whatever the wall clock did.
    hist: dict[str, int] = {}
    for r in rows:
        hist[r["on"]["guard"]["reason"]] = hist.get(r["on"]["guard"]["reason"], 0) + 1
    print(f"\n=== GUARD (requested {args.guard}) ===", flush=True)
    for reason, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:28} {n}", flush=True)
    observed = sum(n for k, n in hist.items() if k not in ("not_run", "not_eligible"))
    print(f"  guard decisions observed on eligible instances: {observed}", flush=True)

    print(
        f"\nVERDICT: gate1={'PASS' if clean else 'FAIL'} "
        f"gate2={'PASS' if net_positive else 'INCONCLUSIVE/FAIL'}",
        flush=True,
    )

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(
                {
                    "time_limit": args.time_limit,
                    "discopt_file": discopt.__file__,
                    "guard": args.guard,
                    "guard_histogram": hist,
                    "comparisons": comparisons,
                    "gate1_cert_clean": clean,
                    "gate2_net_positive": net_positive,
                    "rows": rows,
                },
                indent=2,
                default=str,
            )
        )
        print(f"wrote {args.out}", flush=True)
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
