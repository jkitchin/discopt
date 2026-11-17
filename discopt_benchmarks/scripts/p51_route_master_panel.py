"""highs-parity P5.1: the routed OA master, HiGHS (OFF) vs the in-house lazy master
(ON, ``DISCOPT_CONVEX_ROUTE_OA_MASTER=simplex``). Bars pre-registered here before the run
(the plan's R2a row): BOTH

* cert-clean -- no bound past its reference (``minlplib.solu`` and the arm's own verified
  incumbent), no certified result whose point fails ``verify_point``, no error; and
* certified ON >= certified OFF, 0 models certified OFF but not ON ("lost"), and total
  wall ON <= 1.25 x OFF -- separately on each population below.

Populations (``--population``):

``route``  the section-76.4 route panel (``master_parity_panel._route_models``: routed in-repo
           ``.nl`` x row-scaling spans {0, 3, 6} + 16 portfolios), kernel at its default.
``syn``    every ``syn*``/``rsyn*`` name in ``minlplib.solu`` whose ``.nl`` is in
           ``ref/minlplib/nl`` (fetched from minlplib.org; git-ignored) AND that the convex
           router diverts on this build -- selected here, before the run, since the
           51-instance list of the retired ``auto`` panel was not recorded. Run with the
           convex kernel at its default (``--kernel on``) and off (``--kernel off``,
           ``DISCOPT_CONVEX_KERNEL=0``); the plan asks for both.

In-process, arm order alternating per model, ``--time-limit`` 30 s. Mechanism-fired: ON
rows whose route says ``master=simplex`` must be > 0, OFF rows ``master=highs``. Exceptions
are recorded and fail the run; progress per solve; non-zero exit when nothing ran.

**Flag arms (#2223).** ``--flag NAME --on V --off V`` (or the general ``--arms
"off=;k3=DISCOPT_MILP_LAZY_RESTARTS=3;both=A=1,B=3"``, the first arm the reference) compares
environment settings of the in-house master instead of the two masters: every arm is pinned
to ``DISCOPT_CONVEX_ROUTE_OA_MASTER=simplex`` and the arm order rotates per model. A
repeated arm (``off2=``) measures run-to-run spread. Each arm is judged against the
reference by the CLAUDE.md section 5 bars: cert-clean (no bound past ``minlplib.solu`` or
the arm's own verified incumbent, no certified unverified point, no error, no
certification lost) and not harmful (certified, wall, and bound on rows neither
certifies). Mechanism-fired per arm, read from the lazy master's ``callback_stats``
(free, unlike ``DISCOPT_PROFILE``): rows with ``restarts > 0`` and rows whose master was
handed a ``relaxable_rows`` mask.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import master_parity_panel as MP  # noqa: E402, N812

FLAG = "DISCOPT_CONVEX_ROUTE_OA_MASTER"
SYN_DIR = REPO / "ref" / "minlplib" / "nl"


def _syn_models():
    import discopt.solver as solver
    from discopt.modeling.core import from_nl

    names = sorted(n for n in MP._solu() if n.startswith(("syn", "rsyn")))
    out, skipped = [], []
    for n in names:
        p = SYN_DIR / f"{n}.nl"
        if not p.exists():
            skipped.append((n, "no .nl"))
            continue
        if not solver._convex_minlp_auto_route(from_nl(str(p)))[0]:
            skipped.append((n, "not routed"))
            continue
        out.append((n, n, (lambda p=p: from_nl(str(p)))))
    return out, skipped


def solve_one(lab, inst, build, on, a, solu):
    from discopt.modeling.core import ObjectiveSense
    from discopt.validation.feasibility import verify_point

    # Both arms are pinned explicitly: since highs-parity D3 (2026-11-07) the default
    # master is the in-house one, so an unset variable no longer means HiGHS.
    os.environ[FLAG] = "simplex" if on else "highs"
    try:
        m = build()
        sense = 1.0 if m._objective.sense == ObjectiveSense.MINIMIZE else -1.0
        t = time.perf_counter()
        r = m.solve(time_limit=a.time_limit)
        w = time.perf_counter() - t
    finally:
        os.environ.pop(FLAG, None)
    ok = None
    if r.x is not None:
        flat = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
        ok = bool(verify_point(m, flat).ok)
    viol, checks = [], 0
    refs = [("solu", solu[inst])] if inst in solu else []
    if ok and r.objective is not None:
        refs.append(("incumbent", float(r.objective)))
    if r.bound is not None:
        for lab_ref, ref in refs:
            checks += 1
            if sense * r.bound > sense * ref + 1e-6 * max(1.0, abs(ref)):
                viol.append(f"{lab} on={on} bound {r.bound} past {lab_ref} {ref}")
    if r.gap_certified and ok is False:
        viol.append(f"{lab} on={on} certified an unverified point")
    route = r.algorithm_route or ""
    return {"status": str(r.status), "cert": bool(r.gap_certified), "obj": r.objective,
            "bound": r.bound, "wall": w, "feas_ok": ok, "route": route,
            "fired": ("master=simplex" in route) if on else ("master=highs" in route),
            "checks": checks, "violations": viol}  # fmt: skip


def _parse_arms(spec: str) -> list[tuple[str, dict[str, str]]]:
    arms = []
    for part in spec.split(";"):
        name, _, env = part.partition("=")
        settings = {}
        for kv in filter(None, env.split(",")):
            k, _, v = kv.partition("=")
            settings[k.strip()] = v.strip()
        arms.append((name.strip(), settings))
    if len(arms) < 2 or len({a for a, _ in arms}) != len(arms):
        raise SystemExit(f"--arms needs >= 2 distinct arms, got {spec!r}")
    return arms


def solve_arm(lab, inst, build, env, a, solu):
    """One solve under ``env`` on the in-house master, plus master mechanism counters."""
    import discopt.solvers.milp_simplex as ms
    from discopt.modeling.core import ObjectiveSense
    from discopt.validation.feasibility import verify_point

    saved = {k: os.environ.get(k) for k in [FLAG, *env]}
    os.environ[FLAG] = "simplex"
    os.environ.update(env)
    mech = {"master_calls": 0, "restarts": 0, "masked_calls": 0, "master_nodes": 0}
    orig = ms.solve_milp_with_lazy_cuts

    def counted(*args, **kwargs):
        out = orig(*args, **kwargs)
        cs = out.callback_stats or {}
        mech["master_calls"] += 1
        mech["restarts"] += int(cs.get("restarts", 0) or 0)
        mech["masked_calls"] += int((cs.get("relaxable_rows", 0) or 0) > 0)
        mech["master_nodes"] += int(out.node_count or 0)
        return out

    ms.solve_milp_with_lazy_cuts = counted
    try:
        m = build()
        sense = 1.0 if m._objective.sense == ObjectiveSense.MINIMIZE else -1.0
        t = time.perf_counter()
        r = m.solve(time_limit=a.time_limit)
        w = time.perf_counter() - t
    finally:
        ms.solve_milp_with_lazy_cuts = orig
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ok = None
    if r.x is not None:
        flat = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
        ok = bool(verify_point(m, flat).ok)
    viol, checks = [], 0
    refs = [("solu", solu[inst])] if inst in solu else []
    if ok and r.objective is not None:
        refs.append(("incumbent", float(r.objective)))
    if r.bound is not None:
        for lab_ref, ref in refs:
            checks += 1
            if sense * r.bound > sense * ref + 1e-6 * max(1.0, abs(ref)):
                viol.append(f"{lab} env={env} bound {r.bound} past {lab_ref} {ref}")
    if r.gap_certified and ok is False:
        viol.append(f"{lab} env={env} certified an unverified point")
    route = r.algorithm_route or ""
    return {"status": str(r.status), "cert": bool(r.gap_certified), "obj": r.objective,
            "bound": r.bound, "sense": sense, "wall": w, "feas_ok": ok, "route": route,
            "fell_back": "fell back" in route, "checks": checks, "violations": viol,
            **mech}  # fmt: skip


def _arm_verdict(rows, ref_arm, arm):
    viol = [v for r in rows for v in r[arm]["violations"]]
    cert = {x: sum(r[x]["cert"] for r in rows) for x in (ref_arm, arm)}
    lost = [r["model"] for r in rows if r[ref_arm]["cert"] and not r[arm]["cert"]]
    gained = [r["model"] for r in rows if r[arm]["cert"] and not r[ref_arm]["cert"]]
    wall = {x: sum(r[x]["wall"] for r in rows) for x in (ref_arm, arm)}
    tighter, looser = [], []
    for r in rows:
        o, n = r[ref_arm], r[arm]
        if o["cert"] or n["cert"] or o["bound"] is None or n["bound"] is None:
            continue
        d = n["sense"] * (n["bound"] - o["bound"])
        tol = 1e-6 * max(1.0, abs(o["bound"]))
        if d > tol:
            tighter.append(r["model"])
        elif d < -tol:
            looser.append(r["model"])
    walls = [r[arm]["wall"] for r in rows]
    return {
        "certified": cert, "lost": lost, "gained": gained, "wall": wall,
        "wall_ratio": wall[arm] / wall[ref_arm] if wall[ref_arm] else None,
        "median_wall": float(np.median(walls)) if walls else None,
        "bound_tighter_uncert": tighter, "bound_looser_uncert": looser,
        "violations": viol, "checks": sum(r[arm]["checks"] for r in rows),
        "fell_back": sum(r[arm]["fell_back"] for r in rows),
        "fired_restart_rows": sum(r[arm]["restarts"] > 0 for r in rows),
        "fired_mask_rows": sum(r[arm]["masked_calls"] > 0 for r in rows),
        "master_nodes": sum(r[arm]["master_nodes"] for r in rows),
        "cert_clean": not viol and not lost,
    }  # fmt: skip


def main_arms(a, arms) -> int:
    hdr = MP._header()
    solu = MP._solu()
    skipped = []
    if a.population == "route":
        _routed, models = MP._route_models([float(s) for s in a.spans.split(",")])
    else:
        models, skipped = _syn_models()
    if a.only:
        keep = set(a.only.split(","))
        models = [m for m in models if m[0] in keep]
    names = [n for n, _ in arms]
    print("header", json.dumps(hdr), "models", len(models), "arms", arms, flush=True)
    rows, errors = [], []
    done: set[str] = set()
    if a.resume and Path(a.out).exists():
        # Resume an interrupted run: keep its rows (same arms, recorded commits).
        prev = json.loads(Path(a.out).read_text())
        if [n for n, _ in prev["arms"]] != names:
            raise SystemExit(f"--resume: {a.out} has arms {prev['arms']}, not {arms}")
        rows = prev["rows"]
        done = {r["model"] for r in rows}
        hdr = {**hdr, "resumed_from": prev["header"]}
        print("resume", len(rows), "rows kept", flush=True)

    def dump(extra=None):
        Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "arms": arms,
                                           "rows": rows, "errors": errors, "skipped": skipped,
                                           **(extra or {})},
                                          indent=1, default=str) + "\n")  # fmt: skip

    for i, (lab, inst, build) in enumerate(models, 1):
        if lab in done:
            continue
        k = i % len(arms)
        order = arms[k:] + arms[:k]
        rec = {"model": lab}
        try:
            for name, env in order:
                rec[name] = solve_arm(lab, inst, build, env, a, solu)
        except Exception as e:  # recorded, counted, fails the run
            errors.append(f"{lab}: {type(e).__name__}: {e}"[:300])
            print(f"[{i}/{len(models)}] {lab} ERROR {errors[-1]}", flush=True)
            dump()
            continue
        rows.append(rec)
        line = " | ".join(
            f"{n} {rec[n]['cert']!s:5} {rec[n]['wall']:6.2f}s r{rec[n]['restarts']}"
            f"m{rec[n]['masked_calls']}"
            for n in names
        )
        print(f"[{i}/{len(models)}] {lab:34s} {line}", flush=True)
        dump()
    verdict = {n: _arm_verdict(rows, names[0], n) for n in names[1:]}
    verdict["errors"] = errors
    dump({"verdict": verdict})
    print(json.dumps(verdict, indent=1, default=str), flush=True)
    checks = sum(r[n]["checks"] for r in rows for n in names)
    print("EXECUTED_SOLVES", len(rows) * len(names), "BOUND_CHECKS", checks, flush=True)
    return 0 if rows and checks and not errors else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", choices=("route", "syn"), required=True)
    ap.add_argument("--kernel", choices=("on", "off"), default="on")
    ap.add_argument("--spans", default="0,3,6")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flag", default="", help="flag arms: env var compared ON vs OFF")
    ap.add_argument("--on", default="1")
    ap.add_argument("--off", default="0")
    ap.add_argument("--arms", default="", help='e.g. "off=;k3=DISCOPT_MILP_LAZY_RESTARTS=3"')
    ap.add_argument("--resume", action="store_true", help="keep the rows already in --out")
    a = ap.parse_args(argv)
    MP._assert_discopt_under_test()
    if a.kernel == "off":
        os.environ["DISCOPT_CONVEX_KERNEL"] = "0"
    if a.flag and a.arms:
        raise SystemExit("--flag and --arms are exclusive")
    if a.flag:
        return main_arms(a, [("off", {a.flag: a.off}), ("on", {a.flag: a.on})])
    if a.arms:
        return main_arms(a, _parse_arms(a.arms))
    hdr = MP._header()
    solu = MP._solu()
    skipped = []
    if a.population == "route":
        _routed, models = MP._route_models([float(s) for s in a.spans.split(",")])
    else:
        models, skipped = _syn_models()
    if a.only:
        keep = set(a.only.split(","))
        models = [m for m in models if m[0] in keep]
    print("header", json.dumps(hdr), "models", len(models), "skipped", len(skipped), flush=True)
    rows, errors = [], []
    for i, (lab, inst, build) in enumerate(models, 1):
        order = (False, True) if i % 2 else (True, False)
        rec = {"model": lab}
        try:
            for on in order:
                rec["on" if on else "off"] = solve_one(lab, inst, build, on, a, solu)
        except Exception as e:  # recorded, counted, fails the run
            errors.append(f"{lab}: {type(e).__name__}: {e}"[:300])
            print(f"[{i}/{len(models)}] {lab} ERROR {errors[-1]}", flush=True)
            continue
        rows.append(rec)
        o, n = rec["off"], rec["on"]
        print(f"[{i}/{len(models)}] {lab:34s} OFF {o['status']}/{o['cert']} {o['wall']:6.2f}s "
              f"| ON {n['status']}/{n['cert']} {n['wall']:6.2f}s fired={o['fired']}/{n['fired']}",
              flush=True)  # fmt: skip
        Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows,
                                           "errors": errors, "skipped": skipped},
                                          indent=1, default=str) + "\n")  # fmt: skip
    viol = [v for r in rows for arm in ("off", "on") for v in r[arm]["violations"]]
    checks = sum(r[arm]["checks"] for r in rows for arm in ("off", "on"))
    cert = {arm: sum(r[arm]["cert"] for r in rows) for arm in ("off", "on")}
    lost = [r["model"] for r in rows if r["off"]["cert"] and not r["on"]["cert"]]
    gained = [r["model"] for r in rows if r["on"]["cert"] and not r["off"]["cert"]]
    wall = {arm: sum(r[arm]["wall"] for r in rows) for arm in ("off", "on")}
    fired = {arm: sum(r[arm]["fired"] for r in rows) for arm in ("off", "on")}
    bars = {
        "cert_clean": not viol and not errors,
        "certified_on_ge_off": cert["on"] >= cert["off"],
        "zero_lost": not lost,
        "wall_on_le_1.25x": wall["off"] > 0 and wall["on"] <= 1.25 * wall["off"],
    }
    verdict = {"certified": cert, "lost": lost, "gained": gained, "wall": wall,
               "wall_ratio": wall["on"] / wall["off"] if wall["off"] else None,
               "fired": fired, "checks": checks, "violations": viol, "errors": errors,
               "bars": bars, "pass": all(bars.values())}  # fmt: skip
    Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows,
                                       "skipped": skipped, "verdict": verdict},
                                      indent=1, default=str) + "\n")  # fmt: skip
    print(json.dumps(verdict, indent=1, default=str), flush=True)
    ok = rows and checks and fired["on"] > 0 and fired["off"] > 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
