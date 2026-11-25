"""#2335: HiGHS presolve sparsify (rule 14) on vs off, on highspy >= 1.16.

The MILP route and the ``milp_highs`` masters set ``presolve_rule_off =
MILP_PRESOLVE_RULE_OFF``, which holds bit 13 (#1634, unsound parallel-column
reduction) and bit 14 (sparsify, #1667: the 1.12-1.15 presolve cycle). This panel
measures bit 14 alone:

* ``off``: bit 13 and bit 14 (sparsify off; the route's value before #2335);
* ``on``: bit 13 only (sparsify on; the route's value since #2335).

The arm is applied inside each worker subprocess by patching
``lp_milp_highs.MILP_PRESOLVE_RULE_OFF`` and ``milp_highs.MASTER_PRESOLVE_RULE_OFF``
to rule 13 plus (``off``) or minus (``on``) bit 14, and the
worker records every ``presolve_rule_off`` value HiGHS was actually handed, by
wrapping ``highspy.Highs.setOptionValue``. The parent refuses a row whose recorded
values do not match its arm, and the run exits non-zero when it compared nothing
(CLAUDE.md §6).

Suites:

``uc``      the #1667 unit-commitment hand-off (the model of
            ``test_1667_highs_sparsify_presolve_cycle.py``), HiGHS MILP route;
``check``   the LP/MILP check instances of ``data/lp_milp/manifest.json`` (HiGHS
            check set + i1183; the instances of the #1667 PR panel are among them),
            HiGHS MILP route, via ``lp_milp_panel.worker``;
``miplib``  the MIPLIB-38 panel of ``data/miplib/panel.json``, same route;
``master``  ``tls2``, ``portfol_roundlot`` and the syn/rsyn big-M family (``.nl``
            under ``--nl-dir``), plain ``Model.solve`` with
            ``DISCOPT_CONVEX_ROUTE_OA_MASTER=highs`` (the default master is the
            in-house one, which bit 14 does not reach).

Arms are interleaved per instance with the order rotated per repetition; every solve
is a fresh subprocess. Output is one JSON line per solve (``--out``), written as it
happens. ``summary`` tabulates a finished run.

Correctness checks per row: the panel's oracle (``lp_milp_panel._incorrect``) for
``check``/``miplib``/``uc``; for ``master``, the reference optimum of
``minlplib.solu`` (a certified objective must match it, a bound must not cross it).
Across arms: a certified objective must agree with the other arm's certified
objective.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

BIT14 = 1 << 14
GAP = 1e-4


# ----------------------------------------------------------------------------- worker
def _apply_arm(arm: str) -> int:
    import highspy
    from discopt.solvers import lp_milp_highs as lmh
    from discopt.solvers import milp_highs as mh

    # Both arms are set explicitly from rule 13 (#1634), so the panel measures the same
    # two configurations whatever the shipped defaults are (the route dropped bit 14 in
    # #2335; the masters keep it in MASTER_PRESOLVE_RULE_OFF).
    if arm not in ("on", "off"):
        raise ValueError(arm)
    val = (1 << 13) | (BIT14 if arm == "off" else 0)
    lmh.MILP_PRESOLVE_RULE_OFF = val
    mh.MASTER_PRESOLVE_RULE_OFF = val

    seen: list[int] = []
    orig = highspy.Highs.setOptionValue

    def spy(self, key, value):  # noqa: ANN001
        if key == "presolve_rule_off":
            seen.append(int(value))
        return orig(self, key, value)

    highspy.Highs.setOptionValue = spy
    _apply_arm.seen = seen  # type: ignore[attr-defined]
    return val


def _uc_model():
    sys.path.insert(0, str(REPO / "python" / "tests"))
    import test_1667_highs_sparsify_presolve_cycle as t1667

    # Cut UC_SCRIPT at its FILE print: keep the model build, drop the solve.
    src = t1667.UC_SCRIPT.split('print("FILE"')[0]
    ns: dict = {}
    exec(src, ns)  # noqa: S102 - the repository's own test model
    return ns["m"], t1667.UC_OPT


def worker(kind: str, spec: str, arm: str, tl: float) -> dict:
    val = _apply_arm(arm)
    import discopt

    assert str(REPO / "python") in discopt.__file__, discopt.__file__
    if kind in ("check", "miplib"):
        import lp_milp_panel

        res = lp_milp_panel.worker(spec, "H", tl, GAP, 10**9)
    elif kind == "uc":
        m, _ = _uc_model()
        t0 = time.perf_counter()
        r = m.solve(time_limit=tl)
        res = _row(r, time.perf_counter() - t0)
    elif kind == "master":
        from discopt.modeling.core import from_nl

        m = from_nl(spec)
        t0 = time.perf_counter()
        r = m.solve(time_limit=tl)
        res = _row(r, time.perf_counter() - t0)
    else:
        raise ValueError(kind)
    res["rule_off_value"] = val
    res["rule_off_seen"] = sorted(set(_apply_arm.seen))  # type: ignore[attr-defined]
    res["n_highs_rule_writes"] = len(_apply_arm.seen)  # type: ignore[attr-defined]
    res.pop("solver_stats", None)
    return res


def _row(r, wall: float) -> dict:
    return {
        "status": r.status,
        "objective": r.objective,
        "bound": r.bound,
        "gap_certified": bool(r.gap_certified),
        "nodes": int(r.node_count or 0),
        "wall": wall,
        "algorithm_route": r.algorithm_route,
    }


# ----------------------------------------------------------------------------- items
def _items(a) -> list[dict]:
    out: list[dict] = []
    suites = a.suites.split(",")
    if "uc" in suites:
        out.append({"suite": "uc", "name": "uc_1667", "spec": "-", "oracle": 450623.7692106868,
                    "maximize": False, "oracle_status": "Optimal"})  # fmt: skip
    manifests = (("check", "data/lp_milp/manifest.json"), ("miplib", "data/miplib/panel.json"))
    for suite, man in manifests:
        if suite not in suites:
            continue
        d = json.loads((REPO / "discopt_benchmarks" / man).read_text())
        for x in d["members"]:
            out.append({
                "suite": suite, "name": x["name"], "spec": str(REPO / x["path"]),
                "oracle": x.get("oracle_objective"), "maximize": bool(x.get("maximize", False)),
                "oracle_status": x.get("oracle_status", "Optimal"), "panel": x["panel"],
            })  # fmt: skip
    if "master" in suites:
        nl_dir = Path(a.nl_dir)
        sense = {}
        import csv

        with open(nl_dir.parent / "instancedata.csv") as f:
            for r in csv.DictReader(f, delimiter=";"):
                sense[r["name"]] = r["objsense"]
        opt = {}
        for ln in (nl_dir.parent / "minlplib.solu").read_text().splitlines():
            p = ln.split()
            if len(p) == 3 and p[0] == "=opt=":
                opt[p[1]] = float(p[2])
        for name in (nl_dir.parent / "names.txt").read_text().split():
            out.append({
                "suite": "master", "name": name, "spec": str(nl_dir / f"{name}.nl"),
                "oracle": opt.get(name), "maximize": sense[name] == "max",
                "oracle_status": "Optimal",
            })  # fmt: skip
    if a.only:
        keep = set(a.only.split(","))
        out = [x for x in out if x["name"] in keep]
    return out


def _incorrect(it: dict, res: dict) -> list[str]:
    import lp_milp_panel as lmp

    if it["suite"] in ("check", "miplib") and it.get("panel") == "INF":
        return lmp._incorrect_inf(res, it["oracle_status"])
    if it["oracle"] is None:
        return []
    why = lmp._incorrect(res, it["oracle"], it["maximize"], GAP)
    return why


def _run_sub(it: dict, arm: str, a) -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if it["suite"] == "master":
        env["DISCOPT_CONVEX_ROUTE_OA_MASTER"] = "highs"
        env.pop("DISCOPT_LP_MILP_BACKEND", None)
    else:
        env["DISCOPT_LP_MILP_BACKEND"] = "highs"
    tl = a.master_time_limit if it["suite"] == "master" else a.time_limit
    cmd = [sys.executable, str(HERE), "worker", it["suite"], it["spec"], arm, str(tl)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=4 * tl + 180)
    except subprocess.TimeoutExpired:
        return {"error": "subprocess timeout", "outer_wall": time.perf_counter() - t0}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"error": f"rc={p.returncode}", "stderr_tail": p.stderr[-1500:]}
    res = json.loads(lines[0][len("RESULT ") :])
    res["outer_wall"] = time.perf_counter() - t0
    return res


def cmd_run(a) -> int:
    items = _items(a)
    out = Path(a.out)
    done = set()
    if out.exists():
        for ln in out.read_text().splitlines():
            r = json.loads(ln)
            done.add((r["suite"], r["name"], r["arm"], r["rep"]))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    executed = violations = 0
    with out.open("a") as f:
        for rep in range(a.reps):
            for k, it in enumerate(items):
                arms = ["off", "on"] if (k + rep) % 2 == 0 else ["on", "off"]
                for arm in arms:
                    if (it["suite"], it["name"], arm, rep) in done:
                        continue
                    load = os.getloadavg()[0]
                    res = _run_sub(it, arm, a)
                    why = [] if "error" in res else _incorrect(it, res)
                    if "error" not in res:
                        want_bit = 0 if arm == "on" else BIT14
                        bad = [v for v in res["rule_off_seen"] if (v & BIT14) != want_bit]
                        if bad:
                            raise SystemExit(f"{it['name']} {arm}: HiGHS got {bad}")
                        executed += 1
                    violations += bool(why)
                    row = {"suite": it["suite"], "name": it["name"], "arm": arm, "rep": rep,
                           "load": load, "commit": commit, "incorrect": why, **res}  # fmt: skip
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(
                        f"[{rep}] {it['suite']:6s} {it['name']:24s} {arm:3s} "
                        f"{res.get('status', res.get('error'))!s:12s} "
                        f"obj={res.get('objective')} nodes={res.get('nodes')} "
                        f"wall={res.get('wall', 0) or 0:.2f} "
                        f"writes={res.get('n_highs_rule_writes')} "
                        f"load={load:.1f} {'INCORRECT ' + str(why) if why else ''}",
                        flush=True,
                    )
    print(f"executed={executed} incorrect={violations}", flush=True)
    return 0 if executed > 0 else 1


# ----------------------------------------------------------------------------- summary
def cmd_summary(a) -> int:
    rows = [json.loads(ln) for ln in Path(a.out).read_text().splitlines()]
    by: dict = {}
    for r in rows:
        by.setdefault((r["suite"], r["name"]), {}).setdefault(r["arm"], []).append(r)
    compared = 0
    for suite in sorted({s for s, _ in by}):
        names = sorted(n for s, n in by if s == suite)
        tot = {"off": [], "on": []}
        cert = {"off": 0, "on": 0}
        err = {"off": 0, "on": 0}
        inc = {"off": 0, "on": 0}
        nodes = {"off": 0, "on": 0}
        diffs = []
        lost = {"off": [], "on": []}
        for n in names:
            arms = by[(suite, n)]
            if set(arms) != {"off", "on"}:
                continue
            compared += 1
            med = {}
            for arm in ("off", "on"):
                rs = arms[arm]
                err[arm] += sum("error" in r for r in rs)
                inc[arm] += sum(bool(r["incorrect"]) for r in rs)
                ok = [r for r in rs if "error" not in r]
                c = sum(r["gap_certified"] or r["status"] == "infeasible" for r in ok)
                cert[arm] += c / max(len(rs), 1)
                nodes[arm] += statistics.median([r["nodes"] for r in ok]) if ok else 0
                w = [r["wall"] for r in ok]
                med[arm] = statistics.median(w) if w else float("nan")
                tot[arm].append(med[arm])
                med[arm + "_cert"] = c
                med[arm + "_n"] = len(rs)
            for arm, other in (("off", "on"), ("on", "off")):
                full = med[arm + "_cert"] == med[arm + "_n"]
                if full and med[other + "_cert"] < med[other + "_n"]:
                    lost[other].append(n)
            # certified objectives must agree across arms
            co = [
                r["objective"]
                for arm in ("off", "on")
                for r in arms[arm]
                if "error" not in r and r["gap_certified"] and r["objective"] is not None
            ]
            if co and max(co) - min(co) > 1e-4 * (1 + abs(max(co))) + 1e-6:
                print(f"  CROSS-ARM MISMATCH {suite} {n}: {co}")
                inc["on"] += 1
            diffs.append((med["on"] - med["off"], n, med["off"], med["on"],
                          [r["nodes"] for r in arms["off"] if "error" not in r],
                          [r["nodes"] for r in arms["on"] if "error" not in r]))  # fmt: skip
        print(f"\n== {suite}: {len(diffs)} instances ==")
        for arm in ("off", "on"):
            print(f"  sparsify {arm:3s}: certified {cert[arm]:.2f}  sum median wall "
                  f"{sum(tot[arm]):.2f} s  sum median nodes {nodes[arm]:.0f}  incorrect "
                  f"{inc[arm]}  errors {err[arm]}")  # fmt: skip
        print(f"  certified only by off (lost when on): {lost['on']}")
        print(f"  certified only by on: {lost['off']}")
        diffs.sort()
        for d in diffs[:4] + diffs[-4:]:
            print(f"    {d[1]:24s} off {d[2]:7.2f}s on {d[3]:7.2f}s  nodes off {d[4]} on {d[5]}")
        # run-to-run spread: median per-instance sd of wall within an arm
        sds = []
        for n in names:
            for arm in ("off", "on"):
                w = [r["wall"] for r in by[(suite, n)].get(arm, []) if "error" not in r]
                if len(w) > 1:
                    sds.append(statistics.stdev(w))
        if sds:
            print(
                f"  within-arm wall sd: median {statistics.median(sds):.3f} s, "
                f"max {max(sds):.3f} s over {len(sds)} (instance, arm) cells"
            )
    print(f"\ncompared={compared}")
    return 0 if compared else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("kind")
    w.add_argument("spec")
    w.add_argument("arm")
    w.add_argument("tl", type=float)
    r = sub.add_parser("run")
    r.add_argument("--suites", default="uc,check,miplib,master")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--time-limit", type=float, default=20.0)
    r.add_argument("--master-time-limit", type=float, default=30.0)
    r.add_argument("--nl-dir", default=str(REPO / "ref" / "minlplib" / "nl"))
    r.add_argument("--only", default="")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "worker":
        res = worker(a.kind, a.spec, a.arm, a.tl)
        print("RESULT " + json.dumps(res, default=str), flush=True)
        return 0
    if a.cmd == "run":
        return cmd_run(a)
    return cmd_summary(a)


if __name__ == "__main__":
    raise SystemExit(main())
