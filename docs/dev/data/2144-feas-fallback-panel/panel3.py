#!/usr/bin/env python3
"""#2144 panel: main vs branch(flag OFF) vs branch(flag ON), gdp_parity x {big-m, hull}.

Each (instance, method) cell runs its three arms back to back in a rotating order
(interleaved control, CLAUDE.md 9), each in a fresh ``gdp_baseline.py --worker``
subprocess. ``main`` imports discopt from a worktree at origin/main (PYTHONPATH);
every row records the git_head of the discopt it loaded, and the report refuses
rows whose head does not match their arm (CLAUDE.md 8).

    python -u panel3.py --shard K --nshards N --out rows.jsonl
    python panel3.py --report rows*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/user/discopt")
SCR = Path(__file__).resolve().parent
MAIN_WT = SCR / "main-wt"
SHIM = SCR / "shim"
FLAG = "DISCOPT_GDP_CONFIG_FEAS_FALLBACK"
sys.path.insert(0, str(REPO / "discopt_benchmarks"))
sys.path.insert(0, str(REPO / "discopt_benchmarks" / "scripts"))
ARMS = ("main", "off", "on")
METHODS = ("big-m", "hull")


def _env(arm: str) -> dict:
    env = dict(os.environ)
    pp = [str(SHIM)]
    if arm == "main":
        pp.insert(0, str(MAIN_WT / "python"))
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env[FLAG] = "1" if arm == "on" else "0"
    return env


def _rows(paths) -> list[dict]:
    out = []
    for p in paths:
        p = Path(p)
        if p.exists():
            out += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    return out


def run(args) -> int:
    import gdp_baseline as gb
    from benchmarks import gdplib_runner as gr

    oracle = gr.panel_oracle()
    names = sorted(gr.GDP_PARITY)
    cells = [(n, m) for n in names for m in METHODS]
    mine = [c for i, c in enumerate(cells) if i % args.nshards == args.shard]
    done = {(r["instance"], r["method"], r["arm"]) for r in _rows([args.out])}
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} cells x 3 arms", flush=True)
    for i, (name, method) in enumerate(mine):
        k = (i + args.shard) % 3
        for arm in ARMS[k:] + ARMS[:k]:
            if (name, method, arm) in done:
                continue
            t0 = time.perf_counter()
            load = os.getloadavg()[0]
            cmd = [
                sys.executable,
                "-u",
                str(REPO / "discopt_benchmarks/scripts/gdp_baseline.py"),
                "--worker",
                name,
                f"discopt-{method}",
                str(args.time_limit),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.time_limit + args.grace,
                    env=_env(arm),
                    cwd=str(REPO),
                )
                lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@RESULT ")]
                if lines:
                    rec = json.loads(lines[-1][len("@@RESULT ") :])
                else:
                    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                    rec = {"error": " | ".join(tail)[:400], "returncode": proc.returncode}
            except subprocess.TimeoutExpired:
                rec = {"status": "killed", "objective": None, "bound": None}
            rec.update(
                instance=name,
                method=method,
                arm=arm,
                load1=round(load, 2),
                elapsed=round(time.perf_counter() - t0, 2),
                time_limit=args.time_limit,
            )
            rec["verdict"] = gb.score(rec, oracle.get(name))
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            print(
                f"{name:30s} {method:6s} {arm:4s} {rec['verdict']:18s} st={rec.get('status')} "
                f"obj={rec.get('objective')} bnd={rec.get('bound')} n={rec.get('node_count')} "
                f"fb={rec.get('feas_fallback')} head={str(rec.get('git_head'))[:8]} "
                f"load={rec['load1']} t={rec['elapsed']}",
                flush=True,
            )
    return 0


def report(args) -> int:
    import gdp_baseline as gb
    from benchmarks import gdplib_runner as gr

    oracle = gr.panel_oracle()
    rows = _rows(args.report)
    heads = {"main": args.main_head, "off": args.branch_head, "on": args.branch_head}
    wrong = [r for r in rows if not str(r.get("git_head", "")).startswith(heads[r["arm"]])]
    print(f"rows {len(rows)}; rows whose loaded discopt != arm's head: {len(wrong)}")
    for r in wrong[:5]:
        print("   WRONG HEAD", r["instance"], r["method"], r["arm"], r.get("git_head"))
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["instance"], r["method"]), {})[r["arm"]] = r
    checks = 0
    incorrect, crossings, lost, inc_lost, inc_gained, fired = [], [], [], [], [], []

    def crosses(r):
        o = oracle.get(r["instance"])
        b = r.get("bound")
        if o is None or o.value is None or b is None or not math.isfinite(b):
            return False
        tol = gb.ABS_TOL + gb.REL_TOL * abs(o.value)
        return b > o.value + tol if r.get("minimize", True) else b < o.value - tol

    def has_inc(r):
        return r.get("objective") is not None

    print(f"\n{'instance':28s} {'meth':5s} " + " ".join(f"{a:>26s}" for a in ARMS))
    pairs = {("main", "off"): [], ("off", "on"): [], ("main", "on"): []}
    for (name, method), arms in sorted(cells.items()):
        for r in arms.values():
            checks += 1
            if r["verdict"] == "incorrect":
                incorrect.append(r)
            if crosses(r):
                crossings.append(r)
        cols = []
        for a in ARMS:
            r = arms.get(a)
            if r is None:
                cols.append(f"{'-':>26s}")
                continue
            obj = r.get("objective")
            s = f"{r['verdict'][:6]} {obj if obj is None else round(obj, 2)} n{r.get('node_count')}"
            cols.append(f"{s:>26s}")
        print(f"{name:28s} {method:5s} " + " ".join(cols))
        for a, b in pairs:
            ra, rb = arms.get(a), arms.get(b)
            if ra is None or rb is None:
                continue
            pairs[(a, b)].append((name, method, ra, rb))
            if ra["verdict"] == "solved" and rb["verdict"] != "solved":
                lost.append((a, b, name, method))
            if has_inc(ra) and not has_inc(rb):
                inc_lost.append((a, b, name, method))
            if has_inc(rb) and not has_inc(ra):
                inc_gained.append((a, b, name, method))
        on = arms.get("on")
        if on and (on.get("feas_fallback") or {}).get("calls"):
            fired.append((name, method, on["feas_fallback"]))
    print(f"\nexecuted rows: {checks}")
    print(f"cert-clean: incorrect={len(incorrect)} bound-crossings={len(crossings)}")
    for r in incorrect + crossings:
        print(
            "   ",
            r["instance"],
            r["method"],
            r["arm"],
            r["verdict"],
            r.get("objective"),
            r.get("bound"),
        )
    print(f"fallback fired (ON arm) on {len(fired)} cells: {fired}")
    for a, b in pairs:
        sl = [x for x in lost if x[:2] == (a, b)]
        il = [x for x in inc_lost if x[:2] == (a, b)]
        ig = [x for x in inc_gained if x[:2] == (a, b)]
        better = worse = same = 0
        for name, method, ra, rb in pairs[(a, b)]:
            oa, ob = ra.get("objective"), rb.get("objective")
            if oa is None or ob is None:
                continue
            mn = ra.get("minimize", True)
            d = (ob - oa) if mn else (oa - ob)
            tol = 1e-6 * max(1.0, abs(oa))
            better += d < -tol
            worse += d > tol
            same += abs(d) <= tol
        sa = sum(r["verdict"] == "solved" for *_, r, _ in pairs[(a, b)])
        sb = sum(r["verdict"] == "solved" for *_, _, r in pairs[(a, b)])
        print(
            f"\n{a} -> {b} over {len(pairs[(a, b)])} cells: solved {sa} -> {sb}; "
            f"solved lost {len(sl)} {[x[2:] for x in sl]}; incumbent lost {len(il)} "
            f"{[x[2:] for x in il]}; incumbent gained {len(ig)} {[x[2:] for x in ig]}; "
            f"objective better {better} / worse {worse} / same {same}"
        )
        for name, method, ra, rb in pairs[(a, b)]:
            oa, ob = ra.get("objective"), rb.get("objective")
            if oa is not None and ob is not None and abs(oa - ob) > 1e-6 * max(1.0, abs(oa)):
                print(
                    f"     {name} {method}: {oa:.6g} -> {ob:.6g} "
                    f"(nodes {ra.get('node_count')} -> {rb.get('node_count')})"
                )
    return 1 if (checks == 0 or incorrect or crossings or wrong) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--grace", type=float, default=120.0)
    ap.add_argument("--report", nargs="*")
    ap.add_argument("--main-head", default="a0b3b5e")
    ap.add_argument("--branch-head", default="")
    args = ap.parse_args()
    return report(args) if args.report else run(args)


if __name__ == "__main__":
    sys.exit(main())
