#!/usr/bin/env python3
"""A §5 graduation panel for one ``DISCOPT_*`` flag: ON vs OFF (CLAUDE.md §5).

Written for #1982 (``DISCOPT_FBBT_SIGN_DOMAIN``) and reused by #1978
(``DISCOPT_PY_TREE_SPATIAL_SCORE``); ``--flag`` names the flag. A bound-changing
change graduates only on a corpus panel that is BOTH

* **cert-clean** -- 0 incorrect certified objectives or infeasibility claims vs the
  MINLPLib oracle, no dual bound beyond the oracle, no ``gap_certified`` instance
  lost ON, every ON incumbent independently feasible
  (:func:`discopt.warm_start.check_feasibility`, fail-closed); and
* **net-positive** -- measurably helpful (certified count, bounds where OFF had
  none, nodes), not merely sound.

Each (instance, arm) solve runs in a fresh subprocess with the flag set in its
environment. Arms are interleaved per
instance, alternating which goes first. Progress is printed per instance.

Usage::

    python -u discopt_benchmarks/scripts/flag_onoff_corpus_panel.py \\
        --flag DISCOPT_FBBT_SIGN_DOMAIN --solu minlplib.solu \\
        --extra-dir <dir of .nl> --time-limit 20 --jobs 3 --out panel.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

FLAG = "DISCOPT_FBBT_SIGN_DOMAIN"  # overridden by --flag
REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "python" / "tests" / "data" / "minlplib_nl"

CHILD = r"""
import json, sys, time
import numpy as np
import discopt.modeling as dm
from discopt.modeling.core import objective_sense_sign
from discopt.warm_start import check_feasibility
path, tl = sys.argv[1], float(sys.argv[2])
t = time.time()
out = {}
try:
    m = dm.from_nl(path)
    out["sense"] = int(objective_sense_sign(m))
    r = m.solve(time_limit=tl)
    out.update(status=r.status, obj=r.objective, bound=r.bound,
               cert=bool(getattr(r, "gap_certified", False)), nodes=r.node_count,
               route=str(r.algorithm_route)[:60], stats={k: v for k, v in
               (r.solver_stats or {}).items() if k.startswith("tree/py_spatial")})
    if r.x is not None and r.objective is not None:
        flat = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
        ok, viol = check_feasibility(m, flat, tol=1e-5)
        out["feasible"] = bool(ok)
        if not ok:
            out["violations"] = viol[:3]
except Exception as e:  # recorded, never swallowed: the row carries the error
    out["error"] = f"{type(e).__name__}: {e}"[:300]
out["wall"] = time.time() - t
print("@@" + json.dumps(out))
"""


def solve(path: str, arm: str, tl: float, flag: str = FLAG) -> dict:
    env = dict(os.environ, **{flag: arm})
    try:
        p = subprocess.run(
            [sys.executable, "-c", CHILD, path, str(tl)],
            env=env,
            capture_output=True,
            text=True,
            timeout=tl * 4 + 120,
        )
    except subprocess.TimeoutExpired:
        return {"error": "hard timeout"}
    for line in p.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    return {"error": f"no result (rc={p.returncode}): {p.stderr[-300:]}"}


def run_instance(name: str, path: str, tl: float, first_on: bool, flag: str = FLAG) -> dict:
    arms = ["1", "0"] if first_on else ["0", "1"]
    res = {arm: solve(path, arm, tl, flag) for arm in arms}
    return {"name": name, "off": res["0"], "on": res["1"]}


def load_solu(path: str) -> dict:
    ref: dict = {}
    with open(path) as fh:
        lines = fh.read().splitlines()
    for line in lines:
        p = line.split()
        if len(p) >= 3 and p[0] in ("=opt=", "=best="):
            ref.setdefault(p[1], {})[p[0]] = float(p[2])
        elif len(p) >= 2 and p[0] == "=inf=":
            ref.setdefault(p[1], {})["=inf="] = True
    return ref


def problems(row: dict, ref: dict, tol_rel: float = 1e-4, tol_abs: float = 1e-6) -> list[str]:
    """Soundness verdicts for one instance (the cert-clean bar)."""
    out = []
    r = ref.get(row["name"], {})
    opt = r.get("=opt=")
    best = r.get("=best=", opt)
    for arm in ("off", "on"):
        a = row[arm]
        if "error" in a:
            continue
        s = a.get("sense", 1)
        tol = lambda v: tol_abs + tol_rel * max(1.0, abs(v))  # noqa: E731
        certified = a.get("status") == "optimal" and a.get("cert")
        if certified and opt is not None and abs(a["obj"] - opt) > tol(opt):
            out.append(f"{arm}: certified {a['obj']} != opt {opt}")
        if a.get("status") == "infeasible" and (opt is not None or best is not None):
            out.append(f"{arm}: infeasible claimed, MINLPLib has a solution")
        b = a.get("bound")
        if b is not None and best is not None and s * (b - best) > tol(best):
            out.append(f"{arm}: bound {b} beyond the oracle {best}")
        if a.get("feasible") is False:
            out.append(f"{arm}: incumbent infeasible {a.get('violations')}")
    if row["off"].get("cert") and not row["on"].get("cert"):
        out.append("certification lost ON")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", default=FLAG, help="the DISCOPT_* flag under test")
    ap.add_argument("--solu", required=True)
    ap.add_argument("--extra-dir", default=None, help="more .nl files (e.g. the #1975 class)")
    ap.add_argument("--extra-list", default=None, help="names to take from --extra-dir")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated subset (smoke runs)")
    args = ap.parse_args()

    insts = {p.stem: str(p) for p in sorted(CORPUS.glob("*.nl"))}
    if args.extra_dir:
        names = (
            Path(args.extra_list).read_text().split()
            if args.extra_list
            else [p.stem for p in Path(args.extra_dir).glob("*.nl")]
        )
        for n in names:
            insts.setdefault(n, str(Path(args.extra_dir) / f"{n}.nl"))
    if args.only:
        keep = set(args.only.split(","))
        insts = {n: p for n, p in insts.items() if n in keep}
    ref = load_solu(args.solu)
    print(f"{args.flag}: {len(insts)} instances, {args.time_limit} s, {args.jobs} jobs", flush=True)

    done = 0
    rows = []
    with open(args.out, "w") as fh, ProcessPoolExecutor(args.jobs) as ex:
        futs = {
            ex.submit(run_instance, n, p, args.time_limit, i % 2 == 0, args.flag): n
            for i, (n, p) in enumerate(sorted(insts.items()))
        }
        for f in as_completed(futs):
            row = f.result()
            row["problems"] = problems(row, ref)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done += 1
            off, on = row["off"], row["on"]
            print(
                f"[{done}/{len(insts)}] {row['name']}: "
                f"OFF {off.get('status')} cert={off.get('cert')} "
                f"bound={off.get('bound')} | ON {on.get('status')} cert={on.get('cert')} "
                f"bound={on.get('bound')} {row['problems'] or ''}",
                flush=True,
            )

    n_prob = sum(bool(r["problems"]) for r in rows)
    cert = {a: sum(bool(r[a].get("cert")) for r in rows) for a in ("off", "on")}
    gained = [r["name"] for r in rows if r["on"].get("cert") and not r["off"].get("cert")]
    lost = [r["name"] for r in rows if r["off"].get("cert") and not r["on"].get("cert")]
    new_bound = [
        r["name"]
        for r in rows
        if r["off"].get("bound") is None and r["on"].get("bound") is not None
    ]
    errors = [r["name"] for r in rows if "error" in r["off"] or "error" in r["on"]]
    print(f"\nrows {len(rows)} | instances with a problem: {n_prob}")
    print(f"certified OFF {cert['off']} -> ON {cert['on']} | gained {gained} | lost {lost}")
    print(f"bound where OFF had none: {new_bound}")
    print(f"errors: {errors}")
    n_feas = sum(r[a].get("feasible") is not None for r in rows for a in ("off", "on"))
    print(f"incumbent feasibility checks executed: {n_feas}")
    if not rows or n_feas == 0:
        sys.exit("the panel checked nothing")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"wall {time.time() - t0:.0f} s")
