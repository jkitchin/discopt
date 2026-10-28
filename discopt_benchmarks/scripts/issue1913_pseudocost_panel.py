#!/usr/bin/env python3
"""#1913: CLAUDE.md section 5 panel for ``DISCOPT_KERNEL_PSEUDOCOST``.

``--flag``/``--fired-stat`` run the same panel for another boolean flag (#1941:
``DISCOPT_INCREMENTAL_WIDEN`` with ``incremental/floor_rows``).

The native spatial kernel branched on the lowest-index fractional integer column.
The flag picks that column by pseudocost instead. This compares the two arms on
MINLPLib instances with a proven status (selected exactly as
``scip_wins_screen.py`` selects them, ``--max-vars/--max-cons`` default 200),
fetched from minlplib.org.

Each solve runs in its own subprocess. The arm order alternates per instance
(interleaved A/B, section 9) and ``--workers`` runs whole instances, never the
two arms of one instance on different loads. Checks:

* cert-clean -- no published bound crosses the MINLPLib oracle; no claimed
  optimum disagrees with it; every published incumbent passes ``verify_point``
  on a freshly parsed model; no instance certified OFF loses its certificate ON;
* net-positive -- certificates gained, nodes and wall where both certify, the
  final bound where neither does.

The ON arm reports ``--fired-stat`` (default ``tree/pseudocost_picks``), so an
instance where the rule never acted is visible as such (section 6). ``--report``
prints the verdict and exits non-zero when nothing was compared or a check failed.

    python -u discopt_benchmarks/scripts/issue1913_pseudocost_panel.py \\
        --out <dir> --time-limit 30 --workers 2 [--names-file names.txt]
    python discopt_benchmarks/scripts/issue1913_pseudocost_panel.py --out <dir> \\
        --time-limit 30 --report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scip_wins_screen as screen  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FLAG = "DISCOPT_KERNEL_PSEUDOCOST"
TOL = 1e-4


def _solve_one(nl: str, tl: float, stat: str) -> None:
    """Subprocess entry: solve, verify the incumbent on a fresh model, print RESULT."""
    import discopt
    import numpy as np
    from discopt.modeling.core import from_nl
    from discopt.validation.feasibility import verify_point

    assert discopt.__file__.startswith(str(REPO)), discopt.__file__
    # #2013: record every #1066 route-guard verdict that fired, so a panel over a
    # guard rule shows where it acted (section 6).
    import discopt.solver as _solver_mod

    guard_fired: list[str] = []
    _guard_call = _solver_mod._RouteProgressGuard.__call__

    def _traced_guard(self, context):
        stop = _guard_call(self, context)
        if stop:
            guard_fired.append(f"{self.fired_at:.2f}s: {self.reason}")
        return stop

    _solver_mod._RouteProgressGuard.__call__ = _traced_guard
    t0 = time.perf_counter()
    r = from_nl(nl).solve(time_limit=tl, gap_tolerance=TOL)
    wall = time.perf_counter() - t0
    bad = None
    if r.x is not None and r.objective is not None:
        fresh = from_nl(nl)
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in fresh._variables]
        )
        bad = not verify_point(fresh, flat).ok

    def f(v):
        return None if v is None or not math.isfinite(float(v)) else float(v)

    ss = r.solver_stats or {}
    print(
        "RESULT "
        + json.dumps(
            {
                "status": str(r.status),
                "gap_certified": bool(r.gap_certified),
                "objective": f(r.objective),
                "bound": f(r.bound),
                "nodes": r.node_count,
                "wall": wall,
                "picks": ss.get(stat),
                # #1962: the separator and the integral-point probe fired (§6).
                "cut_rounds": ss.get("tree/cut_rounds"),
                "int_probes": ss.get("tree/int_probes"),
                "spatial_picks": ss.get("tree/spatial_score_picks"),
                "route": getattr(r, "algorithm_route", None),
                "verify_failed": bad,
                "guard_fired": guard_fired,
            }
        ),
        flush=True,
    )


def run(
    out: Path, panel: list[dict], tl: float, workers: int, flag: str = FLAG, stat: str = ""
) -> Path:
    results = out / f"pc_tl{int(tl)}.jsonl"
    done = set()
    if results.exists():
        for ln in results.open():
            j = json.loads(ln)
            done.add((j["name"], j["arm"]))
    todo = [p for p in panel if not {(p["name"], "0"), (p["name"], "1")} <= done]
    print(f"{len(panel)} instances, {len(todo)} to run", flush=True)

    def one(idx_p):
        idx, p = idx_p
        nl = screen._fetch(f"{screen.MINLPLIB}/nl/{p['name']}.nl", out / "nl" / f"{p['name']}.nl")
        rows = []
        for arm in ("1", "0") if idx % 2 else ("0", "1"):
            if (p["name"], arm) in done:
                continue
            # ``--flag A,B`` toggles several flags together (#2013).
            env = dict(os.environ, **dict.fromkeys(flag.split(","), arm))
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, __file__, "--_solve", str(nl), str(tl), stat],
                    capture_output=True,
                    text=True,
                    timeout=tl + 120,
                    env=env,
                )
                lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
                res = (
                    json.loads(lines[-1][7:])
                    if lines
                    else {"status": "CRASH", "rc": proc.returncode, "err": proc.stderr[-1500:]}
                )
            except subprocess.TimeoutExpired:
                res = {"status": "HARD_TIMEOUT"}
            res.update(name=p["name"], arm=arm, opt=p["opt"], sense=p["sense"])
            res["proc_wall"] = time.perf_counter() - t0
            rows.append(res)
        return rows

    executed = 0
    with ThreadPoolExecutor(workers) as ex, results.open("a") as fh:
        for rows in ex.map(one, enumerate(todo)):
            for res in rows:
                executed += 1
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                print(
                    f"[{executed}] {res['name']:28s} arm={res['arm']} {res['status']:12s} "
                    f"obj={res.get('objective')} bound={res.get('bound')} "
                    f"nodes={res.get('nodes')} picks={res.get('picks')} "
                    f"t={res['proc_wall']:.1f}",
                    flush=True,
                )
    print(f"EXECUTED {executed}", flush=True)
    return results


def _crosses(row: dict) -> bool:
    """The published bound is past the oracle optimum (min: above; max: below)."""
    b, opt = row.get("bound"), row.get("opt")
    if b is None or opt is None:
        return False
    slack = 1e-6 + TOL * max(1.0, abs(opt))
    return b > opt + slack if row["sense"] == "min" else b < opt - slack


def _wrong_optimum(row: dict) -> bool:
    if not row.get("gap_certified") or row.get("status") != "optimal":
        return False
    if row.get("opt") is None:
        return row.get("objective") is not None
    return not screen._close(row.get("objective"), row["opt"])


def report(path: Path) -> int:
    by: dict[str, dict] = {}
    for ln in path.open():
        j = json.loads(ln)
        by.setdefault(j["name"], {})[j["arm"]] = j
    checks = problems = 0
    gained, lost, fired = [], [], 0
    both_nodes = [0, 0]
    both_wall = [0.0, 0.0]
    total_wall = [0.0, 0.0]
    neither = []
    for name, arms in sorted(by.items()):
        if set(arms) != {"0", "1"}:
            continue
        off, on = arms["0"], arms["1"]
        for row in (off, on):
            checks += 3
            if row["status"] in ("CRASH", "HARD_TIMEOUT"):
                problems += 1
                print(f"PROBLEM {name} arm={row['arm']} {row['status']}")
            if _crosses(row):
                problems += 1
                print(f"PROBLEM {name} arm={row['arm']} bound {row['bound']} crosses {row['opt']}")
            if _wrong_optimum(row):
                problems += 1
                print(
                    f"PROBLEM {name} arm={row['arm']} certified {row['objective']} != {row['opt']}"
                )
            if row.get("verify_failed"):
                problems += 1
                print(f"PROBLEM {name} arm={row['arm']} incumbent fails verify_point")
        total_wall[0] += off.get("wall") or 0.0
        total_wall[1] += on.get("wall") or 0.0
        if (on.get("picks") or 0) > 0:
            fired += 1
        c0, c1 = bool(off.get("gap_certified")), bool(on.get("gap_certified"))
        if c0 and not c1:
            lost.append(name)
        if c1 and not c0:
            gained.append(name)
        if c0 and c1:
            both_nodes[0] += off.get("nodes") or 0
            both_nodes[1] += on.get("nodes") or 0
            both_wall[0] += off["wall"]
            both_wall[1] += on["wall"]
            if off.get("nodes") != on.get("nodes"):
                print(
                    f"  both certify {name:28s} nodes {off.get('nodes')} -> {on.get('nodes')}"
                    f"  wall {off['wall']:.1f} -> {on['wall']:.1f}  picks={on.get('picks')}"
                )
        if not c0 and not c1:
            neither.append(name)
            print(
                f"  neither      {name:28s} bound {off.get('bound')} -> {on.get('bound')}"
                f"  (opt {off.get('opt')}, sense {off['sense']})  picks={on.get('picks')}"
            )
    print(f"instances {len(by)}, rule fired on {fired}")
    print(f"certified gained {gained}")
    print(f"certified lost   {lost}")
    print(
        f"both certify: nodes {both_nodes[0]} -> {both_nodes[1]}, "
        f"wall {both_wall[0]:.1f} -> {both_wall[1]:.1f} s"
    )
    print(
        f"total wall {total_wall[0]:.1f} -> {total_wall[1]:.1f} s; "
        f"neither certifies: {len(neither)}"
    )
    print(f"executed checks {checks}, problems {problems}")
    return 1 if checks == 0 or problems or lost else 0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_solve":
        _solve_one(sys.argv[2], float(sys.argv[3]), sys.argv[4])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-vars", type=int, default=200)
    ap.add_argument("--max-cons", type=int, default=200)
    ap.add_argument("--names-file", type=Path)
    ap.add_argument("--flag", default=FLAG)
    ap.add_argument("--fired-stat", default="tree/pseudocost_picks")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        sys.exit(report(a.out / f"pc_tl{int(a.time_limit)}.jsonl"))
    names = None
    if a.names_file:
        names = {ln.strip() for ln in a.names_file.read_text().split() if ln.strip()}
    panel = screen.build_panel(a.out, a.max_vars, a.max_cons, names)
    run(a.out, panel, a.time_limit, a.workers, a.flag, a.fired_stat)


if __name__ == "__main__":
    main()
