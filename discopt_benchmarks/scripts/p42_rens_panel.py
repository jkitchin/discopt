"""highs-parity P4.2: the ``DISCOPT_RENS`` graduation panel, bars pre-registered.

**Historical.** The panel ran on build ``56861877`` (data in
``docs/dev/data/highs-parity/p4.2/``); bar 2a failed and ``DISCOPT_RENS`` was retired,
flag and implementation. On a later build the functional gate below fails by design
(the flag is no longer read); check out ``56861877`` to re-run it.

A14 (``milp-competitiveness-plan.md``) left RENS cert-clean with a large primal win
(median primal gap 38.7 % -> 1.6 %, 12 incumbent wins / 0 losses) but failed its
dual-side bar, a single-draw *count* of bound losses that A14 itself showed was partly
rep-to-rep noise. This re-run states the dual bar on magnitude and repetition, as A14
prescribed, BEFORE the run (owner decision 2026-11-05: "A14 bars + magnitude").

Population: the MIPLIB-38 panel (``discopt_benchmarks/data/miplib/panel.json``) and the M
panel (``discopt_benchmarks/data/lp_milp/manifest.json``, panel ``M``), each with a
reference optimum. Arms: ``DISCOPT_RENS`` unset (OFF) and ``=1`` (ON), on the Rust MILP
route (``DISCOPT_LP_MILP_BACKEND=rust``), each solve a fresh subprocess running
``lp_milp_panel.py worker`` with ``DISCOPT_PROFILE=1`` (both arms, for symmetry), arm
order alternating per instance and rep, ``--reps`` (3) repetitions, ``--time-limit`` 20 s,
``gap_tol`` = ``lp_milp_panel.GAP``.

Pre-registered bars (fixed here before the run; a bar found mis-specified afterwards
is not thereby met -- the A14 lesson):

BAR 1, cert-clean (hard, zero slack), over every solve of both arms:
  (a) a ``gap_certified`` result's objective equals the reference optimum within
      ``1e-4 * (1 + |opt|)``;
  (b) no dual bound past the reference optimum by more than ``1e-6 * (1 + |opt|)``;
  (c) no incumbent better than the reference optimum by more than ``1e-6 * (1 + |opt|)``;
  (d) every returned point has ``max_viol <= 1e-6`` on the original rows / box /
      integrality (``lp_milp_loader.dense_max_violation``);
  (e) ``RensRejected == 0`` on every ON solve (the driver's own safety counter).
BAR 2, net-positive, over the *open population* P = instances where not every solve of
both arms is ``gap_certified``. Per instance and arm, the primal gap is
``max(0, sgn*(inc - opt)) / (1 + |opt|)`` (``inf`` without an incumbent) and the dual
gap ``max(0, sgn*(opt - bound)) / (1 + |opt|)``; each is summarised by its median over
reps, and an arm's *spread* is max - min over its reps.
  (2a) solved: certified solves ON >= certified solves OFF (summed over instances, reps);
  (2b) the median over P of the per-instance primal gap is strictly lower ON;
  (2c) incumbent wins > losses over P (a win: ON's primal gap lower than OFF's by more
       than ``max(1e-6, OFF spread)``; a loss: the mirror);
  (2d) dual-bound losses <= 10 % of |P| (floor), where a loss is ON's dual gap above
       OFF's by more than 0.01 (1 % relative) AND by more than OFF's own spread.
Graduate only if BAR 1 and all of BAR 2 pass; otherwise retire flag + implementation
(RINS stays).

Mechanism-fired check: the ON arm's ``RensRun`` summed over the panel must be > 0, and
a functional gate runs first: ``DISCOPT_RENS=not-a-flag-value`` must make the worker fail
(``rens_enabled`` refuses unrecognized values), proving the flag is read by this build.

Discipline (CLAUDE.md sections 6-10): executed counter, exceptions propagate, per-solve
progress, load recorded per solve, non-zero exit when nothing executed or nothing fired.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
PANEL_PY = HERE.parent / "lp_milp_panel.py"
sys.path.insert(0, str(HERE.parent))

import lp_milp_loader as L  # noqa: E402, N812
import lp_milp_panel as P  # noqa: E402, N812

MIPLIB_JSON = REPO / "discopt_benchmarks" / "data" / "miplib" / "panel.json"
M_JSON = REPO / "discopt_benchmarks" / "data" / "lp_milp" / "manifest.json"
FLAG = "DISCOPT_RENS"
_CTR = re.compile(r"^\s+(Rens\w*)\s+(\d+)$")


def population() -> list[dict]:
    mip = json.loads(MIPLIB_JSON.read_text())["members"]
    m = [x for x in json.loads(M_JSON.read_text())["members"] if x.get("panel") == "M"]
    out = []
    for mem in mip + m:
        # The manifests do not carry the objective sense; the instance does.
        out.append({**mem, "maximize": bool(L.read_instance(str(REPO / mem["path"])).maximize)})
    return out


def solve(path: str, on: bool | str, a) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in (FLAG, "DISCOPT_LP_STD_ENGINE")}
    env.update(PYTHONUNBUFFERED="1", DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PROFILE="1")
    if on is True:
        env[FLAG] = "1"
    elif isinstance(on, str):
        env[FLAG] = on
    cmd = [sys.executable, str(PANEL_PY), "worker", path, "R", str(a.time_limit), str(a.gap),
           str(a.max_nodes)]  # fmt: skip
    load = os.getloadavg()[0]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3 * a.time_limit + 180)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"error": f"rc={p.returncode}", "stderr_tail": p.stderr[-1200:], "load": load}
    res = json.loads(lines[0][len("RESULT ") :])
    ctr: dict[str, int] = {}
    for ln in p.stderr.splitlines():
        mm = _CTR.match(ln)
        if mm:
            ctr[mm.group(1)] = ctr.get(mm.group(1), 0) + int(mm.group(2))
    res["rens"] = ctr
    res["load"] = load
    return res


def gaps(res: dict, opt: float, maximize: bool) -> tuple[float, float]:
    scale = 1.0 + abs(opt)
    sgn = -1.0 if maximize else 1.0
    inc, bnd = res.get("objective"), res.get("bound")
    primal = math.inf if inc is None else max(0.0, sgn * (inc - opt)) / scale
    dual = math.inf
    if bnd is not None and math.isfinite(bnd):
        dual = max(0.0, sgn * (opt - bnd)) / scale
    return primal, dual


def cert_problems(res: dict, opt: float, maximize: bool, arm: str) -> list[str]:
    out = []
    scale = 1.0 + abs(opt)
    sgn = -1.0 if maximize else 1.0
    inc, bnd = res.get("objective"), res.get("bound")
    if res.get("gap_certified") and (inc is None or abs(inc - opt) > 1e-4 * scale):
        out.append(f"certified objective {inc} != opt {opt}")
    if bnd is not None and math.isfinite(bnd) and sgn * (bnd - opt) > 1e-6 * scale:
        out.append(f"bound {bnd} past opt {opt}")
    if inc is not None and sgn * (opt - inc) > 1e-6 * scale:
        out.append(f"incumbent {inc} better than opt {opt}")
    mv = res.get("max_viol")
    if inc is not None and (mv is None or mv > 1e-6):
        out.append(f"incumbent max_viol {mv}")
    if arm == "on" and res["rens"].get("RensRejected", 0) != 0:
        out.append(f"RensRejected {res['rens']['RensRejected']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--gap", type=float, default=P.GAP)
    ap.add_argument("--max-nodes", type=int, default=100_000_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--instances", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    P._assert_discopt_under_test()
    hdr = P._header()
    pop = population()
    if a.instances:
        keep = set(a.instances.split(","))
        pop = [x for x in pop if x["name"] in keep]
    print("header", json.dumps(hdr), "instances", len(pop), flush=True)

    # Functional gate: this build reads DISCOPT_RENS (a garbage value must fail).
    g = solve(str(REPO / pop[0]["path"]), "not-a-flag-value", a)
    if "error" not in g:
        raise RuntimeError("DISCOPT_RENS=not-a-flag-value did not fail: the flag is not read")
    print("functional gate: DISCOPT_RENS garbage refused -- ok", flush=True)

    def one(job):
        k, mem, rep = job
        order = (False, True) if (k + rep) % 2 == 0 else (True, False)
        got = {}
        for on in order:
            r = solve(str(REPO / mem["path"]), on, a)
            if "error" in r:
                raise RuntimeError(f"{mem['name']} rep {rep} on={on}: {r}")
            got["on" if on else "off"] = r
        return mem, rep, got

    jobs = [(k, mem, rep) for rep in range(a.reps) for k, mem in enumerate(pop)]
    rows, executed = [], 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for mem, rep, got in ex.map(one, jobs):
            executed += 2
            opt = mem["oracle_objective"]
            maxim = mem["maximize"]
            row = {"name": mem["name"], "panel": mem.get("panel"), "rep": rep, "opt": opt,
                   "maximize": maxim, **got}  # fmt: skip
            row["problems"] = {arm: cert_problems(got[arm], opt, maxim, arm) for arm in got}
            rows.append(row)
            o, n = got["off"], got["on"]
            print(f"[{len(rows)}/{len(jobs)}] {mem['name']:22s} r{rep} "
                  f"OFF {o['status']}/{o['gap_certified']} obj={o['objective']} bnd={o['bound']} "
                  f"| ON {n['status']}/{n['gap_certified']} obj={n['objective']} bnd={n['bound']} "
                  f"rens={n['rens']} load={o['load']:.1f}/{n['load']:.1f}", flush=True)  # fmt: skip
            Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows},
                                              indent=1) + "\n")  # fmt: skip
    verdict = evaluate(rows)
    Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows,
                                       "verdict": verdict}, indent=1) + "\n")  # fmt: skip
    print(json.dumps(verdict, indent=1), flush=True)
    print(f"p42 panel: executed={executed} rens_run={verdict['rens_run_total']}", flush=True)
    return 0 if executed and verdict["rens_run_total"] > 0 else 1


def evaluate(rows: list[dict]) -> dict:
    """The pre-registered bars over the panel rows (see the module docstring)."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["name"], []).append(r)
    problems = [(r["name"], r["rep"], arm, p) for r in rows for arm, ps in r["problems"].items()
                for p in ps]  # fmt: skip
    solved = {arm: sum(bool(r[arm]["gap_certified"]) for r in rows) for arm in ("off", "on")}
    pop = [nm for nm, rs in by.items()
           if not all(r[arm]["gap_certified"] for r in rs for arm in ("off", "on"))]  # fmt: skip
    per = {}
    for nm in pop:
        rs = by[nm]
        g = {arm: [gaps(r[arm], r["opt"], r["maximize"]) for r in rs] for arm in ("off", "on")}
        d = {}
        for arm in ("off", "on"):
            pr = [x[0] for x in g[arm]]
            du = [x[1] for x in g[arm]]
            d[arm] = {"primal": statistics.median(pr), "dual": statistics.median(du),
                      "primal_spread": _spread(pr), "dual_spread": _spread(du)}  # fmt: skip
        per[nm] = d
    med = {arm: statistics.median([per[nm][arm]["primal"] for nm in pop]) if pop else None
           for arm in ("off", "on")}  # fmt: skip
    wins = losses = dual_losses = 0
    detail = {}
    for nm in pop:
        o, n = per[nm]["off"], per[nm]["on"]
        tol = max(1e-6, o["primal_spread"])
        w = _better(n["primal"], o["primal"], tol)
        lo = _better(o["primal"], n["primal"], tol)
        dl = _worse_by(n["dual"], o["dual"], max(0.01, o["dual_spread"]))
        wins += w
        losses += lo
        dual_losses += dl
        detail[nm] = {"off": o, "on": n, "primal_win": w, "primal_loss": lo, "dual_loss": dl}
    cap = math.floor(0.10 * len(pop))
    bar2 = {
        "2a_solved_on_ge_off": solved["on"] >= solved["off"],
        "2b_median_primal_lower": med["off"] is not None and med["on"] < med["off"],
        "2c_wins_gt_losses": wins > losses,
        "2d_dual_losses_le_10pct": dual_losses <= cap,
    }
    return {
        "bar1_problems": problems,
        "bar1_pass": not problems,
        "bar1_checks": len(rows) * 2,
        "solved": solved,
        "population": len(pop),
        "median_primal": med,
        "incumbent_wins": wins,
        "incumbent_losses": losses,
        "dual_losses": dual_losses,
        "dual_loss_cap": cap,
        "bar2": bar2,
        "bar2_pass": all(bar2.values()),
        "graduate": not problems and all(bar2.values()),
        "rens_run_total": sum(r["on"]["rens"].get("RensRun", 0) for r in rows),
        "rens_improved_total": sum(r["on"]["rens"].get("RensImproved", 0) for r in rows),
        "per_instance": detail,
    }


def _spread(v: list[float]) -> float:
    f = [x for x in v if math.isfinite(x)]
    if len(f) != len(v):
        return 0.0 if len(f) == 0 else math.inf
    return max(f) - min(f)


def _better(a: float, b: float, tol: float) -> bool:
    """``a`` lower than ``b`` by more than ``tol`` (inf handled: finite beats inf)."""
    if math.isinf(a):
        return False
    if math.isinf(b):
        return True
    return a < b - tol


def _worse_by(a: float, b: float, tol: float) -> bool:
    return _better(b, a, tol)


if __name__ == "__main__":
    sys.exit(main())
