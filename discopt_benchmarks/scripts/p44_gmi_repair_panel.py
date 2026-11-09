"""highs-parity P4.4: the ``DISCOPT_MILP_ROOT_GMI_REPAIR`` graduation panel, bars
pre-registered here before the run (CLAUDE.md section 5; the plan's Phase 4 standing rule).

The flag (Rust, ``lp/gomory.rs`` + ``bnb/milp_driver.rs``) repairs two defects that left
whole MIPLIB roots with no GMI cut: (1) a SHORT root basis (degenerate artificials on
redundant rows) is completed with unit columns before separation; (2) a cut's coefficient
on a slack column that is not row ``ns + r``'s own slack is rewritten through the row the
column actually belongs to. Entry (``p44_root_gap.py``): ``fiber`` 61.55 % -> 19.76 % and
``sp150x300d`` 50.0 % -> 30.0 % root gap; ``mik`` bit-identical.

Population and arms as ``p42_rens_panel.py``: MIPLIB-38 + the M panel, Rust MILP route,
fresh subprocess per solve, ``DISCOPT_PROFILE=1`` in both arms, arm order alternating per
instance and rep, ``--reps`` 3, ``--time-limit`` 20 s, ``gap_tol = lp_milp_panel.GAP``.

Pre-registered bars (fixed before the run):

BAR 1, cert-clean, every solve of both arms: a certified objective equals the reference
  optimum within ``1e-4 (1 + |opt|)``; no bound past it by more than ``1e-6 (1 + |opt|)``;
  no incumbent better than it by more than that; every returned point ``max_viol <= 1e-6``.
BAR 2, net-positive (the Phase 4 standing rule):
  (2a) certified solves ON >= OFF (summed over instances and reps);
  (2b) total nodes over the instances certified in EVERY solve of both arms (per-instance
       median over reps) is at least 15 % lower ON, OR certified solves ON >= OFF + 2
       with that node total not higher ON;
  (2c) dual-bound losses <= 10 % (floor) of the open population P (instances not
       certified in every solve of both arms), a loss being ON's median dual gap above
       OFF's by more than 0.01 (1 % relative) AND by more than OFF's own rep spread.
Graduate only if BAR 1 and all of BAR 2 pass; otherwise retire flag + implementation.

Mechanism-fired: the ON arm's ``SepGomoryBasisCompleted + SubstSlackByColumn`` summed over
the panel must be > 0 and the OFF arm's must be 0 (proving the flag is read and off means
off). Executed counter; exceptions propagate; per-solve progress; non-zero exit when
nothing executed or nothing fired.
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

import lp_milp_panel as P  # noqa: E402, N812
import p42_rens_panel as R  # noqa: E402, N812  (population, gaps, cert checks)

FLAG = "DISCOPT_MILP_ROOT_GMI_REPAIR"
FIRED = ("SepGomoryBasisCompleted", "SubstSlackByColumn")
_CTR = re.compile(r"^\s+(SepGomoryBasisCompleted|SubstSlackByColumn)\s+(\d+)$")


def solve(path: str, on: bool, a) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in (FLAG, "DISCOPT_LP_STD_ENGINE")}
    env.update(PYTHONUNBUFFERED="1", DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PROFILE="1")
    if on:
        env[FLAG] = "1"
    cmd = [sys.executable, str(PANEL_PY), "worker", path, "R", str(a.time_limit), str(a.gap),
           str(a.max_nodes)]  # fmt: skip
    load = os.getloadavg()[0]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3 * a.time_limit + 180)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"{path} on={on}: rc={p.returncode} {p.stderr[-1500:]}")
    res = json.loads(lines[0][len("RESULT ") :])
    ctr = dict.fromkeys(FIRED, 0)
    for ln in p.stderr.splitlines():
        mm = _CTR.match(ln)
        if mm:
            ctr[mm.group(1)] += int(mm.group(2))
    res["fired"] = ctr
    res["load"] = load
    return res


def cert_problems(res: dict, opt: float, maximize: bool) -> list[str]:
    res = {**res, "rens": {}}  # R.cert_problems' RENS check is inert without counters
    return R.cert_problems(res, opt, maximize, "off")


def evaluate(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["name"], []).append(r)
    problems = [(r["name"], r["rep"], arm, p) for r in rows for arm, ps in r["problems"].items()
                for p in ps]  # fmt: skip
    solved = {arm: sum(bool(r[arm]["gap_certified"]) for r in rows) for arm in ("off", "on")}
    both = [nm for nm, rs in by.items()
            if all(r[arm]["gap_certified"] for r in rs for arm in ("off", "on"))]  # fmt: skip
    nodes = {arm: sum(statistics.median([r[arm]["nodes"] for r in by[nm]]) for nm in both)
             for arm in ("off", "on")}  # fmt: skip
    pop = [nm for nm in by if nm not in both]
    dual_losses, detail = 0, {}
    for nm in pop:
        g = {arm: [R.gaps(r[arm], r["opt"], r["maximize"])[1] for r in by[nm]]
             for arm in ("off", "on")}  # fmt: skip
        med = {arm: statistics.median(g[arm]) for arm in g}
        spread = R._spread(g["off"])
        loss = R._worse_by(med["on"], med["off"], max(0.01, spread))
        dual_losses += loss
        detail[nm] = {"dual_off": med["off"], "dual_on": med["on"], "off_spread": spread,
                      "dual_loss": loss}  # fmt: skip
    cap = math.floor(0.10 * len(pop))
    nodes_ok = nodes["off"] > 0 and nodes["on"] <= 0.85 * nodes["off"]
    plus2 = solved["on"] >= solved["off"] + 2 and nodes["on"] <= nodes["off"]
    bar2 = {
        "2a_solved_on_ge_off": solved["on"] >= solved["off"],
        "2b_nodes_minus15_or_solved_plus2": nodes_ok or plus2,
        "2c_dual_losses_le_10pct": dual_losses <= cap,
    }
    fired = {arm: sum(sum(r[arm]["fired"].values()) for r in rows) for arm in ("off", "on")}
    return {
        "bar1_problems": problems,
        "bar1_pass": not problems,
        "bar1_checks": 2 * len(rows),
        "solved": solved,
        "both_certified": len(both),
        "nodes_both_certified": nodes,
        "population_open": len(pop),
        "dual_losses": dual_losses,
        "dual_loss_cap": cap,
        "bar2": bar2,
        "bar2_pass": all(bar2.values()),
        "graduate": not problems and all(bar2.values()),
        "fired": fired,
        "per_instance": detail,
    }


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
    pop = R.population()
    if a.instances:
        keep = set(a.instances.split(","))
        pop = [x for x in pop if x["name"] in keep]
    print("header", json.dumps(hdr), "instances", len(pop), flush=True)

    def one(job):
        k, mem, rep = job
        order = (False, True) if (k + rep) % 2 == 0 else (True, False)
        got = {("on" if on else "off"): solve(str(REPO / mem["path"]), on, a) for on in order}
        return mem, rep, got

    jobs = [(k, mem, rep) for rep in range(a.reps) for k, mem in enumerate(pop)]
    rows, executed = [], 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for mem, rep, got in ex.map(one, jobs):
            executed += 2
            opt, maxim = mem["oracle_objective"], mem["maximize"]
            row = {"name": mem["name"], "panel": mem.get("panel"), "rep": rep, "opt": opt,
                   "maximize": maxim, **got}  # fmt: skip
            row["problems"] = {arm: cert_problems(got[arm], opt, maxim) for arm in got}
            rows.append(row)
            o, n = got["off"], got["on"]
            print(f"[{len(rows)}/{len(jobs)}] {mem['name']:22s} r{rep} "
                  f"OFF {o['status']}/{o['gap_certified']} bnd={o['bound']} n={o['nodes']} "
                  f"| ON {n['status']}/{n['gap_certified']} bnd={n['bound']} n={n['nodes']} "
                  f"fired={n['fired']} load={o['load']:.1f}", flush=True)  # fmt: skip
            Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows},
                                              indent=1) + "\n")  # fmt: skip
    verdict = evaluate(rows)
    Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows,
                                       "verdict": verdict}, indent=1) + "\n")  # fmt: skip
    v = {k: verdict[k] for k in verdict if k != "per_instance"}
    print(json.dumps(v, indent=1), flush=True)
    print(f"p44 panel: executed={executed} fired={verdict['fired']}", flush=True)
    ok = executed and verdict["fired"]["on"] > 0 and verdict["fired"]["off"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
