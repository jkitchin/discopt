"""P2(a') end-to-end on the three G-G targets, flag ON vs OFF (issue #844).

Reports for each instance and arm: vars/constraints before and after
substitution, substitution wall, solve status/objective/bound/wall, and
soundness against the `=opt=` oracle read from minlplib.solu IN THIS SCRIPT.

The objective SENSE is read from the model, so a MAXIMIZE instance is not
reported as a violation just because the minimize convention was assumed.

Executed-assertion discipline: prints the number of oracle comparisons made and
exits non-zero when it is zero.
"""

import json
import os
import subprocess
import sys
import time

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"
SOLU = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu"
TARGETS = ["watercontamination0202", "gastrans582_cold13", "gastrans040"]
TIME_LIMIT = float(os.environ.get("E2E_TIME_LIMIT", "60"))

CHILD = r"""
import json, os, sys, time
import discopt.modeling as dm
from discopt._rust import model_to_repr

inst, tl = sys.argv[1], float(sys.argv[2])
path = f"{NL}/" + inst + ".nl"
out = {"instance": inst, "flag": os.environ.get("DISCOPT_PRESOLVE_SUBSTITUTE", "0")}
m = dm.from_nl(path)
rep = model_to_repr(m, getattr(m, "_builder", None))
out["vars_before"] = rep.n_vars
out["cons_before"] = rep.n_constraints
out["sense"] = rep.objective_sense
t0 = time.perf_counter()
red, chain = rep.substitute(4)
out["subst_wall"] = time.perf_counter() - t0
out["vars_after"] = red.n_vars
out["cons_after"] = red.n_constraints
t0 = time.perf_counter()
r = m.solve(time_limit=tl)
out["solve_wall"] = time.perf_counter() - t0
out["status"] = r.status
out["objective"] = None if r.objective is None else float(r.objective)
out["bound"] = None if r.bound is None else float(r.bound)
out["gap_certified"] = bool(getattr(r, "gap_certified", False))
print("JSONRESULT " + json.dumps(out))
""".replace("{NL}", NL)


def read_oracle():
    """`{name: (marker, value)}` parsed from minlplib.solu in this run."""
    table = {}
    with open(SOLU) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            marker, name = parts[0], parts[1]
            val = float(parts[2]) if len(parts) > 2 else None
            table[name] = (marker, val)
    return table


def run(inst, flag):
    env = dict(os.environ)
    env["DISCOPT_PRESOLVE_SUBSTITUTE"] = "1" if flag else "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.path.join(env["PYTHONPATH"], "python")
    p = subprocess.run(
        [sys.executable, "-u", "-c", CHILD, inst, str(TIME_LIMIT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=TIME_LIMIT * 8 + 600,
    )
    for line in p.stdout.splitlines():
        if line.startswith("JSONRESULT "):
            return json.loads(line[len("JSONRESULT ") :])
    print(f"  !! no result for {inst} flag={flag}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return None


if __name__ == "__main__":
    oracle = read_oracle()
    checks = 0
    violations = []
    rows = []
    for inst in TARGETS:
        for flag in (False, True):
            t0 = time.time()
            res = run(inst, flag)
            print(
                f"[{time.strftime('%H:%M:%S')}] {inst} flag={int(flag)} "
                f"done in {time.time() - t0:.1f}s",
                flush=True,
            )
            if res is None:
                continue
            marker, opt = oracle.get(inst, (None, None))
            res["oracle_marker"], res["oracle"] = marker, opt
            # Soundness: for MINIMIZE the dual bound must not exceed the optimum;
            # for MAXIMIZE the dual bound must not fall below it.
            if marker == "=opt=" and opt is not None and res["bound"] is not None:
                checks += 1
                tol = 1e-6 * (1.0 + abs(opt))
                if res["sense"] == "minimize" and res["bound"] > opt + tol:
                    violations.append((inst, flag, "bound above optimum", res["bound"], opt))
                if res["sense"] == "maximize" and res["bound"] < opt - tol:
                    violations.append((inst, flag, "bound below optimum", res["bound"], opt))
            if marker == "=opt=" and opt is not None and res["objective"] is not None:
                checks += 1
                tol = 1e-4 * (1.0 + abs(opt))
                if res["sense"] == "minimize" and res["objective"] < opt - tol:
                    violations.append((inst, flag, "false primal", res["objective"], opt))
                if res["sense"] == "maximize" and res["objective"] > opt + tol:
                    violations.append((inst, flag, "false primal", res["objective"], opt))
            rows.append(res)
            print("   " + json.dumps(res), flush=True)

    print("\n=== END-TO-END TABLE ===")
    hdr = (
        f"{'instance':24s} {'flag':4s} {'vars':>16s} {'cons':>16s} {'sub_s':>7s} "
        f"{'status':12s} {'objective':>14s} {'bound':>14s} {'wall_s':>7s}"
    )
    print(hdr)
    for r in rows:
        obj_s = "-" if r["objective"] is None else format(r["objective"], ".6g")
        bnd_s = "-" if r["bound"] is None else format(r["bound"], ".6g")
        print(
            f"{r['instance']:24s} {r['flag']:4s} "
            f"{r['vars_before']:>7d}->{r['vars_after']:<8d} "
            f"{r['cons_before']:>7d}->{r['cons_after']:<8d} "
            f"{r['subst_wall']:7.3f} {r['status']:12s} "
            f"{obj_s:>14s} {bnd_s:>14s} {r['solve_wall']:7.2f}"
        )
    with open("substitute_targets_e2e.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    print(f"\noracle comparisons executed: {checks}")
    if checks == 0:
        print("PROBE EXECUTED NOTHING")
        sys.exit(2)
    if violations:
        print(f"SOUNDNESS VIOLATIONS: {len(violations)}")
        for v in violations:
            print("   ", v)
        sys.exit(1)
    print("soundness: no violations against the =opt= oracle")
