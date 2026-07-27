"""CLAUDE.md §5 differential panel for DISCOPT_PRESOLVE_SUBSTITUTE (issue #844).

Runs both arms (flag OFF, flag ON) over the 66 vendored `.nl` instances in
`python/tests/data/minlplib_nl/` and checks the cert-clean bars:

  * no dual bound crossing the `=opt=` oracle (sense-aware: MINIMIZE bounds must
    not exceed the optimum, MAXIMIZE bounds must not fall below it);
  * no false primal;
  * no certification regression (`gap_certified` True in OFF, False in ON);
  * no lost incumbent (an objective in OFF but none in ON);
  * every incumbent independently feasibility-verified against the PRISTINE
    model via the Rust `evaluate_point` on the un-presolved `ModelRepr`.

Each instance runs in a subprocess so a crash or a hang in one arm cannot
poison the panel. Oracle values are read from `minlplib.solu` IN THIS SCRIPT
and the objective sense is read from each model.

Executed-assertion discipline: prints the number of comparisons made and exits
non-zero when it is zero.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "python" / "tests" / "data" / "minlplib_nl"
SOLU = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu"
TIME_LIMIT = float(os.environ.get("PANEL_TIME_LIMIT", "20"))
FEAS_TOL = 1e-5

CHILD = r"""
import json, os, sys, time
import numpy as np
import discopt.modeling as dm
from discopt._rust import model_to_repr

path, tl = sys.argv[1], float(sys.argv[2])
out = {"instance": os.path.basename(path)[:-3],
       "flag": os.environ.get("DISCOPT_PRESOLVE_SUBSTITUTE", "0")}
m = dm.from_nl(path)
pristine = model_to_repr(m, getattr(m, "_builder", None))
out["sense"] = pristine.objective_sense
out["n_vars"] = pristine.n_vars
t0 = time.perf_counter()
r = m.solve(time_limit=tl)
out["wall"] = time.perf_counter() - t0
out["status"] = r.status
out["objective"] = None if r.objective is None else float(r.objective)
out["bound"] = None if r.bound is None else float(r.bound)
out["gap_certified"] = bool(getattr(r, "gap_certified", False))
out["node_count"] = int(getattr(r, "node_count", 0) or 0)

# Independent feasibility verification of the reported incumbent against the
# PRISTINE (un-presolved) model, using the Rust evaluator rather than whatever
# the solve path used.
out["verified"] = None
if isinstance(r.x, dict):
    flat = []
    ok = True
    for v in m._variables:
        if v.name not in r.x:
            ok = False
            break
        flat.extend(np.asarray(r.x[v.name], dtype=float).reshape(-1).tolist())
    if ok and len(flat) == pristine.n_vars:
        obj, con, bnd = pristine.evaluate_point(flat)
        out["verified"] = {"objective": obj, "con_viol": con, "bnd_viol": bnd}
print("JSONRESULT " + json.dumps(out))
"""


def read_oracle():
    table = {}
    with open(SOLU) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            table[parts[1]] = (parts[0], float(parts[2]) if len(parts) > 2 else None)
    return table


def run(path, flag):
    env = dict(os.environ)
    env["DISCOPT_PRESOLVE_SUBSTITUTE"] = "1" if flag else "0"
    env["PYTHONPATH"] = str(REPO / "python")
    try:
        p = subprocess.run(
            [sys.executable, "-u", "-c", CHILD, str(path), str(TIME_LIMIT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=TIME_LIMIT * 10 + 300,
        )
    except subprocess.TimeoutExpired:
        return {"instance": path.stem, "flag": "1" if flag else "0", "status": "HARD_TIMEOUT"}
    for line in p.stdout.splitlines():
        if line.startswith("JSONRESULT "):
            return json.loads(line[len("JSONRESULT ") :])
    return {
        "instance": path.stem,
        "flag": "1" if flag else "0",
        "status": "CRASH",
        "stderr": p.stderr[-600:],
    }


def main():
    oracle = read_oracle()
    instances = sorted(CORPUS.glob("*.nl"))
    print(f"panel: {len(instances)} instances, {TIME_LIMIT}s each, two arms", flush=True)

    checks = 0
    problems = []
    results = {}
    for i, path in enumerate(instances, 1):
        row = {}
        for flag in (False, True):
            t0 = time.time()
            row["on" if flag else "off"] = run(path, flag)
            print(
                f"[{i:3d}/{len(instances)}] {path.stem:28s} "
                f"{'ON ' if flag else 'OFF'} {time.time() - t0:6.1f}s "
                f"{row['on' if flag else 'off'].get('status')}",
                flush=True,
            )
        results[path.stem] = row
        off, on = row["off"], row["on"]
        marker, opt = oracle.get(path.stem, (None, None))

        for arm_name, arm in (("off", off), ("on", on)):
            if arm.get("status") in ("CRASH", "HARD_TIMEOUT"):
                problems.append((path.stem, arm_name, "run failed", arm.get("status"), None))
                continue
            sense = arm.get("sense")
            # Bound vs oracle.
            if marker == "=opt=" and opt is not None and arm.get("bound") is not None:
                checks += 1
                tol = 1e-6 * (1.0 + abs(opt))
                if sense == "minimize" and arm["bound"] > opt + tol:
                    problems.append((path.stem, arm_name, "bound above optimum", arm["bound"], opt))
                if sense == "maximize" and arm["bound"] < opt - tol:
                    problems.append((path.stem, arm_name, "bound below optimum", arm["bound"], opt))
            # False primal.
            if marker == "=opt=" and opt is not None and arm.get("objective") is not None:
                checks += 1
                tol = 1e-4 * (1.0 + abs(opt))
                if sense == "minimize" and arm["objective"] < opt - tol:
                    problems.append((path.stem, arm_name, "false primal", arm["objective"], opt))
                if sense == "maximize" and arm["objective"] > opt + tol:
                    problems.append((path.stem, arm_name, "false primal", arm["objective"], opt))
            # Independent feasibility of the reported incumbent.
            if arm.get("objective") is not None:
                checks += 1
                v = arm.get("verified")
                if v is None:
                    problems.append((path.stem, arm_name, "incumbent unverifiable", None, None))
                elif max(v["con_viol"], v["bnd_viol"]) > FEAS_TOL:
                    problems.append(
                        (
                            path.stem,
                            arm_name,
                            "incumbent infeasible on pristine model",
                            max(v["con_viol"], v["bnd_viol"]),
                            FEAS_TOL,
                        )
                    )
                elif abs(v["objective"] - arm["objective"]) > 1e-4 * (1 + abs(arm["objective"])):
                    problems.append(
                        (
                            path.stem,
                            arm_name,
                            "reported objective != pristine objective",
                            arm["objective"],
                            v["objective"],
                        )
                    )

        # Differential checks: ON must not lose anything OFF had.
        if off.get("status") not in ("CRASH", "HARD_TIMEOUT") and on.get("status") not in (
            "CRASH",
            "HARD_TIMEOUT",
        ):
            checks += 2
            if off.get("gap_certified") and not on.get("gap_certified"):
                problems.append((path.stem, "diff", "certification regression", True, False))
            if off.get("objective") is not None and on.get("objective") is None:
                problems.append((path.stem, "diff", "lost incumbent", off["objective"], None))

    out_path = REPO / "substitute_diff_panel.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out_path}")

    print(f"\ncomparisons executed: {checks}")
    if checks == 0:
        print("PANEL EXECUTED NOTHING")
        return 2
    if problems:
        print(f"CERT-CLEAN FAILURES: {len(problems)}")
        for p in problems:
            print("   ", p)
        return 1
    print(
        "cert-clean: 0 bound crossings, 0 false primals, 0 cert regressions, "
        "0 lost incumbents, all incumbents verified on the pristine model"
    )

    # Net-positive summary (informational; the gate above is soundness).
    gained = [
        k
        for k, v in results.items()
        if v["off"].get("objective") is None and v["on"].get("objective") is not None
    ]
    cert_gained = [
        k
        for k, v in results.items()
        if not v["off"].get("gap_certified") and v["on"].get("gap_certified")
    ]
    print(f"incumbents gained by ON: {len(gained)} {gained}")
    print(f"certifications gained by ON: {len(cert_gained)} {cert_gained}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
