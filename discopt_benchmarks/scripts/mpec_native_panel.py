"""Cert-clean panel for #2167 slice 3 on NON-ReLU complementarity problems.

The native complementarity branch applies to every pair lowered with
``method="native"``, not only ReLU pairs, so a ReLU win must not hide a
regression elsewhere (issue #2167, risks). The in-repo ``.nl`` corpus has no
complementarity constraint (all 76 files report zero), so this panel builds the
non-ReLU class directly: classic small MPECs and random LCP-constrained
programs, every pair written over variable operands.

Arms per instance, interleaved: ``gdp`` and ``sos1`` (the existing exact
lowerings), ``native0`` (native lowering, ``DISCOPT_COMPL_BRANCH=0``) and
``native1`` (native lowering, ``DISCOPT_COMPL_BRANCH=1``), each in its own
subprocess so the flag is read fresh.

Checks (CLAUDE.md §1, §5): every certified arm agrees with every other
certified arm within the certified gaps; no arm's dual bound exceeds any
arm's incumbent (min sense); every incumbent's source complementarity residual
is <= 1e-6. Ends with the number of executed checks; non-zero exit when it is
zero or any check failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ARM_SRC = r"""
import json, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import numpy as np
import discopt.modeling as dm
from discopt.modeling.core import Model
import discopt.mpec as mpec
from discopt.mpec_report import source_residual_report, _flat_from_result
assert hasattr(mpec, "native_pair_columns"), mpec.__file__
arm, kind = sys.argv[1], sys.argv[2]
seed, n, tl = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])

def build():
    m = Model(kind); pairs = []
    if kind == "distance":
        x = m.continuous("x", lb=0, ub=10); y = m.continuous("y", lb=0, ub=10)
        m.minimize((x - 1) ** 2 + (y - 1) ** 2); pairs = [mpec.complementarity(x, y, name="c")]
    elif kind == "balanced":
        x = m.continuous("x", lb=0, ub=10); y = m.continuous("y", lb=0, ub=10)
        m.subject_to(x == y, name="eq"); m.maximize(x + y)
        pairs = [mpec.complementarity(x, y, name="c")]
    elif kind == "bard1":
        # MacMPEC bard1: KKT of the lower level written with explicit slacks.
        x = m.continuous("x", lb=0, ub=10); y = m.continuous("y", lb=0, ub=10)
        lam = m.continuous("lam", shape=(3,), lb=0, ub=50)
        s = m.continuous("s", shape=(3,), lb=0, ub=60)
        m.subject_to(2 * (y - 1) - 1.5 * x + lam[0] - 0.5 * lam[1] + lam[2] == 0, name="stat")
        m.subject_to(s[0] == 3 * x - y - 3, name="g0")
        m.subject_to(s[1] == -x + 0.5 * y + 4, name="g1")
        m.subject_to(s[2] == -x - y + 7, name="g2")
        m.minimize((x - 5) ** 2 + (2 * y + 1) ** 2)
        pairs = [mpec.complementarity(lam[i], s[i], name=f"c{i}") for i in range(3)]
    elif kind in ("lcp_lp", "lcp_qp"):
        # min c.x (+ 0.5 x'x) s.t. w = M x + q, 0 <= x _|_ w >= 0, x in [0, 10].
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n)); M = A @ A.T / n + 0.5 * rng.standard_normal((n, n))
        q = rng.standard_normal(n) * 2
        c = rng.standard_normal(n)
        x = m.continuous("x", shape=(n,), lb=0, ub=10)
        w = m.continuous("w", shape=(n,), lb=0, ub=200)
        m.subject_to(w == dm.sum(M * x, axis=1) + q, name="lcp")
        obj = dm.sum(c * x)
        if kind == "lcp_qp":
            obj = obj + 0.5 * dm.sum(x * x)
        m.minimize(obj)
        pairs = [mpec.complementarity(x[i], w[i], name=f"c{i}") for i in range(n)]
    return m, pairs

m, pairs = build()
t0 = time.perf_counter()
method = {"gdp": "gdp", "sos1": "sos1", "native0": "native", "native1": "native"}[arm]
r = mpec.solve_mpec(m, pairs, method=method, time_limit=tl)
wall = time.perf_counter() - t0
src = None
if r.x is not None:
    src = float(source_residual_report(m, x_flat=_flat_from_result(m, r)).complementarity.value)
sense = -1.0 if kind == "balanced" else 1.0
print("RESULT " + json.dumps(dict(
    status=str(r.status), objective=r.objective, bound=r.bound, certified=bool(r.gap_certified),
    nodes=int(r.node_count or 0), wall=wall, src_residual=src, sense=sense,
    compl_branches=(r.solver_stats or {}).get("tree/compl_branches"))), flush=True)
"""

PANEL = [("distance", 0, 0), ("balanced", 0, 0), ("bard1", 0, 0)] + [
    (k, s, n) for k in ("lcp_lp", "lcp_qp") for n in (3, 5, 8) for s in (0, 1, 2)
]


def run_arm(arm, inst, tl):
    kind, seed, n = inst
    env = dict(os.environ, DISCOPT_COMPL_BRANCH="1" if arm == "native1" else "0")
    p = subprocess.run(
        [sys.executable, "-u", "-c", ARM_SRC, arm, kind, str(seed), str(n), str(tl)],
        capture_output=True,
        text=True,
        env=env,
        timeout=tl * 4 + 120,
    )
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {
        "status": "crash",
        "error": (p.stderr or p.stdout)[-1500:],
        "objective": None,
        "bound": None,
        "certified": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    arms = ["gdp", "sos1", "native0", "native1"]
    out = open(args.out, "a") if args.out else None  # noqa: SIM115
    n_chk = n_fail = 0
    for k, inst in enumerate(PANEL):
        order = arms[k % 4 :] + arms[: k % 4]
        res = {a: run_arm(a, inst, args.time_limit) for a in order}
        fails = []
        for a, r in res.items():
            if r["status"] == "crash":
                fails.append(f"{a} crashed: {r['error'][-300:]}")
                continue
            if r.get("src_residual") is not None:
                n_chk += 1
                if r["src_residual"] > 1e-6:
                    fails.append(f"{a} incumbent source residual {r['src_residual']:.2e}")
        live = {a: r for a, r in res.items() if r["status"] != "crash"}
        for a, ra in live.items():
            for b, rb in live.items():
                if a == b or ra["bound"] is None or rb["objective"] is None:
                    continue
                n_chk += 1
                s = ra["sense"]
                tol = 1e-6 * max(1.0, abs(rb["objective"]))
                if s * ra["bound"] > s * rb["objective"] + tol:
                    fails.append(f"{a} bound {ra['bound']} crosses {b} incumbent {rb['objective']}")
        # A certified infeasibility carries no objective; it must not meet an
        # incumbent from any arm (a point refutes the proof), and two certified
        # arms must agree on feasibility before their objectives are compared.
        infeas = [a for a, r in live.items() if r["status"] == "infeasible"]
        for a in infeas:
            for b, rb in live.items():
                if rb["objective"] is not None:
                    n_chk += 1
                    fails.append(
                        f"{a} certified infeasible but {b} has incumbent {rb['objective']}"
                    )
        cert = {
            a: r["objective"]
            for a, r in live.items()
            if r["certified"] and r["objective"] is not None
        }
        vals = list(cert.values())
        if len(vals) >= 2:
            n_chk += 1
            if max(vals) - min(vals) > 1e-4 * max(1.0, abs(vals[0])):
                fails.append(f"certified objectives disagree: {cert}")
        n_fail += len(fails)
        rec = {
            "kind": inst[0],
            "seed": inst[1],
            "n": inst[2],
            "order": order,
            "arms": res,
            "fails": fails,
        }
        line = json.dumps(rec)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()
    print(f"# checks executed: {n_chk}; failed: {n_fail}", flush=True)
    return 0 if (n_chk and not n_fail) else 1


if __name__ == "__main__":
    sys.exit(main())
