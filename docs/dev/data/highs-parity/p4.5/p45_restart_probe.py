"""P4.5 entry part 2: root bound after a restart on the root's fixed box (seeded with
the HiGHS optimum). Prints executed count."""

import json
import os
import sys
import tempfile

sys.path.insert(0, "discopt_benchmarks/scripts")
import lp_milp_loader as L
import lp_milp_panel as P
import numpy as np
import p42_rens_panel as R
from discopt import _rust
from discopt.modeling.core import from_nl

names = sys.argv[1].split(",")
pop = {m["name"]: m for m in R.population()}
real = _rust.solve_milp_csc_py
executed = 0
for name in names:
    mem = pop[name]
    inst = L.read_instance(mem["path"])
    opt = mem["oracle_objective"]
    td = tempfile.mkdtemp()
    nl = os.path.join(td, "m.nl")
    perm = L.write_nl(inst, nl)
    m = from_nl(nl)
    h0 = P.solve_arrays(inst, 120.0)
    init = {v: float(h0["x"][perm[k]]) for k, v in enumerate(m._variables)}
    cap = {}

    def spy(*a, **kw):
        if "a" not in cap:
            cap["a"], cap["kw"] = list(a), dict(kw)
        return real(*a, **kw)

    _rust.solve_milp_csc_py = spy
    dump = os.path.join(td, "fix.json")
    os.environ["P45_DUMP"] = dump
    m.solve(time_limit=60, gap_tolerance=P.GAP, max_nodes=2, initial_solution=init)
    os.environ.pop("P45_DUMP")
    _rust.solve_milp_csc_py = real
    a, kw = cap["a"], cap["kw"]
    out0 = real(*a, **kw)
    fx = json.load(open(dump))
    lo = np.array(a[7], dtype=float)
    u = np.array(a[8], dtype=float)
    ns = len(fx["l"])
    lo[:ns] = np.maximum(lo[:ns], fx["l"])
    u[:ns] = np.minimum(u[:ns], fx["u"])
    assert np.all(lo <= u + 1e-9), name
    a2 = list(a)
    a2[7] = np.ascontiguousarray(lo)
    a2[8] = np.ascontiguousarray(u)
    out1 = real(*a2, **kw)
    sgn = -1.0 if inst.maximize else 1.0

    def g(b, opt=opt, sgn=sgn):
        return 100 * max(0.0, sgn * opt - min(b, sgn * opt)) / (1 + abs(opt))

    executed += 1
    print(
        json.dumps(
            {
                "name": name,
                "root_bound": out0[3],
                "restart_bound": out1[3],
                "restart_status": out1[0],
                "gap0": g(out0[3]),
                "gap1": g(out1[3]),
                "closed_pp": g(out0[3]) - g(out1[3]),
            }
        ),
        flush=True,
    )
print(f"p45 restart: executed={executed}")
