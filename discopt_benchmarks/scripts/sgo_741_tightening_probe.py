"""Entry-experiment / reproduction probe for issue #741 (SGO constrained tightening).

Reconstructs the two sound-but-uncertified probes from #736's measurement as
discopt Models — the benchmark corpus is not required; both are classic
Floudas/Himmelblau problems with fully documented coefficients whose oracle
values match the issue table:

* ``ex3_1_2``  == Himmelblau #11 / g04      (5 vars, 6 signomial ineqs, opt -30665.53867)
* ``ex7_2_3``  == HS106 heat exchanger / g10 (8 vars, 6 signomial ineqs, opt 7049.2480205286)

Measures, per instance: classifier acceptance, the ROOT constrained node bound,
and a budgeted tree solve — with the #741 tightened relaxation (default) and
with the frozen pre-#741 reference (``obbt=False``). Also reruns the FALSIFIED
lever-3 experiment (certified secant-argument xi-range tightening): the
certified ranges come out WIDER than box-implied because the Lagrangian corner
certificate is weak in non-coordinate directions — recorded in
``docs/dev/performance-plan.md`` §6.

Task 2 (integer signomial MINLPs) adds the ``nsig`` mode: it loads the in-repo
``cvxnonsep_nsig30`` (15 continuous + 15 integer variables, one mixed-sign
signomial constraint), confirms the classifier now admits it, and runs a
budgeted integer solve — recovering the known integer optimum 130.6287 as an
integer-feasible incumbent with a sound dual bound.

Usage::

    python discopt_benchmarks/scripts/sgo_741_tightening_probe.py [ex3_1_2|ex7_2_3|xi|nsig]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from discopt import Model
from discopt._jax.convexity.signomial_global import (
    _cert_min_linear,
    _constrained_node_bound,
    _obbt_tighten,
    _pack,
    classify_signomial_global,
    solve_signomial_global,
)


def ex3_1_2():
    """Himmelblau #11 / g04 — MINLPLib ex3_1_2 (opt -30665.53867)."""
    m = Model()
    x1 = m.continuous("x1", lb=78.0, ub=102.0)
    x2 = m.continuous("x2", lb=33.0, ub=45.0)
    x3 = m.continuous("x3", lb=27.0, ub=45.0)
    x4 = m.continuous("x4", lb=27.0, ub=45.0)
    x5 = m.continuous("x5", lb=27.0, ub=45.0)
    m.minimize(5.3578547 * x3**2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141)
    e1 = 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5
    m.subject_to(e1 <= 6.665593)
    m.subject_to(e1 >= -85.334407)
    e2 = 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3**2
    m.subject_to(e2 <= 29.48751)
    m.subject_to(e2 >= 9.48751)
    e3 = 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    m.subject_to(e3 <= 15.699039)
    m.subject_to(e3 >= 10.699039)
    return m, -30665.53867


def ex7_2_3():
    """HS106 heat exchanger / g10 — MINLPLib ex7_2_3 (opt 7049.2480205286)."""
    m = Model()
    x1 = m.continuous("x1", lb=100.0, ub=10000.0)
    x2 = m.continuous("x2", lb=1000.0, ub=10000.0)
    x3 = m.continuous("x3", lb=1000.0, ub=10000.0)
    x4 = m.continuous("x4", lb=10.0, ub=1000.0)
    x5 = m.continuous("x5", lb=10.0, ub=1000.0)
    x6 = m.continuous("x6", lb=10.0, ub=1000.0)
    x7 = m.continuous("x7", lb=10.0, ub=1000.0)
    x8 = m.continuous("x8", lb=10.0, ub=1000.0)
    m.minimize(x1 + x2 + x3)
    m.subject_to(0.0025 * x4 + 0.0025 * x6 <= 1.0)
    m.subject_to(-0.0025 * x4 + 0.0025 * x5 + 0.0025 * x7 <= 1.0)
    m.subject_to(-0.01 * x5 + 0.01 * x8 <= 1.0)
    m.subject_to(100.0 * x1 - x1 * x6 + 833.33252 * x4 <= 83333.333)
    m.subject_to(x2 * x4 - x2 * x7 - 1250.0 * x4 + 1250.0 * x5 <= 0.0)
    m.subject_to(x3 * x5 - x3 * x8 - 2500.0 * x5 <= -1250000.0)
    return m, 7049.2480205286


def probe(name, m, oracle, max_nodes=5000, time_limit=300.0):
    struct = classify_signomial_global(m)
    if struct is None:
        print(f"{name}: ABSTAINED by classifier")
        return
    obj_pack = _pack(struct.terms)
    con_packs = [_pack(t) for t in struct.constraint_terms]
    root_lb, _u = _constrained_node_bound(obj_pack, con_packs, struct.u_lb, struct.u_ub)
    print(f"{name}: u-dim={len(struct.u_lb)} cons={len(con_packs)}")
    print(f"  pre-#741 ROOT bound = {root_lb:.6g}   (oracle {oracle:.6g})")
    for obbt in (True, False):
        t0 = time.perf_counter()
        res = solve_signomial_global(
            m, gap_tolerance=1e-4, max_nodes=max_nodes, time_limit=time_limit, obbt=obbt
        )
        wall = time.perf_counter() - t0
        print(
            f"  obbt={obbt}: status={res.status} certified={res.gap_certified} "
            f"obj={res.objective} bound={res.bound} nodes={res.node_count} wall={wall:.1f}s"
        )
        if res.bound is not None and res.bound > oracle + 1e-3 * max(1.0, abs(oracle)):
            print("  *** SOUNDNESS VIOLATION: bound above oracle ***")


def xi_falsification():
    """Rerun the falsified lever-3 entry experiment (certified xi-ranges)."""
    for name, (m, oracle), incumbent in (
        ("ex3_1_2", ex3_1_2(), -30665.169857478555),
        ("ex7_2_3", ex7_2_3(), 7049.264331889627),
    ):
        struct = classify_signomial_global(m)
        obj_pack = _pack(struct.terms)
        con_packs = [_pack(t) for t in struct.constraint_terms]
        tt = _obbt_tighten(obj_pack, con_packs, struct.u_lb, struct.u_ub, incumbent, rounds=8)
        u_lb, u_ub = tt if tt is not None else (struct.u_lb, struct.u_ub)
        cut = incumbent + 1e-9 * max(1.0, abs(incumbent))
        con_specs = [(p, 0.0) for p in con_packs] + [(obj_pack, cut)]
        print(f"== {name} (oracle {oracle:.6g}) certified xi-ranges vs box-implied:")
        seen = set()
        for sig, _lc, ex in [obj_pack] + con_packs:
            for k in range(len(sig)):
                if sig[k] >= 0 or not np.any(np.abs(ex[k]) > 1e-12):
                    continue
                key = ex[k].tobytes()
                if key in seen:
                    continue
                seen.add(key)
                a = ex[k]
                lo = _cert_min_linear(a, con_specs, u_lb, u_ub)
                hi = -_cert_min_linear(-a, con_specs, u_lb, u_ub)
                pos_part = np.where(a >= 0.0, a, 0.0)
                neg_part = np.where(a < 0.0, a, 0.0)
                box_lo = pos_part @ u_lb + neg_part @ u_ub
                box_hi = pos_part @ u_ub + neg_part @ u_lb
                verdict = "TIGHTER" if (lo > box_lo + 1e-9 or hi < box_hi - 1e-9) else "wider/equal"
                print(
                    f"  box [{box_lo:7.2f},{box_hi:7.2f}] -> cert [{lo:7.2f},{hi:7.2f}]  {verdict}"
                )


def nsig_integer_probe(max_nodes=2000, time_limit=150.0):
    """Task 2: admit + soundly solve the integer signomial MINLP cvxnonsep_nsig30."""
    from discopt.modeling.core import from_nl

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "python",
        "tests",
        "data",
        "minlplib_nl",
        "cvxnonsep_nsig30.nl",
    )
    if not os.path.exists(path):
        print("cvxnonsep_nsig30.nl not present; skipping nsig probe")
        return
    m = from_nl(path)
    struct = classify_signomial_global(m)
    if struct is None:
        print("cvxnonsep_nsig30: ABSTAINED (unexpected)")
        return
    oracle = 130.6287
    print(
        f"cvxnonsep_nsig30: u-dim={len(struct.u_lb)} "
        f"integer-coords={len(struct.integer_coords)} cons={len(struct.constraint_terms)}"
    )
    t0 = time.perf_counter()
    res = solve_signomial_global(m, gap_tolerance=1e-4, max_nodes=max_nodes, time_limit=time_limit)
    wall = time.perf_counter() - t0
    print(
        f"  status={res.status} certified={res.gap_certified} obj={res.objective} "
        f"bound={res.bound} nodes={res.node_count} wall={wall:.1f}s   (oracle {oracle})"
    )
    if res.bound is not None and res.bound > oracle + 1e-2:
        print("  *** SOUNDNESS VIOLATION: dual bound above oracle ***")
    if res.x is not None:
        ints = [float(res.x[n][0]) for n in list(res.x)[15:30]]
        allint = all(abs(v - round(v)) < 1e-6 for v in ints)
        print(f"  incumbent integers all integral: {allint}; obj vs oracle: {res.objective}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ex3_1_2"):
        probe("ex3_1_2", *ex3_1_2())
    if which in ("all", "ex7_2_3"):
        probe("ex7_2_3", *ex7_2_3())
    if which in ("all", "xi"):
        xi_falsification()
    if which in ("all", "nsig"):
        nsig_integer_probe()
