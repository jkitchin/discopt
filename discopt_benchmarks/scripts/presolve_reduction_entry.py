"""P2(a) entry experiment: measure discopt's existing presolve reduction on
the three G-G instances, per-pass, vars/cons before->after with wall time.

Run with `python -u` for unbuffered output. Tight per-pass caps so aggregate's
O(n^2) loop returns its achieved-under-budget reduction rather than hanging.
Executed-count discipline: a pass that reduces nothing prints REDUCES-NOTHING.
"""

import sys
import time

from discopt._rust import parse_nl_file

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"
INSTANCES = ["gastrans040", "gastrans582_cold13", "watercontamination0202"]

# Reduction-relevant passes (change var/constraint COUNT), measured alone.
SINGLE_PASSES = ["eliminate", "aggregate", "factorable_elim"]
# Full reducer set to a fixed point (what a real presolve would chain).
FULL_SET = ["eliminate", "aggregate", "factorable_elim", "simplify", "fbbt"]

PASS_CAP_MS = 30000  # per single-pass budget
FULL_CAP_MS = 60000  # full-set budget


def counts(r):
    return r.n_vars, r.n_var_blocks, r.n_constraints


def presolve(r, passes, cap_ms):
    t0 = time.time()
    new_r, stats = r.presolve(
        passes=passes,
        max_iterations=32,
        time_limit_ms=cap_ms,
        work_unit_budget=0,
        fbbt_max_iter=20,
        fbbt_tol=1e-8,
    )
    return new_r, stats, time.time() - t0


def run_one(inst):
    path = f"{NL}/{inst}.nl"
    print(f"\n{'=' * 78}\nINSTANCE: {inst}\n{'=' * 78}", flush=True)
    v0, b0, c0 = counts(parse_nl_file(path))
    print(f"baseline  vars={v0}  cons={c0}", flush=True)

    for pname in SINGLE_PASSES:
        r = parse_nl_file(path)
        vb, bb, cb = counts(r)
        new_r, stats, dt = presolve(r, [pname], PASS_CAP_MS)
        v1, b1, c1 = counts(new_r)
        dv, dc = vb - v1, cb - c1
        ratio = (vb / v1) if v1 else float("inf")
        tag = "REDUCES-NOTHING" if (dv == 0 and dc == 0) else f"-{dv}v -{dc}c ({ratio:.2f}x)"
        print(
            f"  {pname:16s} {vb:>7d}->{v1:<7d}v {cb:>7d}->{c1:<7d}c "
            f"{dt:6.2f}s term={stats['terminated_by']} it={stats['iterations']}  {tag}",
            flush=True,
        )

    r = parse_nl_file(path)
    vb, bb, cb = counts(r)
    new_r, stats, dt = presolve(r, FULL_SET, FULL_CAP_MS)
    v1, b1, c1 = counts(new_r)
    ratio = (vb / v1) if v1 else float("inf")
    print(
        f"  FULL_SET         {vb:>7d}->{v1:<7d}v {cb:>7d}->{c1:<7d}c "
        f"{dt:6.2f}s term={stats['terminated_by']} it={stats['iterations']}  ({ratio:.2f}x)",
        flush=True,
    )


if __name__ == "__main__":
    for inst in sys.argv[1:] or INSTANCES:
        run_one(inst)
