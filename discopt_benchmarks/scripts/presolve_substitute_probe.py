"""P2(a') step-1 probe: batch substitution-graph aggregator, reduction + wall.

Kill criterion, stated BEFORE the run (issue #844 / P2(a')):
    the aggregator must reach >= 50x variable reduction on
    watercontamination0202 within 5 s of substitution wall time.

Also runs an equivalence probe on every instance: sample points inside the
REDUCED box, lift them through the postsolve chain, and require that the
pristine model agrees with the reduced model on (objective, max constraint
violation, max bound violation). Any disagreement above tolerance is a
soundness failure of the transform.

Executed-assertion discipline: prints the number of comparisons actually made
and exits non-zero when it is zero.
"""

import sys
import time

import numpy as np
from discopt._rust import parse_nl_file

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"
DEFAULT = ["gastrans040", "gastrans582_cold13", "watercontamination0202"]
N_SAMPLES = 5
EQ_TOL = 1e-6

CHECKS = 0
FAILURES = []


def sample_point(rep, rng):
    """Uniform sample inside the model box, clipped to a finite window."""
    lo = np.concatenate([np.asarray(rep.var_lb(i), dtype=float) for i in range(rep.n_var_blocks)])
    hi = np.concatenate([np.asarray(rep.var_ub(i), dtype=float) for i in range(rep.n_var_blocks)])
    lo = np.where(np.isfinite(lo), lo, -10.0)
    hi = np.where(np.isfinite(hi), hi, 10.0)
    hi = np.maximum(hi, lo)
    return lo + rng.random(lo.shape) * (hi - lo)


def equivalence_probe(pristine, reduced, chain, rng):
    global CHECKS
    for _ in range(N_SAMPLES):
        x_red = sample_point(reduced, rng)
        x_full = np.asarray(chain.postsolve(list(x_red)), dtype=float)
        o_r, c_r, b_r = reduced.evaluate_point(list(x_red))
        o_p, c_p, b_p = pristine.evaluate_point(list(x_full))
        CHECKS += 3
        scale = 1.0 + abs(o_r)
        if not np.isfinite(o_p) or abs(o_p - o_r) > EQ_TOL * scale:
            FAILURES.append(f"objective {o_p!r} != {o_r!r}")
        if abs(c_p - c_r) > EQ_TOL * (1.0 + abs(c_r)):
            FAILURES.append(f"constraint violation {c_p!r} != {c_r!r}")
        if b_p > b_r + EQ_TOL * (1.0 + abs(b_r)):
            FAILURES.append(f"bound violation {b_p!r} > {b_r!r}")


def run_one(inst, rng):
    path = f"{NL}/{inst}.nl"
    pristine = parse_nl_file(path)
    v0, c0 = pristine.n_vars, pristine.n_constraints
    t0 = time.perf_counter()
    reduced, chain = pristine.substitute(4)
    dt = time.perf_counter() - t0
    v1, c1 = reduced.n_vars, reduced.n_constraints
    ratio = (v0 / v1) if v1 else float("inf")
    print(
        f"{inst:24s} {v0:>7d}->{v1:<7d}v  {c0:>7d}->{c1:<7d}c  "
        f"{dt:7.3f}s  {ratio:8.2f}x  sweeps={chain.n_sweeps}",
        flush=True,
    )
    for i, st in enumerate(chain.sweep_stats()):
        print(
            f"    sweep{i}: elim={st['variables_eliminated']} "
            f"cand={st['candidate_rows']} cycle={st['cycles_rejected']} "
            f"pivot={st['pivots_rejected']} growth={st['growth_rejected']} "
            f"inelig={st['ineligible_rejected']} "
            f"infeas={st['infeasible_detected']} abort={st['aborted']}",
            flush=True,
        )
    equivalence_probe(pristine, reduced, chain, rng)
    return inst, v0, v1, ratio, dt


if __name__ == "__main__":
    rng = np.random.default_rng(20260727)
    rows = [run_one(i, rng) for i in (sys.argv[1:] or DEFAULT)]

    print(f"\nequivalence comparisons executed: {CHECKS}")
    if CHECKS == 0:
        print("PROBE EXECUTED NOTHING")
        sys.exit(2)
    if FAILURES:
        print(f"EQUIVALENCE FAILURES: {len(FAILURES)}")
        for f in FAILURES[:20]:
            print("   ", f)
        sys.exit(1)
    print("equivalence: all comparisons passed")

    water = [r for r in rows if r[0] == "watercontamination0202"]
    if water:
        _, v0, v1, ratio, dt = water[0]
        ok = ratio >= 50.0 and dt <= 5.0
        print(
            f"\nKILL CRITERION (>=50x within 5s on watercontamination0202): "
            f"{ratio:.2f}x in {dt:.3f}s -> {'PASS' if ok else 'FAIL'}"
        )
