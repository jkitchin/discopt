"""Structural census of SCIP's reduction on the G-G instances.

For each instance: classify every EQUALITY constraint by arity (# distinct
scalar-var leaves in a linear body): 1-var (singleton pin -> eliminate.rs),
2-var (affine substitution -> aggregate.rs), >=3-var linear, nonlinear.

This tells us what fraction of SCIP's var reduction is reachable by discopt's
existing eliminate/aggregate transforms vs. requires new algorithm (>=3-var
Gaussian pivoting / doubleton chains where target appears in many rows).

Executed-count discipline: prints total examined + each bucket; fails at zero.
"""

import sys
import time

import discopt.modeling as dm
from discopt._relax.problem_classifier import (
    _extract_linear_coefficients_sparse,
    _NotLinearError,
)

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"


def census(inst):
    path = f"{NL}/{inst}.nl"
    print(f"\n{'=' * 78}\nCENSUS: {inst}\n{'=' * 78}")
    t0 = time.time()
    model = dm.from_nl(path)
    print(f"from_nl in {time.time() - t0:.2f}s")
    n = sum(v.size for v in model._variables)
    ncons = len(model._constraints)
    nvars = len(model._variables)
    print(f"vars(blocks)={nvars} flat={n} cons={ncons}")

    n_eq = 0
    n_ineq = 0
    eq_1var = 0
    eq_2var = 0
    eq_3plus = 0
    eq_nonlinear = 0
    examined = 0
    # Track, for 2-var eqs, how often each variable appears (to detect whether
    # aggregate's "target appears in exactly one eq" precondition holds, i.e.
    # doubleton chains where a var appears in many rows).
    from collections import Counter

    var_eq_degree = Counter()  # flat var index -> # of 2-var-or-less eqs touching it

    t0 = time.time()
    for con in model._constraints:
        examined += 1
        sense = getattr(con, "sense", None)
        if sense == "==":
            n_eq += 1
        else:
            n_ineq += 1
            continue
        try:
            terms, _const = _extract_linear_coefficients_sparse(con.body, model, n)
        except _NotLinearError:
            eq_nonlinear += 1
            continue
        except Exception:
            eq_nonlinear += 1
            continue
        nz = [i for i, c in terms.items() if abs(c) > 1e-12]
        k = len(nz)
        if k <= 1:
            eq_1var += 1
            for i in nz:
                var_eq_degree[i] += 1
        elif k == 2:
            eq_2var += 1
            for i in nz:
                var_eq_degree[i] += 1
        else:
            eq_3plus += 1
    dt = time.time() - t0
    print(f"scanned {examined} constraints in {dt:.2f}s")
    assert examined > 0, "CENSUS EXAMINED ZERO CONSTRAINTS"
    print(f"  equalities:          {n_eq}")
    print(f"  inequalities:        {n_ineq}")
    print(f"  eq 1-var (singleton, eliminate.rs): {eq_1var}")
    print(f"  eq 2-var (doubleton, aggregate.rs): {eq_2var}")
    print(f"  eq >=3-var linear (needs Gaussian): {eq_3plus}")
    print(f"  eq nonlinear:                       {eq_nonlinear}")
    # How many 2-var/1-var eqs have BOTH endpoints appearing in exactly one such eq?
    # (aggregate precondition: target var appears in exactly one expression total).
    solo = sum(1 for _i, d in var_eq_degree.items() if d == 1)
    multi = sum(1 for _i, d in var_eq_degree.items() if d > 1)
    print(f"  vars touching exactly one (1-or-2)-var eq: {solo}")
    print(f"  vars touching >1 such eq (chains):         {multi}")
    reducible_est = eq_1var + eq_2var
    print(
        f"  ESTIMATE reducible-by-existing-transforms (1var+2var eq): {reducible_est} "
        f"= {100.0 * reducible_est / max(n_eq, 1):.1f}% of equalities"
    )


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else ["gastrans040", "gastrans582_cold13"]
    for inst in which:
        census(inst)
