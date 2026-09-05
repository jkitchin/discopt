"""#1182 entry experiment (CLAUDE.md section 4): does the Theorem-1 simplex/CNF
lowering beat the exact GDP/SOS1 path on real corpus instances?

Hypothesis under test (from RFC #1123 / arXiv:2601.03906v1): replacing each
disjunction by its exact continuous simplex lowering removes the binaries and so
gives a *faster certified* solve than discopt's big-M / hull lowering.

Kill criterion, fixed before running: on the in-repo native GDP corpus
(``benchmarks.gdplib_native``, whose optima are SCIP/BARON-certified), the
simplex arm must reach the same certified optimum as big-M/hull on at least one
instance with strictly less wall time or fewer nodes. If it certifies *nothing*
that big-M/hull certifies, or returns an objective better than the certified
optimum (a false primal), the hypothesis is falsified for this class and the
lowering must not ship as a solve path.

Every arm reports the four size quantities separately (requirement 4 of #1182)
and validates the returned point against the ORIGINAL predicates (requirement 1),
never against the weighted rows.

Per CLAUDE.md section 6 this script prints an executed-comparison count and exits
non-zero if it is zero.
"""

from __future__ import annotations

import argparse
import sys
import time

from benchmarks.gdplib_native import NATIVE_BUILDERS
from benchmarks.gdplib_runner import reference_optima
from simplex_proto import reformulate_simplex
from source_check import predicate_report


def _count_jacobian_nonzeros(model) -> int:
    """Structural nonzeros: sum over rows of the number of distinct variables."""
    from discopt.modeling.core import Constraint
    from source_check import variables_in

    nnz = 0
    for c in model._constraints:
        if isinstance(c, Constraint):
            nnz += len(variables_in(c.body))
    return nnz


def _n_binaries(model) -> int:
    from discopt.modeling.core import VarType

    n = 0
    for v in model._variables:
        if v.var_type is VarType.BINARY:
            n += int(_size(v))
    return n


def _size(v) -> int:
    n = 1
    for d in v.shape:
        n *= d
    return n


def _solve(model, *, time_limit, **kw):
    t0 = time.perf_counter()
    res = model.solve(time_limit=time_limit, **kw)
    return res, time.perf_counter() - t0


def run(names, time_limit: float) -> int:
    optima = reference_optima()
    comparisons = 0
    rows = []
    for name in names:
        source = NATIVE_BUILDERS[name]()
        ref = optima[name]

        for arm in ("big-m", "hull", "simplex"):
            model = NATIVE_BUILDERS[name]()
            if arm == "simplex":
                lowered, records, counts = reformulate_simplex(model)
                counts.jacobian_nonzeros = _count_jacobian_nonzeros(lowered)
                target = lowered
            else:
                from discopt._relax.gdp_reformulate import reformulate_gdp

                lowered = reformulate_gdp(model, method=arm)
                from simplex_proto import SizeCounts

                counts = SizeCounts(
                    disjunctions=sum(
                        1
                        for c in model._constraints
                        if type(c).__name__ == "_DisjunctiveConstraint"
                    ),
                    cnf_clauses=0,
                    literal_occurrences=0,
                    aux_variables=_n_binaries(lowered),
                    rows=len(lowered._constraints),
                )
                counts.jacobian_nonzeros = _count_jacobian_nonzeros(lowered)
                target = lowered

            print(f"[{name}/{arm}] solving ({len(target._constraints)} rows, "
                  f"{sum(_size(v) for v in target._variables)} vars)...", flush=True)
            try:
                if arm == "simplex":
                    res, wall = _solve(target, time_limit=time_limit)
                else:
                    # solve the ORIGINAL model so discopt's own gdp pass runs
                    res, wall = _solve(
                        NATIVE_BUILDERS[name](), time_limit=time_limit, gdp_method=arm
                    )
            except Exception as exc:  # a crash is a result, not something to hide
                print(f"[{name}/{arm}] RAISED {type(exc).__name__}: {exc}", flush=True)
                rows.append((name, arm, "raised", None, None, None, None, counts))
                comparisons += 1
                continue

            status = str(getattr(res, "status", "?"))
            obj = getattr(res, "objective", None)
            bound = getattr(res, "bound", None)
            nodes = getattr(res, "node_count", None)
            gap_certified = getattr(res, "gap_certified", None)

            # Requirement 1: validate against the SOURCE predicates.
            point = getattr(res, "x", None)
            rep = predicate_report(source, point) if point else None
            comparisons += 0 if rep is None else rep.comparisons

            rows.append((name, arm, status, obj, bound, nodes, wall, counts, rep,
                         gap_certified))
            print(
                f"[{name}/{arm}] status={status} obj={obj} bound={bound} "
                f"nodes={nodes} wall={wall:.2f}s certified={gap_certified} "
                f"ref={ref} src_disj_violation="
                f"{None if rep is None else rep.max_disjunction_violation}",
                flush=True,
            )
            comparisons += 1

    print("\n=== summary ===")
    hdr = ("instance", "arm", "status", "objective", "bound", "nodes", "wall",
           "clauses", "lits", "aux", "rows", "nnz", "src_viol")
    print(" | ".join(f"{h:>12}" for h in hdr))
    for r in rows:
        name, arm, status, obj, bound, nodes, wall, counts = r[:8]
        rep = r[8] if len(r) > 8 else None
        cells = [
            name, arm, status,
            "-" if obj is None else f"{obj:.6g}",
            "-" if bound is None else f"{bound:.6g}",
            "-" if nodes is None else str(nodes),
            "-" if wall is None else f"{wall:.2f}",
            str(counts.cnf_clauses), str(counts.literal_occurrences),
            str(counts.aux_variables), str(counts.rows),
            str(counts.jacobian_nonzeros),
            "-" if rep is None else f"{rep.max_disjunction_violation:.3g}",
        ]
        print(" | ".join(f"{c:>12}" for c in cells))

    print(f"\nexecuted comparisons: {comparisons}")
    if comparisons == 0:
        print("FAIL: the probe measured nothing (CLAUDE.md section 6)")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=",".join(NATIVE_BUILDERS))
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args()
    sys.exit(run([n for n in args.instances.split(",") if n], args.time_limit))
