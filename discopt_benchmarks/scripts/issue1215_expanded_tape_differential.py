#!/usr/bin/env python
"""Is the Rust-expanded tape BIT-IDENTICAL to the Python DAG walk? (#1215)

`discopt_core::expand` fans an array-valued body out to N scalar rows in Rust,
replacing the Python walk that `_nl_expr_compiler` does per constraint. That is a
CLAUDE.md §5 bound-neutral change, so the bar is exact equality -- objective,
gradient, every constraint value and the full Jacobian -- not agreement within a
tolerance. A fan-out that produced the right VALUES in the wrong ORDER would also
pass a tolerance check and silently mis-attribute every dual, so row order is
compared element-wise rather than as a set.

Discipline: prints the loaded module and asserts a marker unique to this change
(§8); counts every comparison and exits non-zero if that count is zero (§6);
models the Rust path refuses are reported as refusals, which is a legitimate
outcome, while a model that RAISES is a failure (§7).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import discopt.modeling as dm
import discopt.modeling.core as core
import numpy as np
from discopt import _arena_tape
from discopt._arena_tape import try_build_expanded_tape
from discopt._nl_expr_compiler import compile_to_nl_array, compile_to_nl_expr


def m_elementwise(n=200):
    m = dm.Model("elementwise")
    x = m.continuous("x", shape=(n,), lb=0.1, ub=10.0)
    y = m.continuous("y", shape=(n,), lb=0.1, ub=10.0)
    for f in range(4):
        m.subject_to(dm.exp(x) + y <= np.full(n, 9.0 + f), name=f"c{f}")
    m.minimize(dm.sum(x))
    return m


def m_matmul():
    m = dm.Model("matmul")
    x = m.continuous("x", shape=(6,), lb=0.0, ub=5.0)
    a = np.random.default_rng(0).normal(size=(4, 6))
    m.subject_to(a @ x <= np.ones(4), name="Ax")
    m.subject_to(x * x <= np.full(6, 20.0), name="sq")
    m.minimize(-dm.sum(x))
    return m


def m_axis_sum():
    m = dm.Model("axis_sum")
    xs = m.continuous("X", shape=(4, 3), lb=0.0, ub=2.0)
    m.subject_to(dm.sum(xs, axis=1) <= np.array([1.0, 2.0, 3.0, 4.0]), name="rows")
    m.subject_to(dm.sum(xs, axis=0) <= np.array([2.0, 2.0, 2.0]), name="cols")
    m.minimize(-dm.sum(xs))
    return m


def m_broadcast():
    """A scalar broadcast against a vector, and a (n,1) against a (n,m)."""
    m = dm.Model("broadcast")
    x = m.continuous("x", shape=(3, 4), lb=0.1, ub=2.0)
    col = m.continuous("col", shape=(3, 1), lb=0.1, ub=2.0)
    m.subject_to(x * col <= np.full((3, 4), 5.0), name="scale")
    m.subject_to(x + 1.5 <= np.full((3, 4), 6.0), name="shift")
    m.minimize(-dm.sum(x))
    return m


def m_reductions():
    m = dm.Model("reductions")
    x = m.continuous("x", shape=(4,), lb=0.5, ub=2.0)
    m.subject_to(dm.norm(x) <= 3.0, name="ball")
    m.subject_to(dm.prod(x) <= 9.0, name="prod")
    m.minimize(-dm.sum(x))
    return m


def m_indexed_slice():
    m = dm.Model("slice")
    xs = m.continuous("X", shape=(4, 3), lb=0.1, ub=2.0)
    m.subject_to(xs[1] + xs[2] <= np.full(3, 3.0), name="rows12")
    m.subject_to(xs[:, 0] <= np.full(4, 1.5), name="col0")
    m.minimize(-dm.sum(xs))
    return m


def m_scalar_only():
    m = dm.Model("scalar")
    x = m.continuous("x", shape=(4,), lb=0.3, ub=3.0)
    m.subject_to(x[0] * dm.exp(x[1]) + x[2] <= 8.0, name="s0")
    m.subject_to(dm.log(x[3] + 1.0) - x[0] <= 2.0, name="s1")
    m.minimize(x[0] + 2.0 * x[1])
    return m


def m_long_chain(n=6000):
    """Chain flattening must survive the Rust fan-out."""
    m = dm.Model("chain")
    x = m.continuous("x", shape=(5,), lb=0.3, ub=3.0)
    m.subject_to(sum(x[i % 5] * float(i + 1) for i in range(n)) <= 1e9, name="c")
    m.minimize(sum(dm.exp(x[i % 5] * 0.01) for i in range(200)))
    return m


MODELS = [
    m_elementwise,
    m_matmul,
    m_axis_sum,
    m_broadcast,
    m_reductions,
    m_indexed_slice,
    m_scalar_only,
    m_long_chain,
]


def python_tape(model):
    obj = compile_to_nl_expr(model._objective.expression, model)
    cons, sizes = [], []
    for c in model._constraints:
        rows = compile_to_nl_array(c.body, model).reshape(-1)
        sizes.append(int(rows.size))
        cons.extend(rows.tolist())
    return obj, cons, sizes


def bounds(model):
    lo, hi = [], []
    for v in model._variables:
        n = int(v.size)
        for dest, bound in ((lo, v.lb), (hi, v.ub)):
            if np.ndim(bound):
                dest.extend(np.asarray(bound, float).reshape(-1).tolist())
            else:
                dest.extend([float(bound)] * n)
    return np.asarray(lo), np.asarray(hi)


def main() -> int:
    print(f"# discopt loaded from: {core.__file__}")
    marker = "try_build_expanded_tape"
    assert marker in Path(_arena_tape.__file__).read_text(), "marker absent: wrong build"
    print(f"# marker '{marker}' present")
    print(f"# load: {os.getloadavg()[0]:.2f}")

    import pounce

    E = pounce.NlExpr  # noqa: N806 -- the tape node class, named `E` in this layer
    rng = np.random.default_rng(17)
    comparisons = 0
    refusals: list[str] = []
    bad: list[str] = []

    print(
        f"\n{'model':<16s} {'rows':>6s} {'rust':>9s} {'max|dobj|':>11s} "
        f"{'max|dgrad|':>11s} {'max|dcon|':>11s} {'max|djac|':>11s}"
    )

    for fn in MODELS:
        name = fn.__name__[2:]
        m = fn()
        py_obj, py_cons, py_sizes = python_tape(m)
        built = try_build_expanded_tape(m, E)
        if built is None:
            refusals.append(name)
            print(f"{name:<16s} {len(py_cons):6d} {'REFUSED':>9s}")
            continue
        ar_obj, ar_cons, ar_sizes = built

        if ar_sizes != py_sizes:
            bad.append(f"{name}: rows/constraint {ar_sizes} vs {py_sizes}")
            print(f"{name:<16s} {len(py_cons):6d} {'ROWMAP!=':>9s}  {ar_sizes} vs {py_sizes}")
            continue

        lo, hi = bounds(m)
        n = lo.size
        p_py = pounce.build_nl_problem(n, py_obj, constraints=py_cons, x_l=list(lo), x_u=list(hi))
        p_ar = pounce.build_nl_problem(n, ar_obj, constraints=ar_cons, x_l=list(lo), x_u=list(hi))
        do = dg = dc = dj = 0.0
        for _ in range(4):
            x = lo + rng.uniform(0.05, 0.95, size=n) * (hi - lo)
            do = max(do, abs(float(p_py.objective(x)) - float(p_ar.objective(x))))
            dg = max(
                dg,
                float(np.max(np.abs(np.asarray(p_py.gradient(x)) - np.asarray(p_ar.gradient(x))))),
            )
            if py_cons:
                dc = max(
                    dc,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(p_py.constraints(x)) - np.asarray(p_ar.constraints(x))
                            )
                        )
                    ),
                )
                dj = max(
                    dj,
                    float(
                        np.max(np.abs(np.asarray(p_py.jacobian(x)) - np.asarray(p_ar.jacobian(x))))
                    ),
                )
            comparisons += 4
        flag = "" if max(do, dg, dc, dj) == 0.0 else "   <<< NOT BIT-IDENTICAL"
        if flag:
            bad.append(f"{name}: dobj={do} dgrad={dg} dcon={dc} djac={dj}")
        print(
            f"{name:<16s} {len(py_cons):6d} {'built':>9s} {do:11.3e} {dg:11.3e} "
            f"{dc:11.3e} {dj:11.3e}{flag}"
        )

    print(
        f"\n# executed: {comparisons} value/gradient comparisons over "
        f"{len(MODELS) - len(refusals)} Rust-expanded models"
    )
    print(f"# refused (fell back): {', '.join(refusals) or 'none'}")
    if comparisons == 0:
        print("FAIL: every model refused -- nothing was compared")
        return 1
    if bad:
        print("FAIL:")
        for msg in bad:
            print(f"  {msg}")
        return 1
    print("PASS: the Rust-expanded tape is bit-identical to the Python DAG walk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
