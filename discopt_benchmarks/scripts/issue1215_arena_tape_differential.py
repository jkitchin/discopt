#!/usr/bin/env python
"""Is the arena-lowered tape BIT-IDENTICAL to the Python-DAG tape? (#1215)

``_arena_tape`` replaces the Python expression-DAG walk in the tape build with a
forward scan of the Rust arena. That is a *bound-neutral* change in CLAUDE.md §5
terms, so the bar is not "close enough" -- it is that the objective value, the
objective gradient, the constraint values and the full Jacobian come out
**exactly equal** at every probe point. Any drift, in either direction, means the
two paths lowered different mathematics and the change is wrong.

Two arms per model:
  * ``python``: ``compile_to_nl_expr`` / ``compile_to_nl_array`` (the current path)
  * ``arena``:  ``_arena_tape.try_build_arena_tape``

Both are handed to ``pounce.build_nl_problem`` so the comparison is of the tapes,
not of the surrounding evaluator.

Discipline: prints the loaded module and asserts the marker unique to this change
(§8); counts every comparison actually executed and exits non-zero if that count
is zero (§6); no bare ``except`` -- a model that the arena path *refuses* is
reported as a refusal (which is a legitimate outcome), while a model that raises
is a failure (§7).

Usage::

    python -u discopt_benchmarks/scripts/issue1215_arena_tape_differential.py
    python -u discopt_benchmarks/scripts/issue1215_arena_tape_differential.py --corpus
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import discopt.modeling as dm
import discopt.modeling.core as core
import numpy as np
from discopt import _arena_tape
from discopt._arena_tape import try_build_arena_tape
from discopt._nl_expr_compiler import (
    UnsupportedForTape,
    compile_to_nl_array,
    compile_to_nl_expr,
)

# ── models: one per shape the tape actually meets in practice ──────────────


def m_flowsheet(n=200):
    m = dm.Model(f"flowsheet{n}")
    outs = []
    for u in range(n):
        f = m.continuous(f"f{u}", shape=(3,), lb=0.1, ub=10.0)
        t = m.continuous(f"T{u}", lb=300.0, ub=600.0)
        m.subject_to(f[0] + f[1] - f[2] == 0.0, name=f"mb{u}")
        m.subject_to(f[2] * dm.exp(-2000.0 / t) <= 5.0, name=f"rate{u}")
        m.subject_to(dm.log(f[0] + 1.0) + 0.01 * t <= 12.0, name=f"nrg{u}")
        outs.append(f[2])
    m.minimize(dm.sum(outs))
    return m


def m_builtin_sum_chain(n=6000):
    """The chain-flattening path: builtin ``sum()`` over n terms."""
    m = dm.Model(f"chain{n}")
    x = m.continuous("x", shape=(5,), lb=0.3, ub=3.0)
    m.subject_to(sum(x[i % 5] * float(i + 1) for i in range(n)) <= 1e9, name="c")
    m.minimize(sum(dm.exp(x[i % 5] * 0.01) for i in range(200)))
    return m


def m_short_chains():
    """Chains BELOW the flattening threshold keep their binary nesting.

    Includes a right-nested ``a + (b + c)``, where a wrongly-applied flattening
    would reassociate the sum and show up as a last-bit difference.
    """
    m = dm.Model("short")
    x = m.continuous("x", shape=(4,), lb=1e-8, ub=1e8)
    a, b, c, d = (x[i] for i in range(4))
    m.subject_to(a + (b + c) <= 1e9, name="right_nested")
    m.subject_to((a - b) + c <= 1e9, name="left_nested")
    m.subject_to(a + (b - (c + d)) <= 1e9, name="mixed")
    m.minimize(a * 1e-8 + b * 1e8 - c)
    return m


def m_least_squares(n=400):
    """Parameter-estimation shape: residual squares over mutable Parameters."""
    m = dm.Model(f"ls{n}")
    th = m.continuous("theta", shape=(3,), lb=0.01, ub=5.0)
    ts = np.linspace(0.1, 10.0, n)
    ys = 2.0 * np.exp(-0.4 * ts) + 0.3
    terms = [
        (th[0] * dm.exp(-th[1] * float(t)) + th[2] - float(y)) ** 2
        for t, y in zip(ts, ys, strict=True)
    ]
    m.minimize(dm.sum(terms))
    m.subject_to(th[0] + th[2] <= 20.0, name="reg")
    return m


def m_funcs():
    m = dm.Model("funcs")
    x = m.continuous("x", shape=(6,), lb=0.2, ub=0.9)
    m.subject_to(dm.sqrt(x[0]) + dm.log(x[1]) + dm.sin(x[2]) <= 5.0, name="f0")
    m.subject_to(dm.cos(x[3]) + dm.tan(x[4]) + dm.atan(x[5]) <= 5.0, name="f1")
    m.subject_to(dm.exp(x[0]) / (x[1] + 1.0) - x[2] ** 1.7 <= 9.0, name="f2")
    m.subject_to(abs(x[3] - x[4]) <= 1.0, name="f3")
    m.minimize(-dm.sum([x[i] for i in range(6)]))
    return m


def m_shared_subexpr(n=150):
    """Hash-consing + a chain interior that is ALSO used by a non-additive parent."""
    m = dm.Model(f"shared{n}")
    x = m.continuous("x", shape=(4,), lb=0.5, ub=2.0)
    common = x[0] + x[1] + x[2] + x[3]
    for i in range(n):
        m.subject_to(common * dm.exp(x[i % 4] * 0.1) <= 100.0, name=f"c{i}")
    m.minimize(common + common * common)
    return m


def m_integers():
    m = dm.Model("mixed")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=5.0)
    y = m.binary("y", shape=(3,))
    m.subject_to(x[0] * y[0] + x[1] * y[1] + x[2] * y[2] <= 4.0, name="bilin")
    m.minimize(-x[0] - 2.0 * y[1])
    return m


def m_array_body():
    """An array-valued constraint body -- the arena path must REFUSE this."""
    m = dm.Model("arraybody")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=2.0)
    m.subject_to(x <= 1.5, name="arr")
    m.minimize(-x[0])
    return m


def m_custom():
    """``dm.custom`` is opaque to the arena -- must REFUSE, never guess."""
    m = dm.Model("custom")
    x = m.continuous("x", shape=(2,), lb=0.1, ub=2.0)
    f = dm.custom(lambda v: v * 2.0, name="double")
    m.subject_to(f(x[0]) + x[1] <= 5.0, name="c")
    m.minimize(-x[0])
    return m


MODELS = [
    m_flowsheet,
    m_builtin_sum_chain,
    m_short_chains,
    m_least_squares,
    m_funcs,
    m_shared_subexpr,
    m_integers,
    m_array_body,
    m_custom,
]


def python_tape(model):
    obj = compile_to_nl_expr(model._objective.expression, model)
    cons = []
    for c in model._constraints:
        cons.extend(compile_to_nl_array(c.body, model).reshape(-1).tolist())
    return obj, cons


def bounds(model):
    lo, hi = [], []
    for v in model._variables:
        lo.extend(
            np.asarray(v.lb, dtype=float).reshape(-1).tolist()
            if np.ndim(v.lb)
            else [float(v.lb)] * int(v.size)
        )
        hi.extend(
            np.asarray(v.ub, dtype=float).reshape(-1).tolist()
            if np.ndim(v.ub)
            else [float(v.ub)] * int(v.size)
        )
    return np.asarray(lo), np.asarray(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=4)
    args = ap.parse_args()

    print(f"# discopt loaded from: {core.__file__}")
    print(f"# arena module:        {_arena_tape.__file__}")
    marker = "try_build_arena_tape"
    assert marker in Path(_arena_tape.__file__).read_text(), "marker absent: wrong build"
    print(f"# marker '{marker}' present")
    print(f"# load: {os.getloadavg()[0]:.2f}")

    import pounce

    E = pounce.NlExpr  # noqa: N806 -- the tape node class, named `E` throughout this layer
    rng = np.random.default_rng(11)
    comparisons = 0
    refusals = []
    both_refuse = []
    mismatches = []

    print(
        f"\n{'model':<22s} {'rows':>6s} {'arena':>9s} {'max|dobj|':>12s} "
        f"{'max|dgrad|':>12s} {'max|dcon|':>12s} {'max|djac|':>12s}"
    )

    for fn in MODELS:
        m = fn()
        name = fn.__name__[2:]
        try:
            py_obj, py_cons = python_tape(m)
        except UnsupportedForTape as exc:
            # The CURRENT path already refuses this model, so there is no
            # baseline to compare against. The only thing that can be wrong here
            # is the arena path accepting what the Python path rejects -- which
            # would mean it invented mathematics the tape has no equivalent for.
            if try_build_arena_tape(m, E) is not None:
                mismatches.append(f"{name}: arena BUILT a tape the Python path refuses ({exc})")
            both_refuse.append(name)
            print(f"{name:<22s} {'-':>6s} {'both refuse':>11s}   ({exc})")
            continue
        built = try_build_arena_tape(m, E)
        if built is None:
            refusals.append(name)
            print(
                f"{name:<22s} {len(py_cons):6d} {'REFUSED':>9s} "
                f"{'-':>12s} {'-':>12s} {'-':>12s} {'-':>12s}"
            )
            continue
        ar_obj, ar_cons = built
        if len(ar_cons) != len(py_cons):
            mismatches.append(f"{name}: row count {len(py_cons)} vs {len(ar_cons)}")
            print(f"{name:<22s} {len(py_cons):6d} {'ROWS!=':>9s}")
            continue

        lo, hi = bounds(m)
        n_vars = lo.size
        p_py = pounce.build_nl_problem(
            n_vars, py_obj, constraints=py_cons, x_l=list(lo), x_u=list(hi)
        )
        p_ar = pounce.build_nl_problem(
            n_vars, ar_obj, constraints=ar_cons, x_l=list(lo), x_u=list(hi)
        )
        do = dg = dc = dj = 0.0
        for _ in range(args.points):
            x = lo + rng.uniform(0.05, 0.95, size=n_vars) * (hi - lo)
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
        bad = (
            ""
            if (do == 0.0 and dg == 0.0 and dc == 0.0 and dj == 0.0)
            else "  <<< NOT BIT-IDENTICAL"
        )
        if bad:
            mismatches.append(f"{name}: dobj={do} dgrad={dg} dcon={dc} djac={dj}")
        print(
            f"{name:<22s} {len(py_cons):6d} {'built':>9s} {do:12.3e} {dg:12.3e} "
            f"{dc:12.3e} {dj:12.3e}{bad}"
        )

    print(
        f"\n# executed: {comparisons} value/gradient comparisons over "
        f"{len(MODELS) - len(refusals)} arena-built models"
    )
    print(f"# refused (fell back to the Python path): {', '.join(refusals) or 'none'}")
    print(f"# refused by BOTH paths (no tape exists): {', '.join(both_refuse) or 'none'}")
    if comparisons == 0:
        print("FAIL: the probe compared nothing -- every model refused")
        return 1
    if mismatches:
        print("FAIL: arena tape is not bit-identical to the Python tape:")
        for msg in mismatches:
            print(f"  {msg}")
        return 1
    print("PASS: every arena-built tape is bit-identical to the Python-built tape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
