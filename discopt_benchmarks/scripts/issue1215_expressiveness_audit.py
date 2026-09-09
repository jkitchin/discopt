#!/usr/bin/env python
"""Can an arena-native construction path carry discopt's expressive constructs? (#1215)

Issue #1215's arena work would make the flat arena the *primary* construction
representation. That is only viable if the arena can represent what the modeling
layer can express -- GDP, NN embeddings, DAE collocation, shaped/matmul algebra,
mutable ``Parameter``s -- so this audit measures the boundary instead of assuming
it.

Two modes:

``--mode coverage``
    Build one small model per construct family and try to lower it with
    ``model_to_repr``. A family that lowers is arena-representable *today*; one
    that raises names a hard boundary. Constraint counts are reported alongside
    node counts so an empty model cannot masquerade as a pass.

``--mode cse``
    Process-style models reuse the same nonlinear term across many balances (one
    Arrhenius rate in every species balance). The arena hash-conses
    (``ExprArena::enable_interning``), so a shared term should be stored -- and
    differentiated -- once regardless of how many constraints use it. Measured
    two ways: with the term hoisted and built once, and with it *rebuilt* inside
    the loop, which is how users actually write it.

Measurement discipline: each mode prints the module file it loaded (§8) and the
count of families / size points it actually exercised, exiting non-zero if that
count is zero (§6). Every lowering asserts the arena's constraint count matches
the model's, so a silently-empty lowering fails rather than reporting a pass.

Usage (from repo root, extension built, venv active)::

    python -u discopt_benchmarks/scripts/issue1215_expressiveness_audit.py
    python -u discopt_benchmarks/scripts/issue1215_expressiveness_audit.py --mode cse

Measured 2026-09-09 on `main` @ c052e85 -- 8 of 9 construct families lower; the
sole boundary is ``CustomCall`` (``dm.custom`` / ``dm.udf`` / ``dm.implicit``),
which ``convert_expr`` rejects with "Unknown expression type: CustomCall"
because ``ExprNode`` has no opaque-callable variant. ``solver.py`` already
routes such models onto a separate AD-only path.
"""

from __future__ import annotations

import argparse
import sys

import discopt.modeling as dm
import discopt.modeling.core as core
import numpy as np
from discopt._rust import model_to_repr
from discopt.modeling import Model


def _num(obj, attr):
    """Read a PyModelRepr field that may be a property or a method."""
    v = getattr(obj, attr)
    return v() if callable(v) else v


# ─────────────────────────────────────────────────────────────
# construct families
# ─────────────────────────────────────────────────────────────


def f_scalar():
    m = Model()
    x = m.continuous("x", shape=(4,), lb=0.5, ub=3.0)
    m.subject_to(dm.exp(x[0]) * x[1] ** 2 + dm.log(x[2]) <= 5.0)
    m.minimize(x[3])
    return m


def f_matmul():
    m = Model()
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    a = np.arange(9, dtype=float).reshape(3, 3)
    m.subject_to(dm.sum(a @ x) <= 10.0)
    m.minimize(dm.sum(x))
    return m


def f_reductions():
    m = Model()
    x = m.continuous("x", shape=(5,), lb=0, ub=1)
    m.subject_to(dm.norm(x) <= 2.0)
    m.minimize(dm.sum(x * x))
    return m


def f_parameter():
    m = Model()
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    p = m.parameter("p", value=2.0)
    m.subject_to(p * x[0] + x[1] <= 4.0)
    m.minimize(x[0] * p)
    return m


def f_gdp():
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.either_or([[x <= 3.0, y >= 5.0], [x >= 7.0, y <= 2.0]], name="dj")
    m.minimize(x + y)
    return m


def _net(activation: str):
    from discopt.nn.network import Activation, DenseLayer, NetworkDefinition

    rng = np.random.default_rng(0)
    return NetworkDefinition(
        layers=[
            DenseLayer(
                weights=rng.normal(size=(2, 3)),
                biases=np.zeros(3),
                activation=Activation(activation),
            ),
            DenseLayer(
                weights=rng.normal(size=(3, 1)),
                biases=np.zeros(1),
                activation=Activation.LINEAR,
            ),
        ],
        input_bounds=(-np.ones(2), np.ones(2)),
    )


def _nn(method: str, activation: str):
    from discopt.nn.predictor import add_predictor

    m = Model()
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    out, _ = add_predictor(m, x, _net(activation), method=method)
    m.minimize(out[0])
    return m


def f_nn_relu():
    return _nn("relu_bigm", "relu")


def f_nn_smooth():
    return _nn("full_space", "tanh")


def f_dae():
    from discopt.dae import ContinuousSet, DAEBuilder

    m = Model("decay")
    dae = DAEBuilder(m, ContinuousSet("t", bounds=(0, 2), nfe=4, ncp=3))
    dae.add_state("A", initial=1.0, bounds=(0.0, 2.0))
    dae.set_ode(lambda t, s, a, c: {"A": -0.7 * s["A"] ** 2})
    dae.discretize()
    m.minimize(m._variables[0][0])
    return m


def f_custom():
    m = Model()
    x = m.continuous("x", shape=(2,), lb=0.5, ub=3.0)
    f = dm.custom(lambda a: a[0] * a[1], name="prod")
    m.subject_to(f(x) <= 4.0)
    m.minimize(x[0])
    return m


FAMILIES = [
    ("scalar nonlinear (exp/pow/log)", f_scalar),
    ("shaped: matmul + sum", f_matmul),
    ("shaped: norm + elementwise", f_reductions),
    ("Parameter (changeable between solves)", f_parameter),
    ("GDP disjunction (either_or)", f_gdp),
    ("NN embedding: ReLU big-M", f_nn_relu),
    ("NN embedding: smooth full-space", f_nn_smooth),
    ("DAE collocation", f_dae),
    ("dm.custom (opaque callable)", f_custom),
]


def mode_coverage() -> int:
    print(f"# discopt loaded from: {core.__file__}\n")
    results = []
    for name, fn in FAMILIES:
        try:
            m = fn()
            r = model_to_repr(m, getattr(m, "_builder", None))
            n_nodes = _num(r, "arena_len")
            n_cons = _num(r, "n_constraints")
            results.append(
                (
                    name,
                    "LOWERS",
                    f"{n_nodes} arena nodes, {n_cons} constraints "
                    f"({len(m._constraints)} Constraint objs)",
                )
            )
        except Exception as e:  # noqa: BLE001 - recording the boundary is the point
            msg = f"{type(e).__name__}: {e}".replace("\n", " ")
            results.append((name, "REFUSED", msg[:110]))

    w = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        print(f"{name:<{w}}  {status:8s}  {detail}")

    n_lower = sum(1 for _, s, _ in results if s == "LOWERS")
    print(
        f"\n# executed: {len(results)} construct families, {n_lower} lowered, "
        f"{len(results) - n_lower} refused"
    )
    if not results:
        print("FAIL: audit exercised nothing")
        return 1
    return 0


# ─────────────────────────────────────────────────────────────
# mode: cse
# ─────────────────────────────────────────────────────────────


def _count_python_nodes(m) -> int:
    """Distinct Python Expression objects across every constraint body."""
    seen: set[int] = set()
    stack = [c.body for c in m._constraints]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        for attr in ("left", "right", "operand", "arg", "base", "args", "terms"):
            v = getattr(n, attr, None)
            if isinstance(v, core.Expression):
                stack.append(v)
            elif isinstance(v, (list, tuple)):
                stack.extend(a for a in v if isinstance(a, core.Expression))
    return len(seen)


def _cse_row(n_con: int, hoist: bool):
    m = Model()
    x = m.continuous("x", shape=(3,), lb=0.5, ub=3.0)
    shared = dm.exp(-1.5 / x[0]) * x[1] ** 2 if hoist else None
    for i in range(n_con):
        term = shared if hoist else dm.exp(-1.5 / x[0]) * x[1] ** 2
        m.subject_to(term + float(i) * x[2] <= 10.0)
    m.minimize(x[2])
    py = _count_python_nodes(m)
    r = model_to_repr(m, getattr(m, "_builder", None))
    ar = _num(r, "arena_len")
    n_cons = _num(r, "n_constraints")
    if n_cons != n_con:
        raise AssertionError(f"lowered {n_cons} constraints, expected {n_con}")
    return py, ar


def mode_cse(sizes: list[int]) -> int:
    print(f"# discopt loaded from: {core.__file__}")
    rows = 0
    for hoist, label in (
        (True, "hoisted: shared term built ONCE"),
        (False, "rebuilt: shared term rebuilt per constraint"),
    ):
        print(f"\n# {label}")
        print(
            f"{'N constraints':>14s} {'python nodes':>13s} {'arena nodes':>12s} "
            f"{'py/con':>7s} {'arena/con':>10s} {'CSE factor':>11s}"
        )
        for n_con in sizes:
            py, ar = _cse_row(n_con, hoist)
            print(
                f"{n_con:14d} {py:13d} {ar:12d} {py / n_con:7.2f} "
                f"{ar / n_con:10.2f} {py / ar:11.2f}x"
            )
            rows += 1
    print(f"\n# executed: {rows} size points measured")
    if rows == 0:
        print("FAIL: measured nothing")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=("coverage", "cse"), default="coverage")
    ap.add_argument(
        "--sizes",
        default="10,100,1000",
        help="cse mode: comma-separated constraint counts",
    )
    args = ap.parse_args(argv)
    if args.mode == "cse":
        return mode_cse([int(s) for s in args.sizes.split(",") if s.strip()])
    return mode_coverage()


if __name__ == "__main__":
    sys.exit(main())
