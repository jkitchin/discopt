#!/usr/bin/env python3
"""Repro: ``dm.sum(C * v, axis=1)`` yields a FALSE optimality certificate.

Found while re-running ``docs/notebooks/nn_embedding.ipynb`` for issue #1362.
The notebook's committed output had the correct answer (-0.579405); the current
tree returns -0.230589 and calls it ``optimal`` with ``gap_certified=True``.

Mechanism
---------
``sum(Constant((1, n)) * v[n], axis=1)`` -- a SINGLE-ROW row-reduction, the form
``discopt.ml``'s ``full_space`` / ``relu_bigm`` formulations emit for every
affine layer whose output width is 1 -- is not recognised as affine by the
relaxation compiler. It
becomes an ``opaque`` atom whose interval enclosure is *array*-shaped, so

    uniform_relax.py:  lo = float(np.asarray(enc.lo))

raises ``TypeError: only 0-dimensional arrays can be converted to Python
scalars``. That is caught at ``mccormick_lp.py`` and turned into
``MccormickLPResult(status="error")``, logged at DEBUG only.

With no LP bound the node keeps a *finite* non-rigorous lower bound, the
``_nonrigorous_sentinel_fathom`` guard only fires on the sentinel, and the tree
closes at the root: ``status="optimal"``, ``bound == incumbent``,
``gap_certified=True`` -- on a point that is not the optimum.

Run: python scratchpad/nbreview/repro_axis_sum_false_optimum.py
Exits non-zero while the defect is present.
"""

from __future__ import annotations

import numpy as np

import discopt.modeling as dm

CHECKS = 0
FAILURES: list[str] = []


def check(name: str, got: float, want: float, *, tol: float = 1e-4) -> None:
    global CHECKS
    CHECKS += 1
    ok = abs(got - want) <= tol
    print(f"  [{'ok ' if ok else 'BAD'}] {name}: got {got:.6f}, true optimum {want:.6f}")
    if not ok:
        FAILURES.append(f"{name}: {got:.6f} != {want:.6f}")


def case_minimal() -> None:
    """Five variables, one nonconvex row, one affine row. True optimum -8."""
    print("\n1. Minimal repro --- min -z0-z1 s.t. z = x*x, x in [-2,2]^2 (true: -8)")
    C = np.array([[-1.0, -1.0]])
    b = np.array([0.0])

    def build(form: str) -> dm.Model:
        m = dm.Model("m")
        x = m.continuous("x", shape=(2,), lb=-2, ub=2)
        z = m.continuous("z", shape=(2,), lb=0, ub=4)
        out = m.continuous("out", shape=(1,), lb=-10, ub=10)
        m.minimize(out[0])
        m.subject_to(z == x * x)
        if form == "axis_sum":
            m.subject_to(out == dm.sum(C * z, axis=1) + b)
        else:
            m.subject_to(out[0] == -z[0] - z[1])
        return m

    for form in ("scalar", "axis_sum"):
        r = build(form).solve(time_limit=60)
        print(
            f"     {form:9s} status={r.status} obj={float(r.objective):.6f} "
            f"bound={float(r.bound):.6f} nodes={r.node_count} "
            f"gap_certified={r.gap_certified}"
        )
        check(f"minimal/{form}", float(r.objective), -8.0)


def case_relaxation_errors() -> None:
    """The node relaxation itself errors -- the defect one layer down."""
    print("\n2. The node LP relaxation returns status='error' (not a bound)")
    from discopt._relax.mccormick_lp import MccormickLPRelaxer

    global CHECKS
    C = np.array([[-1.0, -1.0]])
    m = dm.Model("m")
    x = m.continuous("x", shape=(2,), lb=-2, ub=2)
    z = m.continuous("z", shape=(2,), lb=0, ub=4)
    out = m.continuous("out", shape=(1,), lb=-10, ub=10)
    m.minimize(out[0])
    m.subject_to(z == x * x)
    m.subject_to(out == dm.sum(C * z, axis=1))

    rel = MccormickLPRelaxer(m)
    lb = np.array([-2.0, -2.0, 0.0, 0.0, -10.0])
    ub = np.array([2.0, 2.0, 4.0, 4.0, 10.0])
    res = rel.solve_at_node(lb, ub)
    CHECKS += 1
    print(f"     root relaxation status={res.status!r} lower_bound={res.lower_bound!r}")
    if res.status == "error":
        FAILURES.append("root McCormick LP relaxation returns status='error'")
        print("     [BAD] the root relaxation cannot be built for an affine axis-sum row")
    else:
        print("     [ok ] the root relaxation builds")


def case_row_count() -> None:
    """It is the ONE-ROW reduction that breaks: 2+ rows relax soundly."""
    print("\n3. Only a single-row reduction is affected (true optimum -8 throughout)")
    global CHECKS
    for nrows in (1, 2, 3):
        C = -np.ones((nrows, 2))
        m = dm.Model("m")
        x = m.continuous("x", shape=(2,), lb=-2, ub=2)
        z = m.continuous("z", shape=(2,), lb=0, ub=4)
        out = m.continuous("out", shape=(nrows,), lb=-10, ub=10)
        m.minimize(out[0])
        m.subject_to(z == x * x)
        m.subject_to(out == dm.sum(C * z, axis=1))
        r = m.solve(time_limit=60)
        CHECKS += 1
        bad = float(r.bound) > -8.0 + 1e-6
        print(
            f"     nrows={nrows}: status={r.status} obj={float(r.objective):.4f} "
            f"bound={float(r.bound):.4f} nodes={r.node_count}"
        )
        if bad:
            FAILURES.append(f"nrows={nrows}: dual bound {float(r.bound):.4f} above -8.0")


def case_nn_embedding() -> None:
    """The real case: docs/notebooks/nn_embedding.ipynb §3 (tanh full_space)."""
    print("\n4. docs/notebooks/nn_embedding.ipynb §3 --- tanh network, full_space")
    from discopt.ml import Activation, DenseLayer, NetworkDefinition, NNFormulation

    np.random.seed(0)
    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.random.randn(4) * 0.1
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.random.randn(1) * 0.1
    net = NetworkDefinition(
        layers=[
            DenseLayer(W1, b1, Activation.TANH),
            DenseLayer(W2, b2, Activation.LINEAR),
        ],
        input_bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
    )

    # Independent reference: a dense grid over the 2-D input box.
    g = np.linspace(-2, 2, 801)
    a, c = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([a.ravel(), c.ravel()], axis=1)
    reference = float((np.tanh(pts @ W1 + b1) @ W2 + b2).min())

    m = dm.Model("smooth_nn")
    nn = NNFormulation(m, net, strategy="full_space")
    nn.formulate()
    m.minimize(nn.outputs[0])
    r = m.solve(time_limit=60)
    print(
        f"     status={r.status} obj={float(r.objective):.6f} bound={float(r.bound):.6f} "
        f"nodes={r.node_count} gap_certified={r.gap_certified}"
    )
    print(f"     grid reference over the input box: {reference:.6f}")
    check("nn_embedding/full_space tanh", float(r.objective), reference, tol=1e-3)

    # The bound is the soundness-critical half: a *dual* bound above the true
    # optimum has cut the optimum out of the box.
    global CHECKS
    CHECKS += 1
    if float(r.bound) > reference + 1e-6:
        FAILURES.append(
            f"dual bound {float(r.bound):.6f} is ABOVE the true optimum {reference:.6f}"
        )
        print("     [BAD] the dual bound is above the true optimum (invalid bound)")
    else:
        print("     [ok ] the dual bound is a valid lower bound")


def main() -> int:
    case_minimal()
    case_relaxation_errors()
    case_row_count()
    case_nn_embedding()
    print(f"\n[executed {CHECKS} assertion(s)]")
    if CHECKS == 0:
        print("PROBE DID NOT FIRE: zero assertions executed")
        return 2
    if FAILURES:
        print(f"DEFECT PRESENT: {len(FAILURES)} failure(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("No failures: the defect appears fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
