"""Issue #1364: a width-1 ``dm.sum(C * v, axis=1)`` row certified a false optimum.

Found re-running ``docs/notebooks/nn_embedding.ipynb`` for #1362. The notebook's
committed output carried the grid-verified optimum; the current tree returned a
worse point, called it ``optimal`` with ``gap_certified=True``, and reported a
dual bound EQUAL to that worse incumbent -- a bound above the true optimum on a
minimization.

Two independent defects, one soundness and one bound-quality, both covered here:

1. **Rust FBBT read a reduction as its own operand's hull.**
   ``presolve/fbbt.rs`` evaluated ``Sum { operand, .. }`` as
   ``node_bounds[operand]`` with a comment calling that "conservative". It is
   not: the node carries ONE interval holding the hull over an array's elements,
   so the enclosure of a sum of ``n`` of them is ``[n*lo, n*hi]``. Returning the
   hull is *narrow* -- an enclosure that does not contain the value -- and on
   ``out == sum(C * z, axis=1)`` it derived ``out >= -4`` where the true floor is
   ``-8``, an invalid tightening that cut the optimum out of the box. (The Python
   evaluator had the same defect and fixed it in #1158.)

2. **The relaxation compiler could not scalarize an axis reduction.**
   ``scalarize.scalar_elements`` returned ``None`` for ``sum(C * v, axis=1)``, so
   the row reached ``uniform_relax`` as one opaque array-valued atom whose
   interval is array-shaped; the McCormick LP build then raised ``TypeError: only
   0-dimensional arrays can be converted to Python scalars`` and every node LP
   came back ``status="error"`` -- no relaxation at all.

``discopt.ml`` emits exactly this row for every affine layer, so every embedded
network with a scalar output was exposed.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.scalarize import scalar_elements
from discopt.tightening import fbbt_box


def _axis_sum_model(coeffs: np.ndarray, *, z_lb: float, z_ub: float):
    """``min out[0]`` s.t. ``out == coeffs @ z``, ``z = x*x``, ``x in [-2, 2]``."""
    n_rows, n_cols = coeffs.shape
    m = dm.Model("axis_sum")
    x = m.continuous("x", shape=(n_cols,), lb=-2, ub=2)
    z = m.continuous("z", shape=(n_cols,), lb=z_lb, ub=z_ub)
    out = m.continuous("out", shape=(n_rows,), lb=-100, ub=100)
    m.minimize(out[0])
    m.subject_to(z == x * x)
    m.subject_to(out == dm.sum(coeffs * z, axis=1))
    return m


# --------------------------------------------------------------------------- #
# 1. FBBT soundness: the derived box must CONTAIN the true range.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "coeffs, z_lb, z_ub, true_lo, true_hi",
    [
        (np.array([[-1.0, -1.0]]), 0.0, 4.0, -8.0, 0.0),
        (np.array([[-1.0, -7.0]]), 0.0, 4.0, -32.0, 0.0),
        (np.array([[-1.0, -2.0, -4.0]]), 0.0, 1.0, -7.0, 0.0),
        (np.array([[1.0, 1.0]]), 0.0, 4.0, 0.0, 8.0),
        # Two and three rows: sound before the fix, and must stay sound.
        (np.array([[-1.0, -1.0], [-1.0, -1.0]]), 0.0, 4.0, -8.0, 0.0),
    ],
)
def test_fbbt_encloses_the_true_range_of_an_axis_sum(coeffs, z_lb, z_ub, true_lo, true_hi):
    n_cols = coeffs.shape[1]
    m = dm.Model("fbbt")
    z = m.continuous("z", shape=(n_cols,), lb=z_lb, ub=z_ub)
    out = m.continuous("out", shape=(coeffs.shape[0],), lb=-100, ub=100)
    m.minimize(out[0])
    m.subject_to(out == dm.sum(coeffs * z, axis=1))

    res = fbbt_box(m)
    lo, hi = float(res.lb[n_cols]), float(res.ub[n_cols])
    # An enclosure may be loose and stay sound; it must never be narrow.
    assert lo <= true_lo + 1e-9, f"FBBT lower bound {lo} cuts off the true floor {true_lo}"
    assert hi >= true_hi - 1e-9, f"FBBT upper bound {hi} cuts off the true ceiling {true_hi}"


def test_fbbt_still_tightens_a_scalar_full_reduction():
    """The fix must not turn every reduction into an abstention."""
    m = dm.Model("full_reduction")
    y = m.continuous("y", shape=(3,), lb=0, ub=1)
    s = m.continuous("s", lb=-100, ub=100)
    m.minimize(s)
    m.subject_to(s == dm.sum(y))
    res = fbbt_box(m)
    lo, hi = float(res.lb[3]), float(res.ub[3])
    assert lo <= 0.0 + 1e-9 and hi >= 3.0 - 1e-9, (lo, hi)
    assert hi <= 3.0 + 1e-9, f"a sum of three [0,1] terms cannot exceed 3, got {hi}"


# --------------------------------------------------------------------------- #
# 2. The solve: no false certificate, and the true optimum is found.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_rows", [1, 2, 3])
def test_axis_sum_reaches_the_true_optimum(n_rows):
    coeffs = -np.ones((n_rows, 2))
    r = _axis_sum_model(coeffs, z_lb=0.0, z_ub=4.0).solve(time_limit=120)
    # min -z0-z1 with z_i = x_i^2, x in [-2,2] -> z_i = 4, objective -8.
    assert r.objective is not None
    assert r.objective <= -8.0 + 1e-4, f"missed the optimum: {r.objective}"
    # The soundness half: a dual bound above the optimum has cut it out of the box.
    assert r.bound <= -8.0 + 1e-6, f"dual bound {r.bound} is above the true optimum -8.0"


def test_axis_sum_agrees_with_the_scalar_spelling():
    """The same model written two ways must give the same certified answer."""
    coeffs = np.array([[-1.0, -1.0]])
    vector = _axis_sum_model(coeffs, z_lb=0.0, z_ub=4.0).solve(time_limit=120)

    m = dm.Model("scalar")
    x = m.continuous("x", shape=(2,), lb=-2, ub=2)
    z = m.continuous("z", shape=(2,), lb=0, ub=4)
    out = m.continuous("out", shape=(1,), lb=-100, ub=100)
    m.minimize(out[0])
    m.subject_to(z == x * x)
    m.subject_to(out[0] == -z[0] - z[1])
    scalar = m.solve(time_limit=120)

    assert scalar.objective == pytest.approx(-8.0, abs=1e-4)
    assert vector.objective == pytest.approx(scalar.objective, abs=1e-4)


# --------------------------------------------------------------------------- #
# 3. Scalarization: the row must reach the relaxation as scalar affine terms.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_rows", [1, 2, 3])
def test_scalar_elements_expands_an_axis_reduction(n_rows):
    m = dm.Model("scalarize")
    z = m.continuous("z", shape=(2,), lb=0, ub=4)
    coeffs = np.arange(1.0, 1.0 + 2 * n_rows).reshape(n_rows, 2)
    elements = scalar_elements(dm.sum(coeffs * z, axis=1))
    assert elements is not None, "the axis reduction did not scalarize"
    assert len(elements) == n_rows


def test_scalar_elements_of_a_full_reduction_is_one_scalar():
    m = dm.Model("scalarize_full")
    y = m.continuous("y", shape=(3,), lb=0, ub=1)
    elements = scalar_elements(dm.sum(y))
    assert elements is not None and len(elements) == 1


def test_the_node_relaxation_builds_for_an_axis_sum_row():
    """It returned ``status="error"`` -- no bound at all -- before #1364."""
    from discopt._relax.mccormick_lp import MccormickLPRelaxer

    coeffs = np.array([[-1.0, -1.0]])
    m = _axis_sum_model(coeffs, z_lb=0.0, z_ub=4.0)
    relaxer = MccormickLPRelaxer(m)
    lb = np.array([-2.0, -2.0, 0.0, 0.0, -100.0])
    ub = np.array([2.0, 2.0, 4.0, 4.0, 100.0])
    res = relaxer.solve_at_node(lb, ub)
    assert res.status != "error", "the McCormick LP could not be built for an affine row"
    assert res.lower_bound is not None
    assert res.lower_bound <= -8.0 + 1e-6, f"invalid root bound {res.lower_bound}"


# --------------------------------------------------------------------------- #
# 4. The real case: a scalar-output network through discopt.ml.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_scalar_output_network_reaches_its_grid_verified_optimum():
    ml = pytest.importorskip("discopt.ml")

    np.random.seed(0)
    w1 = np.random.randn(2, 4) * 0.5
    b1 = np.random.randn(4) * 0.1
    w2 = np.random.randn(4, 1) * 0.5
    b2 = np.random.randn(1) * 0.1
    net = ml.NetworkDefinition(
        layers=[
            ml.DenseLayer(w1, b1, ml.Activation.TANH),
            ml.DenseLayer(w2, b2, ml.Activation.LINEAR),
        ],
        input_bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
    )

    grid = np.linspace(-2, 2, 401)
    a, c = np.meshgrid(grid, grid, indexing="ij")
    pts = np.stack([a.ravel(), c.ravel()], axis=1)
    reference = float((np.tanh(pts @ w1 + b1) @ w2 + b2).min())

    m = dm.Model("smooth_nn")
    nn = ml.NNFormulation(m, net, strategy="full_space")
    nn.formulate()
    m.minimize(nn.outputs[0])
    r = m.solve(time_limit=300)

    assert r.objective == pytest.approx(reference, abs=1e-3), (
        f"reported {r.objective}, dense grid says {reference}"
    )
    assert r.bound <= reference + 1e-6, (
        f"dual bound {r.bound} is above the true optimum {reference}"
    )
