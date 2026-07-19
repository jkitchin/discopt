"""End-to-end tests for the native Rust spatial B&B kernel (issue #764).

Producer (`spatial_producer.build_spatial_kernel_spec`) → PyO3
(`_rust.solve_spatial_tree_py`) → Rust kernel (`bnb::spatial_*`). Validates the
native path certifies the correct global optimum and agrees with the trusted
`model.solve()` path (bound-neutrality) on the incremental-engine subset
(bilinear / monomial / affine-square). Skips gracefully if the extension pre-dates
the binding.
"""

import numpy as np
import pytest

from discopt import Model

_rust = pytest.importorskip("discopt._rust")
if not hasattr(_rust, "solve_spatial_tree_py"):
    pytest.skip("native spatial kernel binding not built", allow_module_level=True)

from discopt._jax.spatial_producer import (  # noqa: E402
    build_spatial_kernel_spec,
    solve_with_native_kernel,
)


def _bilinear_min():
    # min x*y s.t. x+y>=3, x,y in [0,2] -> 2.0 (corner (2,1)/(1,2))
    m = Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(x + y >= 3)
    m.minimize(x * y)
    return m


def _bilinear_max():
    # min -(x*y) s.t. x+y<=3, x,y in [0,2] -> -2.25 (interior x=y=1.5)
    m = Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(x + y <= 3)
    m.minimize(-(x * y))
    return m


def _square_min():
    # min x^2 - x, x in [0,2] -> -0.25 at x=0.5 (monomial p=2)
    m = Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.minimize(x * x - x)
    return m


@pytest.mark.parametrize(
    "build,opt",
    [(_bilinear_min, 2.0), (_bilinear_max, -2.25), (_square_min, -0.25)],
    ids=["bilinear-min", "bilinear-max", "square-min"],
)
def test_native_kernel_certifies_optimum(build, opt):
    res = solve_with_native_kernel(build(), max_nodes=50000, gap_tol=1e-5)
    assert res is not None, "model should be in the incremental-engine subset"
    assert res["status"] == "optimal", res
    assert abs(res["incumbent"] - opt) < 1e-3, f"incumbent {res['incumbent']} != {opt}"
    # Soundness: the global bound never exceeds the true optimum.
    assert res["bound"] <= opt + 1e-6, f"bound {res['bound']} above optimum {opt}"


@pytest.mark.parametrize(
    "build",
    [_bilinear_min, _square_min],
    ids=["bilinear-min", "square-min"],
)
def test_native_kernel_agrees_with_trusted_solve(build):
    """Bound-neutrality: the native kernel's optimum matches model.solve()."""
    native = solve_with_native_kernel(build(), max_nodes=50000, gap_tol=1e-5)
    trusted = build().solve(time_limit=30.0)
    assert native is not None
    assert trusted.objective is not None
    assert abs(native["incumbent"] - trusted.objective) < 1e-3, (
        f"native {native['incumbent']} != trusted {trusted.objective}"
    )


def test_incumbent_point_is_feasible():
    """The accepted incumbent point genuinely satisfies the nonconvex relation."""
    res = solve_with_native_kernel(_bilinear_min(), max_nodes=50000, gap_tol=1e-5)
    x = np.asarray(res["incumbent_x"])
    # w (col 2) == x*y and x+y >= 3.
    assert abs(x[0] * x[1] - x[2]) < 1e-4
    assert x[0] + x[1] >= 3.0 - 1e-4


def test_unsupported_model_returns_none():
    """A model outside the incremental subset (sqrt) yields None (Python fallback)."""
    from discopt.modeling.core import sqrt as dsqrt

    m = Model()
    x = m.continuous("x", lb=1.0, ub=4.0)
    m.minimize(dsqrt(x) + x)
    spec = build_spatial_kernel_spec(m)
    # sqrt is not covered by the incremental engine -> producer declines.
    assert spec is None
