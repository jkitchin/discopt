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


def test_sqrt_model_is_supported():
    """sqrt IS covered by the native kernel (EnvTerm::Sqrt); producer builds a spec."""
    from discopt.modeling.core import sqrt as dsqrt

    m = Model()
    x = m.continuous("x", lb=1.0, ub=4.0)
    m.minimize(dsqrt(x) + x)
    spec = build_spatial_kernel_spec(m)
    assert spec is not None
    assert (np.asarray(spec["term_kind"]) == 3).sum() == 1  # one sqrt term


def test_unsupported_atom_returns_none():
    """A model using an atom the kernel does not implement (exp) yields None."""
    from discopt.modeling.core import exp as dexp

    m = Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.minimize(dexp(x) + x)
    spec = build_spatial_kernel_spec(m)
    assert spec is None


def test_solver_flag_on_matches_flag_off():
    """The DISCOPT_NATIVE_SPATIAL_KERNEL flag (solver.py hand-off) gives the same
    optimum as the default path on a covered model, and OFF is the default."""
    import os

    def build():
        m = Model()
        x = m.continuous("x", lb=0.0, ub=2.0)
        y = m.continuous("y", lb=0.0, ub=2.0)
        m.subject_to(x + y >= 3)
        m.minimize(x * y)
        return m

    prev = os.environ.get("DISCOPT_NATIVE_SPATIAL_KERNEL")
    try:
        os.environ.pop("DISCOPT_NATIVE_SPATIAL_KERNEL", None)
        off = build().solve(time_limit=30)
        os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1"
        on = build().solve(time_limit=30)
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_NATIVE_SPATIAL_KERNEL", None)
        else:
            os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = prev
    assert off.status == "optimal" and on.status == "optimal"
    assert abs(off.objective - on.objective) < 1e-3
    # OFF must not exceed the true optimum; ON must agree.
    assert abs(on.objective - 2.0) < 1e-3


@pytest.mark.slow
def test_tanksize_certifies_natively():
    """#764 definition of done: the native kernel certifies tanksize at the default
    gap tolerance. Seeded with the known feasible value (the fast path, ~30 s); the
    unseeded run also certifies (~190 s, measured 2026-07-19). Locks in the quality
    ratchet: finite slack bounds (0 uncertified nodes) + propagation + honest
    Optimal semantics (bound >= incumbent - gap_tol, never a false certificate)."""
    import os

    nl = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "tanksize.nl")
    if not os.path.exists(nl):
        pytest.skip("tanksize.nl fixture not present")
    import discopt.solver as S
    from discopt import _rust
    from discopt.modeling.core import from_nl

    cap = {}

    def grab(model, lb, ub, n_vars, *a, **k):
        cap.update(
            model=model,
            lb=np.asarray(lb, float)[:n_vars].copy(),
            ub=np.asarray(ub, float)[:n_vars].copy(),
        )
        return None

    prev_env = os.environ.get("DISCOPT_NATIVE_SPATIAL_KERNEL")
    prev_fn = S._try_native_spatial_kernel
    try:
        os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1"
        S._try_native_spatial_kernel = grab
        from_nl(nl).solve(time_limit=10.0)
    finally:
        S._try_native_spatial_kernel = prev_fn
        if prev_env is None:
            os.environ.pop("DISCOPT_NATIVE_SPATIAL_KERNEL", None)
        else:
            os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = prev_env

    spec = build_spatial_kernel_spec(cap["model"], bounds=(cap["lb"], cap["ub"]))
    assert spec is not None
    meta = {k: spec.pop(k) for k in list(spec) if k.startswith("meta_")}
    sign, off = meta["meta_obj_sense_sign"], meta["meta_obj_offset"]
    oracle = 1.2686437540
    res = _rust.solve_spatial_tree_py(
        **spec, max_nodes=50000, gap_tol=1e-4, initial_incumbent=1.2686437614652892
    )
    assert res["status"] == "optimal", res["status"]
    inc = sign * (res["incumbent"] + off)
    bound = sign * (res["bound"] + off)
    # Certificate brackets the oracle; the gap is genuinely closed; every node
    # certified (the finite-slack-bounds fix).
    assert abs(inc - oracle) < 1e-5, f"incumbent {inc} vs oracle {oracle}"
    assert bound <= oracle + 1e-9, f"bound {bound} above oracle {oracle}"
    assert inc - bound <= 1e-4 + 1e-9, f"gap {inc - bound} not closed"
    assert res["n_uncertified"] == 0, f"{res['n_uncertified']} uncertified nodes"


def test_unbounded_relaxation_declines():
    """tanksize's raw .nl box is unbounded, so the McCormick relaxation has infinite
    aux ranges (invalid). The producer must decline soundly (return None) rather than
    emit a spec whose node LP is degenerate — feeding presolved finite bounds is the
    follow-up. This is the guard that keeps the native path from ever being wrong."""
    import os

    nl = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "tanksize.nl")
    if not os.path.exists(nl):
        pytest.skip("tanksize.nl fixture not present")
    from discopt.modeling.core import from_nl

    spec = build_spatial_kernel_spec(from_nl(nl))
    assert spec is None
