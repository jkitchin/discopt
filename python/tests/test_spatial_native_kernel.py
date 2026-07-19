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


def test_native_seed_verify_point_accepts_and_rejects():
    """#764 Task 1: the seed verifier accepts a genuinely feasible point (returning its
    TRUE objective) and rejects an infeasible one. This is the soundness gate that
    stands between a heuristic candidate and the kernel's incumbent cutoff — an
    unverified seed would poison the certificate."""
    import discopt.solver as S

    m = Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(x + y >= 3)
    m.minimize(x * y)
    # Feasible corner (2, 1): x+y=3, objective x*y = 2.0.
    ok, obj = S._native_kernel_verify_point(m, np.array([2.0, 1.0]))
    assert ok is True
    assert obj is not None and abs(obj - 2.0) < 1e-9
    # Infeasible (0, 0): x+y = 0 < 3 -> rejected, no objective vouched.
    ok2, obj2 = S._native_kernel_verify_point(m, np.array([0.0, 0.0]))
    assert ok2 is False and obj2 is None


def test_native_seed_verify_point_respects_integrality():
    """A fractional value on an integer variable is rejected (integrality 1e-5)."""
    import discopt.solver as S

    m = Model()
    n = m.integer("n", lb=0, ub=5)
    m.minimize((n - 2) * (n - 2))
    ok_int, _ = S._native_kernel_verify_point(m, np.array([2.0]))
    assert ok_int is True
    ok_frac, obj_frac = S._native_kernel_verify_point(m, np.array([2.5]))
    assert ok_frac is False and obj_frac is None


def test_driver_seeded_incumbent_is_feasibility_verified():
    """#764 Task 1: with the flag ON the driver seeds the native kernel from a verified
    feasible point; the reported incumbent must itself be independently
    feasibility-verifiable against the original model, and match the true optimum."""
    import os

    import discopt.solver as S

    def build():
        m = Model()
        x = m.continuous("x", lb=0.0, ub=2.0)
        y = m.continuous("y", lb=0.0, ub=2.0)
        m.subject_to(x + y >= 3)
        m.minimize(x * y)
        return m

    prev = os.environ.get("DISCOPT_NATIVE_SPATIAL_KERNEL")
    try:
        os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1"
        r = build().solve(time_limit=30)
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_NATIVE_SPATIAL_KERNEL", None)
        else:
            os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = prev
    assert r.status == "optimal"
    assert abs(r.objective - 2.0) < 1e-4
    assert r.bound <= r.objective + 1e-9  # certificate invariant (min sense)
    m2 = build()
    x_flat = np.array(
        [float(np.asarray(r.x[v.name]).reshape(-1)[0]) for v in m2._variables],
        dtype=np.float64,
    )
    ok, obj = S._native_kernel_verify_point(m2, x_flat)
    assert ok is True, "reported incumbent must be independently feasibility-verified"
    assert abs(obj - r.objective) < 1e-6


@pytest.mark.slow
def test_tanksize_driver_seeded_certifies():
    """#764 Task 1 (definition of done, driver path): a full ``m.solve()`` with the flag
    ON certifies tanksize via the DRIVER-side seed (SubNLP free-binary enumeration ->
    verified feasible point -> ``initial_incumbent``). Asserts the certificate brackets
    the oracle AND the seed engaged — node_count well below the ~78,667 of the unseeded
    run (measured ~11,379 nodes / ~50 s seeded, 2026-07-19)."""
    import os

    nl = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "tanksize.nl")
    if not os.path.exists(nl):
        pytest.skip("tanksize.nl fixture not present")
    from discopt.modeling.core import from_nl

    oracle = 1.2686437540
    prev = os.environ.get("DISCOPT_NATIVE_SPATIAL_KERNEL")
    try:
        os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1"
        r = from_nl(nl).solve(time_limit=200)
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_NATIVE_SPATIAL_KERNEL", None)
        else:
            os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = prev
    assert r.status == "optimal", r.status
    assert abs(r.objective - oracle) < 1e-4, f"objective {r.objective} vs oracle {oracle}"
    assert r.bound <= oracle + 1e-6, f"bound {r.bound} above oracle {oracle}"
    # The seed engaged: far fewer nodes than the unseeded ~78,667.
    assert r.node_count < 40000, f"node_count {r.node_count} suggests the seed did not engage"


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


# ── #789: native-kernel feature-safety routing + final-incumbent verification ──


def _bilinear_binary_minlp():
    """Nonconvex bilinear MINLP with a binary (the model on which the kernel's
    tree incumbent is a false primal — the #789 verification case)."""
    m = Model("bb_mi")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    b = m.binary("b")
    m.subject_to(x * y >= 1.0)
    m.subject_to(x + y <= 4.0 + b)
    m.minimize(x + y + 2.0 * b)
    return m


def _with_kernel_on(monkeypatch):
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "1")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"incumbent_callback": (lambda ctx, model, sol: None)},
        {"node_callback": (lambda ctx, model: None)},
        {"lazy_constraints": (lambda ctx, model, sol: [])},
        {"mccormick_bounds": "none"},
        {"use_learned_relaxations": True},
    ],
)
def test_kernel_feature_safe_declines_return_correct_optimum(monkeypatch, kwargs):
    """#789: with the kernel ON, a solve requesting a Python-engine feature the
    kernel does not honour must route to the Python engine and still certify the
    true optimum (obj 2.0) — never a withheld/None result."""
    _with_kernel_on(monkeypatch)
    m = _bilinear_binary_minlp()
    res = m.solve(time_limit=120.0, **kwargs)
    assert res.status in ("optimal", "feasible"), res.status
    assert res.objective == pytest.approx(2.0, abs=1e-4), res.objective


def test_kernel_feature_safe_predicate_units():
    """The predicate declines each unsupported-feature request and accepts a
    plain solve."""
    from discopt.solver import _native_kernel_feature_safe as safe

    base = dict(
        mccormick_bounds="auto",
        initial_point=None,
        lazy_constraints=None,
        incumbent_callback=None,
        node_callback=None,
        kwargs={},
    )
    assert safe(**base) is True
    assert safe(**{**base, "incumbent_callback": lambda *a: None}) is False
    assert safe(**{**base, "node_callback": lambda *a: None}) is False
    assert safe(**{**base, "lazy_constraints": lambda *a: []}) is False
    assert safe(**{**base, "initial_point": np.zeros(3)}) is False
    assert safe(**{**base, "mccormick_bounds": "none"}) is False
    assert safe(**{**base, "kwargs": {"iteration_callback": lambda *a: None}}) is False
    assert safe(**{**base, "kwargs": {"solution_pool": []}}) is False
    assert safe(**{**base, "kwargs": {"tuning": object()}}) is False


def test_kernel_verifies_final_incumbent_and_declines_false_primal(monkeypatch):
    """#789: on a covered model where the kernel's tree incumbent is INFEASIBLE
    in the original (a false primal), the kernel must verify its own final
    incumbent and decline (route to the Python engine), so the reported result
    is the true optimum — not the #779-withheld None. A plain (feature-safe)
    solve exercises the verification path directly."""
    _with_kernel_on(monkeypatch)
    m = _bilinear_binary_minlp()
    res = m.solve(time_limit=120.0)
    assert res.objective is not None, "withheld/None: false primal escaped kernel verification"
    assert res.objective == pytest.approx(2.0, abs=1e-4), res.objective
    assert not getattr(res, "incumbent_verification_failed", False)
