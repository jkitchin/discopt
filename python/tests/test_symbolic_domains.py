"""Tests for the power/chemeng domain packs and the atom registry (Phases 2/4/5).

Every registered atom is certified sound and correctly curved over its domain,
and each closure is checked against its SymPy derivation where applicable.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from discopt._jax.symbolic import registry  # noqa: E402
from discopt._jax.symbolic.domains import chemeng, power  # noqa: E402
from discopt._jax.symbolic.verification import verify_envelope  # noqa: E402

pytestmark = pytest.mark.relaxation


# --------------------------------------------------------------------------
# Registry: every atom is sound and correctly curved over its domain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", registry.available())
def test_registered_atom_is_sound(name):
    report = registry.certify(name, n_boxes=400, seed=3)
    assert report.sound, (
        f"{name}: lower={report.max_lower_violation:.2e} upper={report.max_upper_violation:.2e}"
    )
    assert report.max_convexity_violation <= 1e-6
    assert report.max_concavity_violation <= 1e-6


def test_certify_all_runs():
    reports = registry.certify_all(n_boxes=200, seed=1)
    assert set(reports) == set(registry.available())
    assert all(r.sound for r in reports.values())


# --------------------------------------------------------------------------
# Chemeng atoms
# --------------------------------------------------------------------------


def test_arrhenius_convex_on_normal_temperature_window():
    """For large E/R the normal T-window is on the convex branch (T < c/2)."""
    c = 6000.0
    relax = chemeng.arrhenius_relax(c)
    f = lambda t: jnp.exp(-c / t)  # noqa: E731
    lb, ub = 300.0, 800.0  # below c/2 = 3000
    xs = jnp.linspace(lb, ub, 100)
    cv, cc = jax.vmap(relax, in_axes=(0, None, None))(xs, lb, ub)
    assert jnp.allclose(cv, f(xs), atol=1e-9)  # cv == f on a convex window
    assert jnp.all(cc >= f(xs) - 1e-7)


def test_arrhenius_rejects_nonpositive_c():
    with pytest.raises(ValueError):
        chemeng.arrhenius_relax(0.0)


def test_saturating_is_concave():
    relax = chemeng.saturating_relax(2.0)
    f = lambda x: x / (2.0 + x)  # noqa: E731
    report = verify_envelope(relax, f, domain=(0.0, 30.0), n_boxes=300, seed=5)
    assert report.sound
    assert report.max_concavity_violation <= 1e-6


# --------------------------------------------------------------------------
# Power atoms
# --------------------------------------------------------------------------


def test_cos_angle_concave_envelope():
    report = verify_envelope(
        power.cos_angle_relax, jnp.cos, domain=(-1.45, 1.45), n_boxes=400, seed=2
    )
    assert report.sound
    assert report.max_concavity_violation <= 1e-6


def test_sin_angle_single_inflection_sound():
    report = verify_envelope(
        power.sin_angle_relax, jnp.sin, domain=(-3.0, 3.0), n_boxes=500, seed=4
    )
    assert report.sound
    assert report.max_convexity_violation <= 1e-6
    assert report.max_concavity_violation <= 1e-6


def test_engine_rejects_globally_periodic_sin():
    """The engine analyzes curvature globally, so multi-inflection sin is rejected.

    The hand-written power pack encodes the within-(-pi, pi) single-inflection
    assumption that the domain-agnostic engine cannot infer — which is exactly
    why the domain packs exist alongside the general engine.
    """
    pytest.importorskip("sympy")
    import sympy as sp
    from discopt._jax.symbolic import derive_envelope
    from discopt._jax.symbolic.envelope_deriver import EnvelopeDerivationError

    x = sp.Symbol("x", real=True)
    with pytest.raises(EnvelopeDerivationError):
        derive_envelope(sp.sin(x), x)


def test_sin_angle_matches_engine_construction_on_single_inflection_proxy():
    """sin's tangent/secant construction matches the engine on an x^3-like proxy.

    A direct check that the runtime single-inflection assembly the power pack
    calls is the same one the engine emits (here for a function the engine *can*
    derive, sharing the runtime): tanh, also convex-then-concave through 0.
    """
    pytest.importorskip("sympy")
    import sympy as sp
    from discopt._jax.symbolic import derive_envelope, lambdify_envelope

    x = sp.Symbol("x", real=True)
    sym = jax.jit(lambdify_envelope(derive_envelope(sp.tanh(x), x)))
    from discopt._jax.symbolic import runtime

    hand = jax.jit(
        lambda t, lb, ub: runtime.single_inflection_envelope(
            t, lb, ub, f=jnp.tanh, fp=lambda u: 1.0 - jnp.tanh(u) ** 2, c=0.0, concavo_convex=False
        )
    )
    import numpy as np

    rng = np.random.default_rng(0)
    # Vectorize boxes + points so the jitted closures compile once.
    boxes = np.sort(rng.uniform(-3.0, 3.0, size=(40, 2)), axis=1)
    boxes[:, 1] = np.maximum(boxes[:, 1], boxes[:, 0] + 1e-2)
    lbs = jnp.asarray(boxes[:, 0])
    ubs = jnp.asarray(boxes[:, 1])
    fr = jnp.asarray(rng.uniform(0.0, 1.0, size=(40, 16)))
    xs = lbs[:, None] + fr * (ubs - lbs)[:, None]
    relax = jax.vmap(jax.vmap(hand, in_axes=(0, None, None)), in_axes=(0, 0, 0))
    relax_s = jax.vmap(jax.vmap(sym, in_axes=(0, None, None)), in_axes=(0, 0, 0))
    cv_h, cc_h = relax(xs, lbs, ubs)
    cv_s, cc_s = relax_s(xs, lbs, ubs)
    assert jnp.allclose(cv_h, cv_s, atol=1e-7)
    assert jnp.allclose(cc_h, cc_s, atol=1e-7)


# --------------------------------------------------------------------------
# End-to-end: relaxation compiler auto-routes x*log(x) to the tight envelope
# --------------------------------------------------------------------------


def test_compiler_routes_xlogx_pattern():
    """The compiler detects ``x * log(x)`` and routes it to the dedicated convex
    envelope (tighter than the generic bilinear of ``x`` and ``log(x)``), end to
    end, on a positive box. Asserts soundness (``cv <= f <= cc``) and that the
    routed result matches the standalone ``xlogx_relax`` closure."""
    import discopt.modeling as M
    import numpy as np
    from discopt._jax.relaxation_compiler import compile_relaxation
    from discopt.modeling.core import Model

    m = Model("entropy")
    x = m.continuous("x", lb=0.5, ub=5.0)
    m.minimize(x * M.log(x))
    expr = x * M.log(x)

    relax_fn = compile_relaxation(expr.expression if hasattr(expr, "expression") else expr, m)
    lb = jnp.array([0.5])
    ub = jnp.array([5.0])
    for xv in np.linspace(0.5, 5.0, 60):
        xa = jnp.array([float(xv)])
        cv, cc = relax_fn(xa, xa, lb, ub)
        true = float(xv) * np.log(float(xv))
        assert float(cv) <= true + 1e-6, f"unsound cv at x={xv}: {float(cv)} > {true}"
        assert float(cc) >= true - 1e-6, f"unsound cc at x={xv}: {float(cc)} < {true}"
        # Routed result == standalone closure (i.e. the dispatch fired).
        cv_x, cc_x = chemeng.xlogx_relax(jnp.array(float(xv)), 0.5, 5.0)
        assert float(cv) == pytest.approx(float(cv_x), abs=1e-7)
        assert float(cc) == pytest.approx(float(cc_x), abs=1e-7)


def test_compiler_xlogx_detector_no_misfire():
    """The x*log(x) detector fires only on a bare variable times log of the SAME
    variable — not on different variables, squares, or other transcendentals."""
    import discopt.modeling as M
    from discopt._jax.relaxation_compiler import _try_extract_xlogx
    from discopt.modeling.core import Model

    m = Model("t")
    x = m.continuous("x", lb=0.5, ub=5.0)
    y = m.continuous("y", lb=0.5, ub=5.0)

    def ex(e):
        return e.expression if hasattr(e, "expression") else e

    assert _try_extract_xlogx(ex(x * M.log(x)), m) is not None
    assert _try_extract_xlogx(ex(M.log(x) * x), m) is not None
    assert _try_extract_xlogx(ex(x * M.log(y)), m) is None  # different variable
    assert _try_extract_xlogx(ex(x * x), m) is None  # no log
    assert _try_extract_xlogx(ex(x * M.exp(x)), m) is None  # not log
    assert _try_extract_xlogx(ex(M.log(x) * M.log(x)), m) is None  # no bare var


def test_compiler_routes_monod_pattern():
    """The compiler detects x/(K+x) on a nonneg-domain variable and routes it to
    the dedicated concave envelope; sound, and matches the standalone closure."""
    import numpy as np
    from discopt._jax.relaxation_compiler import compile_relaxation
    from discopt.modeling.core import Model

    m = Model("monod")
    x = m.continuous("x", lb=0.1, ub=5.0)
    m.maximize(x / (2.0 + x))
    relax_fn = compile_relaxation(x / (2.0 + x), m)
    sat = chemeng.saturating_relax(2.0)
    lb = jnp.array([0.1])
    ub = jnp.array([5.0])
    for xv in np.linspace(0.1, 5.0, 60):
        xa = jnp.array([float(xv)])
        cv, cc = relax_fn(xa, xa, lb, ub)
        f = float(xv) / (2.0 + float(xv))
        assert float(cv) <= f + 1e-6
        assert float(cc) >= f - 1e-6
        cv_s, cc_s = sat(jnp.array(float(xv)), 0.1, 5.0)
        assert float(cv) == pytest.approx(float(cv_s), abs=1e-7)
        assert float(cc) == pytest.approx(float(cc_s), abs=1e-7)


def test_monod_detector_guards_and_no_misfire():
    from discopt._jax.relaxation_compiler import _try_extract_monod
    from discopt.modeling.core import Model

    m = Model("t")
    x = m.continuous("x", lb=0.1, ub=5.0)
    y = m.continuous("y", lb=0.1, ub=5.0)
    assert _try_extract_monod(x / (2.0 + x), m) == (0, pytest.approx(2.0))
    assert _try_extract_monod(x / (x + 2.0), m) == (0, pytest.approx(2.0))
    assert _try_extract_monod(x / (2.0 + y), m) is None  # different variable
    assert _try_extract_monod(x / (2.0 * x), m) is None  # denominator not K+x
    # Negative declared lower bound: envelope invalid -> must not engage.
    mn = Model("n")
    xn = mn.continuous("x", lb=-1.0, ub=5.0)
    assert _try_extract_monod(xn / (2.0 + xn), mn) is None
