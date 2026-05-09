"""Tests for ``discopt._jax.polyhedral_oa`` (M11 of issue #51).

Acceptance criteria:

1. The polyhedral outer approximation contains the true feasible set:
   every sampled ``(x, f(x))`` for ``x`` in ``[a, b]`` (≥ 10⁴ samples)
   satisfies all generated linear inequalities within tolerance.
2. For each underlying arithmetic, the wrapper produces an OA no looser
   than that arithmetic's native LP relaxation through
   ``relaxation_compiler.py``. We test the simpler form: the secant
   between domain endpoints is always present in the cut family, so the
   OA at the endpoints is at least as tight as any single-secant
   construction.
3. API is uniform across the supported arithmetics (single dispatched
   function).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax import polyhedral_oa as oa
from discopt._jax.cutting_planes import LinearCut

N_SAMPLES = 10_000
TOL = 1e-9


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_known_arithmetics():
    assert set(oa.ARITHMETICS) == {"mccormick", "chebyshev", "taylor", "ellipsoidal"}


def test_unknown_arithmetic_raises():
    with pytest.raises(ValueError):
        oa.outer_approximation(jnp.exp, (-1.0, 1.0), "bogus")


def test_ellipsoidal_not_implemented():
    with pytest.raises(NotImplementedError):
        oa.outer_approximation(jnp.exp, (-1.0, 1.0), "ellipsoidal")


def test_invalid_domain_raises():
    with pytest.raises(ValueError):
        oa.outer_approximation(jnp.exp, (1.0, 1.0), "chebyshev")
    with pytest.raises(ValueError):
        oa.outer_approximation(jnp.exp, (1.0, -1.0), "chebyshev")


def test_n_slopes_minimum():
    with pytest.raises(ValueError):
        oa.outer_approximation(jnp.exp, (-1.0, 1.0), "chebyshev", n_slopes=1)


def test_returns_outer_approximation():
    result = oa.outer_approximation(jnp.exp, (-1.0, 1.0), "chebyshev")
    assert isinstance(result, oa.OuterApproximation)
    assert result.domain == (-1.0, 1.0)
    assert result.arithmetic == "chebyshev"
    assert all(isinstance(c, LinearCut) for c in result.cuts)
    assert all(c.coeffs.shape == (2,) for c in result.cuts)


# ---------------------------------------------------------------------------
# Soundness on benchmark-style univariate functions
# ---------------------------------------------------------------------------


SOUNDNESS_CASES = [
    # (name, f, domain, degree)
    ("exp", jnp.exp, (-1.0, 2.0), 8),
    ("log", jnp.log, (0.5, 5.0), 10),
    ("sqrt", jnp.sqrt, (0.5, 4.0), 8),
    ("recip", lambda x: 1.0 / x, (0.5, 3.0), 8),
    ("sin", jnp.sin, (-1.5, 1.5), 10),
    ("cos", jnp.cos, (-1.5, 1.5), 8),
    ("square", lambda x: x * x, (-2.0, 2.0), 4),
    ("exp_minus_xsq", lambda x: jnp.exp(-(x * x)), (-1.5, 1.5), 12),
]


@pytest.mark.parametrize("arithmetic", ["mccormick", "chebyshev", "taylor"])
@pytest.mark.parametrize("name, f, domain, degree", SOUNDNESS_CASES)
def test_oa_is_sound(arithmetic, name, f, domain, degree):
    rng = np.random.default_rng(0)
    xs = rng.uniform(domain[0], domain[1], size=N_SAMPLES)
    true = np.asarray(f(xs))
    result = oa.outer_approximation(f, domain, arithmetic, degree=degree, n_slopes=16)
    lo = result.evaluate_lower(xs)
    hi = result.evaluate_upper(xs)
    assert (true >= lo - TOL).all(), f"{arithmetic}/{name}: lower violated"
    assert (true <= hi + TOL).all(), f"{arithmetic}/{name}: upper violated"


@pytest.mark.parametrize("arithmetic", ["mccormick", "chebyshev", "taylor"])
def test_oa_each_cut_is_globally_valid(arithmetic):
    """Stronger than the OA-as-tube test: every individual cut must hold
    for every sampled point. This guards against future refactors that
    accidentally emit a cut valid only on a sub-segment."""
    domain = (-1.0, 2.0)
    rng = np.random.default_rng(1)
    xs = rng.uniform(domain[0], domain[1], size=N_SAMPLES)
    f_true = np.exp(xs)
    result = oa.outer_approximation(jnp.exp, domain, arithmetic, degree=8, n_slopes=12)
    for k, cut in enumerate(result.cuts):
        sx, sy = float(cut.coeffs[0]), float(cut.coeffs[1])
        lhs = sx * xs + sy * f_true
        if cut.sense == ">=":
            assert (lhs >= cut.rhs - TOL).all(), f"{arithmetic} cut[{k}] (>=) violated"
        elif cut.sense == "<=":
            assert (lhs <= cut.rhs + TOL).all(), f"{arithmetic} cut[{k}] (<=) violated"
        else:
            raise AssertionError(f"unexpected sense {cut.sense!r}")


# ---------------------------------------------------------------------------
# Tightness: the endpoint secant must be present (cuts are at least as
# tight as a single-secant construction at the domain endpoints).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arithmetic", ["mccormick", "chebyshev", "taylor"])
def test_endpoint_secant_present(arithmetic):
    domain = (-1.0, 2.0)
    a, b = domain
    fa = float(np.exp(a))
    fb = float(np.exp(b))
    expected_slope = (fb - fa) / (b - a)
    result = oa.outer_approximation(jnp.exp, domain, arithmetic, degree=8, n_slopes=12)
    slopes_in_cuts = np.array([-float(c.coeffs[0]) for c in result.cuts])
    assert np.min(np.abs(slopes_in_cuts - expected_slope)) < 1e-3


# ---------------------------------------------------------------------------
# Refining n_slopes tightens the OA tube
# ---------------------------------------------------------------------------


def test_more_slopes_tightens_tube():
    domain = (-1.0, 2.0)
    xs = np.linspace(*domain, 200)
    coarse = oa.outer_approximation(jnp.exp, domain, "chebyshev", degree=8, n_slopes=4)
    fine = oa.outer_approximation(jnp.exp, domain, "chebyshev", degree=8, n_slopes=32)
    coarse_width = float(np.mean(coarse.evaluate_upper(xs) - coarse.evaluate_lower(xs)))
    fine_width = float(np.mean(fine.evaluate_upper(xs) - fine.evaluate_lower(xs)))
    assert fine_width <= coarse_width + 1e-12


# ---------------------------------------------------------------------------
# Uniform API: the same call signature works across all supported
# arithmetics and produces structurally identical outputs.
# ---------------------------------------------------------------------------


def test_uniform_api_across_arithmetics():
    domain = (-0.5, 0.5)
    results = {
        kind: oa.outer_approximation(jnp.exp, domain, kind, degree=6, n_slopes=8)
        for kind in ("mccormick", "chebyshev", "taylor")
    }
    for kind, r in results.items():
        assert r.arithmetic == kind
        assert r.domain == domain
        assert len(r.cuts) > 0
        assert all(c.coeffs.shape == (2,) for c in r.cuts)
