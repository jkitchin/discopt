"""M11 regression: polyhedral OA wrapper on benchmark-style univariate
subexpressions.

The wrapper (``discopt._jax.polyhedral_oa``) is not yet wired into the
LP relaxation compiler — that integration ships alongside the M2/M3
kernel integration as a follow-up. This regression exercises the
wrapper's three currently-supported arithmetics on subexpressions
drawn from typical benchmark problems and asserts:

1. Every individual cut in the OA family is globally valid: at every
   sampled ``x`` in the domain (≥ 10⁴ samples), the point ``(x, f(x))``
   satisfies the cut within tolerance (soundness). This is the
   strongest form of the M11 acceptance criterion — not just that the
   sandwich tube contains the function, but that no single emitted cut
   is invalid on any sub-region of the domain.
2. The OA tube tightens (or stays equal) when ``n_slopes`` is
   increased. Refinement never loosens the bound.
3. The wrapper produces structurally identical outputs (same cut
   shape, same domain, same sense conventions) across the three
   supported arithmetics, confirming the unified API.

These properties are the contract M11 makes with downstream consumers.
Breaking any of them is a correctness regression.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax import polyhedral_oa as oa

N_REGRESSION_SAMPLES = 10_000

ARITHMETICS = ("mccormick", "chebyshev", "taylor")


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.parametrize("arithmetic", ARITHMETICS)
@pytest.mark.parametrize(
    "name, f, domain, degree",
    [
        ("exp_neg_xsq", lambda x: jnp.exp(-(x * x)), (-1.5, 1.5), 12),
        ("log_two_plus_xsq", lambda x: jnp.log(2.0 + x * x), (-2.0, 2.0), 10),
        ("sqrt_one_plus_xsq", lambda x: jnp.sqrt(1.0 + x * x), (-2.0, 2.0), 8),
        ("sin_x_cos_x", lambda x: jnp.sin(x) * jnp.cos(x), (-1.5, 1.5), 12),
    ],
)
def test_every_cut_is_globally_valid(arithmetic, name, f, domain, degree):
    rng = np.random.default_rng(0)
    xs = rng.uniform(domain[0], domain[1], size=N_REGRESSION_SAMPLES)
    true = np.asarray(f(xs))
    result = oa.outer_approximation(f, domain, arithmetic, degree=degree, n_slopes=16)
    for k, cut in enumerate(result.cuts):
        sx, sy = float(cut.coeffs[0]), float(cut.coeffs[1])
        lhs = sx * xs + sy * true
        if cut.sense == ">=":
            assert (lhs >= cut.rhs - 1e-9).all(), f"{arithmetic}/{name}: cut[{k}] (>=) violated"
        elif cut.sense == "<=":
            assert (lhs <= cut.rhs + 1e-9).all(), f"{arithmetic}/{name}: cut[{k}] (<=) violated"


@pytest.mark.regression
@pytest.mark.parametrize("arithmetic", ARITHMETICS)
def test_refinement_never_loosens(arithmetic):
    """Increasing ``n_slopes`` produces a tube no wider than a coarser one."""
    domain = (-1.0, 1.5)
    xs_eval = np.linspace(*domain, 256)
    coarse = oa.outer_approximation(jnp.exp, domain, arithmetic, degree=8, n_slopes=4)
    fine = oa.outer_approximation(jnp.exp, domain, arithmetic, degree=8, n_slopes=32)
    coarse_w = float(np.mean(coarse.evaluate_upper(xs_eval) - coarse.evaluate_lower(xs_eval)))
    fine_w = float(np.mean(fine.evaluate_upper(xs_eval) - fine.evaluate_lower(xs_eval)))
    assert fine_w <= coarse_w + 1e-12


@pytest.mark.regression
def test_unified_api_signature_across_arithmetics():
    """All three arithmetics must accept the same call signature and
    return structurally compatible OuterApproximation objects."""
    domain = (-1.0, 1.0)
    results = {
        kind: oa.outer_approximation(jnp.exp, domain, kind, degree=6, n_slopes=8)
        for kind in ARITHMETICS
    }
    cut_count = None
    for kind, r in results.items():
        assert r.arithmetic == kind
        assert r.domain == domain
        assert len(r.cuts) > 0
        for cut in r.cuts:
            assert cut.coeffs.shape == (2,)
            assert cut.sense in ("<=", ">=")
            assert float(cut.coeffs[1]) == 1.0
        if cut_count is None:
            cut_count = len(r.cuts)
        else:
            # Same n_slopes => same number of unique slopes by construction
            # (deduplication may differ slightly across providers).
            assert abs(len(r.cuts) - cut_count) <= 4


@pytest.mark.regression
def test_chebyshev_and_taylor_tubes_are_close():
    """For analytic ``f`` on a narrow box, Chebyshev and Taylor wrappers
    should produce sandwich tubes that agree to within the kernel
    remainders. This is a cross-check between the two providers."""
    domain = (-0.5, 0.5)
    xs_eval = np.linspace(*domain, 200)
    cheb = oa.outer_approximation(jnp.exp, domain, "chebyshev", degree=8, n_slopes=16)
    tayl = oa.outer_approximation(jnp.exp, domain, "taylor", degree=8, n_slopes=16)
    cheb_w = cheb.evaluate_upper(xs_eval) - cheb.evaluate_lower(xs_eval)
    tayl_w = tayl.evaluate_upper(xs_eval) - tayl.evaluate_lower(xs_eval)
    assert float(np.max(np.abs(cheb_w - tayl_w))) < 1e-2
