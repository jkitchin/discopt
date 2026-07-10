"""#85 regression: the failure-triggered dense retry keeps the certificate when
the density-aware LU route is ON (gate probe: nvs21).

Before #85, ``DISCOPT_LU_DENSITY_ROUTE=1`` lost the ``optimal`` certificate on
nvs21: a small class of node LPs fails (``Numerical``/``IterLimit``) on the
sparse route where the historical dense-preferring route succeeds (39 vs 12
failing solves); the failing node keeps its inherited loose lifted-McCormick
relaxation bound, the final dual bound sticks at that value (−15 901 749 vs the
true −5.68522), and the solve returns ``feasible`` — sound (the bound stays
valid) but uncertified. A factorization-time conditioning gate was falsified as
the discriminator (docs/dev/performance-plan.md §9: inverted populations — the
failing solves factorize κ₁≈16 bases while healthy instances succeed at
κ₁ 1e10–1e16), so the fix is failure-triggered: on such a failure the LP is
re-solved once, cold, with the route suppressed (``dense_retry`` in
``crates/discopt-core/src/lp/simplex/primal.rs``).

This test asserts the certificate is retained with the flag ON. It fails before
the #85 change (status ``feasible``, bound −1.59e7) and passes after (status
``optimal``, bound ≈ −5.68522). nvs21 is a gate probe for the failure *class*
(sparse-route LP failure → abandoned node → stuck bound); the class-level
mechanics are unit-tested in Rust (``dense_retry_suppression_scopes_and_restores``
and the ``route_dense_decision`` tests in ``linsolve.rs``).

Marked ``slow`` (a full ~25 s B&B solve); run with ``-m slow``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

_NL = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "nvs21.nl")

# BARON-confirmed optimum (minlplib.solu =opt=). Minimization.
_NVS21_OPT = -5.6847825
# Solver-native tolerances (conftest house values).
_ABS = 1e-4
_REL = 1e-3


@pytest.mark.slow
def test_nvs21_density_route_keeps_certificate(monkeypatch):
    """Route ON: nvs21 must certify ``optimal`` with a closed dual bound."""
    monkeypatch.setenv("DISCOPT_LU_DENSITY_ROUTE", "1")
    result = from_nl(_NL).solve(deterministic=True, max_nodes=200_000)

    # The pre-#85 failure mode: status "feasible" with the dual bound stuck at
    # the root relaxation value (−1.59e7). The certificate must be retained.
    assert result.status == "optimal", (
        f"certificate lost with DISCOPT_LU_DENSITY_ROUTE=1: status={result.status!r}, "
        f"bound={getattr(result, 'bound', None)!r} (pre-#85 signature: feasible / −1.59e7)"
    )

    # The certified optimum must sit at the true optimum (no false-optimal).
    assert result.objective is not None
    err = abs(result.objective - _NVS21_OPT)
    assert err <= _ABS or err <= _REL * abs(_NVS21_OPT), (
        f"objective {result.objective!r} vs oracle {_NVS21_OPT}"
    )

    # The dual bound must have closed to the optimum, not merely be valid-but-
    # loose: bound ≤ objective (min sense, validity) AND bound within tolerance
    # of the optimum (certification). A stuck −1.59e7 bound fails the second.
    bound = getattr(result, "bound", None)
    assert bound is not None
    assert bound <= result.objective + 1e-9, "certificate invariant: bound ≤ incumbent (min)"
    assert abs(bound - _NVS21_OPT) <= 1e-2, (
        f"dual bound {bound!r} did not close to the optimum {_NVS21_OPT} "
        f"(pre-#85 signature: −15901749)"
    )
