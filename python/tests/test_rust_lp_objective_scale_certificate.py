"""The Rust LP route certifies optimality without accounting for objective scale.

``DISCOPT_LP_MILP_BACKEND=rust`` opts out of the #1229 HiGHS route for pure
LP/MILP models. On a model whose objective coefficients span many orders of
magnitude, that route returns ``status="optimal"`` with ``gap_certified=True`` on
a point that is materially suboptimal -- because its termination test is applied
in VARIABLE space, and a residual well inside any sane feasibility tolerance is
amplified by a large objective coefficient into a large objective error.

The family, solved at six ratios (``scale_sweep.py``, 24 executed comparisons)::

    min  C*x + (1/C)*y     s.t.  x + y >= 1,  x, y in [0, 1]

whose optimum is ``x=0, y=1`` with objective ``1/C`` for every ``C``:

    ==========  ==========  =======================  ===========
    C           HiGHS err   Rust err (continuous)    Rust err (int)
    ==========  ==========  =======================  ===========
    1e2         0.0         7.518e-09                0.0
    1e4         0.0         7.518e-07                0.0
    1e6         0.0         **7.518e-05**            0.0
    1e8         0.0         **2.728e-05**            0.0
    1e10        0.0         **2.666e-05**            0.0
    1e12        0.0         **1.331e-05**            0.0
    ==========  ==========  =======================  ===========

It crosses the repo's documented ``abs=1e-6`` tolerance between 1e4 and 1e6.
HiGHS is exact at every ratio, and the INTEGER variant is exact on both backends,
so this is specific to the continuous LP path and is not inherent to the model.

At ``C=1e12`` the route reports a certified objective of **-1.331e-05** for an
objective ``C*x + y/C`` with ``x, y >= 0``, which is provably NON-NEGATIVE on the
feasible box -- the reported value is not merely suboptimal, it is unattainable.
Mechanism at ``C=1e8``, reproduced identically 3/3: the route returns
``x=2.7275725368179037e-13`` instead of ``0``, and ``1e8 * 2.7e-13 = 2.7e-05``.

**Why this is xfail rather than fixed here.** The route's own engine is the
in-house Rust simplex, which is ALSO the default MINLP per-node LP engine, so
changing its termination test is bound-changing for every nonlinear solve and
needs the §5 graduation panel. The #1229 HiGHS route already carries the guard
this one lacks ("HiGHS's *labels* are never trusted for a certificate"), so the
principled fix is scale-aware termination or porting that verification -- not a
tolerance tweak, which CLAUDE.md §3 forbids. Tracked by the issue this test
names; the marker is ``strict`` so that whoever fixes it is forced to remove it.

The DEFAULT path is unaffected: ``DISCOPT_LP_MILP_BACKEND`` defaults to
``highs``, which is exact on every cell above.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest


def _scaled_model(C: float, *, integer: bool = False):
    """``min C*x + y/C`` s.t. ``x + y >= 1``; optimum ``x=0, y=1``, value ``1/C``."""
    m = dm.Model("objective_scale")
    if integer:
        x = m.integer("x", lb=0, ub=1)
        y = m.integer("y", lb=0, ub=1)
    else:
        x = m.continuous("x", lb=0.0, ub=1.0)
        y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(C * x + (1.0 / C) * y)
    m.subject_to(x + y >= 1.0)
    return m


@pytest.mark.parametrize("C", [1e6, 1e8, 1e10, 1e12])
@pytest.mark.xfail(
    strict=True,
    reason="Rust LP route certifies on a variable-space tolerance, ignoring "
    "objective scale; see this module's docstring for the measured family",
)
def test_the_rust_lp_route_certificate_respects_objective_scale(C, monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    r = _scaled_model(C).solve(time_limit=5.0, gap_tolerance=1e-4)
    assert r.gap_certified, "fixture invalid: this cell is supposed to CERTIFY"
    assert r.objective == pytest.approx(1.0 / C, abs=1e-6), (
        f"C={C:g}: certified {r.objective!r} against the true optimum {1.0 / C!r}"
    )


@pytest.mark.parametrize("C", [1e6, 1e8, 1e10, 1e12])
def test_the_default_highs_route_is_exact_at_every_scale(C, monkeypatch):
    """The control: the default route gets every one of these exactly right.

    Without it the xfail above would be indistinguishable from "this model is
    numerically impossible", which is the claim it has to rule out.
    """
    monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)
    r = _scaled_model(C).solve(time_limit=5.0, gap_tolerance=1e-4)
    assert r.objective == pytest.approx(1.0 / C, abs=1e-6), (
        f"C={C:g}: the DEFAULT route is also wrong ({r.objective!r} vs {1.0 / C!r}); "
        f"this is then not a backend-specific defect"
    )


@pytest.mark.parametrize("C", [1e8, 1e12])
def test_the_integer_variant_is_exact_on_both_backends(C, monkeypatch):
    """Scope control: the MILP path is unaffected, so the defect is the LP path."""
    for backend in ("highs", "rust"):
        monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
        r = _scaled_model(C, integer=True).solve(time_limit=5.0, gap_tolerance=1e-4)
        assert r.objective == pytest.approx(1.0 / C, abs=1e-6), (
            f"C={C:g} backend={backend}: integer variant is wrong ({r.objective!r})"
        )


@pytest.mark.parametrize("C", [1e12])
def test_the_reported_objective_is_at_least_attainable(C, monkeypatch):
    """The strongest form: ``C*x + y/C`` with ``x, y >= 0`` cannot be negative.

    A reported value below zero is not a suboptimal answer, it is one no feasible
    point attains. Separated from the approx-optimum test above so the record
    shows this is a distinct and worse symptom.
    """
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    r = _scaled_model(C).solve(time_limit=5.0, gap_tolerance=1e-4)
    if r.objective is None:
        pytest.skip("no incumbent to judge")
    pytest.xfail("reports a negative objective for a provably non-negative one")
