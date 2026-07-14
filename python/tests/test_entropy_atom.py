"""Univariate entropy atom (issue #632 centropy-tightness), gated DISCOPT_ENTROPY_ATOM.

``x*log(x)`` is convex; the factorable path shatters it into a loose decoupled
bilinear. The gated atom emits its exact convex envelope (tangent under + secant
over). These tests pin: (1) SOUND — the relaxation lower bound never exceeds the
true box minimum on any box (a cut feasible point would make it falsely tight);
(2) NEVER LOOSER, and strictly TIGHTER where the convexity bites; (3) byte-identical
when the flag is off, and on a non-entropy model when on.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

import discopt.modeling as dm
from discopt import Model
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms

# Frozen pre-atom (flag-off) bound for the [0.1,1]^2 entropy sum — the decoupled
# bilinear floor. Pins that flag-off behavior is unchanged by this PR.
_OFF_REF_01_1 = -2.3025850929940463


def _entropy_root_lb(lo: float, hi: float, n: int = 2) -> float:
    m = Model("ent")
    x = m.continuous("x", shape=n, lb=lo, ub=hi)
    m.minimize(dm.sum(x[i] * dm.log(x[i]) for i in range(n)))
    terms = classify_nonlinear_terms(m)
    milp, _ = build_milp_relaxation(m, terms, None, incumbent=None)
    return float(milp.solve().objective)


def _true_min(lo: float, hi: float, n: int = 2) -> float:
    """True min of ``sum_i x_i log x_i`` on the box (separable; each at 1/e or a corner)."""
    cands = [lo, hi] + ([1.0 / math.e] if lo <= 1.0 / math.e <= hi else [])
    return n * min(t * math.log(t) for t in cands)


@pytest.fixture
def _flag_on():
    prev = os.environ.get("DISCOPT_ENTROPY_ATOM")
    os.environ["DISCOPT_ENTROPY_ATOM"] = "1"
    yield
    if prev is None:
        os.environ.pop("DISCOPT_ENTROPY_ATOM", None)
    else:
        os.environ["DISCOPT_ENTROPY_ATOM"] = prev


# Boxes where x*log(x)'s curvature bites (interior/near-interior min) vs a corner box.
_CURVATURE_BOXES = [(0.1, 1.0), (0.05, 2.0), (0.5, 5.0), (0.2, 0.9)]
_ALL_BOXES = _CURVATURE_BOXES + [(1.0, 10.0)]


@pytest.mark.parametrize("lo,hi", _ALL_BOXES)
def test_entropy_atom_sound_and_never_looser(_flag_on, lo, hi):
    """ON is a VALID lower bound (<= true min) and NEVER looser than OFF, on every box."""
    on = _entropy_root_lb(lo, hi)
    os.environ.pop("DISCOPT_ENTROPY_ATOM", None)
    off = _entropy_root_lb(lo, hi)
    os.environ["DISCOPT_ENTROPY_ATOM"] = "1"
    tm = _true_min(lo, hi)
    assert on <= tm + 1e-6, f"UNSOUND: ON {on} > true min {tm} on [{lo},{hi}]"
    assert off <= tm + 1e-6, f"UNSOUND (off): {off} > {tm} on [{lo},{hi}]"
    assert on >= off - 1e-9, f"LOOSER than off on [{lo},{hi}]: ON={on} OFF={off}"


@pytest.mark.parametrize("lo,hi", _CURVATURE_BOXES)
def test_entropy_atom_strictly_tighter_where_curvature_bites(_flag_on, lo, hi):
    """Where the min is not at a corner, the convex envelope beats the bilinear floor."""
    on = _entropy_root_lb(lo, hi)
    os.environ.pop("DISCOPT_ENTROPY_ATOM", None)
    off = _entropy_root_lb(lo, hi)
    os.environ["DISCOPT_ENTROPY_ATOM"] = "1"
    assert on > off + 1e-2, f"not tighter on [{lo},{hi}]: ON={on} OFF={off}"


def test_entropy_atom_off_is_unchanged():
    """Flag OFF reproduces the pre-atom bilinear relaxation exactly (deterministic + frozen)."""
    os.environ.pop("DISCOPT_ENTROPY_ATOM", None)
    off1 = _entropy_root_lb(0.1, 1.0)
    off2 = _entropy_root_lb(0.1, 1.0)
    assert off1 == off2
    assert abs(off1 - _OFF_REF_01_1) < 1e-9, f"flag-off behavior changed: {off1} vs {_OFF_REF_01_1}"


def test_non_entropy_product_unaffected_when_on(_flag_on):
    """A bare bilinear x*y is NOT the entropy pattern: ON == OFF exactly."""

    def _bilinear_lb():
        m = Model("bil")
        x = m.continuous("x", lb=0.0, ub=1.0)
        y = m.continuous("y", lb=0.0, ub=1.0)
        m.minimize(x * y)
        terms = classify_nonlinear_terms(m)
        milp, _ = build_milp_relaxation(m, terms, None, incumbent=None)
        return float(milp.solve().objective)

    on = _bilinear_lb()
    os.environ.pop("DISCOPT_ENTROPY_ATOM", None)
    off = _bilinear_lb()
    os.environ["DISCOPT_ENTROPY_ATOM"] = "1"
    assert on == off, f"entropy flag changed a non-entropy product: {on} vs {off}"


def test_entropy_full_solve_still_proves(_flag_on):
    """End-to-end: the entropy model still solves to the true optimum (sound)."""
    m = Model("ent_solve")
    x = m.continuous("x", shape=2, lb=0.1, ub=1.0)
    m.minimize(dm.sum(x[i] * dm.log(x[i]) for i in range(2)))
    r = m.solve(time_limit=20)
    assert r.status == "optimal"
    assert r.objective is not None
    assert abs(r.objective - 2.0 * (-np.exp(-1.0))) < 1e-3
