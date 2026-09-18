"""Issue #1230: the Rust Neumaier--Shcherbina safe bound must not exceed the optimum.

``lp/simplex/refine.rs``'s ``ns_safe_bound``/``ns_safe_bound_csc`` used to round
``bᵀy`` to ``f64``, sum the reduced-cost box terms in plain ``f64``, and subtract no
margin. The #1230 step-1 entry experiment
(``scratchpad/ns_safe_bound_margin.py``, 2026-09-18) measured the consequence: the
bound came out **above** the true optimum on 206 of 980 exactly verified LPs. That
function certifies the default pure-LP route (``lp_milp_highs.ns_bound``) and the
MINLP node kernels (``bnb/spatial_kernel.rs``, ``bnb/convex_kernel.rs``), so the
overshoot was a certificate defect on the default path.

The fix accumulates the whole sum in double-double and subtracts
``refine::NS_MARGIN_REL * (1 + |bᵀy| + Σ|contrib|)``. These tests pin it from the
Python side, where the HiGHS LP route actually calls it.

``DISCOPT_NS_MARGIN=0`` restores the pre-fix evaluation bit-exactly; the flag is read
once per process in Rust, so it is not exercised here (``refine.rs``'s
``issue_1230_legacy_arm_is_preserved_and_is_the_unsound_one`` covers that arm).
"""

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.unit

# The m=4, n=6 LP the step-1 experiment found. Its certifying point was rebuilt over
# the rationals from HiGHS's optimal basis and checked to satisfy all four rows
# *exactly* and to lie in the box *exactly*; ``P_STAR`` is its ``cᵀx`` rounded up, so
# ``P_STAR >= p*`` rigorously and any bound above it is unsound beyond argument.
_Y = [0.06989471698387341, 0.006361153811200572, -1.7243770567161898, 0.2606276336654449]
_C = [
    -5.353293216640445, 0.20617899781232157, -1.2328800557307225,
    2.145883491350228, 0.5722333119416425, 1.2868813860462676,
]  # fmt: skip
_B = [-61.81027684222802, -1395.7897538919822, -1120.5897001915187, -54.096524739668176]
_L = [
    -65.55010680415802, -70.05155575358589, -7.3292205623746165,
    -47.44750931832471, -6.029803960416979, -0.5839316622544222,
]  # fmt: skip
_U = [
    40.01589393093027, 56.357692793905514, 98.28844922141533,
    15.536662598968787, 94.4803442713072, 20.214115334960823,
]  # fmt: skip
_A = [
    [-0.43380007896333744, 1.7840189339049422, -1.4087196089500127, 0.0,
     0.059990290399046416, 3.3723798353481578],
    [43.98676708984297, -0.2013242034249084, -9.25072957527505, 0.0,
     89.29831355000859, -9.17843384442963],
    [3.2555200005351654, 0.0, -9.967630367743432, 0.6251591107181199, 0.0,
     -1.4056139397219884],
    [0.04206968554963261, 0.3175647662262486, 2.647399050106419, 12.369730229020943,
     0.0, 2.985435987614364],
]  # fmt: skip
P_STAR = -0.1849509396488496
#: What the pre-#1230 evaluation returned on this LP: 13042 ulp above ``P_STAR``.
LEGACY_G = -0.18495093964848763


def _std_form():
    from discopt.solvers.lp_milp_highs import StdForm

    return StdForm.from_arrays(_C, sp.csc_matrix(np.array(_A)), _B, _L, _U)


def test_rust_ns_bound_does_not_exceed_the_verified_optimum():
    """The #1230 violator: the bound must land at or below the exact ``p*``."""
    from discopt.solvers.lp_milp_highs import ns_bound

    g = ns_bound(np.array(_Y, dtype=np.float64), _std_form())
    assert g is not None, "the bound must still be certifiable, not abstained away"
    assert g <= P_STAR, f"UNSOUND: bound {g!r} exceeds the verified optimum {P_STAR!r}"
    # The fixture is only a regression test while the old value really was unsound.
    assert LEGACY_G > P_STAR


def test_the_margin_costs_far_less_than_the_certification_threshold():
    """Sound but not useless: the margin must stay well under ``CERT_REL``.

    The constant comes from the Rust module, not a copy, so retuning
    ``refine.rs::NS_MARGIN_REL`` past the certification budget fails here.
    """
    from discopt._rust import NS_MARGIN_REL
    from discopt.solvers.lp_milp_highs import CERT_ABS, CERT_REL, ns_bound

    assert NS_MARGIN_REL < CERT_REL, "the margin must cost less than the certification budget"
    g = ns_bound(np.array(_Y, dtype=np.float64), _std_form())
    assert P_STAR - g <= CERT_ABS + CERT_REL * abs(P_STAR), (
        f"the margin cost {P_STAR - g:.3e} eats the certification budget"
    )


def test_bound_stays_at_or_below_the_exact_evaluation_of_the_same_formula():
    """``exact_ns_bound`` evaluates the same weak-duality formula over the rationals,
    so it is ``<= p*`` for any dual. The float bound must not sit above it."""
    from discopt.solvers.lp_milp_highs import exact_ns_bound, ns_bound

    sf = _std_form()
    y = np.array(_Y, dtype=np.float64)
    g = ns_bound(y, sf)
    g_exact, why = exact_ns_bound(y, sf)
    assert g_exact is not None, f"exact oracle abstained: {why}"
    assert Fraction(g) <= Fraction(g_exact), (
        f"float bound {g!r} sits above the exact evaluation {g_exact!r}"
    )
