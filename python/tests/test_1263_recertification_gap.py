"""#1263: ``feasible -> optimal`` re-certification uses the real gap test.

Three sites (``solve_model``, ``_solve_nlp_bb``, ``_solve_miqp_bb``) granted
``optimal`` / ``gap_certified=True`` when ``|obj - bound| / max(1, |obj|)`` met
``gap_tolerance``. The 1.0 floor makes the relative tolerance an *absolute*
one below magnitude 1 -- the degeneration ``_DEFAULT_ABS_GAP_TOL`` exists to
correct -- so those sites certified results the convergence test
(``_gap_values_converged``) and the reporting function (``_gap_criterion``)
both reject. They now share :func:`_recertify_gap_closed`.

Measured before the fix (169 instances at ``time_limit=20``): seven results
were certified ``optimal`` with no ``gap_criterion``, including ``st_z``
(obj 2.7e-5 over a true optimum of 0).
"""

from __future__ import annotations

import os

import pytest
from discopt.modeling.core import from_nl
from discopt.solver import _gap_criterion, _gap_values_converged, _recertify_gap_closed

REL, ABS = 1e-4, 1e-6
DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")


def _floored(obj, bound):
    """The formula the three sites used before #1263."""
    return abs(obj - bound) / max(1.0, abs(obj)) <= REL


@pytest.mark.parametrize(
    "obj, bound",
    [
        (0.1, 0.09995),  # true relative gap 5e-4, absolute 5e-5
        (3.7e-05, 0.0),  # the gear case _DEFAULT_ABS_GAP_TOL records
        (2.727e-05, -5.6e-10),  # st_z before the fix
    ],
)
def test_small_magnitude_open_gap_is_not_certified(obj, bound):
    assert _floored(obj, bound), "probe no longer exercises the floored formula"
    assert _recertify_gap_closed(obj, bound, False, REL, ABS) is False
    # Maximize mirror: the bound is an upper bound.
    assert _recertify_gap_closed(-obj, -bound, True, REL, ABS) is False
    assert _gap_criterion(obj, bound, REL, ABS) is None


@pytest.mark.parametrize(
    "obj, bound, criterion",
    [
        (100.0, 99.995, "relative"),  # |obj| > 1: the two formulas agree
        (4e-12, 0.0, "absolute"),  # a genuinely zero optimum still certifies
        (0.1, 0.1 - 5e-7, "absolute"),
    ],
)
def test_closed_gap_still_certifies(obj, bound, criterion):
    assert _recertify_gap_closed(obj, bound, False, REL, ABS) is True
    assert _recertify_gap_closed(-obj, -bound, True, REL, ABS) is True
    assert _gap_criterion(obj, bound, REL, ABS) == criterion


def test_predicate_is_the_convergence_test():
    """Same arithmetic as the search's stopping test, sense-ordered."""
    n = 0
    for obj, bound in [(0.1, 0.09995), (5.0, 4.9999), (-0.3, -0.30001), (1e-9, 0.0)]:
        assert _recertify_gap_closed(obj, bound, False, REL, ABS) == _gap_values_converged(
            obj, bound, REL, ABS
        )
        assert _recertify_gap_closed(obj, bound, True, REL, ABS) == _gap_values_converged(
            bound, obj, REL, ABS
        )
        n += 1
    assert n == 4


# --- End to end -------------------------------------------------------------
# The three Python sites were not the only certifiers using the floored test:
# the native spatial kernel's purely absolute ``gap_tol`` (= ``gap_tolerance``)
# and the convex-MINLP OA route's ``denom_floor=1.0`` did the same. Measured
# before the fix (``time_limit=20``): st_z certified at 2.727e-5 (opt 0), and
# portfol_roundlot at 0.028383 (best known 0.0282906, a 0.33% miss); neither
# named a ``gap_criterion``. The oracle values are from ``minlplib.solu``.
#
# portfol_roundlot names its route. On the default path the #1059 auto-route
# hands over to NLP-BB when OA has not certified within its share of the budget.
# That happens on a loaded CI runner, and which certifier answers then depends on
# timing. Pinning OA keeps this test on the certifier it covers. The NLP-BB
# fallback used to raise at its exit gate here; test_nlpbb_unscaled_refine.py
# covers that.
@pytest.mark.parametrize(
    "name, opt, kwargs",
    [
        ("st_z", 0.0, {}),  # native spatial kernel
        ("mathopt5_8", -0.6860722798, {}),  # native spatial kernel
        ("portfol_roundlot", 0.0282906349, {"solver": "mip-nlp", "mip_nlp_method": "oa"}),
    ],
)
def test_certified_result_meets_the_gap_it_names(name, opt, kwargs):
    r = from_nl(os.path.join(DATA, f"{name}.nl")).solve(time_limit=60, **kwargs)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.gap_certified)
    crit = (r.solver_stats or {}).get("gap_criterion")
    assert crit in ("absolute", "relative"), (
        f"{name}: certified optimal with gap_criterion={crit!r} (#1263)"
    )
    assert _gap_values_converged(r.objective, r.bound, REL, ABS)
    # The certificate is honest against the oracle, not just self-consistent.
    assert _gap_values_converged(r.objective, opt, REL, ABS), (r.objective, opt)
