"""#1673 B4: a false convexity certificate turned into a false ``infeasible``.

``min (x0+4)^2 + (x2-2)^2`` s.t. ``x0**-3.5 - x2**2 <= 0``, ``x0 in [1e-3, 200]``,
``x2`` integer in ``[0, 200]``; optimum 21.555127 at ``x2 = 3`` (see
``test_fp_relaxation_soundness.py``). Two defects, both fixed here:

1. The interval-Hessian certificate sized its Gershgorin slack by the WHOLE
   matrix's norm. The ``x0`` row reaches 5e17 on this box, the slack became
   1769, and the ``x2`` row's exact ``-2`` (the concave ``-x2**2``) passed as
   PSD. The row was certified convex and the model was routed to LP/NLP-BB.
2. LP/NLP-BB then read an empty master -- emptied by the invalid OA cuts of
   ``-x2**2`` -- as an infeasibility proof, and reported ``infeasible`` while
   also returning an incumbent of 54.53.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import certificate as CERT
from discopt._relax.convexity.certificate import certify_convex
from discopt._relax.convexity.eigenvalue import (
    gershgorin_certifies_nsd,
    gershgorin_certifies_psd,
    gershgorin_lambda_min,
    interval_magnitude,
    psd_decision_slack,
)
from discopt._relax.convexity.interval import Interval

_OPT = 21.555127


def _build():
    m = dm.Model("fp_b4")
    x0 = m.continuous("x0", lb=0.001, ub=200.0)
    x2 = m.integer("x2", lb=0, ub=200)
    m.minimize((x0 + 4) ** 2 + (x2 - 2) ** 2)
    m.subject_to(x0 ** (-3.5) - x2**2 <= 0)
    return m


def test_the_concave_row_is_not_certified_convex():
    m = _build()
    body = m._constraints[0].body
    assert certify_convex(body, m) is None


def test_a_huge_row_does_not_license_another_rows_negative_diagonal():
    h = np.diag([5e17, -2.0])
    H = Interval(h.copy(), h.copy())
    # The pre-fix gate: matrix-wide slack.
    assert gershgorin_lambda_min(H) >= -psd_decision_slack(interval_magnitude(H))
    assert not gershgorin_certifies_psd(H)
    assert not gershgorin_certifies_nsd(H)
    # A genuinely PSD matrix at the same spread still certifies.
    h_psd = np.diag([5e17, 2.0])
    assert gershgorin_certifies_psd(Interval(h_psd.copy(), h_psd.copy()))


def test_the_default_route_certifies_the_true_optimum():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _build().solve(time_limit=30, gap_tolerance=1e-4)
    assert r.status == "optimal", (r.status, r.objective, r.bound, r.algorithm_route)
    assert r.objective == pytest.approx(_OPT, abs=1e-4)
    assert r.bound <= r.objective + 1e-4


def _matrix_wide_psd(H, floor=None):
    """The certificate's pre-fix Gershgorin verdict, to recreate the misroute."""
    slack = psd_decision_slack(interval_magnitude(H))
    if floor is not None:
        slack = max(slack, float(floor))
    return gershgorin_lambda_min(H) >= -slack


def _misclassify(monkeypatch, m):
    monkeypatch.setattr(CERT, "gershgorin_certifies_psd", _matrix_wide_psd)
    # Probe fired (CLAUDE.md §6): the misclassification is actually in effect.
    assert certify_convex(m._constraints[0].body, m) is not None


def test_lp_nlp_bb_does_not_report_infeasible_beside_an_incumbent(monkeypatch):
    """The explicit ``lp_nlp_bb`` path, fed the pre-fix certificate: its invalid
    OA cuts empty the HiGHS master after a feasible point was found. Before the
    fix the run said ``infeasible`` and returned that point (54.53)."""
    m = _build()
    _misclassify(monkeypatch, m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(
            solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="highs", time_limit=30
        )
    assert r.objective is not None
    assert r.status == "feasible", (r.status, r.objective, r.bound)
    assert not r.gap_certified


def test_the_default_route_falls_back_on_a_misclassified_model(monkeypatch):
    """Defence in depth: even with the false certificate, the route's uncertified
    exit hands the model to the default path, which certifies the optimum."""
    m = _build()
    _misclassify(monkeypatch, m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=30, gap_tolerance=1e-4)
    assert r.algorithm_route.startswith("mip-nlp/lp_nlp_bb"), r.algorithm_route
    assert "fell back" in r.algorithm_route, r.algorithm_route
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(_OPT, abs=1e-4)
