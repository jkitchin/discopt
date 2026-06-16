"""Soundness: ``gap_certified=True`` requires a finite dual bound.

A *certified* optimality gap is a claim that the reported dual (lower) bound is
a rigorous certificate. A non-finite bound (``±inf``) or an absent bound
(``None``) certifies nothing — e.g. a ``time_limit`` termination where no node
ever produced a finite relaxation bound leaves the global lower bound at
``-inf``. Reporting ``gap_certified=True`` there is a *false certification*: a
benchmark/phase gate that counts ``gap_certified`` as "solved" would miscount
the instance.

This surfaced while triaging issue #139 (bucket 2, products with a nonlinear
factor): instances ``nvs20`` and ``st_e36`` drop their objective term from the
MILP relaxation (no valid bound), then on ``time_limit`` reported
``status=time_limit obj=None bound=-inf gap=inf gap_certified=True``.

The guard lives in ``SolveResult.__post_init__`` so every construction path is
covered. Infeasibility certificates are exempt: ``status="infeasible"`` with
``gap_certified=True`` certifies infeasibility, not a gap, and legitimately
carries ``bound=None``.
"""

from __future__ import annotations

import math

import pytest
from discopt.modeling.core import SolveResult


class TestCertifiedGapRequiresFiniteBound:
    """``__post_init__`` downgrades a certification with no finite bound."""

    @pytest.mark.parametrize("bad_bound", [None, float("-inf"), float("inf"), float("nan")])
    def test_non_finite_bound_cannot_be_certified(self, bad_bound):
        r = SolveResult(
            status="time_limit",
            objective=None,
            bound=bad_bound,
            gap=float("inf"),
            gap_certified=True,
        )
        assert r.gap_certified is False
        assert r.bound is None
        assert r.gap is None

    def test_finite_bound_certification_is_preserved(self):
        # A genuine certified lower bound on an open (feasible) problem must
        # survive: 87.35 is a rigorous, if weak, bound below the incumbent.
        r = SolveResult(
            status="feasible",
            objective=258.96,
            bound=87.35,
            gap=(258.96 - 87.35) / 258.96,
            gap_certified=True,
        )
        assert r.gap_certified is True
        assert r.bound == pytest.approx(87.35)
        assert math.isfinite(r.bound)

    def test_optimal_with_matching_bound_is_preserved(self):
        r = SolveResult(
            status="optimal",
            objective=31.0,
            bound=31.0,
            gap=0.0,
            gap_certified=True,
        )
        assert r.gap_certified is True
        assert r.bound == pytest.approx(31.0)

    def test_finite_zero_bound_is_a_valid_certificate(self):
        # bound == 0.0 is finite (a weak but sound lower bound, e.g. ex1252):
        # the guard must NOT mistake a falsy-but-finite value for "no bound".
        r = SolveResult(
            status="time_limit",
            objective=None,
            bound=0.0,
            gap=float("inf"),
            gap_certified=True,
        )
        assert r.gap_certified is True
        assert r.bound == pytest.approx(0.0)

    def test_infeasibility_certificate_is_exempt(self):
        # An infeasibility certificate certifies infeasibility, not a gap, and
        # legitimately carries no bound.
        r = SolveResult(status="infeasible", bound=None, gap_certified=True)
        assert r.gap_certified is True
        assert r.bound is None

    def test_uncertified_result_is_untouched(self):
        r = SolveResult(
            status="time_limit",
            objective=None,
            bound=float("-inf"),
            gap=float("inf"),
            gap_certified=False,
        )
        assert r.gap_certified is False
