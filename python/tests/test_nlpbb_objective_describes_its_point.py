"""The reported objective must be the model's objective at the reported point.

``_solve_nlp_bb`` took ``objective`` from an NLP solve's own converged value at
every site that set it (the tree's incumbent, ``_ref_obj`` from the refine,
``nlp_unscaled.objective`` from the unscaled retry) -- never from the model's
objective at the point actually leaving. The integer snap above those sites does
not touch it at all: it replaces ``sol_flat`` with the rounded point and leaves
the number describing the unrounded one. On this path that same number is also
published as the dual bound.

Measured over the 66-instance in-repo corpus (both routes, 110 pairs carrying
both a point and an objective), identical to every digit across two independent
audit runs:

    nvs07   nlpbb  reported=3.9992714251815236  f(x)=4.000005455142363  -7.34e-04  cert=True
    tspn12  nlpbb  reported=262.64738262946435  f(x)=262.64739525668534 -1.26e-05
    tspn10  nlpbb  reported=225.12606566548436  f(x)=225.12607139977277 -5.73e-06
    tspn08  nlpbb  reported=290.5668500072199   f(x)=290.56685375255876 -3.75e-06
    st_e36  nlpbb  reported=-246.00000007079294 f(x)=-246.0000015571123 +1.49e-06

All five are on the NLP-BB route and all five are OPTIMISTIC -- the reported
value is better than the point achieves.

The corpus trigger is search-path dependent (it reproduces exactly within the
audit's 66-instance sequence but not from a clean single-instance process), so
the test below forces the mechanism instead of the sequence: it makes the
integer snap return a materially different point and asserts the reported number
follows the point rather than the solve.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt._relax.nlp_evaluator import NLPEvaluator


@pytest.fixture(autouse=True)
def _force_nlp_bb(monkeypatch):
    """``nlp_bb=True`` alone does NOT reach ``_solve_nlp_bb``.

    The native convex kernel is consulted first and claims this model, so the
    solve returns with ``result.nlp_bb is False`` and the code under test never
    runs. Measured while writing these tests: the snap hook fired 0 times and
    the invariant assertion passed on the kernel's result instead -- a test that
    reads exactly like a passing one (CLAUDE.md Sec.6). Every test here pins the
    route, and asserts it was reached.
    """
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "0")


def _convex_minlp_for_nlpbb() -> dm.Model:
    """Convex, integer-bearing, and routed to NLP-BB by ``nlp_bb=True``."""
    m = dm.Model("objpoint")
    x = m.continuous("x", lb=0.0, ub=6.0)
    y = m.integer("y", lb=0, ub=4)
    z = m.continuous("z", lb=0.0, ub=60.0)
    m.minimize(z + 2.0 * y + 0.5 * x)
    m.subject_to((x - 3.0) ** 2 <= z)
    m.subject_to(x + y >= 3)
    return m


def _objective_at(model: dm.Model, x_dict) -> float:
    ev = NLPEvaluator(model)
    flat: list[float] = []
    for v in model._variables:
        flat.extend(np.atleast_1d(np.asarray(x_dict[v.name], float)).ravel().tolist())
    f_int = float(ev.evaluate_objective(np.asarray(flat, float)))
    return -f_int if getattr(ev, "_negate", False) else f_int


def test_reported_objective_is_the_objective_at_the_reported_point():
    """The invariant, on an ordinary NLP-BB solve."""
    m = _convex_minlp_for_nlpbb()
    r = m.solve(time_limit=30, nlp_bb=True)
    assert r.nlp_bb is True, "did not reach the NLP-BB route — nothing was tested"
    assert r.objective is not None and r.x is not None, "no incumbent — nothing was tested"
    f = _objective_at(_convex_minlp_for_nlpbb(), r.x)
    assert r.objective == pytest.approx(f, abs=1e-9, rel=1e-12), (
        f"reported objective {r.objective!r} != f(x) = {f!r} (delta {r.objective - f})"
    )


def test_a_snap_that_moves_the_point_moves_the_number_with_it(monkeypatch):
    """Force the mechanism: make the integer snap return a DIFFERENT point.

    Pre-fix the solve reported the pre-snap solve's value beside the post-snap
    point. The perturbation is a whole integer step, so the mismatch is far
    outside any tolerance and cannot be confused with numerical noise.
    """
    original = S._round_incumbent_integers
    fired = {"n": 0}

    def shifting_round(sol_flat, int_offsets, int_sizes):
        rounded, feas = original(sol_flat, int_offsets, int_sizes)
        if int_offsets:
            moved = np.asarray(rounded, float).copy()
            off = int(int_offsets[0])
            # step the first integer column UP by one where the box allows it,
            # which keeps `x + y >= 3` satisfied (y only grows) and changes the
            # objective by a clean +2.0
            moved[off] = float(moved[off]) + 1.0
            fired["n"] += 1
            return moved, feas
        return rounded, feas

    monkeypatch.setattr(S, "_round_incumbent_integers", shifting_round)

    m = _convex_minlp_for_nlpbb()
    r = m.solve(time_limit=30, nlp_bb=True)
    assert r.nlp_bb is True, "did not reach the NLP-BB route — nothing was tested"
    assert fired["n"] > 0, "the snap hook never fired — the mechanism was not exercised"
    assert r.objective is not None and r.x is not None, "no incumbent — nothing was tested"

    f = _objective_at(_convex_minlp_for_nlpbb(), r.x)
    assert r.objective == pytest.approx(f, abs=1e-9, rel=1e-12), (
        f"reported objective {r.objective!r} describes a different point than the one "
        f"returned, whose objective is {f!r} (delta {r.objective - f})"
    )


def test_the_corrected_objective_reaches_the_certificate_test(monkeypatch):
    """A corrected objective must not leave a certificate its own numbers deny.

    Correcting ``objective`` after the tree computed its gap can open a gap the
    tree believed closed. ``_withhold_stale_certificate`` is downstream of the
    correction precisely so that case is withdrawn rather than republished.
    """
    original = S._round_incumbent_integers

    def shifting_round(sol_flat, int_offsets, int_sizes):
        rounded, feas = original(sol_flat, int_offsets, int_sizes)
        if int_offsets:
            moved = np.asarray(rounded, float).copy()
            off = int(int_offsets[0])
            moved[off] = float(moved[off]) + 1.0
            return moved, feas
        return rounded, feas

    monkeypatch.setattr(S, "_round_incumbent_integers", shifting_round)
    r = _convex_minlp_for_nlpbb().solve(time_limit=30, nlp_bb=True)
    assert r.nlp_bb is True, "did not reach the NLP-BB route — nothing was tested"
    if r.objective is None or r.bound is None:
        pytest.skip("no (objective, bound) pair to judge")
    # minimise: a dual bound may never sit above the incumbent it is reported with
    assert r.bound <= r.objective + 1e-6 * max(1.0, abs(r.objective)), (
        f"bound {r.bound} is above the incumbent {r.objective} it is reported against"
    )
    if r.gap_certified:
        gap = abs(r.objective - r.bound) / max(1.0, abs(r.objective))
        assert gap <= 1e-4 + 1e-9, f"certified at a real gap of {gap}"
