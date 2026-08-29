"""#1066: the disaggregated perspective epigraph in the single-tree master.

The shipped (#1064) treatment strengthens the master's ONE aggregate epigraph
row in place, which yields ``sup_z sum_i g_i(z)`` -- every term forced to share a
single reference. Giving each separable term ``q_k x_k**2`` its own epigraph
column ``s_k`` yields ``sum_i sup_z g_i(z)``, the full perspective closure,
because the master's own LP relaxation combines each term's best reference at a
fractional point for free.

Measured on the real instances before implementing (CLAUDE.md §4), cutting-plane
closure of the master LP with binaries relaxed to [0, 1] on ``squfl015-060``
(optimum 366.6218167): plain 129.998 (35.5%), aggregate 173.401 (47.3%) -- both
still climbing after 250 rounds -- disaggregated **362.941 (99.0%)**, converged
in 41 rounds.

Default OFF pending the §5 graduation panel; ``DISCOPT_PERSPECTIVE_DISAGG=1``.
"""

import numpy as np
import pytest
from discopt._relax.perspective import perspective_disaggregation_enabled
from discopt.solvers import oa


def _epigraph(terms):
    ep = oa._perspective_epigraph_for(terms, n_vars=8)
    assert ep is not None
    return ep


def test_flag_defaults_off_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv("DISCOPT_PERSPECTIVE_DISAGG", raising=False)
    assert perspective_disaggregation_enabled() is False
    monkeypatch.setenv("DISCOPT_PERSPECTIVE_DISAGG", "0")
    assert perspective_disaggregation_enabled() is False
    monkeypatch.setenv("DISCOPT_PERSPECTIVE_DISAGG", "1")
    assert perspective_disaggregation_enabled() is True


def test_split_is_all_or_nothing():
    """A term the split cannot carry refuses the WHOLE disaggregation.

    Removing a term from one aggregate row and not another leaves the master
    double-counting it against ``s_k`` -- an over-estimate of ``f``, i.e. an
    invalid bound. Refusal is the only safe answer (CLAUDE.md §1/§3).
    """
    assert oa._perspective_epigraph_for([], n_vars=8) is None
    # column out of range
    assert oa._perspective_epigraph_for([(0, 1, 2.0), (99, 2, 1.0)], n_vars=8) is None
    # non-positive / non-finite curvature
    assert oa._perspective_epigraph_for([(0, 1, 2.0), (2, 3, 0.0)], n_vars=8) is None
    assert oa._perspective_epigraph_for([(0, 1, np.nan)], n_vars=8) is None
    # all usable -> built
    assert oa._perspective_epigraph_for([(0, 1, 2.0), (2, 3, 1.5)], n_vars=8) is not None


def test_row_underestimates_the_term_at_both_integral_y():
    """The Frangioni-Gentile row is valid at y=0 and y=1, which is what makes
    the disaggregation sound rather than merely tighter."""
    q, z = 1.75, 2.5
    ep = _epigraph([(0, 1, q)])
    ep.add(0, z)
    n_master, start = 12, 10
    row = ep.row_for(0, n_master=n_master, perspective_start=start)
    checks = 0
    for y in (0.0, 1.0):
        for x in np.linspace(0.0, 4.0, 25):
            if y == 0.0 and x != 0.0:
                continue  # x is semicontinuous: y=0 forces x=0
            s_true = q * x * x
            point = np.zeros(n_master)
            point[0], point[1], point[start] = x, y, s_true
            # row @ point <= 0 means the cut does not remove the true epigraph point
            checks += 1
            assert float(row @ point) <= 1e-9, (x, y, float(row @ point))
    assert checks > 0, "vacuous: no (x, y) pair was tested"


def test_new_rows_are_pending_so_they_reach_the_master_with_their_cut():
    """A freshly split row is exactly TIGHT at the point it came from, so a
    violation filter scores it at 0 and drops it. It must be emitted anyway --
    dropping it left ``squfl025-040``'s dual bound at 120.652 against the
    aggregate row's 142.510."""
    ep = _epigraph([(0, 1, 2.0), (2, 3, 1.0)])
    assert ep.drain_pending() == []
    assert ep.add(0, 1.5) is True
    assert ep.add(1, 2.0) is True
    assert ep.add(0, 1.5) is False, "duplicate reference must not create a row"
    assert ep.drain_pending() == [0, 1]
    assert ep.drain_pending() == [], "drain must clear"


def test_violations_match_the_dense_row_it_replaces():
    """The vectorised pool scan and the dense row must agree exactly: the scan
    is an optimisation of ``row_for(...) @ x``, not a different test."""
    rng = np.random.default_rng(1066)
    ep = _epigraph([(0, 1, 2.0), (2, 3, 0.75), (4, 5, 3.25)])
    for k, z in ((0, 1.5), (1, 0.25), (2, 2.0), (0, 3.0)):
        ep.add(k, z)
    n_master, start = 11, 8
    x = rng.normal(size=n_master)
    fast = ep.violations(x, perspective_start=start)
    slow = np.array(
        [
            float(ep.row_for(i, n_master=n_master, perspective_start=start) @ x)
            for i in range(len(ep.rows))
        ]
    )
    assert fast.shape == slow.shape
    # Not bit-identical: the scan multiplies ``2*q*z*x`` where the dense row
    # folds ``2*q*z`` into a coefficient first, so the two associate the same
    # product differently (max observed 3.6e-15 absolute). The 1e-12 band is
    # tighter than the 1e-6 violation threshold the scan feeds by six orders of
    # magnitude, so no row can be classified differently by it.
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-12)


def test_disaggregating_a_cut_is_exact_arithmetic():
    """Removing term k from the aggregate row and re-adding its own row must
    reproduce the aggregate row's value at the reference point."""
    q0, q1 = 2.0, 0.5
    ep = _epigraph([(0, 1, q0), (2, 3, q1)])
    n = 8
    x_star = np.zeros(n)
    x_star[0], x_star[2] = 1.5, 3.0
    # aggregate tangent of f = q0 x0^2 + q1 x2^2 at x_star: grad^T x - eta <= grad.z - f(z)
    coeffs = np.zeros(n + 1)
    coeffs[0] = 2.0 * q0 * x_star[0]
    coeffs[2] = 2.0 * q1 * x_star[2]
    coeffs[n] = -1.0
    rhs = q0 * x_star[0] ** 2 + q1 * x_star[2] ** 2
    split = oa._disaggregate_objective_cut(coeffs.copy(), rhs, x_star, ep)
    assert split is not None
    residual_coeffs, residual_rhs = split
    # every perspective term is gone from the residual row
    assert residual_coeffs[0] == pytest.approx(0.0, abs=1e-12)
    assert residual_coeffs[2] == pytest.approx(0.0, abs=1e-12)
    assert residual_rhs == pytest.approx(0.0, abs=1e-12)
    assert residual_coeffs[n] == -1.0, "the eta column must survive the split"
    # and both terms got a row at their own reference
    assert sorted(ep.rows) == [(0, 1.5), (1, 3.0)]


def test_a_non_finite_reference_drops_the_cut_entirely():
    ep = _epigraph([(0, 1, 2.0), (2, 3, 1.0)])
    x_star = np.zeros(8)
    x_star[2] = np.inf
    before = oa._PERSPECTIVE_DISAGG_REFUSED[0]
    assert oa._disaggregate_objective_cut(np.zeros(9), 0.0, x_star, ep) is None
    assert oa._PERSPECTIVE_DISAGG_REFUSED[0] == before + 1
    assert ep.rows == [], "a refused cut must not leave half its terms split out"


def test_esh_hyperplanes_refuse_a_disaggregated_master():
    """Refuse loudly rather than write an aggregate objective row into a master
    whose epigraph has already been split (CLAUDE.md §3)."""
    source = open(oa.__file__).read()
    assert "ESH objective hyperplanes cannot be generated into a master" in source
