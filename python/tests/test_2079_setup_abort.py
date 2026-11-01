"""#2079 follow-up: a native GDP solve's *setup* gives up at its deadline.

After #2080 charged every step to ``time_limit``, the setup itself (model copy,
bound pretransform, big-M lowering, master build) still ran to completion before
the algorithm's first deadline check. On a 200-block x 100-column synthetic GDP
shaped like GDPlib ``stranded_gas``, ``time_limit=3`` returned after 12.2 s with
nothing; after this change it returns at about 3.1 s with ``status="time_limit"``.

The setup polls an abortable scope (``discopt._relax.deadline.abortable_setup``)
and raises :class:`SetupDeadlineExpired`, a ``BaseException`` so no
``except Exception`` on the way out can swallow it. Outside that scope every
checkpoint is a no-op, which is what keeps a solve without a deadline (or one
that finishes its setup in time) unchanged.

The same change replaced three quadratic setup costs with linear ones that read
the same data: a per-group slot box, a dense coefficient vector per row, and a
per-call variable scan in ``_column_indices``. The tests at the bottom pin each
against the code it replaced.
"""

from __future__ import annotations

import copy
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.deadline import (
    SetupDeadlineExpired,
    abortable_setup,
    setup_abortable,
    setup_checkpoint,
)

pytestmark = pytest.mark.smoke


def _blocks(n_blocks: int = 6, width: int = 5) -> dm.Model:
    """A GDP of ``n_blocks`` on/off blocks over one array (the stranded_gas shape)."""
    m = dm.Model("g2079s")
    x = m.continuous("x", shape=(n_blocks * width,), lb=0.0, ub=10.0)
    obj = 0
    for b in range(n_blocks):
        lo, hi = b * width, (b + 1) * width
        on = m.make_disjunct(f"on{b}")
        off = m.make_disjunct(f"off{b}")
        for i in range(lo, hi):
            on.subject_to(x[i] <= 2.0 + 0.5 * (i - lo))
            off.subject_to(x[i] <= 0.0)
        on.subject_to(dm.sum(x[i] for i in range(lo, hi)) >= 1.0)
        m.add_disjunction([on, off], name=f"blk{b}")
        obj = obj + dm.log(1.0 + x[lo] + x[lo + 1]) - 0.01 * x[lo]
    m.minimize(-obj)
    return m


class _After:
    """An expiry predicate that turns true after ``n`` polls, counting them."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.n


# ── the scope ────────────────────────────────────────────────────────────────


def test_checkpoint_is_a_no_op_outside_the_scope():
    assert not setup_abortable()
    setup_checkpoint("anything")  # must not raise


def test_checkpoint_raises_inside_an_expired_scope_and_resets():
    with pytest.raises(SetupDeadlineExpired, match="during the step"):
        with abortable_setup(lambda: True):
            assert setup_abortable()
            setup_checkpoint("the step")
    assert not setup_abortable()


def test_expiry_is_not_swallowed_by_except_exception():
    """The reason it is a BaseException: setup code catches ``Exception`` widely."""

    def setup_step():
        try:
            setup_checkpoint("the step")
        except Exception:  # noqa: BLE001 - the point of the test
            return "swallowed"
        return "ran"

    with pytest.raises(SetupDeadlineExpired):
        with abortable_setup(lambda: True):
            setup_step()


# ── each long setup loop polls ───────────────────────────────────────────────


def test_model_copy_aborts_mid_copy():
    m = _blocks(60, 10)
    poll = _After(1)
    with pytest.raises(SetupDeadlineExpired, match="model copy"):
        with abortable_setup(poll):
            copy.deepcopy(m)
    assert poll.calls >= 2, "the copy never polled"


def test_model_copy_outside_the_scope_is_unchanged():
    """The polled list copy is only taken inside the scope, and copies the same."""
    m = _blocks(4, 5)
    plain = copy.deepcopy(m)
    with abortable_setup(lambda: False):
        polled = copy.deepcopy(m)
    for c in (plain, polled):
        assert [str(k) for k in c._constraints] == [str(k) for k in m._constraints]
        assert [v.name for v in c._variables] == [v.name for v in m._variables]
        assert c._constraints is not m._constraints


def test_gdp_lowering_aborts():
    from discopt.gdp.algorithms.init import GDPSpace

    m = _blocks(30, 5)
    poll = _After(3)
    with pytest.raises(SetupDeadlineExpired):
        with abortable_setup(poll):
            GDPSpace.from_model(m)
    assert poll.calls >= 4


def test_master_build_aborts():
    from discopt.gdp.algorithms.init import GDPSpace
    from discopt.gdp.algorithms.master import build_master

    space = GDPSpace.from_model(_blocks(60, 5))
    poll = _After(0)
    with pytest.raises(SetupDeadlineExpired, match="master build"):
        with abortable_setup(poll):
            build_master(space.lowered)
    assert poll.calls >= 1


# ── the entry point turns an abandoned setup into an honest time_limit ──────


@pytest.mark.parametrize("method", ["loa", "ric", "gloa", "lbb", "ldsda", "enumerate"])
def test_abandoned_setup_returns_time_limit(method, monkeypatch):
    from discopt.gdp.algorithms.base import Deadline
    from discopt.solvers.gdpopt_loa import solve_gdpopt_loa

    # The budget is spent from the start, so the first setup checkpoint fires.
    monkeypatch.setattr(Deadline, "expired", lambda self: True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_gdpopt_loa(_blocks(3, 5), time_limit=60.0, algorithm=method)
    assert r.status == "time_limit"
    assert r.objective is None and r.bound is None and r.x is None
    assert r.gap_certified is False
    assert r.solver_stats["stop_reason"] == "time_limit"
    assert r.solver_stats["setup_abandoned"]


@pytest.mark.parametrize("method", ["loa", "ric"])
def test_setup_in_time_is_not_abandoned(method):
    from discopt.solvers.gdpopt_loa import solve_gdpopt_loa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_gdpopt_loa(_blocks(2, 5), time_limit=120.0, algorithm=method)
    assert "setup_abandoned" not in r.solver_stats
    assert r.objective is not None


def test_master_out_of_time_is_time_limit_not_exhausted(monkeypatch):
    """A master that returns nothing because the clock ran out did not exhaust
    the configurations; ``stop_reason`` says ``time_limit``."""
    from discopt.gdp.algorithms.loa import LOA
    from discopt.gdp.algorithms.master import MasterResult

    clock = [0.0]
    alg = LOA(_blocks(2, 5), time_limit=10.0, clock=lambda: clock[0])

    def timed_out(*a, **k):
        clock[0] = 11.0
        return MasterResult(None, None, None, False, "time_limit")

    monkeypatch.setattr(alg.master, "solve", timed_out)
    assert alg.iterate() is False
    assert alg.stop_reason == "time_limit"


def test_master_proven_infeasible_is_unchanged(monkeypatch):
    from discopt.gdp.algorithms.loa import LOA
    from discopt.gdp.algorithms.master import MasterResult

    clock = [0.0]
    alg = LOA(_blocks(2, 5), time_limit=10.0, clock=lambda: clock[0])

    def infeasible(*a, **k):
        clock[0] = 11.0
        return MasterResult(None, None, None, True, "infeasible")

    monkeypatch.setattr(alg.master, "solve", infeasible)
    alg.iterate()
    assert alg.stop_reason != "time_limit"


# ── the linear-time replacements read the same data ─────────────────────────


def _exprs(m: dm.Model):
    x = m._variables[0]
    y = m.continuous("y", shape=(3, 2), lb=-1, ub=1)
    z = m.binary("z")
    return [x, y, z, x[3], y[1], y[2, 1], y[:, 0], x[2:7]]


def test_column_indexer_matches_column_indices():
    from discopt.validation.feasibility import _column_indices, _ColumnIndexer

    m = _blocks(3, 5)
    exprs = _exprs(m)
    col = _ColumnIndexer(m)
    n = 0
    for e in exprs:
        np.testing.assert_array_equal(col(e), _column_indices(e, m))
        n += 1
    assert n == len(exprs) > 0


def _plain_rows(m: dm.Model) -> list:
    """Every algebraic row of ``m``, disjunct rows unwrapped from their indicators."""
    from discopt.modeling.core import Constraint, _DisjunctiveConstraint

    out, stack = [], list(m._constraints)
    while stack:
        c = stack.pop()
        if isinstance(c, Constraint):
            out.append(c)
        elif isinstance(c, _DisjunctiveConstraint):
            stack.extend(r for d in c.disjuncts for r in d)
        elif hasattr(c, "constraint"):
            stack.append(c.constraint)
    return out


def test_body_terms_match_the_dense_coefficients():
    from discopt._relax.gdp_reformulate import _extract_body_coeffs, _extract_body_terms

    m = _blocks(3, 5)
    x = m._variables[0]
    m.subject_to(2 * x[0] - 3 * x[4] + x[0] + 7 <= 0)
    m.subject_to(x[1] * x[2] <= 1)  # nonlinear: both refuse
    n_vars = sum(int(v.size) for v in m._variables)
    rows = _plain_rows(m)
    checked = 0
    for c in rows:
        dense = _extract_body_coeffs(c.body, m, n_vars)
        terms = _extract_body_terms(c.body, m)
        assert (dense is None) == (terms is None)
        if dense is None:
            continue
        vec = np.zeros(n_vars)
        for k, v in terms[0].items():
            if k < n_vars:
                vec[k] += v
        np.testing.assert_array_equal(vec, dense[0])
        assert terms[1] == dense[1]
        checked += 1
    assert checked >= 10


def test_univariate_matches_the_dense_reading():
    """``_univariate`` (sparse) against the pre-change dense ``flatnonzero`` reading."""
    from discopt._relax.gdp_reformulate import _extract_body_coeffs
    from discopt.gdp.bound_pretransform import _bound_of, _leaf_for_slot, _univariate

    m = _blocks(3, 5)
    x = m._variables[0]
    m.subject_to(3 * x[2] - 6 <= 0)
    m.subject_to(x[3] - x[3] + x[4] >= 0)  # x[3] cancels to 0: univariate in x[4]
    m.subject_to(x[0] + x[1] <= 3)
    n_vars = sum(int(v.size) for v in m._variables)

    def dense(con):
        if np.size(con.rhs) != 1 or float(np.asarray(con.rhs).ravel()[0]) != 0.0:
            return None
        coeffs = _extract_body_coeffs(con.body, m, n_vars)
        if coeffs is None:
            return None
        c_vec, off = coeffs
        nz = np.flatnonzero(c_vec)
        if nz.size != 1:
            return None
        slot = int(nz[0])
        leaf = _leaf_for_slot(con.body, slot, m)
        if leaf is None:
            return None
        return _bound_of(con.sense, slot, leaf, float(c_vec[slot]), off)

    rows = _plain_rows(m)
    hits = 0
    for c in rows:
        a, b = _univariate(c, m, n_vars), dense(c)
        if b is None:
            assert a is None
            continue
        assert a is not None
        assert (a[0], a[2], a[3]) == (b[0], b[2], b[3]) and a[1] is b[1]
        hits += 1
    assert hits >= 10
