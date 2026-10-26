"""Issue #1941 (ex14_1_7): a node LP that every in-house certificate declines, with
no unbounded *nonlinear* column.

ex14_1_7 is a minimax: ``g_k <= x10`` and ``-g_k <= x10`` with ``x10`` free and only
linear. The root LP already solves to the optimum 0 (each ``g_k`` aux column is shared
by its paired rows), yet no bound is certified: the NS safe bound abstains on the free
``x10``, ``trusted_vertex`` is refused because aux bounds reach 1.77e9, and the #1869
(b) handoff requires an unbounded nonlinear column. ``DISCOPT_DECLINED_LP_CERT`` sends
any such declined LP to the verified HiGHS route, whose bound is rigorous for any dual.
Default ON since 2026-10-26; ``=0`` restores the decline.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import pytest
from discopt._relax import mccormick_lp as ML
from discopt.tightening import fbbt_box

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_DECLINED_LP_CERT"
NL = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "ex14_1_7.nl")
OPT = 0.0  # MINLPLib minlplib.solu


def _root(flag: str, monkeypatch):
    monkeypatch.setenv(FLAG, flag)
    m = dm.from_nl(NL)
    bt = fbbt_box(m)  # the solver's root box: x10 >= -341.28, still open above
    return ML.MccormickLPRelaxer(m).solve_at_node(bt.lb, bt.ub, time_limit=10.0)


def test_flag_defaults_on_and_reads_at_call_time(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert ML.declined_lp_cert_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not ML.declined_lp_cert_enabled()


def test_flag_off_root_lp_is_uncertified(monkeypatch):
    r = _root("0", monkeypatch)
    assert r.lower_bound is None, r


def test_flag_on_root_lp_certifies_a_bound_below_the_optimum(monkeypatch):
    r = _root("1", monkeypatch)
    assert r.status == "optimal" and r.lower_bound is not None, r
    assert r.lower_bound <= OPT + 1e-9
    assert r.lower_bound >= OPT - 1e-6  # the relaxation is exact at the root


@pytest.mark.slow
def test_flag_on_solve_certifies_ex14_1_7(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    r = dm.from_nl(NL).solve(time_limit=60)
    assert r.status == "optimal" and r.gap_certified, r
    assert r.bound <= OPT + 1e-9
    assert r.objective >= OPT - 1e-6
