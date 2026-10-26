"""Issue #1869 (b): certify a McCormick node LP whose free columns defeat the in-house
safe bound.

``prolog``'s root LP is bounded at the optimum 0 (its pricing rows bound the bilinear
sums), but the in-house relaxer reported it ``uncertified``: with open columns a
floating dual puts reduced costs on the open side and the Neumaier--Shcherbina bound is
``-inf``. ``DISCOPT_FREE_COLUMN_LP_CERT`` hands such a declined LP to the verified HiGHS
LP route (``lp_milp_highs.solve_lp_std``), whose bound is rigorous for any dual. It also
keeps the relaxer on an all-unbounded model, which the #185 guard dropped because it had
no finite-box continuous variable -- unbounded branching (#1869 (a)) makes one branchable.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import mccormick_lp as ML
from discopt.tightening import fbbt_box

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_FREE_COLUMN_LP_CERT"
PROLOG = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "prolog.nl")
OPT = 0.0  # MINLPLib minlplib.solu


def _root(flag: str, monkeypatch):
    monkeypatch.setenv(FLAG, flag)
    m = dm.from_nl(PROLOG)
    bt = fbbt_box(m)
    return ML.MccormickLPRelaxer(m).solve_at_node(bt.lb, bt.ub)


# #1966: the flag-ON root-LP and end-to-end ``prolog`` certification tests were removed.
# prolog's root LP over the FBBT box is EXACTLY unbounded below over its double data
# (``fl(3340.8) + fl(371.2) > 3712``), so a finite certificate for it would be false;
# test_1966_prolog_root_lp_exactly_unbounded.py pins the ray and the abstention.


def test_flag_off_leaves_the_root_lp_uncertified(monkeypatch):
    r = _root("0", monkeypatch)
    # ``uncertified`` (an LP optimum no route certified) or, since the native routes
    # gate ``optimal`` on an NS bound (#1953), ``iteration_limit``: either way no bound.
    assert r.status in ("uncertified", "iteration_limit") and r.lower_bound is None, r


def test_4stufen_root_bound_is_below_the_best_known_dual(monkeypatch):
    """A second real LP the route now certifies: ``4stufen``'s root, which every
    in-house route declines. Its bound must not exceed MINLPLib's best dual."""
    from discopt.modeling.core import from_nl
    from discopt.solver import flat_variable_bounds

    monkeypatch.setenv(FLAG, "1")
    m = from_nl(os.path.join(os.path.dirname(PROLOG), "4stufen.nl"))
    lb, ub = flat_variable_bounds(m)
    r = ML.MccormickLPRelaxer(m).solve_at_node(lb, ub, time_limit=10.0)
    assert r.status == "optimal" and r.lower_bound <= 112657.2579 + 1e-6


def test_helper_refuses_an_unbounded_lp():
    """``min -x, x >= 0`` has no finite bound: the helper must return nothing."""
    import scipy.sparse as sp
    from discopt._relax.milp_relaxation import MilpRelaxationModel

    milp = MilpRelaxationModel(
        c=np.array([-1.0, 0.0]),
        A_ub=sp.csr_matrix(np.array([[0.0, 1.0]])),
        b_ub=np.array([1.0]),
        bounds=[(0.0, np.inf), (0.0, 1.0)],
    )
    assert ML._verified_free_column_bound(milp, 5.0) == (None, None)


def test_helper_bound_is_below_a_known_lp_optimum():
    """``min x - y, x - y >= -1 (as -x + y <= 1), x, y >= 0``: optimum -1, both open."""
    import scipy.sparse as sp
    from discopt._relax.milp_relaxation import MilpRelaxationModel

    milp = MilpRelaxationModel(
        c=np.array([1.0, -1.0]),
        A_ub=sp.csr_matrix(np.array([[-1.0, 1.0]])),
        b_ub=np.array([1.0]),
        bounds=[(0.0, np.inf), (0.0, np.inf)],
    )
    b, x = ML._verified_free_column_bound(milp, 5.0)
    assert b is not None and b <= -1.0 + 1e-9 and b >= -1.0 - 1e-6
    assert x is not None and x.shape == (2,)


def test_flag_defaults_on(monkeypatch):
    """Graduated under CLAUDE.md section 5 (panel in the gate's docstring)."""
    monkeypatch.delenv(FLAG, raising=False)
    assert ML.free_column_lp_cert_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not ML.free_column_lp_cert_enabled()
