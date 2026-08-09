"""Issue #961: ``status="optimal"`` must always carry a finite ``lower_bound``.

``MccormickLPRelaxer.solve_at_node`` returned ``status="optimal"`` with
``lower_bound=None`` on five corpus instances (beuster, casctanks, st_miqp2/3/4):
the LP solved to optimality but every bound-certification route declined (no
Neumaier-Shcherbina safe bound; the vertex objective refused by the
unbounded-nonlinear-column / conditioning guards, or invalidated by
``_objective_bound_valid=False``), and the decline path passed the raw
``"optimal"`` status through with no bound. Downstream, the baseline generator's
``np.isfinite(None)`` crashed with a TypeError that a bare ``except`` recorded as
a plausible "no bound".

The fix: the decline is reported as its own non-fathoming ``"uncertified"``
status, and the ``MccormickLPResult`` constructor refuses the
``optimal``/no-finite-bound pair outright (a loud contract, per CLAUDE.md §3).
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import numpy as np
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer, MccormickLPResult, _no_bound_status
from discopt.modeling.core import from_nl

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def test_result_constructor_refuses_optimal_without_bound():
    with pytest.raises(ValueError, match="#961"):
        MccormickLPResult(status="optimal")
    with pytest.raises(ValueError, match="#961"):
        MccormickLPResult(status="optimal", lower_bound=float("nan"))
    with pytest.raises(ValueError, match="#961"):
        MccormickLPResult(status="optimal", lower_bound=float("-inf"))


def test_result_constructor_accepts_valid_combinations():
    assert MccormickLPResult(status="optimal", lower_bound=1.5).lower_bound == 1.5
    # Non-optimal statuses legitimately carry no bound.
    for status in ("uncertified", "infeasible", "numerical", "error", "skipped_oversize"):
        assert MccormickLPResult(status=status).lower_bound is None


def test_no_bound_status_relabels_only_optimal():
    assert _no_bound_status("optimal") == "uncertified"
    for status in ("infeasible", "numerical", "unbounded", "iteration_limit", "error"):
        assert _no_bound_status(status) == status


@pytest.mark.slow
def test_certify_decline_reports_uncertified_not_optimal():
    """Regression on a real corpus instance (fails before #961's fix).

    st_miqp3's root relaxation solves to LP optimality but the objective bound is
    refused (``_objective_bound_valid=False``, the #248 garbage-floor guard), so
    no bound can be certified. Before the fix this leaked ``status="optimal"``
    with ``lower_bound=None``; it must surface as ``"uncertified"``.
    """
    model = from_nl(str(_NL_DIR / "st_miqp3.nl"))
    lbs = [np.asarray(v.lb, dtype=np.float64).ravel() for v in model._variables]
    ubs = [np.asarray(v.ub, dtype=np.float64).ravel() for v in model._variables]
    res = MccormickLPRelaxer(model).solve_at_node(np.concatenate(lbs), np.concatenate(ubs))
    if res.status == "optimal":
        assert res.lower_bound is not None and np.isfinite(res.lower_bound)
    else:
        # The certification routes declined on this instance today; the decline
        # must be labeled as such, never "optimal" with no bound.
        assert res.status == "uncertified"
        assert res.lower_bound is None
