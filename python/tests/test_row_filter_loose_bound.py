"""A loose certified root bound is recovered by the #671 row-filtered re-solve.

#1229's ``dual_slack_basis`` free-column fix lets the warm simplex converge on hda's
root McCormick LP instead of hitting its iteration limit. The vertex is primal
feasible at -64675.25, but the Neumaier-Shcherbina bound read off its duals is
-5.71e6, and the failure-triggered row filter no longer fired. Re-solving with the
float64-intractable rows dropped certifies -64675.2492, which is sound: the
filtered LP is a superset of the original.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt.modeling.core import from_nl

_HDA = Path(__file__).parent / "data" / "minlplib_nl" / "hda.nl"
_HDA_OPT = -5964.534084  # minlplib.solu


def _root_bound():
    model = from_nl(str(_HDA))
    lb = np.concatenate([np.asarray(v.lb, dtype=np.float64).ravel() for v in model._variables])
    ub = np.concatenate([np.asarray(v.ub, dtype=np.float64).ravel() for v in model._variables])
    return MccormickLPRelaxer(model).solve_at_node(lb, ub)


def test_hda_root_bound_is_not_the_loose_safe_bound():
    res = _root_bound()
    assert res.status == "optimal"
    assert res.lower_bound <= _HDA_OPT
    # -64675.25 with the recovery; -5.71e6 without it.
    assert res.lower_bound >= -1e5, res.lower_bound


@pytest.mark.parametrize(
    "flag", ["DISCOPT_RELAX_ROW_FILTER_LOOSE_BOUND", "DISCOPT_RELAX_ROW_FILTER"]
)
def test_opt_out_keeps_a_valid_bound(monkeypatch, flag):
    monkeypatch.setenv(flag, "0")
    res = _root_bound()
    assert res.lower_bound is None or res.lower_bound <= _HDA_OPT
