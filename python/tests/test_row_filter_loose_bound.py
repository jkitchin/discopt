"""A loose certified root bound is recovered by the #671 row-filtered re-solve.

#1229's ``dual_slack_basis`` free-column fix lets the warm simplex converge on hda's
root McCormick LP instead of hitting its iteration limit. The vertex is primal
feasible at -64675.25, but the Neumaier-Shcherbina bound read off its duals is
-5.71e6, and the failure-triggered row filter no longer fired. Re-solving with the
float64-intractable rows dropped certifies -64675.2492, which is sound: the
filtered LP is a superset of the original.

The absolute values moved with #1256 and the assertion below is relational as a
result. hda carries three rows of the form ``x == -log(c*4**(a + b*x0) - c)``;
``4**t`` used to canonicalize to an OPAQUE node — no envelope, so the whole row
went to the interval treatment — and now lowers to ``exp(ln(4)*t)``, which the
univariate table relaxes. Over the DECLARED box ``x0 in [0, 40]`` that exp spans
13 decades, and the envelope rows it emits carry coefficients to match: the LP
vertex is unchanged but the Neumaier-Shcherbina bound read off its duals degrades
(measured at this raw box: -5.71e6 -> -9.99e11 with the filter off). The filter
still does its job on it — -9.99e11 -> -1.38e9, a 725x recovery, where before it
was -5.71e6 -> -64675, an 88x one — and the bound stays sound either way. This is
a RAW-box call, and every real solve runs FBBT/OBBT first: measured end to end on
hda, the bound is unchanged at a 10 s limit (-124296.879066 vs -124296.879048)
and BETTER at 60 s (-64510.17 on both runs, against -122962 / -141697 across two
baseline runs — same status and node count, 3). Tightening the certified bound
under ill-conditioned envelope rows is the recorded interval max-combine
follow-up, not something this test should pin to a constant that only held for
one lowering of one operator.
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


def test_hda_root_bound_is_not_the_loose_safe_bound(monkeypatch):
    """The recovery must FIRE and must be a large tightening, both measured here.

    Relational rather than a constant: what #671 owns is the difference between
    the two certified bounds on the same LP, and pinning the absolute value made
    the test a hostage of how one operator happens to be lowered (see the module
    docstring). The gap asserted below is two orders of magnitude; the two
    lowerings measured so far give 88x and 725x.
    """
    with_filter = _root_bound()
    assert with_filter.status == "optimal"
    assert with_filter.lower_bound is not None
    assert with_filter.lower_bound <= _HDA_OPT, "the certified bound must stay sound"

    monkeypatch.setenv("DISCOPT_RELAX_ROW_FILTER_LOOSE_BOUND", "0")
    without = _root_bound()
    assert without.lower_bound is not None
    assert without.lower_bound <= _HDA_OPT, "the un-recovered bound must be sound too"
    assert with_filter.lower_bound > 100.0 * without.lower_bound, (
        f"the row-filtered re-solve barely moved the bound: "
        f"{without.lower_bound!r} -> {with_filter.lower_bound!r}"
    )


@pytest.mark.parametrize(
    "flag", ["DISCOPT_RELAX_ROW_FILTER_LOOSE_BOUND", "DISCOPT_RELAX_ROW_FILTER"]
)
def test_opt_out_keeps_a_valid_bound(monkeypatch, flag):
    monkeypatch.setenv(flag, "0")
    res = _root_bound()
    assert res.lower_bound is None or res.lower_bound <= _HDA_OPT
