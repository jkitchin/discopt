"""A loose certified root bound is recovered by the #671 row-filtered re-solve.

#1229's ``dual_slack_basis`` free-column fix lets the warm simplex converge on hda's
root McCormick LP instead of hitting its iteration limit. The vertex is primal
feasible at -64675.25, but the Neumaier-Shcherbina bound read off its duals is
-5.71e6, and the failure-triggered row filter no longer fired. Re-solving with the
float64-intractable rows dropped certifies -64675.2492, which is sound: the
filtered LP is a superset of the original.

**#1296 superseded the loose-bound trigger on this instance, and the numbers below
are retracted as a description of the current tree.** #1296's tiny-entry gate
withdraws the certificate on hda's root McCormick LP (measured:
``MilpTinyEntryDecert == 1`` with ``DISCOPT_PROFILE=1``), which routes the node
through #671's *failure* trigger — the one the loose-bound flag does not gate. The
loose-bound re-solve therefore no longer fires here, and the two flag arms this
test used to compare are now identical (both −64675.249181289175). Measured over
the whole 66-instance in-repo corpus at this head, the loose-bound flag changes the
certified root bound on **0** instances: it is inert corpus-wide, so there is no
other instance to retarget the old assertion at.

What still discriminates on hda is the row filter *as a whole*: measured at this
head, ``DISCOPT_RELAX_ROW_FILTER=0`` certifies −77002.685837 and the default
certifies −64675.249181, a 1.19x tightening. That is what the assertions below
pin, together with soundness on both arms. The 50x bar is gone because the
pathology it guarded — a root bound six to eleven orders of magnitude below the
vertex — no longer occurs on this LP, not because the bar was inconvenient.

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
#: Minimum tightening the #671 row filter must still deliver on hda, as a factor on
#: the unfiltered bound's distance to zero. Measured at this head: −77002.685837 ->
#: −64675.249181, a 1.19x tightening. The bar is 1.05x so the test states "the
#: filter still earns its keep" with ~4x of headroom while still rejecting an inert
#: filter (1.00x) — see the module docstring for why the old 50x bar is gone.
_MIN_FILTER_GAIN = 1.05


def _root_bound():
    model = from_nl(str(_HDA))
    lb = np.concatenate([np.asarray(v.lb, dtype=np.float64).ravel() for v in model._variables])
    ub = np.concatenate([np.asarray(v.ub, dtype=np.float64).ravel() for v in model._variables])
    return MccormickLPRelaxer(model).solve_at_node(lb, ub)


def test_hda_root_bound_is_not_the_loose_safe_bound(monkeypatch):
    """The row filter must FIRE on hda and must tighten the certified root bound.

    Relational rather than a constant: what #671 owns is the difference between
    the two certified bounds on the same LP, and pinning the absolute value made
    the test a hostage of how one operator happens to be lowered (see the module
    docstring). The arms are the filter as a whole — ``DISCOPT_RELAX_ROW_FILTER``
    — because the narrower loose-bound flag no longer changes this bound, or any
    bound in the corpus, after #1296.

    The comparison is a DIVISION, and an earlier draft of this assertion got that
    wrong in a way worth keeping on record. Both bounds are large and negative
    here, so ``k * without`` moves *away* from zero: written that way the condition
    reads ``|with| < k*|without|``, permitting the filter to make the bound LOOSER
    and still pass. Worse, it cannot fail at all — ``mccormick_lp`` keeps the larger
    of the two certified bounds (``if _filtered_bound > bound``), so
    ``with >= without`` holds by construction, and with ``without < 0`` that gives
    ``k*without < without <= with`` for any ``k > 1``. Dividing is the tightening
    direction: ``without / k`` moves TOWARD zero.
    """
    with_filter = _root_bound()
    assert with_filter.status == "optimal"
    assert with_filter.lower_bound is not None
    assert with_filter.lower_bound <= _HDA_OPT, "the certified bound must stay sound"

    monkeypatch.setenv("DISCOPT_RELAX_ROW_FILTER", "0")
    without = _root_bound()
    assert without.lower_bound is not None
    assert without.lower_bound <= _HDA_OPT, "the unfiltered bound must be sound too"
    # The "fraction of the distance to zero" form below is stated for a NEGATIVE
    # unfiltered bound, which is the regime this LP is in and the reason the
    # filter exists. Fail loudly rather than pass vacuously if that ever changes.
    assert without.lower_bound < 0.0, (
        f"unfiltered bound {without.lower_bound!r} is not negative; the "
        "tightening comparison below no longer says what it means"
    )
    required = without.lower_bound / _MIN_FILTER_GAIN
    assert with_filter.lower_bound > required, (
        f"the row-filtered re-solve barely moved the bound: "
        f"{without.lower_bound!r} -> {with_filter.lower_bound!r} "
        f"(needs > {required!r}, i.e. a {_MIN_FILTER_GAIN:g}x tightening)"
    )


@pytest.mark.parametrize(
    "flag", ["DISCOPT_RELAX_ROW_FILTER_LOOSE_BOUND", "DISCOPT_RELAX_ROW_FILTER"]
)
def test_opt_out_keeps_a_valid_bound(monkeypatch, flag):
    monkeypatch.setenv(flag, "0")
    res = _root_bound()
    assert res.lower_bound is None or res.lower_bound <= _HDA_OPT
