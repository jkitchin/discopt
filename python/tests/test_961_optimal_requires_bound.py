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


@pytest.mark.claim_boundary
def test_root_lp_gate_bites_when_the_matrix_bytes_are_identical():
    """The build-boundary bucket must excuse only a *different* LP, never a drift.

    ``diff_root_lp`` declines to call a root-LP difference a claim change when the
    relaxation fingerprint differs too — the two sides did not solve the same
    matrix (``contvar`` does this across builds; see the docstring there). That
    escape must be conditional: with the recorded bytes matching what this build
    produces, an altered status or bound is still ``changed``. Without this test,
    widening the bucket to a blanket skip would look like a pass.

    The "recorded" side is synthesized from THIS build rather than read out of the
    committed baseline. What is under test is the gate's *conditional*, which needs
    only a row whose bytes and bound the build reproduces — and the committed
    baseline is a linux/x86-64 artifact of one solver commit, so sourcing them from
    it made this assertion fail for two reasons that have nothing to do with the
    gate: a relaxation change (any correct one moves the fingerprints) and a
    platform change (10 of the 66 committed rows do not reproduce on arm64).

    Worse, the old selection predicate was ``diff_root_lp(...).status ==
    "unchanged"``, which does NOT imply byte-identity — ``diff_root_lp`` computes
    the fingerprint only once it already sees a status/bound difference — so the
    ``row["fingerprint"] == current_row(...)["fingerprint"]`` assertion below it
    was load-bearing but unimplied, and whether this test proved anything came down
    to which instance happened to sort first. See #971.
    """
    from support.claim_differential import (
        current_root_lp,
        current_row,
        diff_root_lp,
        load_baseline,
    )

    baseline = load_baseline()
    name = sorted(baseline)[0]
    assert name, "empty baseline -- test is dead"
    status, bound = current_root_lp(name)
    row = {
        "instance": name,
        "fingerprint": current_row(name)["fingerprint"],
        "root_lp_status": status,
        "root_lp_bound": bound,
    }
    assert row["fingerprint"], f"{name}: this build produced no relaxation fingerprint"
    assert status, f"{name}: this build produced no root-LP status"

    same_bytes = dict(row, root_lp_status="a-status-this-build-does-not-produce")
    d = diff_root_lp(name, {name: same_bytes})
    assert d.status == "changed", f"gate did not bite on {name}: {d}"

    drifted = dict(same_bytes, fingerprint="0" * 64)
    d2 = diff_root_lp(name, {name: drifted})
    assert d2.status == "fingerprint_drift", f"drifted bytes not excused on {name}: {d2}"
