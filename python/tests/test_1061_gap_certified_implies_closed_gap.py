"""``gap_certified`` must imply the gap is actually CLOSED (#1061 follow-on).

The OA / mip-nlp route used to derive the flag from ``reported_gap is not None
and not has_unresolved`` -- "a gap was computed", not "the gap is a
certificate". Measured on the 153-instance convex panel (2026-08-20), **19 of
153** instances came back ``gap_certified=True`` on an *open* gap, the widest
being ``syn40m`` at a 430% relative gap. ``result_io.summary_text`` renders that
field as the "(uncertified)" marker, so those rows printed to a user as
certified.

It also made the flag incomparable across routes: the NLP-BB path spells the
strict meaning (``solver.py``: any ``feasible`` exit clears ``_gap_certified``,
keeping bound *validity* in ``_tree_bound_valid``). One field, two meanings, so
a differential panel that switched routes read a certification "regression"
where the newly-selected result was strictly better on bound, incumbent and gap.

The invariant under test is one-directional and route-independent:

    gap_certified  =>  gap is not None and gap <= gap_tolerance

Bound validity is a *different* question and is deliberately NOT asserted here;
it keeps its own signal in the trace (``master_bound_valid`` /
``bound_validity``), which stays ``"global"`` for a valid-but-open bound.
"""

import os

import pytest
from discopt.modeling.core import from_nl

CORPUS = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
GAP_TOL = 1e-4

# Convex big-M/GDP instances that do not close inside a short budget, so the
# time-limited exit carries a valid bound AND an open gap -- the state that
# produced the false certificate. Gate probes only (CLAUDE.md SS2): the
# assertion is the general invariant, not a per-instance expectation.
PROBES = ["syn40m", "rsyn0805m02m"]


@pytest.mark.slow
@pytest.mark.parametrize("name", PROBES)
def test_open_gap_is_never_reported_as_certified(name):
    path = os.path.join(CORPUS, name + ".nl")
    if not os.path.exists(path):
        pytest.skip("MINLPLib corpus not present")

    result = from_nl(path).solve(
        solver="mip-nlp", mip_nlp_method="oa", time_limit=20, gap_tolerance=GAP_TOL
    )

    # SS6: this probe is only meaningful on a run that actually left a gap open.
    # A build that closes it has not exercised the defect, so say so loudly
    # rather than passing silently on a measurement that never happened.
    assert result.gap is not None and result.gap > GAP_TOL, (
        f"probe precondition broken: {name} closed its gap "
        f"(gap={result.gap!r}, status={result.status!r}) -- re-derive the probe"
    )

    assert not result.gap_certified, (
        f"{name} reports gap_certified=True at gap={result.gap:.4g} "
        f"(status={result.status!r}) -- an open gap is not a certificate"
    )

    # The bound is still valid; only the *certificate* claim was withdrawn.
    trace = getattr(result, "mip_nlp_trace", None)
    if isinstance(trace, dict) and trace.get("final_lb") is not None:
        assert trace.get("bound_validity") == "global", (
            "narrowing gap_certified must not downgrade bound validity: "
            f"bound_validity={trace.get('bound_validity')!r}"
        )
        assert trace.get("master_bound_valid") is True


@pytest.mark.slow
def test_gap_certified_agrees_with_optimal_status():
    """``gap_certified`` and ``status == 'optimal'`` must not disagree."""
    path = os.path.join(CORPUS, "syn40m.nl")
    if not os.path.exists(path):
        pytest.skip("MINLPLib corpus not present")
    result = from_nl(path).solve(
        solver="mip-nlp", mip_nlp_method="oa", time_limit=20, gap_tolerance=GAP_TOL
    )
    assert result.gap_certified == (result.status == "optimal"), (
        f"status={result.status!r} but gap_certified={result.gap_certified!r}"
    )
