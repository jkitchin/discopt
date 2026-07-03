"""Regression test: st_e17 certifies via convex-NLP-failure -> clear -> spatial B&B.

st_e17 (MINLPLib) is convex — ``min 29.4 x0 + 18 x1`` s.t.
``6 - (x0 - 0.2458 x0**2 / x1) <= 0`` with ``x1 in [1e-5, 30]`` (a
quadratic-over-linear / rotated-SOC constraint). It is therefore routed to the
convex NLP fast path, but the NLP fails to converge: the ``x0**2 / x1`` term is
badly conditioned as ``x1`` reaches toward its 1e-5 lower bound, so the solver
returns a non-certified iterate (status ``iteration_limit``) far from the
optimum.

The solver now treats that as a signal: when the convex NLP does not certify and
the model has a sign-definite non-constant denominator (which the relaxation
otherwise drops to ``general_nl``), it clears the denominator — exact and
value-preserving — and falls back to the sound spatial B&B, which certifies it:

    before:  iteration_limit, obj ~ 203.96, bound None   (uncertified)
    after:   optimal 376.29, bound 376.28                (certified)

Convex models whose division the NLP *can* handle (e.g. the rotated SOC
``x**2/z <= y`` in ``nlp_cvx_204``) are unaffected — they certify on the fast
path and are never cleared. The companion scale guard in
``factorable_reform._clear_divisions`` keeps the cleared constraint sound under
the small ``x1`` denominator.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_NL = Path(__file__).parent / "data" / "minlplib" / "st_e17.nl"
# Brute-forced true optimum (independently verified):
#   min 29.4 x0 + 18 x1 s.t. x0 - 0.2458 x0**2/x1 >= 6.
_OPT = 376.291932


@pytest.mark.correctness
def test_st_e17_certifies_to_optimum():
    """st_e17 certifies its true optimum with a valid dual bound."""
    assert _NL.exists(), f"missing {_NL}"
    r = dm.from_nl(str(_NL)).solve(time_limit=90, gap_tolerance=1e-4)
    assert r.status == "optimal", f"status={r.status} (was 'iteration_limit' before the fix)"
    assert r.objective is not None
    assert abs(r.objective - _OPT) <= 1e-1, f"obj={r.objective} != {_OPT}"
    # Soundness invariant: a valid dual bound never exceeds the optimum.
    assert r.bound is not None, "certified solve must report a dual bound"
    assert r.bound <= r.objective + 1e-2, f"invalid dual bound {r.bound} > obj {r.objective}"


@pytest.mark.slow
def test_st_e17_returned_point_is_feasible():
    """The certified incumbent satisfies the original ratio constraint."""
    r = dm.from_nl(str(_NL)).solve(time_limit=90, gap_tolerance=1e-4)
    assert r.x is not None
    x0 = float(r.x["x0"])
    x1 = float(r.x["x1"])
    # Original constraint: 6 - (x0 - 0.2458 x0**2/x1) <= 0.
    body = 6 - (x0 - 0.2458 * x0**2 / x1)
    assert body <= 1e-4, f"certified point violates original constraint: body={body}"
