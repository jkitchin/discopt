"""Regression: per-node orchestration overhead (issue #723).

The convex-MINLP dispatch (``solve_model``) already memoizes the eigenvalue/LP
-heavy model convexity classification, but ``_solve_nlp_bb`` used to *re-run* a
full classification of the same model+bounds purely to set its ``gap_certified``
flag — a redundant second classify per solve, and a third/fourth inside the
RENS root primal heuristic's sub-solve (which re-enters ``solve_model``).

#723 threads the caller's already-computed convex verdict into ``_solve_nlp_bb``
instead. This is bound-neutral (identical model, identical bounds, and the reuse
only fires when GDP reformulation is a no-op), so it must not change the number
of nodes explored or the certified objective — only how many times the
classifier runs.

The pin here is a strict upper bound on the classifier call count for a convex
MIQP that routes through NLP-BB + RENS: it was 4 (two sites x {parent, RENS
sub-solve}) before the fix and is 2 after. The test fails on the pre-#723 code.
"""

from __future__ import annotations

from pathlib import Path

import discopt._jax.convexity as _cvx
import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.solver import solve_model

pytestmark = pytest.mark.unit

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def test_convexity_dedup_on_nlp_bb_path(monkeypatch):
    """classify_model runs <= 2x on the convex NLP-BB + RENS path (was 4x).

    fac2 is a convex MINLP that routes through ``_solve_nlp_bb`` and fires the
    RENS root primal heuristic (which re-enters ``solve_model``). Pre-#723 that
    meant four full classifications per solve: two dispatch sites x {parent,
    RENS sub-solve}. #723 threads the caller's verdict into ``_solve_nlp_bb``,
    removing the redundant second site at each entry -> two classifications.

    The solve stays bound-neutral: identical node count and certified objective.
    """
    calls = {"n": 0}
    orig = _cvx.classify_model

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    # Patch the package-level binding the solver imports at call time.
    monkeypatch.setattr(_cvx, "classify_model", counting)

    m = from_nl(str(_NL_DIR / "fac2.nl"))
    res = solve_model(m, time_limit=60.0, gap_tolerance=1e-4)

    assert str(res.status) == "optimal"
    # The certified objective pins the optimum (correctness). Exact node-count
    # bound-neutrality is verified same-machine by check_cert_neutrality; the raw
    # count is platform-FP-dependent for this nonconvex instance (39 local vs 41
    # CI), so here we only assert it did not blow up.
    assert res.node_count < 100
    assert res.objective == pytest.approx(331837498.18201387, rel=1e-9)
    # Pre-#723 this was 4; the dedup brings it to 2 (one per solve_model entry).
    assert calls["n"] <= 2, f"convexity classified {calls['n']}x (expected <= 2)"


def test_interval_mul_zero_times_inf_still_sound():
    """#723 fast-path in Interval.__mul__ keeps the 0*inf NaN convention (C-36)."""
    from discopt._jax.convexity.interval import Interval

    # [0,0] . [-inf, inf] -> [0,0] (not NaN): the 0-factor convention.
    # (up to one ULP of outward rounding on each side).
    r = Interval.from_bounds(0.0, 0.0) * Interval.from_bounds(-np.inf, np.inf)
    assert not np.isnan(r.lo) and not np.isnan(r.hi)
    assert r.lo <= 0.0 <= r.hi
    assert r.lo == pytest.approx(0.0, abs=1e-300)
    assert r.hi == pytest.approx(0.0, abs=1e-300)

    # [0,5] . [-inf, inf] -> [-inf, inf]: genuine +-inf corners still dominate.
    r2 = Interval.from_bounds(0.0, 5.0) * Interval.from_bounds(-np.inf, np.inf)
    assert r2.lo == -np.inf and r2.hi == np.inf

    # Finite fast path is identical to the pointwise product enclosure.
    r3 = Interval.from_bounds(-2.0, 3.0) * Interval.from_bounds(-1.0, 4.0)
    assert r3.lo <= -8.0 <= r3.hi  # -2 * 4 = -8 is the min corner
    assert r3.lo <= 12.0 <= r3.hi  # 3 * 4 = 12 is the max corner
