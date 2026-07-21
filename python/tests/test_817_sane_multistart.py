"""Regression tests for #817 (primal-path increment): the sane-start multistart
for the pure-continuous single-NLP path finds feasible incumbents on
unbounded-variable models where the single |x|<=10 box-center start stalls.

`_solve_continuous` starts the local NLP at the box-center clipped to |x|<=10.
On models with unbounded variables that magnitude sends the NLP into a bad,
infeasible basin (emfl050_3_3/pooling_foulds3tp/squfl010-080persp stall at obj
594/-320/217k, all infeasible), so the solve returns NO incumbent — while a start
at |x|<=0.5 converges to the feasible optimum (10.4/-8.0/503.6). With
``DISCOPT_SANE_MULTISTART=1`` the path tries small start magnitudes first and
keeps the best VERIFIED-feasible result.

Default OFF pending the §5 differential panel; flag-OFF behavior is unchanged.
The multistart runs the real corpus instances, so it is corpus-gated + slow.
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import pytest

BENCH = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))


def _solve(inst: str, flag: str, tl: float):
    env_prev = os.environ.get("DISCOPT_SANE_MULTISTART")
    os.environ["DISCOPT_SANE_MULTISTART"] = flag
    try:
        return dm.from_nl(str(BENCH / f"{inst}.nl")).solve(time_limit=tl)
    finally:
        if env_prev is None:
            os.environ.pop("DISCOPT_SANE_MULTISTART", None)
        else:
            os.environ["DISCOPT_SANE_MULTISTART"] = env_prev


@pytest.mark.slow
@pytest.mark.skipif(
    not (BENCH / "pooling_foulds3tp.nl").exists(),
    reason="pooling_foulds3tp.nl (benchmark corpus) absent",
)
def test_817_multistart_finds_incumbent_pooling():
    """pooling_foulds3tp (bilinear, opt -8.0): flag OFF finds NO incumbent; flag
    ON finds the feasible optimum."""
    off = _solve("pooling_foulds3tp", "0", 12.0)
    on = _solve("pooling_foulds3tp", "1", 20.0)
    assert off.objective is None, (
        f"baseline unexpectedly found an incumbent ({off.objective}); repro drifted"
    )
    assert on.objective is not None and abs(on.objective - (-8.0)) < 1e-2, (
        f"#817: sane multistart failed to find pooling optimum -8.0 (got {on.objective})"
    )


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(
    not (BENCH / "emfl050_3_3.nl").exists(),
    reason="emfl050_3_3.nl (benchmark corpus) absent",
)
def test_817_multistart_finds_incumbent_emfl():
    """emfl050_3_3 (facility, opt 10.402): flag ON finds the feasible optimum
    (needs a longer budget than the big-model root setup leaves at 12s)."""
    on = _solve("emfl050_3_3", "1", 30.0)
    assert on.objective is not None and abs(on.objective - 10.402) < 1e-1, (
        f"#817: sane multistart failed to find emfl optimum 10.402 (got {on.objective})"
    )
    # Any incumbent returned must be genuinely feasible (never a false primal).
    if on.objective is not None and on.x is not None:
        import numpy as np
        from discopt._jax.nlp_evaluator import cached_evaluator
        from discopt._jax.primal_heuristics import _check_constraint_feasibility

        model = dm.from_nl(str(BENCH / "emfl050_3_3.nl"))
        ev = cached_evaluator(model)
        flat = np.concatenate(
            [
                np.atleast_1d(np.asarray(on.x[v.name], dtype=np.float64)).ravel()
                for v in model._variables
            ]
        )
        assert _check_constraint_feasibility(ev, flat, tol=1e-3), (
            "#817: returned incumbent is infeasible in the original model"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not (BENCH / "ex4_1_1.nl").exists(),
    reason="ex4_1_1.nl (benchmark corpus) absent",
)
def test_817_multistart_does_not_regress_control():
    """A finite-bounded control (ex4_1_1, opt -7.487): flag ON must return the same
    optimum as OFF (the multistart early-stops on the first converged start)."""
    off = _solve("ex4_1_1", "0", 8.0)
    on = _solve("ex4_1_1", "1", 8.0)
    assert off.objective is not None and on.objective is not None
    assert abs(on.objective - off.objective) < 1e-4, (
        f"#817 over-multistart regressed a control: OFF={off.objective} ON={on.objective}"
    )
