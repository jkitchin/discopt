"""A measured zero is a result; the profile must be able to say it.

The layer profile answers "where did the wall clock go". After the NLP work
moved off JAX, the most valuable thing it can report is ``jax_time_fraction ==
0.0`` -- and that is exactly the value it could not express. Both producers
wrote ``x / wt if x else None``, so a measured 0.0 became ``None``;
``layer_profiling_summary`` filters ``None`` out; ``mean_jax_fraction`` came
back ``nan``, which reads as "never instrumented".

Measured on the global50 panel before this fix: ``jax_time_fraction`` was
``None`` on 50 of 50 instances, and ``rust_time_fraction`` on 7 of 50 (the fast
instances where Rust time rounded to zero). The field would have read exactly
the same if JAX had been running the whole time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from benchmarks.metrics import (  # noqa: E402
    SolveResult,
    SolveStatus,
    layer_profiling_summary,
    time_fraction,
)


def test_a_measured_zero_stays_zero():
    """The bug in one line: 0.0 in, 0.0 out -- not None."""
    assert time_fraction(0.0, 10.0) == 0.0


def test_an_unreported_layer_is_still_none():
    """The other direction. None must remain reserved for "no such measurement",
    which is the honest answer for every non-discopt solver adapter."""
    assert time_fraction(None, 10.0) is None


def test_ordinary_fractions_are_unchanged():
    assert time_fraction(2.5, 10.0) == pytest.approx(0.25)


def _result(**kw) -> SolveResult:
    base = {"instance": "i", "solver": "discopt", "status": SolveStatus.OPTIMAL}
    base.update(kw)
    return SolveResult(**base)


def test_all_zero_jax_aggregates_to_zero_not_nan():
    """The global50 case. Fifty runs that provably never touched JAX must
    summarise as 0.0; ``nan`` there is the instrument declining to report its
    own strongest result."""
    runs = [
        _result(jax_time_fraction=0.0, python_time_fraction=0.9, rust_time_fraction=0.1)
        for _ in range(50)
    ]
    summary = layer_profiling_summary(runs)

    assert summary["mean_jax_fraction"] == 0.0, (
        f"50 runs measured at exactly 0.0 summarised as {summary['mean_jax_fraction']}"
    )
    assert summary["n_profiled"] == 50.0


def test_nan_still_means_no_data():
    """``nan`` must keep its one honest meaning, or the assertion above is
    satisfiable by a summary that reports 0.0 for everything."""
    summary = layer_profiling_summary([_result()])
    assert summary["mean_jax_fraction"] != summary["mean_jax_fraction"]  # nan
    assert summary["n_profiled"] == 0.0


def test_pounce_has_a_bucket_in_the_summary():
    """POUNCE is where the NLP cost went after JAX left the solve path -- 78% of
    a tspn08 solve per ``discopt._timing``. Before this change the benchmark
    record had no field for it, so the dominant layer was invisible."""
    runs = [_result(pounce_time_fraction=0.78, python_time_fraction=0.2)]
    assert layer_profiling_summary(runs)["mean_pounce_fraction"] == pytest.approx(0.78)


def test_the_four_layers_are_not_peers():
    """A guard on the documented invariant, so nobody later "fixes" the profile
    by making the buckets sum to 1.0. jax <= python and pounce <= rust; summing
    all four double-counts, which is the pre-#921 defect this schema replaced.
    """
    r = _result(
        rust_time_fraction=0.6,
        python_time_fraction=0.4,
        pounce_time_fraction=0.55,
        jax_time_fraction=0.3,
    )
    assert r.jax_time_fraction <= r.python_time_fraction
    assert r.pounce_time_fraction <= r.rust_time_fraction
    assert r.rust_time_fraction + r.python_time_fraction == pytest.approx(1.0)
    total = sum(
        (
            r.rust_time_fraction,
            r.python_time_fraction,
            r.pounce_time_fraction,
            r.jax_time_fraction,
        )
    )
    assert total > 1.0, "if the four ever sum to 1.0 they have been made peers again"
