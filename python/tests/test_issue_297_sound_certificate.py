"""Regression for issue #297: st_ph10 must not return a false-optimal certificate.

`st_ph10` (MINLPLib, minimization, known optimum -10.5) was deterministically
returned as ``optimal`` with ``gap=0`` at an incumbent of -28.0556 — *below*
discopt's own (correct) dual lower bound of -10.5. For a minimization the
incumbent objective must be >= the lower bound, so ``obj < bound`` is an internally
inconsistent, unsound certificate (the most severe class).

This guards the soundness invariant robustly (no dependence on the exact optimum
being reached): a returned certificate must never place the incumbent below the
dual bound, and a ``gap=0`` ``optimal`` must sit at the true optimum -10.5.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")


@pytest.mark.requires_pounce
def test_st_ph10_certificate_is_sound():
    """The returned certificate must be self-consistent: the incumbent is never
    below the dual bound, and a ``gap=0`` ``optimal`` is at the true optimum."""
    path = os.path.join(_DATA, "st_ph10.nl")
    if not os.path.exists(path):
        pytest.skip("st_ph10 instance unavailable")
    r = dm.from_nl(path).solve(time_limit=10, gap_tolerance=1e-4)

    # Minimization: a sound solve never reports an incumbent below its dual bound.
    if r.objective is not None and r.bound is not None:
        assert r.objective >= r.bound - 1e-4 * (1.0 + abs(r.bound)), (
            f"unsound: incumbent {r.objective} is below the dual bound {r.bound} "
            "(issue #297 false-optimal)"
        )

    # A certified optimum must be the real optimum, not a fabricated gap=0 point.
    if r.status == "optimal":
        assert r.objective == pytest.approx(-10.5, abs=1e-2), (
            f"certified optimal at {r.objective}, expected the true optimum -10.5"
        )
