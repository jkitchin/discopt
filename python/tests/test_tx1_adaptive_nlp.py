"""TX1 regression suite — adaptive back-off for the strided in-tree node NLP.

``docs/dev/tenx-plan.md`` §3 (TX1). The strided node-NLP is a *pure primal
heuristic*: it fires only where the McCormick LP relaxer supplies the node dual
bound and the model is nonconvex (``solve_model``'s ``_gate_node_nlp``). There its
objective is never a bound — the LP is — so adaptively throttling it can only
change *incumbent arrival* (node counts, wall), never the certificate. TX0
measured this bucket as idle waste on integer-heavy nonconvex models (nvs09:
14.3 s skippable, identical proof/bound).

``DISCOPT_ADAPTIVE_NLP`` graduated to **default-ON** in G2
(``docs/dev/baron-gap-plan.md`` §4). Flag-graduation convention: the adaptive
back-off is now the default; ``DISCOPT_ADAPTIVE_NLP=0`` restores today's fixed
``node_nlp_stride``.

Soundness contract asserted here:
  * On a **convex** model the flag is inert — the gated regime requires
    ``not _model_is_convex``, so bound + node_count are byte-identical on/off.
  * On a **nonconvex heuristic** model (nvs09) the flag preserves the optimum and
    yields a valid dual bound (``bound <= optimum`` for min), never a false
    certificate.
  * The back-off mechanism actually fires (effective stride grows when the
    node-NLP stops improving the incumbent).

Marked ``slow``: real vendored solves.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

# nvs09 BARON-confirmed optimum (minlplib.solu; matches this repo's measured
# certified objective to 1e-6).
_NVS09_OPT = -43.13433691803531
_ABS = 1e-6
_REL = 1e-4


def _solve(name: str, *, adaptive: bool, tl: float, monkeypatch):
    monkeypatch.setenv("DISCOPT_ADAPTIVE_NLP", "1" if adaptive else "0")
    model = from_nl(os.path.join(_DATA, f"{name}.nl"))
    return model.solve(time_limit=tl)


def test_adaptive_nlp_default_direction(monkeypatch):
    """G2 flip: the adaptive back-off is now default-ON; ``=0`` restores fixed.

    Fast, load-immune unit test on the tuning default itself (no solve).
    ``docs/dev/baron-gap-plan.md`` §4 graduated ``DISCOPT_ADAPTIVE_NLP`` from
    default-OFF to default-ON; the flag-graduation convention keeps ``=0`` as the
    explicit escape hatch back to today's fixed ``node_nlp_stride``.
    """
    from discopt.solver_tuning import SolverTuning

    # Unset env -> the new default is ON (adaptive back-off engaged).
    monkeypatch.delenv("DISCOPT_ADAPTIVE_NLP", raising=False)
    assert SolverTuning().adaptive_nlp is True

    # =0 restores the fixed stride (explicit opt-out still honored).
    monkeypatch.setenv("DISCOPT_ADAPTIVE_NLP", "0")
    assert SolverTuning().adaptive_nlp is False

    # =1 is still ON (idempotent with the default).
    monkeypatch.setenv("DISCOPT_ADAPTIVE_NLP", "1")
    assert SolverTuning().adaptive_nlp is True


@pytest.mark.slow
def test_adaptive_nlp_convex_inert(monkeypatch):
    """On a convex model the flag never engages: bound + node_count identical."""
    off = _solve("cvxnonsep_psig30", adaptive=False, tl=30, monkeypatch=monkeypatch)
    on = _solve("cvxnonsep_psig30", adaptive=True, tl=30, monkeypatch=monkeypatch)
    assert off.status == on.status == "optimal"
    # Byte-identical certificate: the adaptive path is gated to nonconvex nodes,
    # so a convex solve must be untouched (both bound and search tree).
    assert on.node_count == off.node_count
    assert on.bound == off.bound


@pytest.mark.slow
def test_adaptive_nlp_nonconvex_sound_and_backoff_fires(monkeypatch, caplog):
    """On nvs09 the flag preserves the optimum + a valid bound, and backs off."""
    with caplog.at_level(logging.DEBUG, logger="discopt.solver"):
        res = _solve("nvs09", adaptive=True, tl=30, monkeypatch=monkeypatch)

    # Not an error / not false-infeasible.
    assert res.status in ("optimal", "feasible")
    assert res.objective is not None
    tol = _ABS + _REL * abs(_NVS09_OPT)
    # Incumbent is the true optimum (never beats it — no false-feasible).
    assert res.objective >= _NVS09_OPT - tol
    assert abs(res.objective - _NVS09_OPT) <= tol
    # Dual bound is a valid lower bound (min sense): never crosses the optimum.
    if res.bound is not None:
        assert res.bound <= _NVS09_OPT + tol

    # The adaptive back-off actually engaged (effective stride grew at least once).
    assert any(
        "TX1 adaptive node-NLP" in rec.getMessage() and "back off" in rec.getMessage()
        for rec in caplog.records
    ), "expected the adaptive node-NLP back-off to fire on nvs09"
