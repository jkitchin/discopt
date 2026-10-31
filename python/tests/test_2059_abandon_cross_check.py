"""#2059: an abandoned HiGHS ``lp_nlp_bb`` master may confirm its bound.

When the #1066 guard hands the convex-MINLP route over, the master's termination
hook answers ``"abandon"``. Until #2059 the #1634 presolve-free cross-solve then
got no time at all, so the route's bound was always withdrawn and the solve
published only the fallback's -- on portfol_classical050_1 at 120 s, the
continuous bound -0.097767 instead of the route's -0.096817.

``DISCOPT_ROUTE_ABANDON_CROSS_CHECK`` (default ON since #2059) gives that
cross-solve a share of the remaining budget (``oa._ABANDON_CROSS_CHECK_FRACTION``).
These tests pin:

* ON (the default): the cross-solve runs, the bound it publishes is a valid
  lower bound (never above the known optimum), and the abandoned master still
  hands over promptly;
* ``=0``: exactly the old behaviour -- no cross-solve, bound withdrawn;
* the share is validated and stays inside [0, 1].
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

pytest.importorskip("highspy")

import discopt.modeling as dm  # noqa: E402
from discopt.solvers import milp_highs, oa  # noqa: E402

#: The n=20 big-M portfolio's optimum (``test_1658``'s reproducer).
_OPT = 0.0104420249


def _portfolio(n=20, seed=21, K=3):
    rng = np.random.default_rng(seed)
    mu = np.round(rng.uniform(0.04, 0.14, n), 3)
    B = np.round(rng.normal(0, 1, (n, 2)) * [0.15, 0.08], 3)
    d = np.round(rng.uniform(0.15, 0.35, n) ** 2, 4)
    m = dm.Model(f"portfolio_{n}")
    x = m.continuous("x", shape=(n,), lb=0, ub=0.4)
    z = m.binary("z", shape=(n,))
    f = m.continuous("f", shape=(2,), lb=-1, ub=1)
    for k in range(2):
        m.subject_to(f[k] == dm.sum(lambda i: B[i, k] * x[i], over=range(n)))
    m.minimize(f[0] ** 2 + f[1] ** 2 + dm.sum(lambda i: d[i] * x[i] ** 2, over=range(n)))
    m.subject_to(dm.sum(lambda i: x[i], over=range(n)) == 1)
    m.subject_to(dm.sum(lambda i: mu[i] * x[i], over=range(n)) >= 0.10)
    m.subject_to(dm.sum(lambda i: z[i], over=range(n)) <= K)
    for i in range(n):
        m.subject_to(x[i] <= 0.4 * z[i])
        m.subject_to(x[i] >= 0.1 * z[i])
    return m


def _abandon_at_first_check_in(monkeypatch, flag):
    from discopt.solvers.mip_nlp import solve_mip_nlp

    if flag is None:
        monkeypatch.delenv("DISCOPT_ROUTE_ABANDON_CROSS_CHECK", raising=False)
    else:
        monkeypatch.setenv("DISCOPT_ROUTE_ABANDON_CROSS_CHECK", flag)
    calls = []

    def hook(ctx):
        calls.append(ctx["elapsed"])
        return True

    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(
            _portfolio(),
            method="lp_nlp_bb",
            milp_solver="highs",
            time_limit=60,
            mip_nlp_options={"termination_hook": hook},
        )
    wall = time.perf_counter() - t
    assert calls, "the hook was never consulted (CLAUDE.md section 6)"
    stats = r.mip_nlp_trace["summary"]["callback_stats"]
    assert stats["abandoned"] is True, "the hook stop was not an abandon"
    return r, stats["presolve_cross_check"], wall


def test_opt_out_still_withdraws_the_abandoned_bound(monkeypatch):
    r, cross, _ = _abandon_at_first_check_in(monkeypatch, "0")
    assert cross.get("ran") is not True, cross
    assert "no time budget" in str(cross.get("withdrawn")), cross
    assert r.bound is None, r.bound


@pytest.mark.parametrize("flag", [None, "1"], ids=["default", "on"])
def test_on_confirms_and_publishes_the_abandoned_bound(monkeypatch, flag):
    """Fails before #2059: the abandoned master's cross-solve never ran, whatever
    the flag, so ``r.bound`` was ``None``."""
    r, cross, wall = _abandon_at_first_check_in(monkeypatch, flag)
    assert cross.get("ran") is True, cross
    assert "withdrawn" not in cross, cross
    assert r.bound is not None and np.isfinite(r.bound), (r.bound, cross)
    # Soundness: a published lower bound never passes the optimum or the incumbent.
    assert r.bound <= _OPT + 1e-7, (r.bound, _OPT)
    if r.objective is not None:
        assert r.bound <= r.objective + 1e-9, (r.bound, r.objective)
    # The share is a ceiling on a 60 s budget: the hand-over stays prompt.
    assert wall < 20.0, wall


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_share_outside_the_unit_interval_is_refused(bad):
    with pytest.raises(ValueError, match="abandon_cross_check_fraction"):
        milp_highs.solve_milp_with_lazy_cuts(
            np.array([1.0]),
            bounds=[(0.0, 1.0)],
            integrality=np.array([1]),
            lazy_callback=lambda x: [],
            abandon_cross_check_fraction=bad,
        )


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, oa._ABANDON_CROSS_CHECK_FRACTION),
        ("0", 0.0),
        ("1", oa._ABANDON_CROSS_CHECK_FRACTION),
        ("on", oa._ABANDON_CROSS_CHECK_FRACTION),
    ],
)
def test_flag_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("DISCOPT_ROUTE_ABANDON_CROSS_CHECK", raising=False)
    else:
        monkeypatch.setenv("DISCOPT_ROUTE_ABANDON_CROSS_CHECK", raw)
    assert oa._route_abandon_cross_check_fraction() == expected
    assert 0.0 <= oa._ABANDON_CROSS_CHECK_FRACTION <= 1.0
