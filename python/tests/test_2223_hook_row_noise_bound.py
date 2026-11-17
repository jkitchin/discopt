"""Issue #2223: rows the lazy and node hooks append are never seen by the #1296 gate.

The #1296 tiny-entry gate (``has_unscalable_tiny_entry``) runs once, on the caller's
rows at entry. Rows a lazy or fractional separator returns are folded in later and
are never gated, so a noise-floor entry that would withdraw the certificate of a
static solve reaches the lazy master's node LPs unchecked. The question #2223 asked
is whether that can produce a bound above the optimum.

Measured (planning probe, then this test's population): the #1296 big-M witness and
random ill-scaled models whose every noise entry lives only in hook rows, against
the same models in well-scaled variables (``y' = eps * y``) solved statically:
92 bound comparisons, 0 bounds above the optimum, 0 wrong certificates. The static
arm of the same witness is withdrawn by the gate (bound ``None``), so the gate is
what protects the static path; the lazy path's node LPs are verified against the
original data (#1945), and its root presolve and root cuts ran before the rows
existed. This pins that measurement so a change to either can not silently turn it
into a false bound.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solvers import SolveStatus

ms = pytest.importorskip("discopt.solvers.milp_simplex")


def _instance(rng):
    nb, k = 6, 3
    eps = 10.0 ** -int(rng.integers(11, 14))
    W = rng.integers(1, 10, (k, nb)).astype(float)
    cap = W.sum(1) * 0.3
    v = rng.integers(1, 10, nb).astype(float)
    ct = rng.uniform(0.1, 3.0, k)
    n = nb + 2 * k  # x (binary), y (continuous, ill-scaled), t (continuous)
    c = np.concatenate([-v, np.zeros(k), ct])
    A1 = np.zeros((2 * k, n))
    b1 = np.zeros(2 * k)
    for i in range(k):
        A1[i, :nb] = W[i]
        A1[i, nb + i] = -1.0
        b1[i] = cap[i]
        A1[k + i, nb + i] = 1.0
        A1[k + i, nb + k + i] = -1.0
    A2 = A1.copy()
    A2[:k, nb : nb + k] *= eps
    for i in range(k):
        A2[k + i, nb + k + i] = -1.0 / eps
    bounds = [(0, 1)] * nb + [(0, None)] * k + [(0, 10)] * k
    integ = np.array([1] * nb + [0] * (2 * k))
    return c, A1, A2, b1, bounds, integ, nb


@pytest.mark.parametrize("node", [False, True], ids=["lazy_rows", "lazy_and_node_rows"])
def test_noise_entries_in_hook_rows_never_lift_the_bound_past_the_optimum(node):
    rng = np.random.default_rng(1296)
    compared = 0
    for _ in range(12):
        c, A1, A2, b1, bounds, integ, nb = _instance(rng)
        ref = ms.solve_milp(
            c, A_ub=A1, b_ub=b1, bounds=bounds, integrality=integ, gap_tolerance=1e-9
        )
        assert ref.status == SolveStatus.OPTIMAL
        opt = float(ref.objective)
        rows = [(A2[i].copy(), float(b1[i])) for i in range(A2.shape[0])]

        def sep(x, rows=rows):
            out = [r for r in rows if r[0] @ x > r[1] + 1e-7 * max(1.0, np.abs(r[0]).max())]
            return out or None

        extra = dict(node_callback=sep, node_hook_rounds=2, node_hook_cut_cap=50) if node else {}
        A0 = np.zeros((1, len(c)))
        A0[0, :nb] = 1.0
        r = ms.solve_milp_with_lazy_cuts(
            c,
            A_ub=A0,
            b_ub=np.array([float(nb)]),
            bounds=bounds,
            integrality=integ,
            lazy_callback=sep,
            gap_tolerance=1e-9,
            time_limit=20,
            **extra,
        )
        # The noise rows really went through the hook (CLAUDE.md section 6).
        assert r.callback_stats["lazy_cuts"] + r.callback_stats.get("node_cuts", 0) > 0
        if r.bound is not None:
            compared += 1
            assert r.bound <= opt + 1e-6 * max(1.0, abs(opt)), (r.bound, opt, r.status)
        if r.status == SolveStatus.OPTIMAL:
            assert r.objective == pytest.approx(opt, abs=1e-5 * max(1.0, abs(opt)))
    assert compared > 0, "no bound was compared"
