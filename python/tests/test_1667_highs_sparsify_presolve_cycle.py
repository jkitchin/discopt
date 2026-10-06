"""#1667: ``m.solve(time_limit=20)`` on the HiGHS MILP route never returned.

A 10-unit x 24-h unit commitment (``u`` binary, start-up/shut-down ``v``, ``w``
continuous) solved by plain HiGHS from ``m.to_mps()`` in about 3 s; through the
route it was still inside ``h.run()`` after 300 s. The route hands HiGHS its slack
standard form ``a x + s = b``, and HiGHS's MIP presolve cycles on it in
``fastPresolveLoop`` -> ``rowPresolve`` (stack sample), a loop that polls neither
``time_limit`` nor the interrupt callbacks. Switching off sparsify, the aggregator
or doubleton equations each broke the cycle; the route now switches off sparsify
(:data:`lp_milp_highs.MILP_PRESOLVE_RULE_OFF`). Upstream bug: #1671.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
from discopt.solvers import lp_milp_highs as L
from discopt.solvers import milp_highs as MH

UC_OPT = 450623.7692106868


def test_sparsify_is_off_on_every_milp_route():
    assert L.MILP_PRESOLVE_RULE_OFF & (1 << 14)  # kPresolveRuleSparsify (#1667)
    assert L.MILP_PRESOLVE_RULE_OFF & (1 << 13)  # parallel rows/cols stays off (#1634)
    assert MH.MILP_PRESOLVE_RULE_OFF == L.MILP_PRESOLVE_RULE_OFF  # OA / GDP masters too


UC_SCRIPT = textwrap.dedent(
    """\
    import time
    import numpy as np
    import discopt, discopt.modeling as dm

    T, K, G = 24, 3, 10
    bus = np.array([0, 0, 0, 1, 1, 1, 2, 1, 2, 2])
    Pmin = np.array([150, 150, 80, 70, 70, 70, 70, 15, 15, 15.0])
    Pmax = np.array([400, 400, 220, 180, 180, 180, 180, 60, 60, 60.0])
    a_nl = np.array([3200, 3250, 2000, 2650, 2680, 2710, 2740, 1200, 1220, 1240.0])
    mc = np.array([[19.0, 20.5, 22.5], [19.5, 21.0, 23.0], [23.0, 25.0, 27.5],
                   [31.0, 33.0, 36.0], [31.5, 33.5, 36.5], [32.0, 34.0, 37.0],
                   [32.5, 34.5, 37.5], [55.0, 60.0, 68.0], [56.0, 61.0, 69.0],
                   [58.0, 63.0, 71.0]])
    S_up = np.array([12000, 12000, 5000, 2200, 2200, 2300, 2300, 300, 300, 300.0])
    UT = np.array([8, 8, 6, 4, 4, 4, 4, 1, 1, 1])
    DT = np.array([8, 8, 6, 3, 3, 3, 3, 1, 1, 1])
    RU = np.array([100, 100, 70, 90, 90, 90, 90, 60, 60, 60.0])
    SU = np.maximum(Pmin, RU)
    RS = np.array([50, 50, 35, 60, 60, 60, 60, 45, 45, 45.0])
    u0 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p0 = np.array([300, 280, 150, 120, 0, 0, 0, 0, 0, 0.0])
    seg = np.repeat((Pmax - Pmin)[:, None] / K, K, axis=1)
    load = np.array([900, 850, 820, 810, 830, 900, 1050, 1220, 1320, 1350, 1340, 1320,
                     1300, 1290, 1300, 1340, 1430, 1540, 1580, 1540, 1440, 1290, 1120, 980.0])
    solar = 550 * np.clip(np.sin(np.pi * (np.arange(T) - 6) / 13), 0, None)
    Dbus = np.array([0.10, 0.35, 0.55])[:, None] * load - np.r_[0, 0, 1][:, None] * solar
    cf = np.array([0.62, 0.66, 0.68, 0.70, 0.68, 0.63, 0.55, 0.46, 0.38, 0.32, 0.28, 0.25,
                   0.22, 0.20, 0.20, 0.22, 0.26, 0.31, 0.37, 0.44, 0.50, 0.55, 0.58, 0.60])
    wind = 450.0 * cf
    lines, xl, Fmax = [(0, 1), (0, 2), (1, 2)], 0.1, np.array([600.0, 520.0, 600.0])
    R = 0.06 * load + 0.10 * wind

    m = dm.Model("uc_1667")
    u = m.binary("u", shape=(G, T))
    v = m.continuous("v", shape=(G, T), lb=0, ub=1)
    w = m.continuous("w", shape=(G, T), lb=0, ub=1)
    up = lambda g, t: u[g, t - 1] if t > 0 else float(u0[g])
    for g in range(G):
        for t in range(T):
            m.subject_to(v[g, t] - w[g, t] == u[g, t] - up(g, t))
            m.subject_to(v[g, t] + w[g, t] <= 1)
            m.subject_to(dm.sum(lambda k: v[g, k], over=range(max(0, t - UT[g] + 1), t + 1))
                         <= u[g, t])
            m.subject_to(dm.sum(lambda k: w[g, k], over=range(max(0, t - DT[g] + 1), t + 1))
                         <= 1 - u[g, t])
    s = m.continuous("s", shape=(G, T, K), lb=0, ub=float(seg.max()))
    r = m.continuous("r", shape=(G, T), lb=0, ub=float(RS.max()))
    wu = m.continuous("wind", shape=(T,), lb=0, ub=450.0)
    th = m.continuous("theta", shape=(2, T), lb=-1.0, ub=1.0)
    P = lambda g, t: Pmin[g] * u[g, t] + dm.sum(lambda k: s[g, t, k], over=range(K))
    for t in range(T):
        m.subject_to(wu[t] <= float(wind[t]))
        for g in range(G):
            m.subject_to(P(g, t) + r[g, t] <= Pmax[g] * u[g, t])
            pp = P(g, t - 1) if t > 0 else float(p0[g])
            m.subject_to(P(g, t) - pp <= RU[g] * up(g, t) + SU[g] * v[g, t])
            m.subject_to(pp - P(g, t) <= RU[g] * u[g, t] + SU[g] * w[g, t])
            for k in range(K):
                m.subject_to(s[g, t, k] <= float(seg[g, k]) * u[g, t])
            m.subject_to(r[g, t] <= RS[g] * u[g, t])
        m.subject_to(dm.sum(lambda g: r[g, t], over=range(G)) >= float(R[t]))
        ang = [0.0, th[0, t], th[1, t]]
        F = [100 * (ang[i] - ang[j]) / xl for i, j in lines]
        for l in range(3):
            m.subject_to(F[l] <= Fmax[l])
            m.subject_to(-F[l] <= Fmax[l])
        for n in range(3):
            gen = dm.sum(lambda g: P(g, t), over=[g for g in range(G) if bus[g] == n])
            gen = gen + wu[t] if n == 0 else gen
            out = (sum(F[l] for l, (i, j) in enumerate(lines) if i == n)
                   - sum(F[l] for l, (i, j) in enumerate(lines) if j == n))
            m.subject_to(gen - out == float(Dbus[n, t]))
    energy = dm.sum(lambda g: dm.sum(lambda t: dm.sum(
        lambda k: float(mc[g, k]) * s[g, t, k], over=range(K)), over=range(T)), over=range(G))
    m.minimize(dm.sum(lambda g: dm.sum(lambda t: a_nl[g] * u[g, t] + S_up[g] * v[g, t],
                                       over=range(T)), over=range(G)) + energy)
    print("FILE", discopt.__file__, flush=True)
    t0 = time.perf_counter()
    res = m.solve(time_limit=20)
    wall = time.perf_counter() - t0
    print("RESULT", res.status, repr(res.objective), repr(res.bound), wall, flush=True)
    """
)


def test_unit_commitment_returns_within_its_time_limit():
    """Before the fix this never returned; the subprocess bounds the test, not the solve."""
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", UC_SCRIPT],
            capture_output=True,
            text=True,
            timeout=150,
            env=dict(os.environ),
        )
    except subprocess.TimeoutExpired:
        pytest.fail("m.solve(time_limit=20) did not return within 150 s (#1667)")
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, proc.stdout
    _, status, obj, bound, wall = line[0].split()
    assert status == "optimal"
    assert float(obj) == pytest.approx(UC_OPT, rel=1e-6)
    assert float(bound) <= float(obj) + 1e-6
    assert float(bound) >= float(obj) * (1 - 1e-4) - 1e-6
    # The time limit, plus the route's post-solve certificate cross-checks.
    assert float(wall) < 20.0 + 40.0
