"""Tests for the rectangular AC-OPF builder (Wave 2, W5 capstone).

AC-OPF is the flagship nonconvex QCQP application. Correctness is validated two
ways: the power-injection formulas must equal ``Re/Im[V_i conj(Y V)]`` exactly,
and discopt's global optimum on a small case must match an independent power-flow
solve (scipy). The line losses make the optimal generation strictly exceed the
load, so the answer is non-trivial.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt.opf import admittance_matrix, build_ac_opf_rectangular, two_bus_example


def test_power_injection_formula_matches_complex_power():
    ac = two_bus_example()
    G, B = admittance_matrix(ac)
    Y = G + 1j * B
    n = len(ac.buses)
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = rng.uniform(0.95, 1.05, n)
        f = rng.uniform(-0.1, 0.1, n)
        V = e + 1j * f
        S = V * np.conj(Y @ V)
        P = np.zeros(n)
        Q = np.zeros(n)
        for i in range(n):
            for k in range(n):
                P[i] += G[i, k] * (e[i] * e[k] + f[i] * f[k]) + B[i, k] * (
                    f[i] * e[k] - e[i] * f[k]
                )
                Q[i] += G[i, k] * (f[i] * e[k] - e[i] * f[k]) - B[i, k] * (
                    e[i] * e[k] + f[i] * f[k]
                )
        np.testing.assert_allclose(P, S.real, atol=1e-9)
        np.testing.assert_allclose(Q, S.imag, atol=1e-9)


def test_admittance_matrix_is_symmetric_singular():
    ac = two_bus_example()
    G, B = admittance_matrix(ac)
    Y = G + 1j * B
    assert np.allclose(Y, Y.T)
    # A pure series-line network has a singular Y-bus (rows sum to zero).
    assert np.allclose(Y.sum(axis=1), 0.0)


@pytest.mark.slow
def test_two_bus_opf_matches_independent_power_flow():
    from scipy.optimize import fsolve

    ac = two_bus_example()
    G, B = admittance_matrix(ac)
    Y = G + 1j * B

    # Independent reference: solve the bus-2 power balance for (e2, f2), then the
    # slack generation Pg1 = P1 is the (cost) objective.
    def eqs(x):
        V = np.array([1.0, x[0]]) + 1j * np.array([0.0, x[1]])
        S = V * np.conj(Y @ V)
        return [S.real[1] + 0.5, S.imag[1] + 0.2]

    e2, f2 = fsolve(eqs, [1.0, 0.0])
    V = np.array([1.0, e2]) + 1j * np.array([0.0, f2])
    pg1_ref = float((V * np.conj(Y @ V)).real[0])
    assert pg1_ref > 0.5  # losses make generation exceed the 0.5 pu load

    res = build_ac_opf_rectangular(ac).solve(time_limit=90)
    assert res.status == "optimal"
    assert abs(float(res.objective) - pg1_ref) < 1e-3
