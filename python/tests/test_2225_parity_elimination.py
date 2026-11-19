"""#2225: exact integer certificates through continuous columns, in the IIS engine.

A mod-2 certificate (``_parity_infeasible``) needs every column of its rows held
integer. When an IIS probe relaxes an integrality, the contradiction can hide behind
the relaxed (continuous) columns: they cancel in a combination of equality rows, and
what is left is an integer row ``d . x = beta`` with ``gcd(d)`` not dividing ``beta``,
or a parity contradiction. ``_elimination_certificate`` eliminates those columns by
exact rational row operations and re-checks any certificate on the original data
(``_verify_integer_certificate``). Bounds play no part, so the proof holds for every
engine and every box.

Before: such probes went to the MILP solver. The Rust engine searched 10^6 nodes
without deciding the MIPcc27 ``enlight4`` probes and the entry wrote the whole model
(enlight11 too). After: the probe is decided without a solve, and both instances get
complete, certified, checker-passing IISs (enlight4 19 elements in 3 s, enlight11 96 in
8 s; HiGHS's engine had 188, incomplete, at 120 s).
"""

from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from discopt import _iis_milp
from discopt._iis_milp import (
    _elimination_certificate,
    _find_elimination_certificate,
    _verify_integer_certificate,
)

DATA = Path(__file__).parent / "data" / "iis_2225_enlight4_probe.npz"


def test_no_certificate_on_systems_feasible_by_construction():
    """Soundness: right-hand sides built as ``A x*`` with ``x*`` integer on the held
    columns. Neither the raw search nor the verified result may claim infeasibility;
    the same systems with one right-hand side moved by 1 are often certified, so the
    search is exercised (each such certificate is a verified proof)."""
    rng = np.random.default_rng(2225)
    feasible = certified = 0
    for _ in range(2000):
        m, n = int(rng.integers(2, 8)), int(rng.integers(2, 9))
        A = rng.integers(-3, 4, size=(m, n)).astype(float)
        A[rng.random((m, n)) < 0.4] = 0.0
        held = rng.random(n) < 0.6
        x = np.where(held, rng.integers(-5, 6, n), rng.integers(-20, 21, n) / 4.0)
        b = A @ x
        if not np.all(b == np.round(b)):
            continue
        As = sp.csr_matrix(A)
        assert _find_elimination_certificate(As, b, b, held, 200_000) is None
        assert _elimination_certificate(As, b, b, held) is None
        feasible += 1
        b2 = b.copy()
        b2[0] += 1.0
        if _elimination_certificate(As, b2, b2, held) is not None:
            certified += 1
    assert feasible > 400 and certified > 100, (feasible, certified)


def test_gcd_certificate_through_a_continuous_column():
    """x0 + x1 + y = 1, x1 + x2 - y = 1, x0 + x2 = 1 (x integer, y continuous): the
    first two give x0 + 2 x1 + x2 = 2, less the third, 2 x1 = 1."""
    A = sp.csr_matrix(np.array([[1, 1, 0, 1], [0, 1, 1, -1], [1, 0, 1, 0]], dtype=float))
    b = np.ones(3)
    held = np.array([True, True, True, False])
    lam = _elimination_certificate(A, b, b, held)
    assert lam is not None and _verify_integer_certificate(A, b, b, held, lam)
    # y integer as well: still infeasible (the same certificate needs no elimination).
    assert _elimination_certificate(A, b, b, np.ones(4, dtype=bool)) is not None
    # x1 continuous too: 2 x1 = 1 has a solution, nothing to certify.
    assert _elimination_certificate(A, b, b, np.array([True, False, True, False])) is None


def test_verifier_rejects_what_is_not_a_proof():
    A = sp.csr_matrix(np.array([[1, 1, 0, 1], [0, 1, 1, -1], [1, 0, 1, 0]], dtype=float))
    b = np.ones(3)
    held = np.array([True, True, True, False])
    good = {0: Fraction(1), 1: Fraction(1), 2: Fraction(-1)}
    assert _verify_integer_certificate(A, b, b, held, good)
    # Leaves the continuous column in the combination.
    assert not _verify_integer_certificate(A, b, b, held, {0: Fraction(1), 2: Fraction(-1)})
    # A free-sign multiplier on a row that is not an equality.
    hi = b.copy()
    hi[2] = 5.0
    assert not _verify_integer_certificate(A, b, hi, held, good)
    assert not _verify_integer_certificate(A, b, b, held, {})


def _probe_engine():
    z = np.load(DATA)
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    rlo, rhi, lb, ub, integ = z["row_lo"], z["row_hi"], z["col_lb"], z["col_ub"], z["is_int"]
    m = A.shape[0]
    big = 1e20
    eng = _iis_milp._Engine(
        A.tocsr(),
        np.where(np.isfinite(rlo), rlo, -big),
        np.where(np.isfinite(rhi), rhi, big),
        np.where(np.isfinite(lb), lb, -big),
        np.where(np.isfinite(ub), ub, big),
        integ.astype(bool),
        {i: np.array([i]) for i in range(m)},
        probe_tl=5.0,
        engine="rust",
    )
    cons = {i for i in range(m) if np.isfinite(rlo[i]) or np.isfinite(rhi[i])}
    bounds = {(j, "lower") for j in range(lb.size) if np.isfinite(lb[j])}
    bounds |= {(j, "upper") for j in range(ub.size) if np.isfinite(ub[j])}
    ints = set(np.flatnonzero(integ).tolist())
    return eng, cons, bounds, ints


def test_enlight4_probe_is_decided_without_a_solve():
    """The captured MIPcc27 enlight4 probe: three free continuous columns, eight
    equality rows. Before, the Rust engine's probe searched until its budget and came
    back undecided; now the certificate decides it with no solve."""
    eng, cons, bounds, ints = _probe_engine()
    t = time.perf_counter()
    verdict = eng.probe(cons, bounds, ints)
    assert verdict == (True, None)
    assert eng.n_solves == 0 and eng.n_parity == 1 and eng.n_parity_elim == 1
    assert time.perf_counter() - t < 2.0
    # The final confirmation of such a set is answered by the same certificate.
    assert eng.confirm_infeasible(cons, bounds, ints) is True
    assert eng.n_solves == 0


@pytest.mark.requires_highs
def test_highs_agrees_on_the_probe():
    """The oracle for the fixture: HiGHS proves the probe infeasible."""
    pytest.importorskip("highspy")
    from discopt._iis_backend import INFEASIBLE, make_backend

    z = np.load(DATA)
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    args = (z["row_lo"], z["row_hi"], z["col_lb"], z["col_ub"], z["is_int"])
    b = make_backend("highs", A, *args)
    b.configure(*args)
    assert b.run("choose", time.perf_counter() + 60.0) == INFEASIBLE
