"""The exact lattice (integer Farkas) test used by the MILP IIS engine.

``lattice_infeasible(rows, held)`` decides whether integer equality rows have a
solution with the ``held`` columns integer and every other column real, bounds
ignored. Its "infeasible" verdict lets the IIS filter drop elements with no solve,
so it is checked against an independent brute force on random small systems.
"""

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._iis_lattice import lattice_core, lattice_infeasible
from discopt._iis_milp import milp_iis


def _brute_feasible(rows, held, ncols, box=4):
    """Some integer assignment of the held columns in [-box, box] leaves the
    real columns' linear system consistent (exact, via rational rank)."""
    from fractions import Fraction

    hcols = sorted(held)
    ccols = [j for j in range(ncols) if j not in held]
    for z in itertools.product(range(-box, box + 1), repeat=len(hcols)):
        val = dict(zip(hcols, z))
        # residual system over the real columns: C y = r
        M = [
            [Fraction(r.get(j, 0)) for j in ccols]
            + [Fraction(b - sum(v * val[j] for j, v in r.items() if j in val))]
            for r, b in rows
        ]
        # Gauss-Jordan; inconsistent iff a row reduces to [0 ... 0 | nonzero]
        rk = 0
        for c in range(len(ccols)):
            p = next((i for i in range(rk, len(M)) if M[i][c] != 0), None)
            if p is None:
                continue
            M[rk], M[p] = M[p], M[rk]
            for i in range(len(M)):
                if i != rk and M[i][c] != 0:
                    f = M[i][c] / M[rk][c]
                    M[i] = [a - f * bb for a, bb in zip(M[i], M[rk])]
            rk += 1
        if all(row[-1] == 0 for row in M[rk:]):
            return True
    return False


def test_infeasible_verdicts_are_sound_against_brute_force():
    """Every "infeasible" verdict on random small systems: no solution anywhere
    in the brute-force box. (A "feasible" verdict is not checked here: its
    solution can lie far outside any small box.)"""
    rng = np.random.default_rng(2024)
    n_infeasible = n_feasible = 0
    for _ in range(400):
        ncols = int(rng.integers(1, 4))
        nrows = int(rng.integers(1, 4))
        rows = []
        for _ in range(nrows):
            coef = {j: int(v) for j in range(ncols) if (v := rng.integers(-3, 4))}
            rows.append((coef, int(rng.integers(-4, 5))))
        held = {j for j in range(ncols) if rng.random() < 0.7}
        verdict = lattice_infeasible(rows, held)
        assert verdict is not None
        if verdict:
            assert not _brute_feasible(rows, held, ncols), (rows, held)
            n_infeasible += 1
        else:
            n_feasible += 1
    assert n_infeasible >= 20 and n_feasible >= 20, (n_infeasible, n_feasible)


def test_feasible_by_construction_is_never_called_infeasible():
    """Right-hand sides built from an integer point (held columns) and a rational
    one (the rest): the system has a solution, so the verdict must be False."""
    from fractions import Fraction

    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(400):
        ncols = int(rng.integers(1, 7))
        held = {j for j in range(ncols) if rng.random() < 0.6}
        point = {
            j: Fraction(int(rng.integers(-9, 10)))
            if j in held
            else Fraction(int(rng.integers(-9, 10)), int(rng.integers(1, 4)))
            for j in range(ncols)
        }
        rows = []
        for _ in range(int(rng.integers(1, 6))):
            coef = {j: int(v) for j in range(ncols) if (v := rng.integers(-5, 6))}
            b = sum(v * point[j] for j, v in coef.items())
            if b.denominator != 1:  # the test takes integer data only
                coef = {j: v * b.denominator for j, v in coef.items()}
                b *= b.denominator
            rows.append((coef, int(b)))
        assert lattice_infeasible(rows, held) is False, (rows, held, point)
        checked += 1
    assert checked == 400


def test_known_cases():
    # 2x + 4y = 3: infeasible over the integers.
    assert lattice_infeasible([({0: 2, 1: 4}, 3)], {0, 1}) is True
    # ... feasible once y is real.
    assert lattice_infeasible([({0: 2, 1: 4}, 3)], {0}) is False
    # x + y = 1, x - y = 0: 2x = 1 (mod-2 contradiction).
    assert lattice_infeasible([({0: 1, 1: 1}, 1), ({0: 1, 1: -1}, 0)], {0, 1}) is True
    # 2x + 2y - z = 1, z - 2w = 0, z real: projecting z out leaves 2x + 2y - 2w = 1.
    rows = [({0: 2, 1: 2, 2: -1}, 1), ({2: 1, 3: -2}, 0)]
    assert lattice_infeasible(rows, {0, 1, 3}) is True
    assert lattice_infeasible(rows, {0, 1}) is False
    assert lattice_infeasible([], {0}) is False


def test_core_is_minimal():
    rows = {
        10: ({0: 2, 1: 2, 2: -1}, 1),
        11: ({2: 1, 3: -2}, 0),
        12: ({4: 1, 5: 1}, 3),  # irrelevant
    }
    cons, ints = lattice_core(rows, {0, 1, 3, 4, 5}, orders=4)
    assert cons == {10, 11}
    sub = [rows[c] for c in sorted(cons)]
    assert lattice_infeasible(sub, ints) is True
    for c in cons:
        assert lattice_infeasible([rows[k] for k in sorted(cons - {c})], ints) is False
    for j in ints:
        assert lattice_infeasible(sub, ints - {j}) is False


@pytest.mark.smoke
def test_engine_uses_lattice_where_parity_cannot():
    """Rows with a continuous column were outside the mod-2 test; the lattice test
    projects the column out, gives the candidate, and proves the filter's probes
    without a solve."""
    pytest.importorskip("highspy")
    m = dm.Model("lat")
    x = m.integer("x", lb=-50, ub=50)
    y = m.integer("y", lb=-50, ub=50)
    w = m.integer("w", lb=-50, ub=50)
    z = m.continuous("z", lb=-100, ub=100)
    v = m.integer("v", shape=(6,), lb=0, ub=4)
    m.subject_to(2 * x + 2 * y - z == 1, name="a")
    m.subject_to(z - 2 * w == 0, name="b")
    for k in range(6):
        m.subject_to(v[k] + x <= 40, name=f"noise{k}")
    m.minimize(0 * x)
    res = milp_iis(m)
    names = sorted(m._constraints[i].name for i in res.constraints)
    assert names == ["a", "b"]
    assert res.parity_rows == 2 and res.n_lattice >= 1
    assert res.complete and res.proven_irreducible
    # No bound is needed: the lattice argument ignores them.
    assert res.bounds == []
    assert len(res.certificates) == len(res.constraints) + len(res.integrality)
