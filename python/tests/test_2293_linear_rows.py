"""#2293: affine constraint rows reach POUNCE as coefficients, not as a taped expression.

A three-term linear row (``x[k+1,s] - x[k,s] - h*x[k+1,s+1] == 0``) cost ~600 B in
the POUNCE problem plus ~600 B of transient ``NlExpr`` nodes when taped. The
transient part stayed behind as glibc fragmentation, so the evaluator held
~2.2 KiB/row. ``build_nl_problem(..., con_linear=...)`` stores the row the way a
parsed ``.nl`` file's ``J`` segment is stored: ``(var, coef)`` pairs, no tree, no
tape. Measured on 229,376 such rows: 2228 -> 1138 B/row of RSS.

The split is admitted only where it is **bit-identical** to taping the row
(``_arena_tape._linear_row``), so the evaluator must return the same constraint
values, Jacobian and Hessian to the last bit with the split on or off. These
tests check that exactly (``np.array_equal``, no tolerance). They also pin which
row shapes are admitted. A shape admitted wrongly would show up as a last-bit
difference here, and nowhere else.

The admission tests read only the arena and run on any POUNCE. The evaluator
differentials need a POUNCE whose ``build_nl_problem`` takes ``con_linear`` and
skip otherwise.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest

pounce = pytest.importorskip("pounce")

import discopt._tape_nlp_evaluator as T  # noqa: E402
from discopt._arena_tape import _linear_row, _num, _Program  # noqa: E402
from discopt._repr_cache import readonly_model_repr  # noqa: E402

needs_con_linear = pytest.mark.skipif(
    not T.pounce_takes_con_linear(),
    reason="this POUNCE's build_nl_problem has no con_linear= (needs the #2293 POUNCE change)",
)

CORPUS = Path(__file__).parent / "data" / "minlplib_nl"


def _row_shapes_model():
    """One row per shape the admission rule distinguishes, in a known order."""
    m = dm.Model("shapes")
    x = m.continuous("x", shape=(6,), lb=-2, ub=3)
    y = m.continuous("y", shape=(2, 3), lb=-1, ub=1)
    rows = {
        "lin_chain": x[1] - x[0] - 0.01 * x[2],  # 3 leaves, binary nesting
        "lin_long": x[0] + 2 * x[1] - x[2] * 3.5 + x[3] - x[4],  # >= 4 leaves: n-ary sum
        "lin_single": x[5],
        "lin_neg": -x[4] + x[1],
        "lin_index2d": y[0, 1] - y[1, 2] + 0.5 * y[1, 0],
        "trailing_const": x[0] + x[1] - 5.0,
        "leading_const": 5.0 + x[0] - x[1],  # would reassociate: not split
        "middle_const": x[0] + 5.0 - x[1],  # would reassociate: not split
        "div_const": x[0] / 3.0 + x[1],  # x/c is not (1/c)*x: not split
        "mixed": x[0] * x[1] + x[2] - x[3],  # one nonlinear leaf: not split
        "nonlinear": dm.exp(x[2]) - x[0],
    }
    for name, body in rows.items():
        m.subject_to(body == 0, name=name)
    m.minimize(dm.sum([(x[i] - 0.5) ** 2 for i in range(6)]) + y[0, 0] * y[1, 1])
    return m, list(rows)


def _admitted(model) -> list[bool]:
    rep = readonly_model_repr(model)
    prog = _Program(rep.tape_program())
    return [_linear_row(prog, int(c)) is not None for c in _num(rep, "constraint_ids")]


def test_admission_rule_by_row_shape():
    m, names = _row_shapes_model()
    got = dict(zip(names, _admitted(m)))
    assert got == {
        "lin_chain": True,
        "lin_long": True,
        "lin_single": True,
        "lin_neg": True,
        "lin_index2d": True,
        "trailing_const": True,
        "leading_const": False,
        "middle_const": False,
        "div_const": False,
        "mixed": False,
        "nonlinear": False,
    }


def test_linear_row_coefficients():
    m, names = _row_shapes_model()
    rep = readonly_model_repr(m)
    prog = _Program(rep.tape_program())
    ids = [int(c) for c in _num(rep, "constraint_ids")]
    by_name = dict(zip(names, ids))
    assert _linear_row(prog, by_name["lin_chain"]) == ([1, 0, 2], [1.0, -1.0, -0.01], 0.0)
    assert _linear_row(prog, by_name["lin_long"]) == (
        [0, 1, 2, 3, 4],
        [1.0, 2.0, -3.5, 1.0, -1.0],
        0.0,
    )
    # y is the second block (6 + row-major offset).
    assert _linear_row(prog, by_name["lin_index2d"]) == ([7, 11, 9], [1.0, -1.0, 0.5], 0.0)
    assert _linear_row(prog, by_name["trailing_const"]) == ([0, 1], [1.0, 1.0], -5.0)


def _evaluators(model, monkeypatch):
    """``(taped, split)`` evaluators for one model, built fresh each."""
    out = []
    for split in (False, True):
        monkeypatch.setattr(T, "pounce_takes_con_linear", lambda split=split: split)
        ev = T.TapeNLPEvaluator(model)
        out.append(ev)
    return out


def _assert_bit_identical(a, b, n_points: int = 4) -> int:
    """Same structure and the same values to the last bit; returns comparisons made."""
    rng = np.random.default_rng(2293)
    lo, hi = a.variable_bounds
    # Infinite or sentinel-sized sides get a window next to the finite one.
    big = 1e15
    lo_ok, hi_ok = np.abs(lo) < big, np.abs(hi) < big
    lo = np.where(lo_ok, lo, np.where(hi_ok, hi - 3.0, -3.0))
    hi = np.where(hi_ok, hi, lo + 3.0)
    compared = 0
    for s, t in zip(a.jacobian_structure(), b.jacobian_structure()):
        assert np.array_equal(s, t)
        compared += 1
    for s, t in zip(a.hessian_structure(), b.hessian_structure()):
        assert np.array_equal(s, t)
        compared += 1
    for _ in range(n_points):
        x = rng.uniform(lo, hi)
        lam = rng.uniform(-1.0, 1.0, a.n_constraints)
        pairs = [
            (a.evaluate_objective(x), b.evaluate_objective(x)),
            (a.evaluate_gradient(x), b.evaluate_gradient(x)),
            (a.evaluate_constraints(x), b.evaluate_constraints(x)),
            (a.evaluate_jacobian_values(x), b.evaluate_jacobian_values(x)),
            (a.evaluate_hessian_values(x, 1.0, lam), b.evaluate_hessian_values(x, 1.0, lam)),
        ]
        for u, v in pairs:
            u, v = np.asarray(u), np.asarray(v)
            assert u.shape == v.shape
            # array_equal, not allclose: the split is admitted only where it
            # is exact, and a tolerance would hide a wrongly admitted shape.
            assert np.array_equal(u, v, equal_nan=True), np.nanmax(np.abs(u - v))
            compared += 1
    return compared


@needs_con_linear
def test_split_is_used_and_bit_identical_on_every_row_shape(monkeypatch):
    m, names = _row_shapes_model()
    taped, split = _evaluators(m, monkeypatch)
    assert taped._con_linear is None
    # The shared problem consumed the CSR; it is released with the roots, so
    # check on the problem that the linear rows really went in as coefficients.
    assert split._problem.nnz_jac == taped._problem.nnz_jac
    assert _assert_bit_identical(taped, split) == 4 + 4 * 5


@needs_con_linear
def test_split_keeps_the_csr_until_the_problem_is_built(monkeypatch):
    m, _ = _row_shapes_model()
    monkeypatch.setattr(T, "pounce_takes_con_linear", lambda: True)
    seen = {}
    real = T.TapeNLPEvaluator._new_problem

    def spy(self):
        seen["csr"] = self._con_linear
        return real(self)

    monkeypatch.setattr(T.TapeNLPEvaluator, "_new_problem", spy)
    T.TapeNLPEvaluator(m)
    indptr, indices, data = seen["csr"]
    # 6 admitted rows of 11, with 3+5+1+2+3+2 terms.
    assert len(indptr) == 12 and int(indptr[-1]) == 16 == len(indices) == len(data)


@needs_con_linear
def test_odd_discretization_is_bit_identical(monkeypatch):
    """The issue's model class: implicit-Euler rows, linear and mixed."""
    m = dm.Model("ode")
    N, S, h = 6, 5, 0.01
    x = m.continuous("x", shape=(N + 1, S), lb=-1, ub=2)
    p = m.continuous("p", shape=(3,), lb=-1, ub=1)
    for k in range(N):
        y = x[k + 1]
        for s in range(S):
            if s % 2:
                rhs = y[(s + 1) % S]
            else:
                rhs = dm.exp(p[s % 3]) * y[s] * y[(s + 1) % S]
            m.subject_to(y[s] - x[k, s] - h * rhs == 0)
    m.minimize(dm.sum([(x[N, s] - 0.5) ** 2 for s in range(S)]))
    taped, split = _evaluators(m, monkeypatch)
    assert _assert_bit_identical(taped, split) > 0


def _corpus_models():
    from discopt.modeling.core import from_nl

    for path in sorted(CORPUS.glob("*.nl")):
        try:
            yield path.stem, from_nl(str(path))
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            yield path.stem, exc


@needs_con_linear
def test_corpus_is_bit_identical(monkeypatch):
    """Every in-repo corpus instance the arena path lowers, split vs taped."""
    compared = split_rows = 0
    refused = []
    for name, model in _corpus_models():
        if isinstance(model, Exception):
            refused.append((name, repr(model)))
            continue
        try:
            taped, split = _evaluators(model, monkeypatch)
        except T.UnsupportedForTape:
            continue
        if split._problem.m == 0:
            continue
        compared += _assert_bit_identical(taped, split, n_points=2)
        split_rows += int(sum(_admitted(model))) if model._constraints else 0
    # The probe must have fired on real instances, and real linear rows.
    assert compared > 100, compared
    assert split_rows > 100, split_rows
