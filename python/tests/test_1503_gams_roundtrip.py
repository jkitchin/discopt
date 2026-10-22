"""``to_gams`` -> ``from_gams`` round trip over the ``.nl`` round-trip zoo (#1503).

Before #1503:

* a model holding a ``dm.Parameter`` exported as ``<unsupported:Parameter>``
  inside the equation text -- a ``.gms`` file that is not the model, written
  without complaint;
* the GAMS reader rejected the writer's own output: every element of an array
  variable is written as a quoted label, ``x('1')``, which raised
  ``GamsParseError: Unexpected token STRING('1')``, and every fractional or
  symbolic power is written ``rPower(x, r)``, which raised ``Cannot resolve
  indexed ref: rPower``.

The load-bearing property is **value fidelity**, not that bytes were written:
at sampled points the re-imported model's objective and every constraint row
must equal the original's (the same oracle ``test_export_cli_roundtrip.py``
applies to ``.nl``).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

pytest.importorskip("jax", reason="JAX is the optional discopt[jax] extra (#1873)")

from discopt._relax.dag_compiler import compile_expression
from discopt._relax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Constant
from discopt.modeling.gams_parser import parse_gams

pytestmark = pytest.mark.unit


# ── Zoo (mirrors TestNLRoundTrip in test_export_cli_roundtrip.py) ─────────


def _minlp():
    m = dm.Model("rt_minlp")
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.1, ub=3.0)
    b = m.binary("b")
    k = m.integer("k", lb=1, ub=4)
    m.minimize(dm.exp(x) + dm.log(y) + x * y + 2.5 * b + k**2 + x / y + dm.tan(0.3 * x))
    m.subject_to(dm.sin(x) + dm.cos(y) <= 1.5)
    m.subject_to(x + 2 * y + b + k <= 10)
    m.subject_to(x**2 + y**2 == 3)
    m.subject_to(dm.sqrt(x) * k >= 0.5)
    m.subject_to(dm.log1p(y) + dm.sigmoid(x) + dm.softplus(y) + dm.log2(x) <= 8)
    return m


def _arrays():
    m = dm.Model("rt_array")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=2.0)
    big_x = m.continuous("X", shape=(2, 2), lb=0.1, ub=1.5)
    s = m.continuous("s", lb=0.2, ub=1.0)
    m.minimize(x[0] + x[1] + x[2] + big_x[0, 0] + s)
    m.subject_to(Constant(np.array([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]])) @ x <= 4.0)
    m.subject_to(x @ Constant(np.array([1.0, -1.0, 2.0])) >= -5.0)
    m.subject_to(big_x @ Constant(np.array([[1.0, 2.0], [0.5, 1.0]])) <= 5.0)
    m.subject_to(dm.sum(x) <= 5.0)
    m.subject_to(dm.sum(big_x, axis=1) <= 3.0)
    m.subject_to(dm.exp(x) + s <= 9.0)
    m.subject_to(big_x[1, 0] * big_x[0, 1] - x[2] <= 1.0)
    return m


def _maximize():
    m = dm.Model("rt_max")
    x = m.continuous("x", lb=0.5, ub=2.0)
    m.maximize(dm.log(x) - x**2)
    return m


def _abs_neg():
    m = dm.Model("rt_absneg")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(abs(x - 1) + dm.exp(x) * (-x))
    m.subject_to(x - (-2.0) >= 0.5)  # a negative literal after an operator
    return m


def _const_offset():
    m = dm.Model("rt_const")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x + 2 * y + 5)
    m.subject_to(x + y >= 0.5)
    return m


def _sum_over():
    m = dm.Model("rt_sumover")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=1.0)
    m.minimize(x[0] + x[1] + x[2])
    m.subject_to(dm.sum(lambda i: x[i] ** 2, over=range(3)) <= 2.0)
    m.subject_to(dm.exp(dm.sum(lambda i: 0.5 * x[i], over=range(3))) <= 4.0)
    return m


def _fractional_powers():
    # rPower: a fractional constant exponent and a symbolic one, on array elements.
    m = dm.Model("rt_rpower")
    y = m.continuous("y", shape=(2,), lb=0.1, ub=4.0)
    z = m.continuous("z", lb=0.5, ub=2.0)
    m.minimize(y[0] ** 1.5 + y[1] ** z)
    m.subject_to(y[0] ** 0.5 + y[1] >= 1.5)
    m.subject_to(z**0.25 * y[1] <= 3.0)
    return m


def _scalar_parameter():
    # The issue's repro.
    m = dm.Model("rt_param")
    x = m.continuous("x", lb=0, ub=5)
    p = m.parameter("p", value=2.0)
    m.subject_to(p - x <= 0)
    m.minimize((x - p) ** 2)
    return m


def _array_parameter():
    m = dm.Model("rt_param_arr")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=5.0)
    p = m.parameter("p", value=np.array([1.0, -2.5, 3.0000001]))
    m.subject_to(dm.sum([p[i] * x[i] for i in range(3)]) >= -6.0)
    m.subject_to(p @ x <= 20.0)
    m.minimize(x[0] + p[1] * x[1] + x[2])
    return m


ZOO = {
    f.__name__.lstrip("_"): f
    for f in (
        _minlp,
        _arrays,
        _maximize,
        _abs_neg,
        _const_offset,
        _sum_over,
        _fractional_powers,
        _scalar_parameter,
        _array_parameter,
    )
}


# ── Oracle ───────────────────────────────────────────────────────────────


def _bounds(model):
    lbs, ubs = [], []
    for v in model._variables:
        n = max(1, int(np.prod(v.shape)))
        lbs.append(np.broadcast_to(np.asarray(v.lb, dtype=float).ravel(), (n,)))
        ubs.append(np.broadcast_to(np.asarray(v.ub, dtype=float).ravel(), (n,)))
    return np.concatenate(lbs), np.concatenate(ubs)


def _offsets(model):
    out, off = {}, 0
    for v in model._variables:
        n = max(1, int(np.prod(v.shape)))
        out[v.name] = (off, n)
        off += n
    return out, off


def _rows(model, x):
    """Every scalar row as ``g(x)`` of ``g <= 0`` / ``g == 0`` (``>=`` flipped)."""
    vals = []
    for c in model._constraints:
        body = np.atleast_1d(compile_expression(c.body, model)(x)).ravel()
        vals.append(-body if c.sense == ">=" else body)
    return np.concatenate(vals) if vals else np.zeros(0)


@pytest.mark.parametrize("name", sorted(ZOO))
def test_to_gams_from_gams_round_trip_is_value_faithful(name):
    m = ZOO[name]()
    text = m.to_gams()
    assert "unsupported" not in text
    r = parse_gams(text)

    assert r._objective.sense == m._objective.sense
    src_off, _ = _offsets(m)
    dst_off, n_dst = _offsets(r)
    for vname, (_, n) in src_off.items():
        assert vname in dst_off and dst_off[vname][1] == n, vname
    src_types = {v.name: v.var_type for v in m._variables}
    for v in r._variables:
        if v.name in src_types:
            assert v.var_type == src_types[v.name], v.name

    lb, ub = _bounds(m)
    ev_src, ev_dst = NLPEvaluator(m), NLPEvaluator(r)
    rng = np.random.default_rng(0)
    compared = 0
    for _ in range(4):
        x = lb + rng.random(lb.size) * (ub - lb)
        x2 = np.zeros(n_dst)
        for vname, (o, n) in src_off.items():
            o2, _ = dst_off[vname]
            x2[o2 : o2 + n] = x[o : o + n]
        assert ev_dst.evaluate_objective(x2) == pytest.approx(
            ev_src.evaluate_objective(x), rel=1e-12, abs=1e-12
        )
        want, got = _rows(m, x), _rows(r, x2)
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
        compared += 1 + want.size
    assert compared > 0


def test_parameter_is_written_as_its_value_not_a_placeholder():
    text = _scalar_parameter().to_gams()
    assert "<unsupported" not in text
    assert "obj_eq.. obj_var =e= power((x - 2), 2);" in text


def test_array_parameter_without_index_is_refused():
    m = dm.Model("p_arr")
    x = m.continuous("x", lb=0, ub=1)
    p = m.parameter("p", value=np.array([1.0, 2.0]))
    m.minimize(x)
    m.subject_to(x <= 1)
    # An unindexed shaped parameter in a scalar row cannot be one GAMS number.
    from discopt.export.gams import _GamsWriter

    with pytest.raises(ValueError, match="array parameter 'p'"):
        _GamsWriter(m, None)._expr_to_gams(p)


def test_an_unknown_expression_node_is_refused_not_written():
    from discopt.export.gams import _GamsWriter
    from discopt.modeling.core import Expression

    class _Alien(Expression):
        pass

    m = dm.Model("alien")
    m.continuous("x", lb=0, ub=1)
    with pytest.raises(ValueError, match="Cannot write expression type _Alien"):
        _GamsWriter(m, None)._expr_to_gams(_Alien())


_RPOWER_DATA = (
    "Scalar a; a = rPower({base}, 0.5);\n"
    "Variables z; Positive Variable x; x.up = 10;\n"
    "Equations e, o; e.. x =g= a; o.. z =e= x;\n"
    "Model mm /all/; Solve mm using lp minimizing z;"
)


def test_rpower_folds_in_a_data_statement():
    res = parse_gams(_RPOWER_DATA.format(base=4)).solve(time_limit=30)
    assert res.objective == pytest.approx(2.0, abs=1e-6)


def test_rpower_of_a_negative_base_is_refused_in_a_data_statement():
    """GAMS rPower is undefined for a negative base; Python would return a
    complex number. The fold must refuse rather than produce one."""
    from discopt.modeling.gams_parser import GamsParseError

    with pytest.raises(GamsParseError):
        parse_gams(_RPOWER_DATA.format(base=-4))


def test_public_file_round_trip(tmp_path):
    """The same through the public API: ``Model.to_gams(path)`` / ``dm.from_gams``."""
    m = _fractional_powers()
    path = tmp_path / "rt.gms"
    assert m.to_gams(str(path)) is None
    r = dm.from_gams(str(path))
    assert {v.name for v in m._variables} <= {v.name for v in r._variables}
    assert len(r._constraints) == len(m._constraints)
