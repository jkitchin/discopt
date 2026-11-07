"""highs-parity P2.0 (iii): exact speedups of the declared-box and factorable passes.

P2.1/P2.0 found the HiGHS LP route's remaining Python wall in two whole-model passes
that run on every LP: ``tighten_nonlinear_bounds`` (``_declared_box_tightening``) and
``has_factorable_work``. Three changes, each exact by construction and measured
identical on L/M/INF + the in-repo ``.nl`` corpus (197 instances: ``lb``, ``ub`` and
stats bit-for-bit; ``has_factorable_work`` 225/225, 49 True in both):

* ``has_factorable_work`` returns False at once on a model whose objective and every
  row are scalar-linear over ``Constant`` coefficients (a ``Parameter`` is not
  treated as a constant there, so such a row gets the full scan);
* the univariate-polynomial rule expands each row once, not once per round;
* the defined-variable rule reads a scalar variable leaf's enclosure from its box.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import factorable_reform as F
from discopt._relax import nonlinear_bound_tightening as N
from discopt._relax.affine_array import is_scalar_linear, scalar_linear_of


def _lp():
    m = dm.Model("lp")
    x = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(4)]
    m.subject_to(x[0] + 2.0 * x[1] - x[2] / 4.0 <= 4.0)
    m.subject_to(-(x[3]) + 3.0 * (x[0] - x[1]) == 1.0)
    m.minimize(3.0 * x[0] + x[1] ** 1)
    return m, x


def test_is_scalar_linear_matches_scalar_linear_of_grammar():
    m, x = _lp()
    p = m.parameter("p", value=2.0)
    n = sum(int(v.size) for v in m._variables)
    cases = [x[0] + 2.0 * x[1], x[0] * x[1], dm.exp(x[0]), x[0] / p, p * x[0], x[0] ** 2,
             dm.sum([x[0], -x[1], 4.0]), x[0] / 0.0]  # fmt: skip
    checked = 0
    for e in cases:
        assert is_scalar_linear(e) == (scalar_linear_of(e, m._flat_var_offset, n) is not None), e
        checked += 1
    assert checked == len(cases)
    assert not is_scalar_linear(x[0] / p, allow_parameters=False)
    assert not is_scalar_linear(p * x[0], allow_parameters=False)


def test_linear_model_has_no_factorable_work_without_scanning(monkeypatch):
    m, _x = _lp()
    calls = []
    monkeypatch.setattr(F, "distribute_charged", lambda *a, **k: calls.append(1))
    assert F.has_factorable_work(m) is False
    assert calls == []


def test_nonlinear_row_still_scanned():
    m, x = _lp()
    y = m.continuous("y", lb=1.0, ub=2.0)
    m.subject_to(x[0] / y + x[1] <= 3.0)  # a clearable variable denominator
    assert F.has_factorable_work(m) is True


def test_polynomial_rule_expands_each_row_once(monkeypatch):
    m = dm.Model("poly")
    x = m.continuous("x", lb=-1e20, ub=1e20)
    y = m.continuous("y", lb=1.0, ub=2.0)
    m.subject_to(x**2 - 4.0 * y == 0.0)
    m.minimize(x + y)
    calls = []
    real = N._poly_expand
    monkeypatch.setattr(N, "_poly_expand", lambda *a, **k: calls.append(1) or real(*a, **k))
    lb = np.array([-1e20, 1.0])
    ub = np.array([1e20, 2.0])
    rule = N.UnivariatePolynomialRootBoundRule()
    md = N._cached_flat_metadata(m)
    out = [rule.tighten(m, lb, ub, md) for _ in range(3)]
    assert len(calls) == 1
    for lo, hi in out:
        assert np.array_equal(lo, out[0][0]) and np.array_equal(hi, out[0][1])
        assert np.isfinite(lo[0]) and np.isfinite(hi[0])  # the rule did bound x


_NL = Path(__file__).parent / "data"


@pytest.mark.parametrize("name", ["gkocis", "haverly", "dispatch", "st_e40"])
def test_defined_variable_rule_unchanged_by_the_leaf_shortcut(monkeypatch, name):
    """The whole pass, with the rule firing, is bit-identical with the shortcut off."""
    from discopt.modeling.core import from_nl
    from discopt.solver import flat_variable_bounds

    path = next(p for d in ("minlplib_nl", "minlplib") if (p := _NL / d / f"{name}.nl").exists())
    on = N.tighten_nonlinear_bounds(from_nl(str(path)), *flat_variable_bounds(from_nl(str(path))))
    assert "defined_variable_forward" in on[2].applied_rules
    # Off: every leaf goes through ``evaluate_interval``, as before the shortcut.
    from discopt._relax.convexity.interval_eval import evaluate_interval

    monkeypatch.setattr(N, "_leaf_interval", evaluate_interval)
    m = from_nl(str(path))
    off = N.tighten_nonlinear_bounds(m, *flat_variable_bounds(m))
    assert np.array_equal(on[0], off[0]) and np.array_equal(on[1], off[1]) and on[2] == off[2]
