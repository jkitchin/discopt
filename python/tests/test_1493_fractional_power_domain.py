"""``x ** p`` with a non-integer ``p`` has the domain ``x >= 0``, like ``sqrt`` (#1493).

``sqrt(x) <= 1.5`` on ``x in [-5, 5]`` with ``min x`` certified 0, while the same
model written ``x ** 0.5 <= 1.5`` returned ``feasible`` 1.25 with bound -5.
``FunctionDomainBoundRule`` clamped a ``sqrt`` argument to its domain but did not
recognise the power spelling, so the power form kept the undefined half ``[-5, 0)``:
the local NLP evaluates NaN there and the relaxation has no envelope for a
fractional power of a sign-straddling base. Every evaluator returns NaN for a
negative base with a non-integer exponent (checked below), so the implied bound
removes no evaluable point.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

pytest.importorskip("jax", reason="JAX is the optional discopt[jax] extra (#1873)")

from discopt._relax.nlp_evaluator import cached_evaluator
from discopt._relax.nonlinear_bound_tightening import (
    FunctionDomainBoundRule,
    build_flat_variable_metadata,
)

pytestmark = pytest.mark.unit


def _tighten(model):
    meta = build_flat_variable_metadata(model)
    n = len(meta.flat_var_types)
    return FunctionDomainBoundRule().tighten(model, np.full(n, -np.inf), np.full(n, np.inf), meta)


@pytest.mark.parametrize("p", [0.5, 1 / 3, 1.5, 2.5, -0.5])
def test_fractional_power_base_gets_the_sqrt_domain(p):
    m = dm.Model("fp")
    x = m.continuous("x", lb=-np.inf, ub=np.inf)
    m.minimize(x**p + x)
    lb, ub = _tighten(m)
    assert lb[0] == 0.0 and ub[0] == np.inf


def test_affine_base_is_inverted():
    m = dm.Model("fp")
    z = m.continuous("z", lb=-np.inf, ub=np.inf)
    m.subject_to((2.0 * z + 1.0) ** 0.5 <= 3)  # 2z + 1 >= 0 -> z >= -0.5
    m.minimize(z)
    lb, _ = _tighten(m)
    assert lb[0] == pytest.approx(-0.5)


@pytest.mark.parametrize("p", [2, 3, -1, -2, 2.0])
def test_integer_powers_are_not_restricted(p):
    m = dm.Model("ip")
    x = m.continuous("x", lb=-np.inf, ub=np.inf)
    m.minimize(x**p + x)
    lb, ub = _tighten(m)
    assert lb[0] == -np.inf and ub[0] == np.inf


def test_multivariable_base_is_left_alone():
    m = dm.Model("mv")
    x = m.continuous("x", shape=(2,), lb=-np.inf, ub=np.inf)
    m.minimize((x[0] - x[1]) ** 0.5 + x[0])
    lb, ub = _tighten(m)
    assert np.all(lb == -np.inf) and np.all(ub == np.inf)


@pytest.mark.parametrize("p", [0.5, 1 / 3, 1.5, -0.5])
def test_the_removed_half_is_not_evaluable(p):
    """Feasible-point check: every point the bound removes evaluates to NaN."""
    m = dm.Model("ev")
    x = m.continuous("x", lb=-5, ub=5)
    m.minimize(x**p)
    ev = cached_evaluator(m)
    checked = 0
    for xv in np.linspace(-5, -1e-9, 25):
        assert np.isnan(ev.evaluate_objective(np.array([xv])))
        checked += 1
    assert np.isfinite(ev.evaluate_objective(np.array([2.0])))
    assert checked == 25


def _model(form, sense, box):
    m = dm.Model("eq")
    x = m.continuous("x", lb=box[0], ub=box[1])
    f = dm.sqrt(x) if form == "sqrt" else x**0.5
    m.subject_to(f <= 1.5)
    if sense == "min":
        m.minimize(x)
    elif sense == "max":
        m.maximize(x)
    else:
        m.minimize(x - 3 * f)
    return m


@pytest.mark.parametrize("sense", ["min", "max", "mixed"])
@pytest.mark.parametrize("box", [(-5.0, 5.0), (-1.0, 3.0), (0.5, 5.0)])
def test_sqrt_and_half_power_spellings_agree(sense, box):
    rs = _model("sqrt", sense, box).solve(time_limit=20)
    rp = _model("pow", sense, box).solve(time_limit=20)
    assert rs.status == "optimal" and rs.gap_certified, (rs.status, rs.objective)
    assert rp.status == rs.status and rp.gap_certified == rs.gap_certified
    assert rp.objective == pytest.approx(rs.objective, abs=1e-6)
    # Closed form: x in [max(lo, 0), min(hi, 2.25)].
    lo, hi = max(box[0], 0.0), min(box[1], 2.25)
    xs = np.linspace(lo, hi, 20001)
    ref = {"min": lo, "max": hi, "mixed": float(np.min(xs - 3 * np.sqrt(xs)))}[sense]
    assert rp.objective == pytest.approx(ref, abs=1e-4)
