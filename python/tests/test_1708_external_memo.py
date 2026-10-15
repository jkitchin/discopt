"""#1708: ``dm.external`` callbacks run once per distinct point, not once per visit.

Measured before the fix on the issue's 2-variable MINLP under ``solver="mip-nlp"``:
``fn`` ran 2236 times at 85 distinct (bit-exact) inputs, ``jac`` 1872 at 71,
``hess`` 1769 at 69. Outer approximation re-solves the same fixed-integer NLP from
the same warm start many times, and each solve walks the same ~43-iterate path,
so every repeat sat >= 40 distinct points back -- beyond the 8-point memo that
#1619 B-13 introduced. The memo is per solve, so it can safely hold the whole
history; these tests pin that, the opt-out, and the byte bound.
"""

from __future__ import annotations

import importlib

import discopt.modeling as dm
import numpy as np
import pytest


def _issue_model(memoize=True):
    calls = {"f": [], "j": [], "h": []}

    def f(u):
        calls["f"].append(np.asarray(u).tobytes())
        return np.array([u[0] * np.exp(-0.1 * u[1]), u[0] + u[1] ** 2])

    def jac(u):
        calls["j"].append(np.asarray(u).tobytes())
        e = np.exp(-0.1 * u[1])
        return np.array([[e, -0.1 * u[0] * e], [1.0, 2 * u[1]]])

    def hess(u):
        calls["h"].append(np.asarray(u).tobytes())
        e = np.exp(-0.1 * u[1])
        return np.array([[[0, -0.1 * e], [-0.1 * e, 0.01 * u[0] * e]], [[0, 0], [0, 2.0]]])

    blk = dm.external(f, jac=jac, hess=hess, shape=(2,), name="unit", memoize=memoize)
    m = dm.Model("toy")
    u = m.continuous("u", shape=(2,), lb=[1.0, 0.0], ub=[5.0, 3.0])
    on = m.binary("on")
    y = blk(u)
    ye = m.continuous("ye", shape=(2,), lb=0.0, ub=20.0)
    ylo, yhi = np.array([0.5, 1.0]), np.array([5.0, 14.0])
    for k in range(2):  # ye = on * y, exact for binary on
        m.subject_to(ye[k] <= yhi[k] * on)
        m.subject_to(ye[k] >= ylo[k] * on)
        m.subject_to(ye[k] <= y[k] - ylo[k] * (1 - on))
        m.subject_to(ye[k] >= y[k] - yhi[k] * (1 - on))
    m.subject_to(ye[1] <= 6.0)
    m.maximize(3 * ye[0] - 0.5 * ye[1] - 1.0 * on)
    return m, calls


def test_mip_nlp_calls_each_callable_once_per_distinct_point():
    m, calls = _issue_model()
    r = m.solve(time_limit=60, solver="mip-nlp")
    assert r.objective == pytest.approx(11.5, abs=1e-6)
    checked = 0
    for key, seq in calls.items():
        assert seq, key
        # Before #1708: 2236 / 1872 / 1769 calls at 85 / 71 / 69 distinct inputs.
        assert len(seq) == len(set(seq)), (key, len(seq), len(set(seq)))
        checked += 1
    assert checked == 3


def test_memoize_false_calls_through():
    from discopt.modeling.external import _checked

    n = {"k": 0}

    def f(x):
        n["k"] += 1
        return np.array([x[0]])

    call = _checked(f, role="fn", label="t", want=lambda xs: (1,), memo_size=0)
    for _ in range(3):
        call(np.array([1.0]))
    assert n["k"] == 3

    # and through the public builder
    blk = dm.external(f, jac=lambda x: np.eye(1), shape=(1,), memoize=False)
    assert blk._discopt_external.name == "f"


def test_memo_keeps_a_long_history():
    from discopt.modeling.external import _MEMO_SIZE, _checked

    n = {"k": 0}

    def f(x):
        n["k"] += 1
        return np.array([x[0]])

    call = _checked(f, role="fn", label="t", want=lambda xs: (1,))
    pts = [np.array([float(i)]) for i in range(min(_MEMO_SIZE, 200))]
    for _ in range(3):
        for p in pts:
            call(p)
    assert n["k"] == len(pts)


def test_memo_respects_the_byte_budget(monkeypatch):
    ext = importlib.import_module("discopt.modeling.external")

    monkeypatch.setattr(ext, "_MEMO_MAX_BYTES", 3 * 8 * 100)  # three 100-float outputs
    n = {"k": 0}

    def f(x):
        n["k"] += 1
        return np.full(100, x[0])

    call = ext._checked(f, role="fn", label="t", want=lambda xs: (100,))
    for i in range(4):
        call(np.array([float(i)]))
    assert n["k"] == 4
    call(np.array([3.0]))  # most recent: kept
    call(np.array([1.0]))  # within the three newest: kept
    assert n["k"] == 4
    call(np.array([0.0]))  # oldest: evicted by the byte budget
    assert n["k"] == 5


def test_memo_keeps_the_newest_entry_even_over_budget(monkeypatch):
    ext = importlib.import_module("discopt.modeling.external")

    monkeypatch.setattr(ext, "_MEMO_MAX_BYTES", 1)
    n = {"k": 0}

    def f(x):
        n["k"] += 1
        return np.array([x[0], x[0]])

    call = ext._checked(f, role="fn", label="t", want=lambda xs: (2,))
    call(np.array([1.0]))
    call(np.array([1.0]))
    assert n["k"] == 1


@pytest.mark.parametrize(
    "bad, exc",
    [(-1, ValueError), (1.5, TypeError), ("yes", TypeError)],
)
def test_invalid_memoize_is_rejected(bad, exc):
    with pytest.raises(exc, match="memoize"):
        dm.external(lambda x: x, jac=lambda x: np.eye(1), memoize=bad)


def test_opaque_custom_with_integers_points_at_mip_nlp():
    jnp = pytest.importorskip("jax.numpy")

    c = dm.custom(lambda x: jnp.sin(x[0]) * x[1], name="c")
    m = dm.Model("c")
    x = m.continuous("x", shape=(2,), lb=0, ub=2)
    b = m.binary("b")
    m.subject_to(x[1] <= 2 * b)
    m.maximize(c(x) - 0.1 * b)
    with pytest.raises(ValueError, match="solver='mip-nlp'"):
        m.solve(time_limit=20)
