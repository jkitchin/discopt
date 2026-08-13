"""Standard derivative-free test functions with published optima.

One definition shared by the DIRECT integration tests, the entry experiment, and
the docs notebook, so a number quoted in one place cannot drift from another.

Each function is given twice from a single body written against an array module:
``jnp_body`` (opaque to the relaxation layer, for wrapping in ``dm.custom``) and
``np_body`` (for driving the search engine directly). Writing them once and
substituting the module makes the two provably the same algebra.

**Boxes for the origin-centred functions are deliberately asymmetric.** DIRECT's
first evaluation is the centre of the box, so a symmetric box around an optimum
at the origin is "solved" at evaluation 1 — every variant then scores a perfect
result having searched nothing. The first draft of the entry-experiment panel had
that flaw and produced three meaningless exact zeros
(``docs/dev/direct-entry-2026-08-12.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TestFunc:
    """A benchmark function, its box, and its published global minimum."""

    name: str
    n: int
    lb: np.ndarray
    ub: np.ndarray
    fstar: float
    jnp_body: Callable  # written with jax.numpy -> opaque to the relaxation layer
    np_body: Callable  # numpy twin, same algebra

    def relative_error(self, value: float) -> float:
        """Relative error against the published optimum (absolute when f* == 0)."""
        return abs(value - self.fstar) / max(1.0, abs(self.fstar))


def _branin(v, xp):
    x1, x2 = v[0], v[1]
    return (
        (x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6) ** 2
        + 10 * (1 - 1 / (8 * math.pi)) * xp.cos(x1)
        + 10
    )


def _six_hump_camel(v, xp):
    x1, x2 = v[0], v[1]
    return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2


def _goldstein_price(v, xp):
    x1, x2 = v[0], v[1]
    a = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
    b = 30 + (2 * x1 - 3 * x2) ** 2 * (
        18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
    )
    return a * b


def _shubert(v, xp):
    s1 = sum((i + 1) * xp.cos((i + 2) * v[0] + (i + 1)) for i in range(5))
    s2 = sum((i + 1) * xp.cos((i + 2) * v[1] + (i + 1)) for i in range(5))
    return s1 * s2


def _rastrigin(v, xp):
    return 10 * v.shape[0] + xp.sum(v**2 - 10 * xp.cos(2 * math.pi * v))


def _ackley(v, xp):
    n = v.shape[0]
    return (
        -20 * xp.exp(-0.2 * xp.sqrt(xp.sum(v**2) / n))
        - xp.exp(xp.sum(xp.cos(2 * math.pi * v)) / n)
        + 20
        + math.e
    )


def _sphere(v, xp):
    return xp.sum(v**2)


def _linear(v, xp):
    """The survey's 'global drag' function, ``1 + x1 + ... + xn`` on the unit cube."""
    return 1.0 + xp.sum(v)


_H_C = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_H3_P = np.array(
    [
        [0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0381, 0.5743, 0.8828],
    ]
)
_H6_A = np.array(
    [
        [10.0, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3.0, 3.5, 1.7, 10, 17, 8],
        [17.0, 8, 0.05, 10, 0.1, 14],
    ]
)
_H6_P = 1e-4 * np.array(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
)


def _hartman3(v, xp):
    return -xp.sum(_H_C * xp.exp(-xp.sum(_H3_A * (v - _H3_P) ** 2, axis=1)))


def _hartman6(v, xp):
    return -xp.sum(_H_C * xp.exp(-xp.sum(_H6_A * (v - _H6_P) ** 2, axis=1)))


def _make(name, n, lo, hi, fstar, body) -> TestFunc:
    import jax.numpy as jnp

    return TestFunc(
        name=name,
        n=n,
        lb=np.broadcast_to(np.asarray(lo, dtype=np.float64), (n,)).copy(),
        ub=np.broadcast_to(np.asarray(hi, dtype=np.float64), (n,)).copy(),
        fstar=fstar,
        jnp_body=lambda v, _b=body: _b(v, jnp),
        np_body=lambda v, _b=body: _b(v, np),
    )


#: name -> factory. Lazy so importing this module does not import jax.
_REGISTRY: dict[str, Callable[[], TestFunc]] = {
    # Two global minima; DIRECT's sampled points should cluster at both.
    "six_hump_camel": lambda: _make(
        "six_hump_camel", 2, [-3.0, -2.0], [3.0, 2.0], -1.0316285, _six_hump_camel
    ),
    # Three global minima (survey Fig. 8).
    "branin": lambda: _make("branin", 2, [-5.0, 0.0], [10.0, 15.0], 0.397887, _branin),
    # Sharply scaled; probes scale sensitivity.
    "goldstein_price": lambda: _make("goldstein_price", 2, -2.0, 2.0, 3.0, _goldstein_price),
    # 18 global minima; the survey's pathological over-refinement case at eps=0.
    "shubert": lambda: _make("shubert", 2, -10.0, 10.0, -186.7309, _shubert),
    # Asymmetric boxes: see the module docstring.
    "rastrigin_2": lambda: _make("rastrigin_2", 2, -4.12, 6.12, 0.0, _rastrigin),
    "ackley_2": lambda: _make("ackley_2", 2, -25.768, 39.768, 0.0, _ackley),
    "sphere_2": lambda: _make("sphere_2", 2, -4.0, 6.0, 0.0, _sphere),
    "hartman_3": lambda: _make("hartman_3", 3, 0.0, 1.0, -3.86278, _hartman3),
    # The survey's Hartman-6 result is why DIRECT-GL is not the default.
    "hartman_6": lambda: _make("hartman_6", 6, 0.0, 1.0, -3.32237, _hartman6),
    # Survey §2.3.1 / Fig. 15: the global-drag case.
    "linear_2": lambda: _make("linear_2", 2, 0.0, 1.0, 1.0, _linear),
    "linear_5": lambda: _make("linear_5", 5, 0.0, 1.0, 1.0, _linear),
}


def get(name: str) -> TestFunc:
    """One test function by name."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown test function {name!r}; have {sorted(_REGISTRY)}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def build_model(tf: TestFunc, name: str | None = None):
    """A discopt model whose objective is ``tf`` behind an opaque ``dm.custom``.

    Wrapping in ``dm.custom`` is the point: it puts the model on the path that has
    no algebraic relaxation, which is the class ``solver="direct"`` exists to
    serve.
    """
    import discopt.modeling as dm

    m = dm.Model(name or f"direct_{tf.name}")
    x = m.continuous("x", shape=tf.n, lb=tf.lb, ub=tf.ub)
    m.minimize(dm.custom(tf.jnp_body, name=tf.name)(x))
    return m, x
