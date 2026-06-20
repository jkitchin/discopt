"""Adversarial soundness suite for the auto-firing cut detectors.

Motivated by the PR #245 review, whose blocker was a *hardcoded* cut
(``inject_complementarity``) fired from a structural match without checking the
sign/direction its validity depends on. The fuzzers below probe the hardcoded-cut
detectors (complementarity, Fortet binaries, GP log-lift) and the gas recognizer
with sign-, bound-, and sense-flipped models. The oracle is
**optimum-preservation**: every detector is value-preserving, so an unsound cut
that removes the true optimum changes the solved objective — the assertions catch
exactly the class of failure the review surfaced.

A second block gives the soundness *gates* (``verify_envelope`` / Jensen) negative
tests, ensuring they actually reject planted-unsound / planted-non-convex inputs.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("sympy")

import discopt.modeling as dm  # noqa: E402
from discopt._jax.symbolic import cut_recognizer as R  # noqa: E402
from discopt._jax.symbolic.verification import verify_envelope  # noqa: E402

pytestmark = pytest.mark.relaxation

TL = 25


def _opt(model):
    r = model.solve(time_limit=TL, gap_tolerance=1e-4)
    return r.status, (None if r.objective is None else float(r.objective))


def _assert_inject_preserves_optimum(build, label):
    """Solve baseline vs inject_all_patterns; the optimum must be unchanged."""
    s0, o0 = _opt(build())
    s1, o1 = _opt(build_injected := build())
    R.inject_all_patterns(build_injected)
    s1, o1 = _opt(build_injected)
    if o0 is None:
        # baseline infeasible/unbounded -> injection must not invent a feasible opt
        assert o1 is None, f"{label}: injection changed an unsolved baseline to {o1}"
        return
    assert o1 is not None, f"{label}: injection made a solvable model unsolved"
    assert abs(o0 - o1) < 1e-2, f"{label}: optimum moved {o0} -> {o1} (unsound cut)"


# --------------------------------------------------------------------------
# Complementarity adversarial cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sense, xlb, rhs, coeff",
    [
        (">=", 0.0, 0.0, 1.0),  # x*y >= 0: vacuous on nonneg box -> must NOT fire
        ("<=", -3.0, 0.0, 1.0),  # negative lb: cut needs x,y>=0 -> must NOT fire
        ("==", 0.0, 5.0, 1.0),  # x*y == 5: not complementarity -> must NOT fire
        ("==", 0.0, 0.0, -2.0),  # -2*x*y == 0: equality sign-agnostic -> may fire, sound
        ("<=", 0.0, 0.0, 1.0),  # x*y <= 0: valid complementarity -> fires, sound
    ],
)
def test_complementarity_adversarial(sense, xlb, rhs, coeff):
    def build():
        m = dm.Model("c")
        x = m.continuous("x", lb=xlb, ub=6.0)
        y = m.continuous("y", lb=0.0, ub=4.0)
        m.minimize(-(x + y))  # push to the corner so an illegal cut would bite
        term = coeff * x * y
        if sense == ">=":
            m.subject_to(term >= rhs)
        elif sense == "<=":
            m.subject_to(term <= rhs)
        else:
            m.subject_to(term == rhs)
        return m

    _assert_inject_preserves_optimum(build, f"compl[{sense},xlb={xlb},rhs={rhs},c={coeff}]")


def test_complementarity_square_not_misread_as_bilinear():
    """x*x == 0 (single variable) is not a complementarity product."""
    m = dm.Model("sq")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize(-(x + y))
    m.subject_to(x * x == 0)  # forces x=0, but is not x*y complementarity
    assert R.inject_complementarity(m) == 0


# --------------------------------------------------------------------------
# Fortet binary-product adversarial cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize("coeff", [2.0, -3.0])
@pytest.mark.parametrize("sense", ["min", "max"])
def test_fortet_adversarial_sign_and_sense(coeff, sense):
    def build():
        m = dm.Model("b")
        b = m.binary("b", shape=(3,))
        obj = coeff * b[0] * b[1] * b[2] + 0.5 * b[0]
        m.minimize(obj if sense == "min" else -obj)
        m.subject_to(b[0] + b[1] + b[2] >= 1)
        return m

    _assert_inject_preserves_optimum(build, f"fortet[c={coeff},{sense}]")


def test_fortet_mixed_continuous_binary_not_fired():
    m = dm.Model("mix")
    b = m.binary("b", shape=(2,))
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(b[0] * b[1] * x)  # not a pure binary product
    assert R.inject_binary_products(m) == 0


# --------------------------------------------------------------------------
# GP log-lift adversarial cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sense", ["min", "max"])
def test_gp_adversarial_sense(sense):
    def build():
        m = dm.Model("gp")
        x = m.continuous("x", lb=0.5, ub=4.0)
        y = m.continuous("y", lb=0.5, ub=4.0)
        obj = 2.0 * x**1.5 * y**0.5
        m.minimize(obj if sense == "min" else -obj)
        m.subject_to(x * y >= 3.0)
        return m

    _assert_inject_preserves_optimum(build, f"gp[{sense}]")


def test_gp_negative_coefficient_monomial_not_fired():
    m = dm.Model("gpneg")
    x = m.continuous("x", lb=0.5, ub=4.0)
    y = m.continuous("y", lb=0.5, ub=4.0)
    m.minimize(-2.0 * x**1.5 * y**0.5)  # negative coeff -> not a posynomial monomial
    m.subject_to(x + y <= 6.0)
    assert R.inject_gp_cuts(m) == 0


# --------------------------------------------------------------------------
# Gas recognizer robustness (derives from the model's own equations)
# --------------------------------------------------------------------------


def _gas(ratio_flip=False, wey_flip=False):
    m = dm.Model("g")
    p = m.continuous("p", shape=(3,), lb=30.0, ub=70.0)
    pd = m.continuous("pd", lb=45.0, ub=70.0)
    w = m.continuous("w", lb=0.0, ub=100.0)
    b = m.continuous("b", lb=1.0, ub=2.0)
    m.minimize(0.828 * w * (b**0.2857 - 1))
    ps, c0, c2 = 50.0, 3.5, 5.0
    if wey_flip:
        m.subject_to(w**2 == c0 * (p[0] ** 2 - ps**2))
    else:
        m.subject_to(w**2 == c0 * (ps**2 - p[0] ** 2))
    m.subject_to(w**2 == c2 * (p[1] ** 2 - p[2] ** 2))
    m.subject_to(40.0**2 == c2 * (p[2] ** 2 - pd**2))
    if ratio_flip:
        m.subject_to(p[0] == b * p[1])
    else:
        m.subject_to(p[1] == b * p[0])
    m.subject_to(w == 40.0)
    return m


@pytest.mark.parametrize("ratio_flip, wey_flip", [(False, False), (True, False), (False, True)])
def test_gas_recognizer_optimum_preserved_under_sign_flips(ratio_flip, wey_flip):
    _assert_inject_preserves_optimum(
        lambda: _gas(ratio_flip=ratio_flip, wey_flip=wey_flip),
        f"gas[ratio_flip={ratio_flip},wey_flip={wey_flip}]",
    )


# --------------------------------------------------------------------------
# Soundness-gate negative tests (the gates must catch their own targets)
# --------------------------------------------------------------------------


def test_verify_envelope_rejects_unsound_containment():
    f = lambda x: x**2  # noqa: E731
    # cv overshoots f (cv = f + 1): convex but NOT containing -> unsound.
    bad = lambda x, lb, ub: (f(x) + 1.0, f(x))  # noqa: E731
    rep = verify_envelope(bad, f, domain=(-2.0, 2.0), n_boxes=200)
    assert not rep.sound
    assert rep.max_lower_violation > 0.5


def test_verify_envelope_rejects_noncovex_underestimator():
    f = lambda x: x**2  # noqa: E731

    # contains f but cv is non-convex (min of a constant and f): curvature gate.
    def bad(x, lb, ub):
        return jnp.minimum(jnp.ones_like(x), f(x)), jnp.maximum(-jnp.ones_like(x), f(x))

    rep = verify_envelope(bad, f, domain=(-2.0, 2.0), n_boxes=200)
    assert rep.max_lower_violation <= 1e-7  # it does contain f
    assert not rep.sound  # ...but curvature fails -> not sound
    assert rep.max_convexity_violation > 1e-3
