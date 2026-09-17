"""Public SymPy bridge — component E of #1248, via #1278.

#1248's acceptance for this piece is "a round trip over the elementary function
set, ``entropy`` included, matching values and gradients at random points". That
is what the first two tests do, and they are the reason the bridge defines
``sympy.Function`` subclasses for the six intrinsics SymPy has no function for
rather than rewriting them: ``log2(x)`` as ``log(x)/log(2)`` is mathematically
equal and does not round-trip, and ``entropy``/``sigmoid``/``softplus`` have no
closed SymPy spelling at all.

The rest of the file is about the other half of the design: this bridge REFUSES
rather than guesses. The private translator it supersedes
(``_relax/symbolic/cut_recognizer.model_to_sympy``) substitutes ``sp.Dummy`` for a
node it cannot key — correct for a pattern matcher, silently wrong for a round
trip — and maps calls by ``getattr(sp, name)``, which raises ``AttributeError`` on
exactly the intrinsics #1248 asks for.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest

sp = pytest.importorskip("sympy")

from discopt.symbolic import (  # noqa: E402 - after importorskip by design
    DISCOPT_FUNCTIONS,
    SymbolicTranslationError,
    from_sympy,
    numeric_modules,
    sympy_function_for,
    to_sympy,
)

pytestmark = [pytest.mark.smoke]

#: ``(discopt builder, numpy reference, a box where it is real and finite)``.
#: Every unary intrinsic the modeling layer can emit, minus the two with no
#: continuous meaning here (``sign`` is refused by the relaxer; ``abs`` is not
#: differentiable at 0 and is checked separately on a box away from it).
ELEMENTARY = {
    "exp": (dm.exp, np.exp, (-1.5, 1.5)),
    "log": (dm.log, np.log, (0.2, 3.0)),
    "log2": (dm.log2, np.log2, (0.2, 3.0)),
    "log10": (dm.log10, np.log10, (0.2, 3.0)),
    "log1p": (dm.log1p, np.log1p, (-0.5, 3.0)),
    "sqrt": (dm.sqrt, np.sqrt, (0.2, 3.0)),
    "sin": (dm.sin, np.sin, (-2.0, 2.0)),
    "cos": (dm.cos, np.cos, (-2.0, 2.0)),
    "tan": (dm.tan, np.tan, (-1.0, 1.0)),
    "asin": (dm.asin, np.arcsin, (-0.9, 0.9)),
    "acos": (dm.acos, np.arccos, (-0.9, 0.9)),
    "atan": (dm.atan, np.arctan, (-2.0, 2.0)),
    "sinh": (dm.sinh, np.sinh, (-1.5, 1.5)),
    "cosh": (dm.cosh, np.cosh, (-1.5, 1.5)),
    "tanh": (dm.tanh, np.tanh, (-1.5, 1.5)),
    "asinh": (dm.asinh, np.arcsinh, (-2.0, 2.0)),
    "acosh": (dm.acosh, np.arccosh, (1.2, 3.0)),
    "atanh": (dm.atanh, np.arctanh, (-0.8, 0.8)),
    "erf": (dm.erf, lambda t: np.vectorize(math.erf)(t), (-1.5, 1.5)),
    "entropy": (dm.xlogx, lambda t: t * np.log(t), (0.05, 3.0)),
    "sigmoid": (dm.sigmoid, lambda t: 1.0 / (1.0 + np.exp(-t)), (-2.0, 2.0)),
    "softplus": (dm.softplus, lambda t: np.logaddexp(0.0, t), (-2.0, 2.0)),
    "abs": (dm.abs, np.abs, (0.5, 3.0)),
}

_RNG = np.random.default_rng(20260916)


def _one_var(builder, lo, hi):
    m = dm.Model("bridge")
    x = m.continuous("x", lb=lo, ub=hi)
    return m, x, builder(x)


# --------------------------------------------------------------------------- #
# #1248's acceptance: a round trip matching values AND gradients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(ELEMENTARY))
def test_round_trip_matches_values_at_random_points(name):
    builder, np_f, (lo, hi) = ELEMENTARY[name]
    _m, _x, expr = _one_var(builder, lo, hi)
    s, syms = to_sympy(expr)
    assert len(syms) == 1, syms

    (sym,) = syms
    f = sp.lambdify(sym, s, modules=[numeric_modules(), "numpy"])
    pts = _RNG.uniform(lo, hi, size=40)
    checked = 0
    for t in pts:
        want = float(np_f(np.asarray(t)))
        got = float(f(t))
        assert got == pytest.approx(want, rel=1e-10, abs=1e-12), (name, t, got, want)
        checked += 1
    assert checked == 40, checked


@pytest.mark.parametrize("name", sorted(ELEMENTARY))
def test_round_trip_matches_gradients_at_random_points(name):
    """The derivative is the half a rewrite would silently get wrong: a
    ``sympy.Function`` with no ``fdiff`` differentiates to ``Subs(Derivative(...))``
    and evaluates to nothing."""
    builder, np_f, (lo, hi) = ELEMENTARY[name]
    _m, _x, expr = _one_var(builder, lo, hi)
    s, syms = to_sympy(expr)
    (sym,) = syms

    d = sp.diff(s, sym)
    assert not d.has(sp.Derivative), (name, d)
    df = sp.lambdify(sym, d, modules=[numeric_modules(), "numpy"])

    pts = _RNG.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), size=25)
    checked = 0
    for t in pts:
        h = 1e-6 * max(1.0, abs(t))
        fd = (float(np_f(np.asarray(t + h))) - float(np_f(np.asarray(t - h)))) / (2 * h)
        got = float(df(t))
        scale = max(1.0, abs(fd))
        assert abs(got - fd) / scale < 1e-5, (name, t, got, fd)
        checked += 1
    assert checked == 25, checked


@pytest.mark.parametrize("name", sorted(ELEMENTARY))
def test_the_expression_comes_back_as_a_discopt_expression_with_the_same_values(name):
    """The other direction: `from_sympy` must rebuild something the solver's own
    tape evaluates to the same numbers."""
    from discopt._tape_nlp_evaluator import make_evaluator

    builder, np_f, (lo, hi) = ELEMENTARY[name]
    m, _x, expr = _one_var(builder, lo, hi)
    s, syms = to_sympy(expr)
    rebuilt = from_sympy(s, syms)

    m.minimize(rebuilt)
    ev = make_evaluator(m)
    checked = 0
    for t in _RNG.uniform(lo, hi, size=20):
        got = float(ev.evaluate_objective(np.array([float(t)])))
        want = float(np_f(np.asarray(t)))
        assert got == pytest.approx(want, rel=1e-10, abs=1e-12), (name, t, got, want)
        checked += 1
    assert checked == 20


def test_a_compound_expression_round_trips_through_the_solver():
    """Arithmetic, several intrinsics and two variables at once — and the rebuilt
    model must SOLVE to the same optimum, not merely evaluate the same."""

    def build(use_bridge):
        m = dm.Model("compound")
        x = m.continuous("x", lb=0.2, ub=2.0)
        y = m.continuous("y", lb=0.2, ub=2.0)
        m.subject_to(x + y == 2.0)
        body = dm.exp(x) + dm.xlogx(y) + x * y - dm.sqrt(x) + y**2 / 3.0
        if use_bridge:
            s, syms = to_sympy(body)
            body = from_sympy(s, syms)
        m.minimize(body)
        return m

    a = build(False).solve(time_limit=60)
    b = build(True).solve(time_limit=60)
    assert a.status == b.status == "optimal", (a.status, b.status)
    assert b.objective == pytest.approx(a.objective, rel=1e-8, abs=1e-8)


def test_integer_constants_stay_exact():
    """`x**2` must stay polynomial and `x/3` must not become a float division —
    the private translator floats every constant, which is what this fixes."""
    m = dm.Model("exact")
    x = m.continuous("x", lb=0.0, ub=1.0)
    s, _syms = to_sympy(x**2 + x / 3)
    assert s.is_polynomial(), s
    assert sp.Rational(1, 3) in s.atoms(sp.Rational), s.atoms(sp.Number)


def test_symbols_are_shared_across_calls():
    m = dm.Model("shared")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    s1, syms = to_sympy(x + y)
    s2, syms = to_sympy(x * y, syms)
    assert len(syms) == 2, syms
    assert s1.free_symbols == s2.free_symbols


def test_a_vector_variables_elements_get_distinct_symbols():
    m = dm.Model("vec")
    v = m.continuous("v", shape=3, lb=0.0, ub=1.0)
    s, syms = to_sympy(v[0] * v[1] + v[2])
    assert len(syms) == 3, syms
    assert {str(k) for k in syms} == {"v_0", "v_1", "v_2"}, sorted(str(k) for k in syms)


# --------------------------------------------------------------------------- #
# Registered atoms (#1248 A)
# --------------------------------------------------------------------------- #
def test_a_registered_atom_survives_the_round_trip_as_an_atom():
    """Flattening a registered atom into its lowering would silently undo #1248 A:
    the model would still be correct and the relaxer would lose the envelope."""
    from discopt.operators import atom_of, clear_registered, registry_snapshot

    with registry_snapshot():
        clear_registered()
        rk = dm.register_function("rk_bridge", lambda t: t * (1 - t) * (3.0 + 1.5 * (2 * t - 1)))
        m = dm.Model("atom")
        x = m.continuous("x", lb=0.0, ub=1.0)
        s, syms = to_sympy(rk(x))
        assert type(s).__name__ == "rk_bridge", s

        back = from_sympy(s, syms)
        tag = atom_of(back)
        assert tag is not None and tag[0] == "rk_bridge", tag


def test_an_unregistered_name_is_refused_rather_than_lowered():
    from discopt.operators import clear_registered, registry_snapshot

    with registry_snapshot():
        clear_registered()
        with pytest.raises(SymbolicTranslationError, match="no registered atom"):
            sympy_function_for("never_registered")


def test_a_dropped_registration_is_refused_on_the_way_back():
    """The binding is by name and resolved at translation time, so a registration
    dropped in between must raise rather than quietly produce something else."""
    from discopt.operators import clear_registered, registry_snapshot

    with registry_snapshot():
        clear_registered()
        fn = dm.register_function("transient", lambda t: t * t)
        m = dm.Model("atom")
        x = m.continuous("x", lb=0.0, ub=1.0)
        s, syms = to_sympy(fn(x))
        clear_registered()
        with pytest.raises(SymbolicTranslationError, match="no longer registered"):
            from_sympy(s, syms)


# --------------------------------------------------------------------------- #
# Refusals — the point of the module
# --------------------------------------------------------------------------- #
def test_an_unbound_symbol_is_refused_not_invented():
    m = dm.Model("unbound")
    x = m.continuous("x", lb=0.0, ub=1.0)
    s, syms = to_sympy(x)
    stray = sp.Symbol("z", real=True)
    with pytest.raises(SymbolicTranslationError, match="not in symbol_map"):
        from_sympy(s + stray, syms)


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda z: sp.Piecewise((z, z > 0), (0, True)),
        lambda z: sp.floor(z),
        lambda z: sp.ceiling(z),
        lambda z: sp.gamma(z),
        lambda z: sp.zeta(z),
    ],
)
def test_a_sympy_object_with_no_discopt_spelling_is_refused(expr_fn):
    m = dm.Model("refuse")
    x = m.continuous("x", lb=0.5, ub=1.0)
    _s, syms = to_sympy(x)
    (sym,) = syms
    with pytest.raises(SymbolicTranslationError):
        from_sympy(expr_fn(sym), syms)


def test_an_array_constant_is_refused():
    m = dm.Model("arr")
    x = m.continuous("x", shape=2, lb=0.0, ub=1.0)
    p = m.parameter("p", np.array([1.0, 2.0]))
    with pytest.raises(SymbolicTranslationError, match="array-valued"):
        to_sympy(x[0] * p)


def test_a_non_expression_is_refused():
    with pytest.raises(SymbolicTranslationError, match="expected a discopt Expression"):
        to_sympy(3.0)


def test_the_six_added_functions_are_exactly_the_ones_sympy_lacks():
    """A guard on the table itself: if SymPy ever grows one of these, the
    hand-written subclass should go away rather than shadow it."""
    table = DISCOPT_FUNCTIONS()
    assert set(table) == {"entropy", "sigmoid", "softplus", "log2", "log10", "log1p"}
    for name in table:
        assert not hasattr(sp, name), (
            f"sympy now has {name!r}; drop the hand-written subclass and map to it"
        )
