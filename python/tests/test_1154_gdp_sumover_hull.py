"""#1154: the GDP walkers must handle ``SumOverExpression`` — or refuse loudly.

``dm.sum(f(i) for i in S)`` builds a :class:`SumOverExpression`, an n-ary node
holding its already-expanded term list in ``.terms``. Six walkers in
``discopt._relax.gdp_reformulate`` had no case for it, so a disjunct body built
that way was under-reported as variable-free (``_collect_variables``), nonlinear
(``_is_linear``), unbounded (``_bound_expression``) and un-evaluable at the
origin (``_body_at_zero``) — three loud refusals across ``auto``/``big-m``/
``hull``.

The contract this file pins is a **desugaring equivalence**: every walker must
answer identically on ``Σ[t1, …, tn]`` and on the left-folded chain
``t1 + … + tn``, which is the expression the modeling layer could equally have
produced. Nothing about the reformulation mathematics is new; only the node type
is newly recognised.

Two guards sit alongside it:

* with ``DISCOPT_GDP_SUMOVER`` OFF — the opt-out, which stays supported after
  the §5 panel graduated the flag default-ON — the walkers are byte-identical to
  pre-#1154: every route still refuses, and **no route reports a bound above the
  true optimum**; and
* :func:`_assert_hull_saw_every_variable` refuses loudly when the hull collector
  misses a variable an *independent* DAG walker can see. That is the general
  fix for the PR #1150 defect: widening ``_is_linear`` alone left ``all_vars``
  empty, so hull emitted ``Σ[3 terms] - (0 * y0) <= 0`` — the disjunct body
  imposed globally with its selector coefficient collapsed to zero — and both
  ``auto`` and ``hull`` certified ``bound = -3.0`` on a model whose true minimum
  is ``-30.0``. A dual bound above the optimum of a minimization is a false
  certificate (CLAUDE.md §1).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import (
    HullVariableCoverageError,
    _body_at_zero,
    _bound_expression,
    _collect_variables,
    _hull_linear_substitute,
    _is_linear,
    _substitute_vars,
)
from discopt.modeling.core import SumOverExpression, _wrap
from discopt.solver_tuning import SolverTuning

SUMOVER_ENV = "DISCOPT_GDP_SUMOVER"

#: The repro from #1154 / PR #1150. ``x_i in [0, 10]`` and the objective is
#: ``-(x0 + x1 + x2)``, so the second disjunct (``x0 >= 8``) leaves x1, x2 free
#: at their upper bounds and the true minimum is exactly -30.0.
TRUE_OPTIMUM = -30.0


@pytest.fixture(autouse=True)
def _sumover_on(monkeypatch):
    """Pin the flag ON for this file, independent of the shipped default.

    The default is ON (graduated 2026-09-04), so this is belt-and-braces — but it
    keeps the ON-arm tests meaningful if the default is ever flipped back, and it
    stops an ambient ``DISCOPT_GDP_SUMOVER=0`` in the developer's environment from
    turning them into silent no-ops. :func:`test_the_shipped_default_is_on` is what
    actually asserts the default.
    """
    monkeypatch.setenv(SUMOVER_ENV, "1")


def test_the_shipped_default_is_on(monkeypatch):
    """The §5 panel graduated the flag; a silent revert must fail here.

    Read with the env var *removed*, so this sees the field default rather than
    whatever the surrounding environment happens to say.
    """
    monkeypatch.delenv(SUMOVER_ENV, raising=False)
    assert SolverTuning().gdp_sumover is True
    monkeypatch.setenv(SUMOVER_ENV, "0")
    assert SolverTuning().gdp_sumover is False, "the =0 opt-out must stay live"


def _repro_model():
    m = dm.Model("sumover_disjunct")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(3)]
    m.either_or([[dm.sum(x[i] - 1 for i in range(3)) <= 0.0], [x[0] >= 8.0]])
    m.minimize(-(x[0] + x[1] + x[2]))
    return m


def _fold(terms):
    """The left-folded ``t1 + t2 + ... + tn`` chain equivalent to ``Σ[terms]``."""
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return acc


def _term_sets(m):
    """Term lists spanning the shapes the walkers have to agree on."""
    x, y, z = m._variables[0], m._variables[1], m._variables[2]
    return {
        "affine": [x - 1, y - 1, z - 1],
        "scaled": [_wrap(2.0) * x, _wrap(-3.0) * y, z / _wrap(4.0)],
        "with_constants": [x, _wrap(5.0), -y],
        "nested_sumover": [dm.sum(v for v in (x, y)), z - 2],
        "single": [x + 1],
        "nonlinear": [x * y, z],
    }


# ── The desugaring equivalence, walker by walker ────────────────────────────


def _sumover_model():
    m = dm.Model("walkers")
    m.continuous("x", lb=-2.0, ub=5.0)
    m.continuous("y", lb=0.0, ub=3.0)
    m.continuous("z", lb=1.0, ub=4.0)
    return m


def test_is_linear_matches_the_folded_chain():
    m = _sumover_model()
    checked = 0
    for name, terms in _term_sets(m).items():
        got = _is_linear(SumOverExpression(list(terms)))
        want = _is_linear(_fold(terms))
        assert got == want, f"{name}: Σ -> {got}, folded chain -> {want}"
        checked += 1
    # ``nonlinear`` must actually exercise the False arm, or the loop above is
    # only ever asserting True == True (CLAUDE.md §6).
    assert _is_linear(SumOverExpression(_term_sets(m)["nonlinear"])) is False
    assert _is_linear(SumOverExpression(_term_sets(m)["affine"])) is True
    assert checked == 6, f"only {checked} term sets compared"


def test_collect_variables_matches_the_folded_chain():
    m = _sumover_model()
    checked = 0
    for name, terms in _term_sets(m).items():
        got = set(_collect_variables(SumOverExpression(list(terms))))
        want = set(_collect_variables(_fold(terms)))
        assert got == want, f"{name}: Σ -> {sorted(got)}, folded -> {sorted(want)}"
        assert got, f"{name}: the probe collected nothing, so it compared nothing"
        checked += 1
    assert checked == 6


def test_bound_expression_matches_the_folded_chain():
    m = _sumover_model()
    checked = 0
    for name, terms in _term_sets(m).items():
        got = _bound_expression(SumOverExpression(list(terms)), m)
        want = _bound_expression(_fold(terms), m)
        assert got == want, f"{name}: Σ -> {got}, folded -> {want}"
        assert np.all(np.isfinite(got)) or name == "nonlinear", name
        checked += 1
    # Positive control: the #1154 body really is finitely bounded, which is what
    # unblocks big-M. x_i in [-2,5]/[0,3]/[1,4] -> Σ(v - 1) in [-4, 9].
    lo, hi = _bound_expression(SumOverExpression(_term_sets(m)["affine"]), m)
    assert (lo, hi) == (-4.0, 9.0), (lo, hi)
    assert checked == 6


def test_bound_expression_does_not_produce_nan_on_mixed_infinities():
    """An unbounded-below and an unbounded-above term in one sum.

    Plain float addition would give ``-inf + inf = nan``, which compares False
    against every finite threshold and so reads as "bounded" at the big-M gate.
    """
    m = dm.Model("infinities")
    lo_free = m.continuous("lo_free", lb=-np.inf, ub=0.0)
    hi_free = m.continuous("hi_free", lb=0.0, ub=np.inf)
    lo, hi = _bound_expression(SumOverExpression([lo_free, hi_free]), m)
    assert not np.isnan(lo) and not np.isnan(hi)
    assert lo == -np.inf and hi == np.inf


def test_body_at_zero_matches_the_folded_chain():
    m = _sumover_model()
    all_vars = {v.name: v for v in m._variables}
    checked = 0
    for name, terms in _term_sets(m).items():
        got = _body_at_zero(SumOverExpression(list(terms)), all_vars)
        want = _body_at_zero(_fold(terms), all_vars)
        assert got == want, f"{name}: Σ -> {got}, folded -> {want}"
        checked += 1
    # Positive control: Σ(v - 1) over three variables is -3 at the origin, and a
    # nonzero g(0) is exactly what the FSG perspective needs (#1043).
    assert _body_at_zero(SumOverExpression(_term_sets(m)["affine"]), all_vars) == -3.0
    assert checked == 6


def test_substitute_vars_matches_the_folded_chain():
    m = _sumover_model()
    sub = m.continuous("x_sub", lb=-2.0, ub=5.0)
    var_map = {"x": sub}
    all_vars = {v.name: v for v in m._variables}
    checked = 0
    for name, terms in _term_sets(m).items():
        got = _substitute_vars(SumOverExpression(list(terms)), var_map)
        want = _substitute_vars(_fold(terms), var_map)
        # Compare by what the substitution was *for*: which variables survive.
        assert set(_collect_variables(got)) == set(_collect_variables(want)), name
        assert "x" not in _collect_variables(got), f"{name}: 'x' survived substitution"
        checked += 1
    # And it is a real rebuild, not the node passed through unchanged.
    node = SumOverExpression(_term_sets(m)["affine"])
    assert _substitute_vars(node, var_map) is not node
    # A map that touches nothing still short-circuits to the same object.
    assert _substitute_vars(node, {"absent": sub}) is node
    assert _body_at_zero(node, all_vars) == -3.0  # unchanged by substitution
    assert checked == 6


def test_hull_linear_substitute_matches_the_folded_chain():
    m = _sumover_model()
    y_k = m.binary("y_k")
    disagg = {name: m.continuous(f"v_{name}", lb=-5.0, ub=5.0) for name in ("x", "y", "z")}
    checked = 0
    for name, terms in _term_sets(m).items():
        if name == "nonlinear":
            continue  # the linear route is not taken for a nonlinear body
        got = _hull_linear_substitute(SumOverExpression(list(terms)), disagg, y_k)
        want = _hull_linear_substitute(_fold(terms), disagg, y_k)
        got_names = set(_collect_variables(got))
        assert got_names == set(_collect_variables(want)), name
        # The whole point: originals are gone, disaggregated vars are in, and
        # the constant part is carried on the selector.
        assert got_names.isdisjoint({"x", "y", "z"}), f"{name}: an original survived"
        checked += 1
    assert checked == 5
    # ``with_constants`` carries a literal 5.0, so y_k must appear.
    scaled = _hull_linear_substitute(
        SumOverExpression(_term_sets(m)["with_constants"]), disagg, y_k
    )
    assert "y_k" in _collect_variables(scaled)


# ── End to end: the routes, and the bound they may never report ────────────


@pytest.mark.parametrize("method", ["auto", "big-m", "hull"])
def test_every_route_solves_the_sumover_disjunct_to_the_true_optimum(method):
    result = _repro_model().solve(gdp_method=method, time_limit=60)
    assert result.status == "optimal", (method, result.status)
    assert result.objective == pytest.approx(TRUE_OPTIMUM, abs=1e-4)
    assert result.bound is not None
    assert result.bound <= TRUE_OPTIMUM + 1e-6, (
        f"[{method}] dual bound {result.bound} is ABOVE the true minimum "
        f"{TRUE_OPTIMUM}: an invalid certificate"
    )


@pytest.mark.parametrize("method", ["auto", "big-m", "hull"])
def test_no_route_ever_reports_a_bound_above_the_true_optimum(method, monkeypatch):
    """The #1150 regression guard — and it must hold in BOTH flag arms.

    With the flag OFF every route refuses; with it ON every route solves. What
    neither arm may ever do is answer with a dual bound above -30.0, which is
    what the reverted #1150 widening did.
    """
    checked = 0
    for arm in ("0", "1"):
        monkeypatch.setenv(SUMOVER_ENV, arm)
        try:
            result = _repro_model().solve(gdp_method=method, time_limit=60)
        except (ValueError, NotImplementedError) as exc:
            # A loud refusal is an acceptable answer; a wrong bound is not.
            assert arm == "0", f"[{method}] refused with the flag ON: {exc}"
            checked += 1
            continue
        assert result.bound is None or result.bound <= TRUE_OPTIMUM + 1e-6, (
            f"[{method}, flag={arm}] bound {result.bound} > true optimum {TRUE_OPTIMUM}"
        )
        checked += 1
    assert checked == 2


@pytest.mark.parametrize("method", ["auto", "big-m", "hull"])
def test_flag_off_still_refuses_loudly(method, monkeypatch):
    """The OFF arm is the pre-#1154 behaviour: a refusal, never a wrong answer."""
    monkeypatch.setenv(SUMOVER_ENV, "0")
    with pytest.raises((ValueError, NotImplementedError)):
        _repro_model().solve(gdp_method=method, time_limit=60)


def test_hull_refuses_when_the_collector_misses_a_variable(monkeypatch):
    """The independent-walker cross-check, which is what #1150 lacked.

    With the flag OFF the hull collector cannot see into ``Σ[...]`` but
    ``_iter_model_leaves`` can, so the guard fires by name on exactly the
    variables that would otherwise have been emitted un-disaggregated.
    """
    monkeypatch.setenv(SUMOVER_ENV, "0")
    with pytest.raises(HullVariableCoverageError) as excinfo:
        _repro_model().solve(gdp_method="hull", time_limit=60)
    message = str(excinfo.value)
    # x0 also appears in the second disjunct as a plain variable, so it *is*
    # collected; x1 and x2 exist only inside the summation.
    assert "'x1'" in message and "'x2'" in message, message


def test_the_guard_is_silent_once_the_walkers_can_see_the_node():
    """The same model, flag ON: the guard must not fire on a body it can read."""
    result = _repro_model().solve(gdp_method="hull", time_limit=60)
    assert result.status == "optimal"
    assert result.objective == pytest.approx(TRUE_OPTIMUM, abs=1e-4)


# ── The incumbent is a real point of the original model ────────────────────


@pytest.mark.parametrize("method", ["auto", "big-m", "hull"])
def test_the_returned_point_is_feasible_for_the_original_disjunction(method):
    """§5 feasible-point verification: check the answer, not just the number."""
    result = _repro_model().solve(gdp_method=method, time_limit=60)
    assert result.x is not None
    xs = [float(result.x[f"x{i}"]) for i in range(3)]
    for xi in xs:
        assert -1e-6 <= xi <= 10.0 + 1e-6, xs
    arm_a = sum(xi - 1.0 for xi in xs) <= 1e-6
    arm_b = xs[0] >= 8.0 - 1e-6
    assert arm_a or arm_b, f"[{method}] {xs} satisfies neither disjunct"
    assert -sum(xs) == pytest.approx(result.objective, abs=1e-4)
