"""#1334: the two warm starts that still crashed a solve, and the route the issue missed.

#1324 set the contract -- a warm start is a HINT, so it may be ignored but must
never be able to fail a solve that succeeds without it. The adversary agent found
two ways through it, and a route scan while fixing them found a third:

1. **Non-finite point, ``nlp_bb=True``.** ``primal_point_from_result`` let a NaN
   through (``np.clip`` passes it: every comparison against NaN is False, so it
   survived the #1316 out-of-bounds clamp untouched), and the NLP-BB injection
   block tests integrality with ``round()`` BEFORE it checks ``np.isfinite``.
   ``round(nan)`` raises ``ValueError: cannot convert float NaN to integer``.

2. **Any warm start, ``solver="amp"`` on a GDP model.** The route set
   ``amp_kwargs["initial_point"]`` *before* ``reformulate_gdp`` appended the
   selector binaries, so ``_normalize_initial_point`` raised ``AMP initial_point
   has length 1; expected 3``. The same class #1255/#1324 fixed elsewhere, on a
   route neither covered.

3. **``solver="mip-nlp"``, same cause, not in the issue.** Found by scanning every
   route rather than the two reported: ``NLP initial point has shape (1,);
   expected (3,)``. Fixing the class rather than the two named instances is what
   CLAUDE.md §2 asks for, and is why the fix is one shared
   ``warm_start.prepare_warm_start`` rather than three more inline blocks.
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.warm_start import primal_point_from_result

_DONE = ("optimal", "feasible", "time_limit", "node_limit")


def _int_model():
    """A discrete column, so the injection block's ``round()`` runs."""
    m = dm.Model("i1334_int")
    y = m.integer("y", lb=0, ub=3)
    z = m.continuous("z", lb=0.0, ub=4.0)
    m.subject_to(y + z <= 5.0)
    m.minimize((y - 1.4) ** 2 + (z - 2.2) ** 2)
    return m, y, z


def _gdp_model():
    """The issue's repro: the lowering appends one selector per disjunct, so the
    user's point is 1 column wide and the solved model is 3."""
    m = dm.Model("i1334_gdp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.either_or([[x <= 2.0], [x >= 6.0]])
    m.minimize((x - 4.0) ** 2)
    return m, x


# ─────────────────────────────────────────────────────────────
# 1. non-finite warm start (issue part 1)
# ─────────────────────────────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nlp_bb_survives_a_non_finite_warm_start(bad):
    """The issue's repro verbatim. Before the fix ``bad=nan`` raised
    ``ValueError: cannot convert float NaN to integer`` from the injection
    block's ``round()``."""
    m, _y, _z = _int_model()
    r0 = m.solve()
    assert r0.status in _DONE
    r0.x["y"] = np.asarray(float(bad))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = m.solve(nlp_bb=True, warm_start=r0)

    assert r1.status in _DONE
    # And the poisoned hint must not have poisoned the answer.
    assert r1.objective == pytest.approx(0.16, abs=1e-5)


@pytest.mark.smoke
def test_a_nan_warm_start_is_repaired_at_the_producer_not_passed_on():
    """``primal_point_from_result`` is where the NaN got in, so that is where it
    is answered: a NaN says the previous solve did not determine that variable,
    which is exactly the state of a variable nobody supplied."""
    m, _y, _z = _int_model()
    r0 = m.solve()
    r0.x["y"] = np.asarray(float("nan"))

    with pytest.warns(UserWarning, match="NaN"):
        point = primal_point_from_result(m, r0)

    assert np.all(np.isfinite(point)), "the producer must not emit a non-finite point"


@pytest.mark.smoke
def test_an_explicit_non_finite_initial_solution_is_still_refused_loudly():
    """NOT a crash to fix: ``initial_solution`` is the caller naming a value, and
    a named NaN is an error in the caller's input, refused by name before any
    route runs. Pinned so the #1334 filtering never gets widened into silently
    accepting it."""
    m, y, _z = _int_model()
    with pytest.raises(ValueError, match="non-finite"):
        m.solve(initial_solution={y: float("nan")})


# ─────────────────────────────────────────────────────────────
# 2/3. a reformulation-widened warm start, on every route
# ─────────────────────────────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize(
    "route",
    [
        pytest.param({"solver": "amp"}, id="amp"),
        pytest.param({"solver": "mip-nlp"}, id="mip-nlp"),
        pytest.param({}, id="default"),
        pytest.param({"nlp_bb": True}, id="nlp_bb"),
    ],
)
@pytest.mark.parametrize("kind", ["warm_start", "initial_solution"])
def test_a_warm_start_cannot_fail_a_gdp_solve_on_any_route(route, kind):
    """``solver="amp"`` raised ``AMP initial_point has length 1; expected 3`` and
    ``solver="mip-nlp"`` raised ``NLP initial point has shape (1,); expected
    (3,)`` on a model both solve fine with no warm start."""
    m_ref, _x_ref = _gdp_model()
    r0 = m_ref.solve(time_limit=30)
    assert r0.status in _DONE

    m, x = _gdp_model()
    baseline = m.solve(time_limit=30, **route)

    m2, x2 = _gdp_model()
    hint = {"warm_start": r0} if kind == "warm_start" else {"initial_solution": {x2: 6.0}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warmed = m2.solve(time_limit=30, **route, **hint)

    assert warmed.status in _DONE
    # A hint may change the search; it may not change the answer.
    assert warmed.status == baseline.status
    assert warmed.objective == pytest.approx(baseline.objective, rel=1e-4, abs=1e-6)


# ─────────────────────────────────────────────────────────────
# the shared helper both halves now go through
# ─────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_prepare_warm_start_drops_rather_than_raises():
    """Every arm returns a usable vector or ``None``. Nothing it is handed may
    raise, because every caller is on a solve path that works without it."""
    from discopt.warm_start import prepare_warm_start

    m, x = _gdp_model()
    n_user = sum(int(v.size) for v in m._variables)
    assert n_user == 1

    checks = 0

    # None in, None out.
    assert prepare_warm_start(m, None, route="t") is None
    checks += 1

    # Exactly the model's width: returned as a float64 vector.
    exact = prepare_warm_start(m, [6.0], route="t")
    checks += 1
    assert exact is not None and exact.shape == (1,) and exact.dtype == np.float64

    # Non-finite: dropped, never raised, whichever kind.
    for bad in (np.nan, np.inf, -np.inf):
        assert prepare_warm_start(m, [bad], route="t") is None
        checks += 1

    # Wider than the model: cannot describe it, so dropped.
    assert prepare_warm_start(m, [1.0, 2.0, 3.0, 4.0], route="t") is None
    checks += 1

    assert checks == 6, f"probe ran {checks} assertions, expected 6"


@pytest.mark.smoke
def test_prepare_warm_start_completes_across_the_gdp_lowering():
    """The positive half: a 1-column user point becomes the lowered model's
    3 columns, with the selector repaired to the disjunct the point implies."""
    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt.warm_start import prepare_warm_start

    m, _x = _gdp_model()
    lowered = reformulate_gdp(m, method="big-m")
    n_lowered = sum(int(v.size) for v in lowered._variables)
    assert n_lowered > 1, "the lowering must have appended columns for this to test anything"

    completed = prepare_warm_start(lowered, [6.0], route="t")
    assert completed is not None
    assert completed.shape == (n_lowered,)
    assert np.all(np.isfinite(completed))
    assert completed[0] == pytest.approx(6.0)
    # x = 6 satisfies the second disjunct only, and ``sum(selectors) == 1``.
    selectors = completed[1:]
    assert sum(round(float(s)) for s in selectors) == 1


# ─────────────────────────────────────────────────────────────
# the "Minor": reformulation auxiliaries are visible in ``result.x``
# ─────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_declared_x_returns_only_the_callers_own_variables():
    """``result.x`` is keyed by the columns the SOLVER solved, so a GDP model's
    result carries ``_gdp_aux_*`` selectors the caller never wrote. They stay in
    ``x`` -- for a disjunctive model the selector is the only record of which
    disjunct was chosen -- and ``declared_x`` gives the caller's own."""
    m, _x = _gdp_model()
    r = m.solve(time_limit=30)
    assert r.status in _DONE

    generated = [k for k in r.x if k not in {v.name for v in m._variables}]
    assert generated, "this model must reformulate for the test to mean anything"

    assert set(r.declared_x(m)) == {v.name for v in m._variables}
    assert set(r.declared_x(m)) == {"x"}
    # Nothing is dropped from ``x`` itself.
    assert set(generated).issubset(set(r.x))


@pytest.mark.smoke
def test_declared_x_is_exact_not_a_name_prefix_rule():
    """A user variable whose own name starts with an underscore is the caller's,
    and must survive. The split reads the model's variable list rather than
    guessing from the name, so it cannot fall behind a new reform pass."""
    m = dm.Model("i1334_underscore")
    u = m.continuous("_my_var", lb=0.0, ub=3.0)
    m.minimize((u - 1.0) ** 2)
    r = m.solve(time_limit=30)
    assert r.status in _DONE
    assert "_my_var" in r.declared_x(m)


@pytest.mark.smoke
def test_declared_x_on_a_result_with_no_point():
    """An infeasible/errored solve reports no point; the accessor must not raise."""
    m, _x = _gdp_model()
    assert dm.SolveResult(status="infeasible").declared_x(m) == {}
