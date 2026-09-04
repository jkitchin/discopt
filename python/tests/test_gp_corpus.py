"""Regression corpus for the geometric-programming / log-convexity subsystem (#111).

These tests characterise the GP subsystem end to end:

* :func:`discopt.gp.is_log_convex` — the **log-space** convexity verdict, kept
  separate from :func:`discopt._relax.convexity.classify_model` (the **x-space**
  verdict). A genuine GP is convex only under ``y = log x``, so the two
  disagree: ``is_log_convex`` is ``True`` while x-space ``classify_model`` is
  ``False``. This separation is a soundness requirement — folding log-convexity
  into the x-space verdict would mis-gate the x-space convex fast path.
* **Automatic routing** — a plain ``model.solve()`` (no ``solver=`` argument) on
  a recognised GP is dispatched through the exact log-space convex solve, so the
  result carries ``convex_fast_path is True``, a valid ``bound`` equal to the
  objective, and a certified zero ``gap``. ``solver="bb"`` opts out.
* **Negative controls** — signomials (mixed-sign coefficients) and integer-
  variable models are not GPs: ``is_log_convex`` is ``False`` and they are not
  auto-routed.

Why no MINLPLib ``.nl`` corpus
------------------------------
MINLPLib does not ship raw-posynomial geometric programs. A scan of the cached
instance set found ``classify_gp`` recognises none of them: the ``cvxnonsep_*``
convex family are either sums of individually-convex monomials (already convex in
x-space, handled by the #40 signomial-monomial recogniser) or carry integer
variables, and the ``*_r`` reformulated variants are stored already-convexified
(they contain ``log``). The genuine log-only-convex structure therefore lives in
programmatically-built classic GPs, which is what this corpus uses — each with a
closed-form optimum (Boyd & Vandenberghe Ch. 4.5).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import pytest
from discopt._relax.convexity import classify_model
from discopt.gp import classify_gp, is_log_convex, solve_gp
from discopt.modeling.core import Model

# Strictly-positive box shared by the continuous GP variables.
POS = dict(lb=1e-3, ub=1e3)


# ──────────────────────────────────────────────────────────────────────
# Corpus: classic GPs with closed-form optima
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GPCase:
    """A GP model factory plus its analytic optimum."""

    name: str
    build: Callable[[], Model]
    optimum: float


def _monomial_balance() -> Model:
    # minimize x/y + y/x over x, y > 0. Optimum 2 at x == y (AM-GM).
    m = Model("balance")
    x = m.continuous("x", **POS)
    y = m.continuous("y", **POS)
    m.minimize(x / y + y / x)
    return m


def _posynomial_objective() -> Model:
    # minimize x + 1/(x*y) + y. Stationarity gives x == y == 1, value 3.
    m = Model("posyobj")
    x = m.continuous("x", **POS)
    y = m.continuous("y", **POS)
    m.minimize(x + 1.0 / (x * y) + y)
    return m


def _constrained_posynomial() -> Model:
    # minimize x + y s.t. 1/(x*y) <= 1 (i.e. x*y >= 1). AM-GM: min 2 at x==y==1.
    # A posynomial <= monomial inequality binding at the optimum.
    m = Model("cobb")
    x = m.continuous("x", **POS)
    y = m.continuous("y", **POS)
    m.minimize(x + y)
    m.subject_to(1.0 / (x * y) <= 1.0)
    return m


def _box_volume() -> Model:
    # maximize x*y (monomial) s.t. x*y <= 6, x/y <= 3, y/x <= 3.
    # The volume bound is tight => optimum 6.
    m = Model("boxvol")
    x = m.continuous("x", **POS)
    y = m.continuous("y", **POS)
    m.maximize(x * y)
    m.subject_to(x * y <= 6.0)
    m.subject_to(x / y <= 3.0)
    m.subject_to(y / x <= 3.0)
    return m


CORPUS = (
    GPCase("monomial_balance", _monomial_balance, 2.0),
    GPCase("posynomial_objective", _posynomial_objective, 3.0),
    GPCase("constrained_posynomial", _constrained_posynomial, 2.0),
    GPCase("box_volume", _box_volume, 6.0),
)


def _ids(cases: tuple[GPCase, ...]) -> list[str]:
    return [c.name for c in cases]


# ──────────────────────────────────────────────────────────────────────
# Log-convex verdict, distinct from the x-space verdict
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", CORPUS, ids=_ids(CORPUS))
def test_corpus_is_log_convex(case: GPCase) -> None:
    """Every corpus GP is recognised as log-convex (a GP in standard form)."""
    model = case.build()
    assert is_log_convex(model) is True
    assert classify_gp(model) is not None


@pytest.mark.parametrize("case", CORPUS, ids=_ids(CORPUS))
def test_corpus_is_not_x_space_convex(case: GPCase) -> None:
    """SOUNDNESS SEPARATION: a GP is log-convex but NOT x-space convex.

    ``classify_model`` (with the certificate, the exact setting that gates the
    x-space convex fast path) must keep returning ``is_convex == False`` for
    these genuinely log-only-convex models — otherwise the x-space fast path
    would be taken on a problem that is not convex in x.
    """
    model = case.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        is_convex, _mask = classify_model(model, use_certificate=True)
    assert is_convex is False, (
        f"{case.name}: classify_model promoted a log-only-convex GP to x-space "
        f"convex — this would mis-gate the x-space convex fast path."
    )


# ──────────────────────────────────────────────────────────────────────
# Automatic routing through the log-space convex solve
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", CORPUS, ids=_ids(CORPUS))
def test_corpus_auto_routes_to_gp_fast_path(case: GPCase) -> None:
    """A plain ``model.solve()`` auto-routes a GP through the exact log solve.

    The result reaches the closed-form optimum with ``convex_fast_path`` set, a
    valid ``bound`` equal to the objective, and a certified zero gap.
    """
    model = case.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.solve()
    assert result.status == "optimal"
    assert result.objective == pytest.approx(case.optimum, abs=1e-4)
    assert result.convex_fast_path is True
    assert result.bound == pytest.approx(result.objective, abs=1e-9)
    assert result.gap == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("case", CORPUS, ids=_ids(CORPUS))
def test_corpus_auto_route_matches_solve_gp(case: GPCase) -> None:
    """Auto-route and the explicit ``solve_gp`` entry point agree."""
    auto = case.build()
    direct = case.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto_result = auto.solve()
        direct_result = solve_gp(direct)
    assert direct_result is not None
    assert auto_result.objective == pytest.approx(direct_result.objective, abs=1e-6)


@pytest.mark.slow
def test_bb_opt_out_skips_gp_fast_path() -> None:
    """``solver="bb"`` forces classic branch-and-bound, not the GP fast path."""
    model = _monomial_balance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Bounded budget: this is a nonconvex division GP (x/y + y/x) over a
        # wide [1e-3, 1e3] box. The optimum (2.0) is found at the root, but
        # *certifying* it via spatial B&B + the sound alphaBB/McCormick bound
        # exhausts the tree only after many minutes — the prior unbounded form
        # relied on an unsound NLP pruning bound (removed in #120) to terminate
        # fast. The classic-path assertions below hold the instant the
        # incumbent is found (at the root), so a small budget suffices and keeps
        # this off the fast-CI critical path; the cap only bounds the
        # never-reached certification, not the assertions.
        result = model.solve(solver="bb", time_limit=5.0)
    assert result.status in ("optimal", "feasible")
    # The classic path does not set the convex single-NLP fast-path flag. This is
    # what the test is actually about, and it holds today.
    assert result.convex_fast_path is False
    # #1039: the objective assertion that used to live here now has its own test
    # below -- it fails, and it fails for a soundness reason, so it is pinned as a
    # strict xfail rather than having its tolerance widened past the defect.


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "#1151: reported objective is BELOW the true global minimum. The error is "
        "the absolute constraint tolerance amplified by 1/denominator, so it is "
        "unbounded as the denominator shrinks. Fix is in the Rust incumbent path."
    ),
)
def test_bb_reported_objective_is_attained_by_its_own_incumbent() -> None:
    """A reported objective must be the objective AT the returned point.

    #1039 bucket E listed this as an accuracy miss --
    ``assert 1.998683979470214 == 2.0 +- 1.0e-04`` -- and the obvious repair
    would have been to widen the tolerance to 2e-3. That would have masked a
    soundness defect.

    ``minimize x/y + y/x`` over a positive box has global minimum exactly 2 by
    AM-GM: no feasible point attains less. The solve returns ``status=optimal``
    with ``objective=1.998683979470214``, which is 1.3e-3 BELOW that minimum --
    a value no feasible point achieves. The returned point itself is fine
    (x=0.0014052502011193727, y=0.0014073586395206353, inside the box, true
    objective 2.000002247829649); it is the reported number that is wrong, by
    -1.318268e-03 against its own incumbent.

    Mechanism, measured (``scratchpad/issue1039/probe_gp2.py``, ``probe_gp3.py``):
    the error is the absolute feasibility tolerance divided by the quotient's
    denominator. Only the division case misbehaves -- an affine objective and a
    *bilinear* one (which also needs a McCormick auxiliary) both agree with the
    oracle to 4e-16, so this is specific to the quotient reformulation, not to
    auxiliaries in general. Scaling the box floor confirms the 1/y law, with
    ``|delta| * denominator`` flat at ~1.9e-6, i.e. ~2x the 1e-6 absolute
    tolerance for the two quotient terms:

        box floor   denominator   delta            |delta|*denominator
        1e-3        0.00140525    -1.318268e-03    1.852e-06
        1e-2        0.0106986     -1.850776e-04    1.980e-06
        1e-1        8.13524       -4.360956e-13    3.548e-12
        1e+0        479.758       -4.440892e-16    2.131e-13

    (the last two land at large denominators, so the amplification vanishes).

    A trace over the whole solve found ZERO Python frames returning the bad
    value, so it is produced in the Rust B&B incumbent path and passed through --
    which is why this is pinned here rather than fixed in this PR: the fix is a
    bound-changing solver change and needs the §5 differential regime, not a
    test-repair PR.

    ``strict=True`` on purpose: when the incumbent objective is recomputed from
    the original expression at the incumbent point (the general fix -- one
    evaluation, and it can only make the number correct, since the point is
    feasible and its true objective is a valid upper bound), this test XPASSes
    and fails the suite, which is the signal to remove the xfail.
    """
    model = _monomial_balance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.solve(solver="bb", time_limit=5.0)

    x, y = model._variables[0], model._variables[1]
    xv, yv = float(result.value(x)), float(result.value(y))
    # Oracle written outside the system: plain Python arithmetic on the returned
    # point, so a shared mistake in the solver cannot make this pass.
    oracle = xv / yv + yv / xv

    assert oracle >= 2.0 - 1e-9, f"the returned point itself is infeasible: f={oracle}"
    assert result.objective == pytest.approx(oracle, abs=1e-9), (
        f"reported objective {result.objective!r} is not attained by its own "
        f"incumbent ({xv!r}, {yv!r}), whose true objective is {oracle!r} "
        f"(delta {result.objective - oracle:+.6e})"
    )


def test_unknown_solver_is_rejected() -> None:
    """An unrecognised ``solver=`` value raises rather than silently routing."""
    model = _monomial_balance()
    with pytest.raises(ValueError, match="Unknown solver"):
        model.solve(solver="xyz")


# ──────────────────────────────────────────────────────────────────────
# Negative controls: not GPs, must not be log-convex or auto-routed
# ──────────────────────────────────────────────────────────────────────


def test_signomial_is_not_log_convex() -> None:
    """A mixed-sign signomial objective is not a GP."""
    m = Model("signomial")
    x = m.continuous("x", **POS)
    y = m.continuous("y", **POS)
    m.minimize(x * y - x)  # negative-coefficient term => signomial, not posynomial
    assert is_log_convex(m) is False
    assert solve_gp(m) is None


def test_integer_variable_is_not_log_convex() -> None:
    """An integer variable disqualifies the (continuous) GP fast path."""
    m = Model("intvar")
    x = m.integer("x", lb=1, ub=10)
    y = m.continuous("y", **POS)
    m.minimize(x / y + y / x)
    assert is_log_convex(m) is False
    assert solve_gp(m) is None


def test_nonpositive_variable_is_not_log_convex() -> None:
    """A variable whose lower bound is not strictly positive is not a GP."""
    m = Model("nonpos")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(x * x)
    assert is_log_convex(m) is False


# ──────────────────────────────────────────────────────────────────────
# #1151 — the reported objective must be attained by the reported point
# ──────────────────────────────────────────────────────────────────────


def _division_gp(floor: float) -> Model:
    """``minimize x/y + y/x`` over ``[floor, 1e3]^2``. Global minimum exactly 2
    by AM-GM (``t + 1/t >= 2`` for ``t > 0``), attained on the diagonal."""
    m = Model("balance")
    x = m.continuous("x", lb=floor, ub=1e3)
    y = m.continuous("y", lb=floor, ub=1e3)
    m.minimize(x / y + y / x)
    return m


@pytest.mark.slow
@pytest.mark.parametrize("floor", [1e-3, 1e-2, 1e-1, 1.0], ids=lambda f: f"floor{f:g}")
def test_bb_reported_objective_is_attained_by_its_own_incumbent(floor: float) -> None:
    """#1151: ``objective`` and ``x`` must agree, at every denominator floor.

    The defect this pins was a **false certificate**, not an accuracy miss. The
    solver reported ``objective = 1.998683979470214`` at ``status = optimal`` on
    a problem whose global minimum is exactly 2 — a value no feasible point
    attains, its own incumbent included (true objective there: 2.000002247829649).

    The mechanism: a quotient in the objective is lifted to an aux ``w == x/y``
    whose defining equality is cleared to ``w*y - x == 0``. A residual ``eps`` on
    that row is an error of ``eps/y`` in ``w`` — and ``w`` is what the objective
    reads. The incumbent verifier's row-scale term over-read the (deliberately
    ``1/dmin``-scaled) row's magnitude by ``1/|x_j|``, restoring exactly the
    amplification the scaling removes, so the error grew without bound as the
    denominator shrank:

    ==========  ==========  =============  ==============
    box floor   denominator  reported−true  |delta|×denom
    ==========  ==========  =============  ==============
    1e-3        0.00140525   -1.318268e-03   1.852e-06
    1e-2        0.0106986    -1.850776e-04   1.980e-06
    1e-1        8.13524      -4.360956e-13   3.548e-12
    1e+0        479.758      -4.440892e-16   2.131e-13
    ==========  ==========  =============  ==============

    ``|delta| x denominator`` flat at ~1.9e-6 (twice the 1e-6 absolute tolerance,
    for the two quotient terms) is the signature: no fixed tolerance widening
    bounds it, because the amplification is unbounded as ``y -> 0``.

    The assertion is therefore a *relative* one, keyed on the objective's own
    magnitude and on nothing about an intermediate denominator.
    """
    model = _division_gp(floor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Bounded budget, and deliberately shorter than an honest certificate
        # needs on the tight floors (measured: 37.6 s at 1e-3, 227.8 s at 1e-2 —
        # ``scratchpad/issue1151/cert_cost.py``). The assertions are about the
        # reported VALUE, which is right from the first incumbent, not about
        # reaching ``optimal``; the defect certified in 0.6 s, so a short budget
        # discriminates just as sharply and keeps this off the slow critical path.
        result = model.solve(solver="bb", time_limit=20.0)

    assert result.objective is not None, "no incumbent to check"
    assert result.x is not None, "an incumbent value with no point is not checkable"
    xv = float(result.x["x"])
    yv = float(result.x["y"])
    # Plain-Python oracle: the ORIGINAL objective, no solver machinery involved.
    oracle = xv / yv + yv / xv
    tol = 1e-6 * (1.0 + abs(oracle))

    assert result.objective == pytest.approx(oracle, abs=tol), (
        f"reported objective {result.objective!r} is not the objective at the "
        f"reported point ({oracle!r}); delta {result.objective - oracle:.6e}"
    )
    # And the reported value must not be below the true global minimum at all.
    assert result.objective >= 2.0 - tol, (
        f"reported objective {result.objective!r} is BELOW the global minimum 2.0 "
        f"— a false certificate"
    )
    if result.bound is not None:
        assert result.bound <= 2.0 + 1e-6, (
            f"dual bound {result.bound!r} exceeds the true global minimum 2.0"
        )
