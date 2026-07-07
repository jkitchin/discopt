"""Regression corpus for the geometric-programming / log-convexity subsystem (#111).

These tests characterise the GP subsystem end to end:

* :func:`discopt.gp.is_log_convex` — the **log-space** convexity verdict, kept
  separate from :func:`discopt._jax.convexity.classify_model` (the **x-space**
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
from discopt._jax.convexity import classify_model
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
    assert result.objective == pytest.approx(2.0, abs=1e-4)
    # The classic path does not set the convex single-NLP fast-path flag.
    assert result.convex_fast_path is False


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
