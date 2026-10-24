"""#1497 follow-up: a solve no longer swallows a crashed reformulation pass.

PR #1511 made ``integer.bilinear`` / ``integer.multilinear`` / ``binary.multilinear``
raise on a defect and absorb only their documented give-ups
(``IntegerProductNotApplicable`` / ``_Unsupported``). But ``solve_model`` still
wrapped each call site in ``except Exception`` + a DEBUG log, so during a SOLVE a
defect in the pass was still read as "nothing to reformulate" and the model was
solved un-reformed with no visible trace (CLAUDE.md §3/§7).

Each test injects a failure into the pass internals and asserts two things:

* a ``RuntimeError`` (a defect) now fails ``Model.solve`` -- it was swallowed before;
* the documented not-applicable exception still leaves the solve running, on the
  un-reformed model, to the correct optimum.

Every test counts the injected calls and asserts the count is non-zero, so a model
that stopped reaching the pass would fail here rather than pass vacuously (§6).
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax import binary_multilinear_reform as bml
from discopt._relax import disjunctive_config_bound as dcb
from discopt._relax import integer_product_reform as ipr


class _Boom(RuntimeError):
    """The injected defect; a distinct type so the assertion cannot match by luck."""


def _binary_cubic() -> dm.Model:
    # Pure-binary degree-3 polynomial: the binary-multilinear witness fires.
    m = dm.Model("bml_1497")
    x = m.binary("x")
    y = m.binary("y")
    z = m.binary("z")
    m.minimize(-(x * y * z) + 0.5 * x + 0.25 * y)
    m.subject_to(x + y + z >= 1)
    return m


def _integer_bilinear() -> dm.Model:
    # Distinct-variable integer x continuous product: the nonconvexity witness
    # ``has_nonconvex_integer_bilinear`` fires and the lift is a pure MILP.
    m = dm.Model("ipx_1497")
    n = m.integer("n", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=3)
    m.minimize(-(n * y) + 2 * n + y)
    m.subject_to(n + y <= 6)
    return m


def _integer_trilinear() -> dm.Model:
    m = dm.Model("iml_1497")
    a = m.integer("a", lb=0, ub=3)
    b = m.integer("b", lb=0, ub=3)
    c = m.integer("c", lb=0, ub=3)
    m.subject_to(a + b + c <= 3)
    m.minimize(-(a * b * c))
    return m


def _counting_raiser(exc_factory, calls):
    def _raise(*_a, **_k):
        calls.append(1)
        raise exc_factory()

    return _raise


# ── binary.multilinear ─────────────────────────────────────────────────────


def test_binary_multilinear_defect_fails_the_solve(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        bml, "_reformulate", _counting_raiser(lambda: _Boom("injected bml defect"), calls)
    )
    with pytest.raises(_Boom, match="injected bml defect"):
        _binary_cubic().solve(time_limit=30)
    assert calls, "the binary-multilinear pass was never reached; the test proves nothing"


def test_binary_multilinear_unsupported_still_skips(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        bml, "_reformulate", _counting_raiser(lambda: bml._Unsupported("injected"), calls)
    )
    res = _binary_cubic().solve(time_limit=30)
    assert calls, "the binary-multilinear pass was never reached; the test proves nothing"
    assert res.status == "optimal"
    # x=y=z=1 -> -1 + 0.5 + 0.25; the only other candidates are >= 0.
    assert res.objective == pytest.approx(-0.25, abs=1e-6)


# ── integer.bilinear ───────────────────────────────────────────────────────


def test_integer_bilinear_defect_fails_the_solve(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        ipr, "_rewrite", _counting_raiser(lambda: _Boom("injected ipx defect"), calls)
    )
    with pytest.raises(_Boom, match="injected ipx defect"):
        _integer_bilinear().solve(time_limit=30)
    assert calls, "the integer-bilinear pass was never reached; the test proves nothing"


def test_integer_bilinear_not_applicable_still_skips(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        ipr,
        "_rewrite",
        _counting_raiser(lambda: ipr.IntegerProductNotApplicable("injected"), calls),
    )
    res = _integer_bilinear().solve(time_limit=30)
    assert calls, "the integer-bilinear pass was never reached; the test proves nothing"
    assert res.status == "optimal"
    # -(n*y) + 2n + y over n in {0..5}, y in [0,3], n + y <= 6: n=3, y=3 -> -9+6+3=0;
    # n=0,y=0 -> 0; n=5,y=1 -> -5+10+1=6; n=4,y=2 -> -8+8+2=2. Optimum 0.
    assert res.objective == pytest.approx(0.0, abs=1e-6)


# ── integer.multilinear (flag-gated, DISCOPT_INTEGER_MULTILINEAR_REFORM) ────


def test_integer_multilinear_defect_fails_the_solve(monkeypatch):
    monkeypatch.setenv("DISCOPT_INTEGER_MULTILINEAR_REFORM", "1")
    calls: list[int] = []
    monkeypatch.setattr(
        ipr, "_rewrite", _counting_raiser(lambda: _Boom("injected iml defect"), calls)
    )
    with pytest.raises(_Boom, match="injected iml defect"):
        _integer_trilinear().solve(time_limit=30)
    assert calls, "the integer-multilinear pass was never reached; the test proves nothing"


def test_integer_multilinear_not_applicable_still_skips(monkeypatch):
    monkeypatch.setenv("DISCOPT_INTEGER_MULTILINEAR_REFORM", "1")
    calls: list[int] = []
    monkeypatch.setattr(
        ipr,
        "_rewrite",
        _counting_raiser(lambda: ipr.IntegerProductNotApplicable("injected"), calls),
    )
    res = _integer_trilinear().solve(time_limit=30)
    assert calls, "the integer-multilinear pass was never reached; the test proves nothing"
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-1.0, abs=1e-6)


# ── disjunctive config bound (#732, flag-gated) inside the multilinear block ──


def _config_reform(calls, indicators):
    """The real multilinear reform with its configuration-indicator set pinned, so
    the test controls which adoption branch the solver takes: a non-empty set is
    adopted only at ``time_limit >= 180`` (where the config bound engages), an
    empty one is adopted at any budget (where a short budget hits ``_DcbSkip``)."""
    real = ipr.reformulate_integer_multilinear

    def _wrapped(model, **kw):
        out = real(model, **kw)
        if out is not model:
            calls.append(1)
            setattr(out, "_ipx_config_indicators", frozenset(indicators))
        return out

    return _wrapped


def _trilinear_with_continuous_residual() -> dm.Model:
    # A residual continuous nonlinearity keeps the reform off the pure-MILP route,
    # which is the branch the disjunctive config bound lives on.
    m = dm.Model("dcb_1497")
    a = m.integer("a", lb=0, ub=3)
    b = m.integer("b", lb=0, ub=3)
    t = m.continuous("t", lb=0.5, ub=2)
    m.subject_to(a + b <= 4)
    m.minimize(-(a * b * t) + dm.exp(t))
    return m


def test_disjunctive_config_bound_defect_fails_the_solve(monkeypatch):
    monkeypatch.setenv("DISCOPT_INTEGER_MULTILINEAR_REFORM", "1")
    monkeypatch.setenv("DISCOPT_DISJUNCTIVE_CONFIG_BOUND", "1")
    reform_calls: list[int] = []
    monkeypatch.setattr(ipr, "reformulate_integer_multilinear", _config_reform(reform_calls, {0}))
    calls: list[int] = []
    monkeypatch.setattr(
        dcb,
        "compute_disjunctive_config_bound",
        _counting_raiser(lambda: _Boom("injected dcb defect"), calls),
    )
    # time_limit >= 180 s: the adoption and engagement gates both open. The
    # injected failure fires at the root, long before any budget is spent.
    with pytest.raises(_Boom, match="injected dcb defect"):
        _trilinear_with_continuous_residual().solve(time_limit=200)
    assert reform_calls and calls, "the config-bound pass was never reached"


def test_disjunctive_config_bound_engagement_skip_is_still_silent(monkeypatch):
    """The ``_DcbSkip`` engagement gate is a decision, not an error: under a short
    budget the pass is declined and the solve proceeds exactly as before."""
    monkeypatch.setenv("DISCOPT_INTEGER_MULTILINEAR_REFORM", "1")
    monkeypatch.setenv("DISCOPT_DISJUNCTIVE_CONFIG_BOUND", "1")
    reform_calls: list[int] = []
    monkeypatch.setattr(ipr, "reformulate_integer_multilinear", _config_reform(reform_calls, ()))
    calls: list[int] = []
    monkeypatch.setattr(
        dcb,
        "compute_disjunctive_config_bound",
        _counting_raiser(lambda: _Boom("must not be called"), calls),
    )
    res = _trilinear_with_continuous_residual().solve(time_limit=30)
    assert reform_calls, "the multilinear reform never fired; the skip was not exercised"
    assert not calls
    assert res.objective is not None


# ── the pre-B&B structure passes in the same block ─────────────────────────


def _factorable_nonconvex() -> dm.Model:
    # ``x*x*y`` is a mixed repeated-factor product (factorable work), the model is
    # nonconvex, and it has bounded continuous variables for the periodic/domain
    # rules and the dependency scan to walk.
    m = dm.Model("pre_1497")
    x = m.continuous("x", lb=1, ub=3)
    y = m.continuous("y", lb=1, ub=3)
    m.minimize(x * x * y - 4 * x * y)
    m.subject_to(x + y <= 5)
    return m


def _is_periodic_site(_a, k):
    # The periodic/domain reduction is the only caller that passes ``rules=``;
    # the declared-box tightening at dispatch calls the same function first.
    return "rules" in k


def _always(_a, _k):
    return True


@pytest.mark.parametrize(
    "module, attr, is_site",
    [
        (
            "discopt._relax.nonlinear_bound_tightening",
            "tighten_nonlinear_bounds",
            _is_periodic_site,
        ),
        ("discopt._relax.dependent_vars", "find_functionally_dependent_names", _always),
        ("discopt.solvers._root_presolve", "tighten_root_bounds_with_fbbt", _always),
    ],
)
def test_structure_pass_defect_fails_the_solve(monkeypatch, module, attr, is_site):
    import importlib

    mod = importlib.import_module(module)
    real = getattr(mod, attr)
    calls: list[int] = []

    def _site_call_raises(*a, **k):
        # Only the FIRST call at the pre-B&B site under test is broken; every other
        # caller (root FBBT also runs, unguarded, at tree setup) gets the real
        # function. Otherwise another site's raise -- or another site's own
        # catch-all -- would decide the outcome, and the test could pass while the
        # site under test still swallowed its failure.
        if is_site(a, k):
            calls.append(1)
            if len(calls) == 1:
                raise _Boom(f"injected {attr} defect")
        return real(*a, **k)

    monkeypatch.setattr(mod, attr, _site_call_raises)
    with pytest.raises(_Boom, match=f"injected {attr} defect"):
        _factorable_nonconvex().solve(time_limit=30)
    assert calls, f"{attr} was never reached; the test proves nothing"


def test_structure_passes_unpatched_still_solve():
    res = _factorable_nonconvex().solve(time_limit=30)
    assert res.status == "optimal"
    assert res.objective is not None
