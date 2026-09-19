"""Issue #1343: the Rust term-classifier fallback must be observable.

Before this change ``_classify_nonlinear_terms_rust`` returned ``None`` from three
bare ``except Exception`` arms and three structural declines, all indistinguishable
from a Rust success. These tests pin that every route is named, counted, and — for
the arms that catch an exception — carries what was raised.

The deep-recursion test is the one that matters most structurally: classification
runs on a worker thread whose context copy is discarded, so an implementation that
published the route from inside the classification closure would record nothing on
exactly the deep models the fallback exists for.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import pytest
from discopt._relax import term_classifier as tc


@pytest.fixture(autouse=True)
def _clean_counters():
    tc.reset_classifier_route_counts()
    yield
    tc.reset_classifier_route_counts()


def _polynomial_model() -> dm.Model:
    m = dm.Model("poly")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.minimize(x[0] * x[1] + x[2] ** 3)
    return m


def _general_nl_model() -> dm.Model:
    m = dm.Model("general_nl")
    x = m.continuous("x", lb=0.1, ub=2.0)
    y = m.continuous("y", lb=0.1, ub=2.0)
    m.minimize(dm.sin(x) + x * y)
    return m


def _expandable_square_model() -> dm.Model:
    m = dm.Model("expandable_square")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.minimize((x[0] - x[1]) ** 2)
    return m


# --------------------------------------------------------------------------
# The success route is counted too — "Rust ran" must be a positive fact
# --------------------------------------------------------------------------


def test_rust_route_is_counted_when_the_fast_path_produces_the_catalog():
    terms = tc.classify_nonlinear_terms(_polynomial_model())

    assert terms.bilinear == [(0, 1)]
    assert tc.classifier_route_counts() == {tc.ROUTE_RUST: 1}


def test_counts_accumulate_and_reset():
    model = _polynomial_model()
    tc.classify_nonlinear_terms(model)
    tc.classify_nonlinear_terms(model)

    assert tc.classifier_route_counts()[tc.ROUTE_RUST] == 2

    tc.reset_classifier_route_counts()
    assert tc.classifier_route_counts() == {}


def test_returned_counts_are_a_copy_not_the_live_mapping():
    tc.classify_nonlinear_terms(_polynomial_model())

    snapshot = tc.classifier_route_counts()
    snapshot["rust"] = 999
    snapshot["injected"] = 1

    assert tc.classifier_route_counts() == {tc.ROUTE_RUST: 1}


# --------------------------------------------------------------------------
# Structural declines: named, counted, no detail
# --------------------------------------------------------------------------


def test_general_nl_decline_names_its_reason():
    tc.classify_nonlinear_terms(_general_nl_model())

    assert tc.classifier_route_counts() == {tc.ROUTE_PY_GENERAL_NL_OBJECTS: 1}
    assert tc.classifier_route_details() == {}


def test_expandable_square_decline_names_its_reason():
    tc.classify_nonlinear_terms(_expandable_square_model())

    assert tc.classifier_route_counts() == {tc.ROUTE_PY_EXPANDABLE_SQUARE: 1}


# --------------------------------------------------------------------------
# The exception arms: counted, attributed, and the exception is NOT discarded
# --------------------------------------------------------------------------


def test_a_raising_rust_classifier_is_counted_and_its_exception_recorded(monkeypatch, caplog):
    """The §7 case: a Rust failure must not look like a Rust success."""
    import discopt._rust as _rust

    def _boom(_model):
        raise RuntimeError("arena exploded")

    monkeypatch.setattr(_rust, "model_to_repr", _boom)

    model = _polynomial_model()
    with caplog.at_level(logging.WARNING, logger=tc.__name__):
        terms = tc.classify_nonlinear_terms(model)

    # The fallback still produces the right answer ...
    assert terms == tc._classify_nonlinear_terms_python(model)
    # ... but it is no longer silent.
    assert tc.classifier_route_counts() == {tc.ROUTE_PY_CLASSIFY_RAISED: 1}
    detail = tc.classifier_route_details()[tc.ROUTE_PY_CLASSIFY_RAISED]
    assert "RuntimeError" in detail and "arena exploded" in detail
    assert any("arena exploded" in r.getMessage() for r in caplog.records)


def test_a_missing_rust_extension_is_counted_but_not_warned(monkeypatch, caplog):
    """An unbuilt extension is an environment fact, not a defect: debug, not warning."""
    import builtins

    real_import = builtins.__import__

    def _no_rust(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "discopt._rust" and fromlist and "model_to_repr" in fromlist:
            raise ImportError("no _rust here")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_rust)

    model = _polynomial_model()
    with caplog.at_level(logging.WARNING, logger=tc.__name__):
        tc.classify_nonlinear_terms(model)

    assert tc.classifier_route_counts() == {tc.ROUTE_PY_IMPORT_FAILED: 1}
    assert "no _rust here" in tc.classifier_route_details()[tc.ROUTE_PY_IMPORT_FAILED]
    assert caplog.records == []


def test_a_raising_degree_cross_check_is_counted_and_recorded(monkeypatch):
    """The third bare except — the one #1343's own body first missed."""
    import discopt._rust as _rust

    real_model_to_repr = _rust.model_to_repr

    class _ReprWithBrokenDegreeCheck:
        def __init__(self, inner):
            self._inner = inner

        def classify_nonlinear_terms(self):
            # An empty catalog forces the degree cross-check to run.
            return {"general_nl_count": 0}

        def is_objective_linear(self):
            raise ValueError("degree analysis unavailable")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        _rust, "model_to_repr", lambda m: _ReprWithBrokenDegreeCheck(real_model_to_repr(m))
    )

    tc.classify_nonlinear_terms(_polynomial_model())

    assert tc.classifier_route_counts() == {tc.ROUTE_PY_DEGREE_CHECK_RAISED: 1}
    assert (
        "degree analysis unavailable"
        in (tc.classifier_route_details()[tc.ROUTE_PY_DEGREE_CHECK_RAISED])
    )


def test_an_incomplete_rust_catalog_is_counted(monkeypatch):
    """Nonlinear model, empty payload -> the blind-spot guard, now named."""
    import discopt._rust as _rust

    real_model_to_repr = _rust.model_to_repr

    class _ReprWithEmptyCatalog:
        def __init__(self, inner):
            self._inner = inner

        def classify_nonlinear_terms(self):
            return {"general_nl_count": 0}

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        _rust, "model_to_repr", lambda m: _ReprWithEmptyCatalog(real_model_to_repr(m))
    )

    tc.classify_nonlinear_terms(_polynomial_model())

    assert tc.classifier_route_counts() == {tc.ROUTE_PY_CATALOG_INCOMPLETE: 1}


# --------------------------------------------------------------------------
# The thread boundary
# --------------------------------------------------------------------------


def test_the_route_is_recorded_on_the_deep_recursion_worker_thread(monkeypatch):
    """Classification runs on a worker thread whose context copy is discarded.

    ``_run_with_deep_recursion``'s own docstring: "writes made by ``fn`` still
    cannot leak back here." An implementation that published the route from inside
    the classification closure would count nothing here while passing every test
    above — an instrument that silently measures nothing on exactly the deep models
    it exists for (CLAUDE.md §6).
    """
    import sys

    deep = sys.getrecursionlimit() + 10_000
    monkeypatch.setattr(tc, "_classify_recursion_headroom", lambda _model: deep)

    model = _general_nl_model()
    terms = tc.classify_nonlinear_terms(model)

    assert terms.general_nl, "the deep path must still classify"
    assert tc.classifier_route_counts() == {tc.ROUTE_PY_GENERAL_NL_OBJECTS: 1}
