"""The nonlinear-term classifier must not RecursionError on a deep body.

`classify_nonlinear_terms`'s Python fallback (`distribute_products` ->
`_classify_node`) recurses once per expression node, so a model whose objective
or a constraint is one deep chain blew the default 1000-frame limit. The caller
in `solve_model` catches `Exception` and logs "reformulation skipped", so the
blowup was read as "no reformulation available" and the model silently lost its
whole term catalog -- measured on 14 corpus instances including the graphpart,
unitcommit and acopf_*_qcqp families.

The fix runs the walk with size-scaled recursion headroom on a large stack,
reusing the runner proven for the convexity (#266) and factorable (#271) walks.
"""

from __future__ import annotations

import sys

import discopt.modeling as dm
import pytest
from discopt._relax.term_classifier import (
    _classify_recursion_headroom,
    classify_nonlinear_terms,
)


def _n_terms() -> int:
    """Chain length, sized off the *ambient* recursion limit rather than a
    constant: this repo's pytest config raises the limit to 3000, so a fixed
    1500-term chain silently stopped being deep enough to reproduce anything.
    The walk enters ~2 frames per node, so twice the limit is safely past it."""
    return 2 * sys.getrecursionlimit()


def _deep_bilinear_model(n_terms: int | None = None) -> dm.Model:
    """A model whose objective is one left-deep `+` chain of `x_i * x_{i+1}`."""
    n_terms = _n_terms() if n_terms is None else n_terms
    m = dm.Model()
    x = m.continuous("x", shape=(n_terms + 1,), lb=-1.0, ub=1.0)
    expr = x[0] * x[1]
    for i in range(1, n_terms):
        expr = expr + x[i] * x[i + 1]
    m.minimize(expr)
    return m


def test_shallow_model_takes_the_inline_path():
    """Control: a small model must not pay for the deep-stack path."""
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=-1.0, ub=1.0)
    m.minimize(x[0] * x[1] + x[2] * x[2])
    assert _classify_recursion_headroom(m) == 0


def test_deep_model_requests_headroom_beyond_the_default_limit():
    """Guard the guard: if the estimate stayed under the limit the next test
    would pass without ever engaging the fix."""
    m = _deep_bilinear_model()
    assert _classify_recursion_headroom(m) > sys.getrecursionlimit()


def test_deep_body_classifies_instead_of_raising():
    """Before the fix this raised RecursionError (and was swallowed upstream)."""
    m = _deep_bilinear_model()
    terms = classify_nonlinear_terms(m)
    assert terms.bilinear, "deep chain of x_i*x_{i+1} produced no bilinear terms"
    assert len(terms.bilinear) == _n_terms()


def test_deep_body_raises_without_the_headroom_runner():
    """Pin the failure the fix removes: the same walk at the default limit still
    blows up, so this test fails loudly if the deep path is ever bypassed."""
    from discopt._relax.term_classifier import _classify_nonlinear_terms_python

    m = _deep_bilinear_model()
    with pytest.raises(RecursionError):
        _classify_nonlinear_terms_python(m)
