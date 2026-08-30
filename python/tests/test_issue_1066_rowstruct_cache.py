"""#1066: the nonlinear row scan re-derived its bound-independent half per node.

A rule's row scan splits in two: deriving the row's algebraic shape, then
intersecting that shape against the current box. Only the second half reads the
box, yet both halves ran at every tightening pass. Measured on
``portfol_classical050_1`` after the linear-FBBT fix of the same issue, the first
half was 15.9 s of a 46.6 s solve -- 10.5M ``_constant_value`` calls and 55.6M
``isinstance`` calls, redone identically at every one of 160 passes.

These tests pin the two properties that make caching that half legitimate: the
cached path returns **bit-identical** bounds to the uncached scan (CLAUDE.md 5
bound-neutrality), and the cache is genuinely exercised rather than silently
bypassed -- otherwise the differential test would pass by comparing the uncached
scan against itself (CLAUDE.md 6).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import nonlinear_bound_tightening as nbt
from discopt._relax.model_utils import flat_variable_bounds
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit


def _model(name: str = "sepquad") -> Model:
    """Rows the separable-quadratic rule accepts, plus one it must reject."""
    m = Model(name)
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-4.0, ub=6.0)
    z = m.continuous("z", lb=-3.0, ub=3.0)
    m.subject_to(x**2 + 2.0 * y**2 + 3.0 * z <= 9.0)
    m.subject_to(2.0 * x + y**2 <= 7.0)
    m.subject_to(-x + 0.5 * z**2 + 1.5 <= 4.0)
    m.subject_to(dm.exp(x) + y <= 20.0)  # not separable quadratic
    m.minimize(x + y + z)
    return m


def _tighten(model: Model):
    lb, ub = flat_variable_bounds(model)
    return nbt.tighten_nonlinear_bounds(model, lb, ub)


def test_cached_rows_give_bit_identical_bounds_to_the_uncached_scan(monkeypatch):
    """The cache is an optimisation of the scan, not a different scan."""
    cached_lb, cached_ub, cached_stats = _tighten(_model("cached"))

    bypassed = {"n": 0}

    def no_cache(model, rule_name, constraint, metadata, build):
        bypassed["n"] += 1
        return build()

    monkeypatch.setattr(nbt, "_cached_row_structure", no_cache)
    plain_lb, plain_ub, plain_stats = _tighten(_model("plain"))

    # CLAUDE.md 6: a differential that never ran the control arm proves nothing.
    assert bypassed["n"] > 0, "control arm never reached the cache seam"
    assert "separable_quadratic_upper_bound" in cached_stats.applied_rules
    np.testing.assert_array_equal(cached_lb, plain_lb)
    np.testing.assert_array_equal(cached_ub, plain_ub)
    assert cached_stats.applied_rules == plain_stats.applied_rules


def test_the_cache_is_populated_and_reused_across_passes():
    """Guards the differential above from passing because the cache is dead."""
    m = _model()
    _tighten(m)

    cache = getattr(m, "_nl_struct_cache", None)
    assert cache is not None, "structural cache was never attached to the model"
    assert cache["rowstruct"], "no row decomposition was cached -- the fix is a no-op"
    n_first = len(cache["rowstruct"])

    # A second pass over the same model must reuse the entries, not re-add them.
    lb, ub = flat_variable_bounds(m)
    nbt.tighten_nonlinear_bounds(m, lb * 0.5, ub * 0.5)
    assert len(cache["rowstruct"]) == n_first, "second pass re-derived cached rows"


def test_a_row_the_rule_rejects_is_cached_as_a_rejection():
    """A non-match is as bound-independent as a match, and on a real instance
    most rows are non-matches -- caching only the matches would leave the cost."""
    m = _model()
    _tighten(m)
    values = m._nl_struct_cache["rowstruct"].values()
    assert any(v is None for v in values), "a rejected row was not cached as a rejection"
    assert any(v is not None for v in values), "no row was accepted; the model is wrong"


def test_a_caller_with_its_own_metadata_bypasses_the_cache():
    """The decomposition depends on ``metadata``, which arrives as a parameter
    and is therefore not covered by the model-structural cache token."""
    m = _model()
    _tighten(m)
    cache = m._nl_struct_cache
    n_before = len(cache["rowstruct"])

    foreign = nbt.build_flat_variable_metadata(m)
    assert foreign is not cache["metadata"]

    built = []
    rule = nbt.SeparableQuadraticUpperBoundRule()
    out = nbt._cached_row_structure(
        m, rule.name, m._constraints[0], foreign, lambda: built.append(1) or "rebuilt"
    )
    assert built == [1], "foreign metadata must rebuild rather than read the cache"
    assert out == "rebuilt"
    assert len(cache["rowstruct"]) == n_before, "a bypassed call must not write the cache"
