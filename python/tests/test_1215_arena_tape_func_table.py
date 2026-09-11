"""The arena tape's MathFunc table must name methods POUNCE actually has (#1215).

``_arena_tape`` lowers a unary ``FunctionCall`` by calling
``getattr(nl_expr, _FUNC_METHOD[idx])()``. A name in that table that is not a
real ``pounce.NlExpr`` method does not degrade to a fallback -- it raises
``AttributeError`` in the middle of a solve, for any model that happens to use
that operator and no other.

That is exactly what shipped: ``MathFunc::Log2`` was mapped to ``"log2"``,
which ``NlExpr`` does not define (it has ``log`` and ``log10``). Every model
using ``log2`` crashed on the default path. The failure is a *class* -- one
wrong string in a lookup table -- so the first test here checks the whole table
against the live class rather than checking ``log2`` alone, and the second
checks that the two lowering paths agree on the operator that was wrong.
"""

import math

import discopt.modeling as dm
import pytest
from discopt import Model
from discopt._arena_tape import _FUNC_LOG2, _FUNC_METHOD

pounce = pytest.importorskip("pounce")


@pytest.mark.unit
def test_every_mapped_method_exists_on_nl_expr():
    """Each entry in the table must be callable on a real ``NlExpr``."""
    x = pounce.NlExpr.var(0)
    checked = 0
    missing = []
    for idx, method in sorted(_FUNC_METHOD.items()):
        fn = getattr(x, method, None)
        if fn is None:
            missing.append((idx, method))
            continue
        # Present is not enough -- it must also be a no-argument unary call,
        # which is how the scan invokes it.
        fn()
        checked += 1
    assert not missing, f"_FUNC_METHOD names no such NlExpr method: {missing}"
    # The probe must actually have run: an empty table would pass vacuously.
    assert checked == len(_FUNC_METHOD) >= 10, f"only {checked} methods checked"


@pytest.mark.unit
def test_log2_is_not_mapped_to_a_method():
    """``log2`` has no ``NlExpr`` method and must be lowered in the scan."""
    assert _FUNC_LOG2 not in _FUNC_METHOD
    assert not hasattr(pounce.NlExpr.var(0), "log2")


def _log2_model():
    m = Model()
    x = m.continuous("x", lb=1.0, ub=8.0)
    m.subject_to(dm.log2(x) >= 1.0)
    m.minimize(dm.log2(x) + x)
    return m


@pytest.mark.unit
def test_log2_solves_on_the_arena_path(monkeypatch):
    """The bug: this raised ``AttributeError: ... has no attribute 'log2'``."""
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", "1")
    r = _log2_model().solve()
    assert str(r.status) == "optimal", r.status
    # log2(x) >= 1 forces x >= 2, and the objective rises with x, so x* = 2.
    assert r.objective == pytest.approx(1.0 + 2.0, abs=1e-5)


@pytest.mark.unit
def test_log2_agrees_with_the_legacy_path(monkeypatch):
    """Bound-neutrality: the arena lowering must not move the certificate."""
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", "0")
    legacy = _log2_model().solve()
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", "1")
    arena = _log2_model().solve()

    assert str(legacy.status) == str(arena.status) == "optimal"
    assert arena.objective == pytest.approx(legacy.objective, abs=1e-6)
    assert arena.node_count == legacy.node_count, (
        f"arena lowering changed the tree: {arena.node_count} vs {legacy.node_count}"
    )


@pytest.mark.unit
def test_the_scan_lowers_log2_as_log_over_ln2():
    """The scan's form must match ``_nl_expr_compiler``'s, not merely be close."""
    E = pounce.NlExpr
    x = E.var(0)
    scan_form = E.log(x) * E.const_(1.0 / math.log(2.0))
    for probe in (1.0, 2.0, 8.0, 1234.5):
        assert scan_form.eval([probe]) == pytest.approx(math.log2(probe), rel=1e-12)
