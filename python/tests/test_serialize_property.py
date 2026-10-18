"""Property-based round trip of the native format over generated models (WP-4, D8).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-4. Every hand-written round
trip test covers the cases someone thought of; these cover the cross product
``strategies_models.models()`` draws -- every operator, function name, index kind,
relation kind and a scoped fix, nested and shared. For each generated model:

* ``dumps`` is a fixed point: re-dumping the reloaded model gives the same text;
* the reloaded model has the same ``transformations.model_fingerprint``;
* ``pickle`` gives the same fingerprint (and the same document);
* ``to_nl`` gives the same text, or refuses both models the same way.

Profiles (``conftest.py``): ``ci`` (default, 50 examples) and ``slow`` (2000):
``HYPOTHESIS_PROFILE=slow pytest python/tests/test_serialize_property.py``.
"""

from __future__ import annotations

import pickle
import warnings

import strategies_models as sm
from discopt import serialize
from discopt.serialize import dumps, loads
from discopt.transformations import model_fingerprint
from hypothesis import HealthCheck, given, settings


def _loads_quietly(text: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return loads(text)


def _fingerprint(g: sm.GeneratedModel, model=None) -> tuple:
    model = g.model if model is None else model
    if g.fixed is None:
        return model_fingerprint(model)
    var, value = g.fixed
    # The reloaded variable of the same name holds the same pin, so compare the
    # original inside its scope with the reload as loaded.
    if model is g.model:
        with var.fixed(value):
            return model_fingerprint(model)
    return model_fingerprint(model)


@given(sm.models())
def test_generated_model_round_trips(g):
    text = sm.dumps_generated(g)
    reloaded = _loads_quietly(text)
    assert dumps(reloaded, provenance=False) == text
    assert model_fingerprint(reloaded) == _fingerprint(g)


@given(sm.models())
def test_generated_model_pickle_matches_dumps(g):
    text = sm.dumps_generated(g)
    reloaded = _loads_quietly(text)
    copy = pickle.loads(pickle.dumps(reloaded))
    assert model_fingerprint(copy) == model_fingerprint(reloaded)
    assert dumps(copy, provenance=False) == text


def _to_nl(model):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ("text", model.to_nl())
    except Exception as exc:  # noqa: BLE001 - the outcome is what is compared
        return ("refused", type(exc).__name__)


@given(sm.models())
def test_generated_model_exports_the_same_nl(g):
    """``.nl`` is an independent writer: equal text after a round trip means the
    reload is the same problem to a second consumer, not only to the fingerprint."""
    text = sm.dumps_generated(g)
    reloaded = _loads_quietly(text)
    if g.fixed is None:
        before = _to_nl(g.model)
    else:
        var, value = g.fixed
        with var.fixed(value):
            before = _to_nl(g.model)
    assert _to_nl(reloaded) == before


# ── the strategy must exercise the format (entry experiment, §6) ───────────

#: Measured at WP-4 entry: 200 examples, derandomized, reached 39/41 (95 %).
COVERAGE_FLOOR = 0.80


def test_strategy_covers_the_op_table():
    """The kill criterion, kept as a test: a generator that stops reaching most of
    the node ops and function names is no longer a property test of the format."""
    seen: set = set()
    count = [0]

    @settings(
        max_examples=200,
        database=None,
        derandomize=True,
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )
    @given(sm.models())
    def collect(g):
        seen.update(sm.ops_of(sm.dumps_generated(g)))
        count[0] += 1

    collect()
    assert count[0] == 200
    universe = {f"op:{o}" for o in serialize._KNOWN_OPS} | {
        f"fn:{f}" for f in serialize._known_funcs()
    }
    hit = universe & seen
    assert len(hit) / len(universe) >= COVERAGE_FLOOR, sorted(universe - hit)
    # Every node op, every index kind and every row kind, without exception.
    assert {f"op:{o}" for o in serialize._KNOWN_OPS} <= seen
    assert {"ix:int", "ix:slice", "ix:tuple", "ix:bool"} <= seen
    assert {
        "row:algebraic",
        "row:indicator",
        "row:sos",
        "row:disjunction",
        "row:logical",
    } <= seen
    assert {"rel:complementarity", "rel:builder_block", "sum:axis"} <= seen


def test_function_tables_cover_every_known_function():
    assert set(sm.UNARY_FUNCS) | set(sm.BINARY_FUNCS) == set(serialize._known_funcs())
    assert set(sm.BINARY_OPS) == serialize._KNOWN_BINARY_OPS
    assert set(sm.UNARY_OPS) == serialize._KNOWN_UNARY_OPS
