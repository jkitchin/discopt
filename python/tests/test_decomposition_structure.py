"""Tests for the shared decomposition structure layer.

Covers the annotation API on ``Model`` and ``detect_decomposition`` /
``restricted_bounds`` in ``discopt.decomposition.structure``.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition import (
    DecompositionStructure,
    detect_decomposition,
    restricted_bounds,
)
from discopt.decomposition.structure import flat_bounds

# ── Annotation API ────────────────────────────────────────────


def test_stage_annotations_recorded():
    m = dm.Model("stages")
    x = m.continuous("x", lb=0, ub=1)
    y = m.binary("y")
    m.first_stage(y)
    m.second_stage(x)
    assert m._decomp_stages == {"y": 1, "x": 2}


def test_indexed_stage_annotation_resolves_to_whole_variable():
    """Annotating an indexed element (``y[i]``) must resolve to the base variable.

    Regression: ``str(y[0])`` is ``"y[3][0]"``, which silently never matches the
    variable name ``"y"`` — so the annotated variable used to fall into the
    recourse subproblem and trip the "integer in recourse" guard.
    """
    m = dm.Model("idx")
    y = m.binary("y", shape=(3,))
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y[0])  # indexed element -> whole variable "y"
    m.second_stage(x[1])
    assert m._decomp_stages == {"y": 1, "x": 2}
    s = detect_decomposition(m)
    assert "y" in set(s.complicating_vars)


def test_set_stage_and_block_chainable():
    m = dm.Model("chain")
    x = m.continuous("x", lb=0, ub=1)
    out = m.set_stage(x, 3).set_block(x, 7)
    assert out is m
    assert m._decomp_stages["x"] == 3
    assert m._decomp_blocks["x"] == 7


def test_mark_coupling_by_object_and_name():
    m = dm.Model("coupling")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    c = x + y <= 1
    m.subject_to(c, name="link")
    m.mark_coupling(c)
    # id(c) and its name are both recorded.
    assert id(c) in m._coupling_keys
    assert "link" in m._coupling_keys

    m2 = dm.Model("coupling2")
    a = m2.continuous("a", lb=0, ub=1)
    b = m2.continuous("b", lb=0, ub=1)
    m2.subject_to(a + b <= 1, name="link2")
    m2.mark_coupling("link2")
    assert "link2" in m2._coupling_keys


# ── Auto-detection ────────────────────────────────────────────


def test_separable_two_block_model():
    """Two independent blocks with no shared constraint → 2 blocks."""
    m = dm.Model("two_block")
    x1 = m.continuous("x1", lb=0, ub=10)
    x2 = m.continuous("x2", lb=0, ub=10)
    y1 = m.continuous("y1", lb=0, ub=10)
    y2 = m.continuous("y2", lb=0, ub=10)
    m.subject_to(x1 + x2 <= 5)
    m.subject_to(y1 + y2 <= 5)
    m.minimize(x1 + x2 + y1 + y2)

    s = detect_decomposition(m)
    assert isinstance(s, DecompositionStructure)
    assert s.is_separable
    assert s.num_blocks == 2
    # x-vars and y-vars land in different blocks.
    assert s.block_of_var["x1"] == s.block_of_var["x2"]
    assert s.block_of_var["y1"] == s.block_of_var["y2"]
    assert s.block_of_var["x1"] != s.block_of_var["y1"]
    assert s.coupling_constraints == []


def test_bridge_coupling_detected():
    """Two blocks joined by a single linking constraint → detected as coupling."""
    m = dm.Model("bridge")
    x1 = m.continuous("x1", lb=0, ub=10)
    x2 = m.continuous("x2", lb=0, ub=10)
    y1 = m.continuous("y1", lb=0, ub=10)
    y2 = m.continuous("y2", lb=0, ub=10)
    m.subject_to(x1 + x2 <= 5)  # block A
    m.subject_to(y1 + y2 <= 5)  # block B
    m.subject_to(x1 + y1 <= 4)  # the bridge
    m.minimize(x1 + x2 + y1 + y2)

    s = detect_decomposition(m)
    # Removing the bridge separates A and B.
    assert s.coupling_constraints == [2]
    assert s.is_separable
    assert s.num_blocks == 2
    assert s.source == "detected"


def test_complicating_defaults_to_integers():
    m = dm.Model("milp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.binary("y")
    z = m.integer("z", lb=0, ub=5)
    m.subject_to(x + y + z <= 8)
    m.minimize(x + y + z)

    s = detect_decomposition(m)
    assert set(s.complicating_vars) == {"y", "z"}


def test_explicit_annotation_overrides_default():
    m = dm.Model("override")
    x = m.continuous("x", lb=0, ub=10)
    y = m.binary("y")
    m.first_stage(x)  # override: continuous x is complicating, not y
    m.subject_to(x + y <= 5)
    m.minimize(x + y)

    s = detect_decomposition(m)
    assert s.complicating_vars == ["x"]
    assert s.source in ("annotated", "mixed")


def test_explicit_coupling_argument_used():
    m = dm.Model("explicit_coupling")
    x1 = m.continuous("x1", lb=0, ub=10)
    y1 = m.continuous("y1", lb=0, ub=10)
    m.subject_to(x1 <= 5)
    m.subject_to(y1 <= 5)
    m.subject_to(x1 + y1 <= 4)
    m.minimize(x1 + y1)

    s = detect_decomposition(m, coupling=[2])
    assert s.coupling_constraints == [2]
    assert s.num_blocks == 2


# ── Bound helpers ─────────────────────────────────────────────


def test_flat_bounds_declared_order():
    m = dm.Model("bounds")
    m.continuous("a", lb=-1, ub=1)
    m.continuous("b", shape=(2,), lb=0, ub=3)
    lb, ub = flat_bounds(m)
    assert lb.tolist() == [-1, 0, 0]
    assert ub.tolist() == [1, 3, 3]


def test_restricted_bounds_pins_variables():
    m = dm.Model("fix")
    m.continuous("a", lb=-5, ub=5)
    m.continuous("b", shape=(2,), lb=-5, ub=5)
    lb, ub = restricted_bounds(m, {"a": 2.0, "b": np.array([1.0, -1.0])})
    assert lb[0] == ub[0] == 2.0
    assert lb[1] == ub[1] == 1.0
    assert lb[2] == ub[2] == -1.0


def test_restricted_bounds_scalar_broadcast():
    m = dm.Model("fix2")
    m.continuous("v", shape=(3,), lb=0, ub=10)
    lb, ub = restricted_bounds(m, {"v": 4.0})
    assert lb.tolist() == [4, 4, 4]
    assert ub.tolist() == [4, 4, 4]


def test_restricted_bounds_size_mismatch_raises():
    m = dm.Model("fix3")
    m.continuous("v", shape=(3,), lb=0, ub=10)
    with pytest.raises(ValueError):
        restricted_bounds(m, {"v": np.array([1.0, 2.0])})


def test_summary_is_string():
    m = dm.Model("s")
    m.continuous("x", lb=0, ub=1)
    m.subject_to(m._variables[0] <= 1)
    m.minimize(m._variables[0])
    s = detect_decomposition(m)
    assert isinstance(s.summary(), str)
    assert "DecompositionStructure" in s.summary()


# ── Phase 4 (T4.2, S3): .dec interop and truncation flag ──────


def test_dec_roundtrip_lossless(tmp_path):
    from discopt.decomposition.graph.export import read_dec, write_dec
    from discopt.decomposition.structure import detect_decomposition

    m = dm.Model("blocks")
    x = m.binary("x", shape=(4,))
    m.minimize(2 * x[0] + 3 * x[1] + 2 * x[2] + 4 * x[3])
    m.subject_to(x[0] + x[1] >= 1)
    m.subject_to(x[2] + x[3] >= 1)
    conf = x[0] + x[2] <= 1
    m.subject_to(conf)
    m.mark_coupling(conf)

    s1 = detect_decomposition(m)
    path = str(tmp_path / "model.dec")
    write_dec(s1, m, path)
    s2 = read_dec(path, m)
    assert s1.coupling_constraints == s2.coupling_constraints
    assert s1.num_blocks == s2.num_blocks
    # dec_file short-circuit path matches read_dec.
    s3 = detect_decomposition(m, dec_file=path)
    assert s3.coupling_constraints == s1.coupling_constraints


def test_detection_truncated_defaults_false():
    from discopt.decomposition.structure import detect_decomposition

    m = dm.Model("small")
    x = m.continuous("x", lb=0, ub=1)
    m.subject_to(x <= 1)
    m.minimize(x)
    s = detect_decomposition(m)
    assert s.detection_truncated is False
