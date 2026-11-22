"""#2293: per-row memory of the structural keys the tape evaluator path keeps alive.

A trivial linear row cost ~2.3 KiB of live memory once its tape evaluator was
built. The pounce NlProblem and the cached Rust repr are most of that. About
460 B/row was Python bookkeeping in this repo:

* ``evaluator_fingerprint`` read ``c.__dict__`` to avoid forcing a lazy body.
  On CPython 3.12+ that read allocates a dict for an object whose attributes
  are stored inline, and the dict stays for the object's lifetime (72 B/row);
* the evaluator fingerprint and the read-only repr key held four ``id()``
  columns as tuples of boxed ints (160 B/row);
* the tape evaluator held its variable bounds as lists of Python floats.

These tests pin the cheap forms, and check that each one still answers the way
the old one did.
"""

from __future__ import annotations

import gc
import tracemalloc
import weakref

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import _repr_cache as rc
from discopt import _structure_memo as sm
from discopt._evaluator_cache import built_body, evaluator_fingerprint, id_column
from discopt.modeling.core import Constraint

N_ROWS = 4000


def _chain_model(n: int = N_ROWS):
    m = dm.Model("m2293")
    x = m.continuous("x", shape=(n + 1,), lb=-1, ub=2)
    for k in range(n):
        m.subject_to(x[k + 1] - x[k] - 0.01 * x[(k + 2) % (n + 1)] == 0)
    m.minimize(dm.sum([(x[i] - 0.5) ** 2 for i in range(3)]))
    return m, x


def _live_bytes_per_row(fn, n_rows: int) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        kept = fn()
        gc.collect()
        live, _peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del kept
    return live / n_rows


def test_constraint_has_no_class_level_body():
    # `built_body` relies on this: with no class attribute, `object.__getattribute__`
    # can only find the instance's own `body`.
    assert "body" not in vars(Constraint)
    assert not hasattr(Constraint, "body")


def test_built_body_matches_dict_read_and_never_forces_a_lazy_row():
    m, x = _chain_model(5)
    checked = 0
    for c in m._constraints:
        assert built_body(c) is c.__dict__.get("body")
        checked += 1
    assert checked == 5

    m2 = dm.Model("lazy")
    y = m2.continuous("y", shape=(3,), lb=0, ub=1)
    m2.add_linear_constraints(np.eye(3), y, "<=", np.ones(3))
    lazy = m2._builder_linear_constraints(lazy=True)
    assert len(lazy) == 3
    for c in lazy:
        assert built_body(c) is None
        assert "body" not in c.__dict__  # still unbuilt: built_body did not force it
        body = c.body  # forces it
        assert built_body(c) is body


def test_id_column_equality_matches_tuple_of_ids():
    a, b, c = object(), object(), object()
    assert id_column([a, b]) == id_column((a, b))
    assert id_column([a, b]) != id_column([b, a])
    assert id_column([a, b]) != id_column([a, b, c])
    assert id_column([]) == b""
    assert np.frombuffer(id_column([a, b, c]), dtype=np.uintp).tolist() == [id(a), id(b), id(c)]


def test_evaluator_fingerprint_live_memory_per_row():
    m, _ = _chain_model()
    per_row = _live_bytes_per_row(lambda: evaluator_fingerprint(m), N_ROWS)
    # Measured on this model: 160 B/row before #2293 (a 72 B dict per row from
    # the `__dict__` read, plus two columns of boxed ids). Packed it is 2 x 8 bytes
    # plus the 8-byte slot of the pinned `bodies` tuple.
    assert per_row < 48, per_row


def test_evaluator_fingerprint_still_sees_every_change():
    m, x = _chain_model(10)
    a = evaluator_fingerprint(m)
    assert evaluator_fingerprint(m) == a
    assert hash(evaluator_fingerprint(m)) == hash(a)

    m._constraints[3].body = m._constraints[3].body + 0.0  # replaced in place (#2099)
    b = evaluator_fingerprint(m)
    assert b != a

    m.subject_to(x[0] <= 1)
    assert evaluator_fingerprint(m) != b


def test_repr_cache_key_live_memory_per_row():
    m, _ = _chain_model()
    per_row = _live_bytes_per_row(lambda: rc._state_key(m), N_ROWS)
    # Measured on this model: 136 B/row before #2293, 72 after. The 72 is the two
    # packed id columns (16), the sense/rhs/name/bodies/constraints tuple slots
    # (40) and the 16 B/variable of bound bytes. 100 separates the two.
    assert per_row < 100, per_row


def test_repr_cache_still_misses_on_a_replaced_body():
    m, _ = _chain_model(10)
    r1 = rc.readonly_model_repr(m)
    assert rc.readonly_model_repr(m) is r1
    m._constraints[2].body = m._constraints[2].body + 0.0
    assert rc.readonly_model_repr(m) is not r1


def test_structure_memo_pins_the_bodies_it_keys_on():
    # The key holds `id(c.body)`; without a pin, a replaced body is freed and a
    # new expression can take its id while the entry lives (#1329).
    m, x = _chain_model(10)
    sm.structure_memo(m, "probe", lambda _m: 1)
    old = m._constraints[4].body
    ref = weakref.ref(old)
    # An unrelated replacement: `old + 0.0` would keep `old` alive as its child.
    m._constraints[4].body = x[0] - 1.0
    del old
    gc.collect()
    assert ref() is not None, "structure memo no longer pins the body id it keys on"

    calls = []
    sm.structure_memo(m, "probe", lambda _m: calls.append(1) or 2)
    assert calls == [1], "a replaced body must miss the memo"


@pytest.mark.requires_pounce
def test_tape_bounds_are_float64_arrays_and_reach_the_problem_unchanged():
    pytest.importorskip("pounce")
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    m = dm.Model("b")
    x = m.continuous("x", shape=(3,), lb=[-np.inf, -1.0, 0.0], ub=[np.inf, 2.0, 0.0])
    y = m.integer("y", lb=-3, ub=5)
    m.subject_to(x[0] * x[1] + y <= 4)
    m.minimize(dm.exp(x[2]) + x[0] ** 2 + y)
    ev = TapeNLPEvaluator(m)
    for arr in (ev._lo, ev._hi, ev._tape_lo, ev._tape_hi):
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64
    lb, ub = ev.variable_bounds
    np.testing.assert_array_equal(ev._lo, lb.astype(np.float64))
    np.testing.assert_array_equal(ev._hi, ub.astype(np.float64))

    import pounce

    # The problem pounce built from the arrays has the bounds a list would give.
    E = pounce.NlExpr
    ref = pounce.build_nl_problem(
        4, E.const_(0.0), constraints=None, x_l=[float(v) for v in lb], x_u=[float(v) for v in ub]
    )
    np.testing.assert_array_equal(np.asarray(ev._problem.x_l), np.asarray(ref.x_l))
    np.testing.assert_array_equal(np.asarray(ev._problem.x_u), np.asarray(ref.x_u))
