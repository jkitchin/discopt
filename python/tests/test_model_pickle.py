"""``pickle`` of a :class:`Model` in the states real models reach.

Before ``Model.__getstate__`` existed, ``copy.deepcopy`` had been taught (#1479) to
handle a solved model (a POUNCE module in the evaluator caches), a ``from_nl``
model (the Rust ``PyModelRepr`` in ``_nl_repr``) and a fast-API model (the Rust
``PyModelBuilder``) -- but ``pickle`` went through none of that, so
``pickle.dumps`` still raised ``TypeError: cannot pickle 'module' object`` /
``cannot pickle 'builtins.PyModelRepr' object`` on the first two. The first three
tests below fail on the pre-fix tree for exactly those reasons.

A pickle round trip is bound-neutral: the unpickled model must solve to the same
objective in the same number of nodes, not merely to a close one.
"""

from __future__ import annotations

import copy
import pathlib
import pickle

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt.modeling.core import _UnpickledNlRepr, from_nl

CORPUS = sorted((pathlib.Path(__file__).parent / "data" / "minlplib_nl").glob("*.nl"))


def _roundtrip(m):
    return pickle.loads(pickle.dumps(m))


def _minlp():
    m = dm.Model("solved")
    x = m.continuous("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=3)
    m.minimize((x - 1.3) ** 2 + y)
    m.subject_to(x + y >= 2)
    return m


def _fast_api_model():
    m = dm.Model("fast")
    x = m.continuous("x", shape=(2,), lb=0, ub=4)
    m.add_linear_objective(np.array([1.0, 1.0]), x)
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([3.0]))
    return m


def test_pickle_of_a_solved_model_is_bound_neutral():
    m = _minlp()
    ref = m.solve(time_limit=30)
    assert any(k.endswith("_cache") and k != "_flat_var_offsets_cache" for k in vars(m))

    c = _roundtrip(m)
    assert not [k for k in vars(c) if k.endswith("_cache") and k != "_flat_var_offsets_cache"]
    assert c._flat_var_offsets_cache is None
    assert dt.model_fingerprint(c) == dt.model_fingerprint(m)
    assert all(v.model is c for v in c._variables)
    # The last solve result travels, as it does through `clone()`.
    assert c._last_solve_result is not None
    assert c._last_solve_result.objective == ref.objective

    got = c.solve(time_limit=30)
    assert got.status == ref.status
    assert got.objective == ref.objective
    assert got.node_count == ref.node_count


def test_pickle_of_a_from_nl_model():
    assert CORPUS, "in-repo .nl corpus missing"
    checked = 0
    for path in CORPUS:
        m = from_nl(str(path))
        c = _roundtrip(m)
        assert dt.model_fingerprint(c) == dt.model_fingerprint(m), path.name
        assert all(v.model is c for v in c._variables), path.name
        assert getattr(c, "_source_nl_path", None) == getattr(m, "_source_nl_path", None)
        # The parse does not survive, but its presence still keeps the
        # referenced-variables guard conservative, exactly as on the original.
        assert isinstance(c._nl_repr, _UnpickledNlRepr), path.name
        assert m._referenced_variable_names() is None
        assert c._referenced_variable_names() is None, path.name
        checked += 1
    assert checked == len(CORPUS)


def test_pickle_of_a_fast_api_model_replays_the_builder():
    m = _fast_api_model()
    ref = m.solve(time_limit=10)
    c = _roundtrip(m)
    assert c._builder is not None and c._builder is not m._builder
    assert c.num_constraints == m.num_constraints == 1
    assert dt.model_fingerprint(c) == dt.model_fingerprint(m)
    assert c.solve(time_limit=10).objective == pytest.approx(ref.objective, abs=1e-9)


def test_pickle_of_a_solved_from_nl_model():
    path = CORPUS[0]
    m = from_nl(str(path))
    m.solve(time_limit=10)
    c = _roundtrip(m)
    assert dt.model_fingerprint(c) == dt.model_fingerprint(m)


def test_unpickled_nl_repr_refuses_use_loudly():
    c = _roundtrip(from_nl(str(CORPUS[0])))
    with pytest.raises(AttributeError, match="does not survive pickling"):
        c._nl_repr.n_complementarities  # noqa: B018


def test_unpickled_bounds_stay_read_only():
    c = _roundtrip(_minlp())
    assert not c._variables[0].lb.flags.writeable


def test_copy_copy_is_still_a_shallow_copy():
    # `relax_objective_defining_equality` relies on `copy.copy(model)` sharing
    # everything it does not reassign; `__getstate__` must not change that.
    m = _fast_api_model()
    m.solve(time_limit=10)
    c = copy.copy(m)
    assert c is not m
    assert vars(c).keys() == vars(m).keys()
    assert all(vars(c)[k] is vars(m)[k] for k in vars(m))
