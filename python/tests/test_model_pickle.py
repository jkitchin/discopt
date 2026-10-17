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
import time

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt.modeling.core import _DetachedNlRepr, from_nl

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
        assert isinstance(c._nl_repr, _DetachedNlRepr), path.name
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
    with pytest.raises(AttributeError, match="does not survive"):
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


# ── adversarial findings: copies that differed from, or failed on, the original ──


def test_a_solved_model_with_a_relaxation_analysis_pickles():
    """The analysis cache is not named ``*_cache`` and holds local closures."""
    m = dm.Model("nlp")
    x = m.continuous("x", shape=(2,), lb=-2, ub=2)
    m.minimize(dm.exp(x[0]) + x[1] ** 2 - x[0] * x[1])
    ref = m.solve(time_limit=30)
    assert "_uniform_relax_analysis" in vars(m)
    c = _roundtrip(m)
    assert "_uniform_relax_analysis" not in vars(c)
    got = c.solve(time_limit=30)
    assert (got.status, got.objective, got.node_count) == (
        ref.status,
        ref.objective,
        ref.node_count,
    )


def test_the_solve_deadline_does_not_cross_a_pickle():
    m = dm.Model("nlp")
    x = m.continuous("x", lb=-2, ub=2)
    m.minimize((x - 1) ** 2)
    # ``Model.solve`` deletes it on the way out; a direct ``solve_model`` call
    # leaves it behind, as an absolute ``perf_counter`` reading of THIS process.
    m._solve_deadline = time.perf_counter() + 5.0
    assert "_solve_deadline" not in vars(_roundtrip(m))


def _copy_by(how: str, m):
    return {
        "clone": m.clone,
        "deepcopy": lambda: copy.deepcopy(m),
        "pickle": lambda: _roundtrip(m),
    }[how]()


def _decomposed():
    m = dm.Model("dec")
    x = m.continuous("x", shape=(4,), lb=0, ub=1)
    m.subject_to(x[0] + x[1] <= 1)
    m.subject_to(x[0] + x[3] <= 1)
    m.minimize(dm.sum(x))
    m.mark_coupling(m._constraints[1])
    m.set_constraint_block(m._constraints[0], 0)
    return m


@pytest.mark.parametrize("how", ["clone", "deepcopy", "pickle"])
def test_id_keyed_decomposition_marks_follow_the_copy(how):
    """Before: the copy kept the ORIGINAL's ``id(constraint)`` keys, so its
    coupling set came back empty and ``dumps`` refused it."""
    from discopt.decomposition.structure import _annotated_coupling
    from discopt.serialize import dumps

    m = _decomposed()
    c = _copy_by(how, m)
    assert _annotated_coupling(c) == _annotated_coupling(m) == {1}
    assert [k for k in c._block_labels_con if isinstance(k, int)] == [id(c._constraints[0])]
    assert dumps(c, provenance=False) == dumps(m, provenance=False)


def _atom_model(fn):
    m = dm.Model("atom")
    x = m.continuous("x", shape=(3,), lb=-2, ub=2)
    m.minimize(sum(fn(x[i]) * (i + 1) for i in range(3)) + dm.sum(x) ** 2)
    m.subject_to(dm.sum(x) >= -1)
    return m


def _tagged(m) -> int:
    from discopt.operators import atom_of
    from discopt.serialize import _children

    seen, stack, n = set(), [m._objective.expression], 0
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        n += atom_of(e) is not None
        stack.extend(_children(e))
    return n


@pytest.mark.parametrize("how", ["clone", "deepcopy", "pickle"])
def test_a_registered_atom_keeps_its_envelope_through_a_copy(how):
    """Before: the copy duplicated the registration, ``atom_of``'s identity check
    failed, and the copy relaxed term by term -- a looser bound, measured 31 vs 5
    nodes on the adversarial probe -- and pickling raised on its closures."""
    from discopt.operators import register_function, registry_snapshot

    with registry_snapshot():
        fn = register_function("test_pickle_cubic", lambda t: t**3 - 2 * t)
        m = _atom_model(fn)
        kw = dict(deterministic=True, max_nodes=3000, time_limit=120)
        ref = _atom_model(fn).solve(**kw)
        assert _tagged(m) == 3
        c = _copy_by(how, m)
        assert _tagged(c) == 3
        got = c.solve(**kw)
        assert (got.objective, got.node_count) == (ref.objective, ref.node_count)


def test_a_pickled_atom_is_not_trusted_after_its_registration_changed():
    """The name is all that crosses a pickle; a different definition under it
    must not lend its envelope to the old body (that would be a false bound)."""
    from discopt.operators import clear_registered, register_function, registry_snapshot

    with registry_snapshot():
        fn = register_function("test_pickle_swap", lambda t: t**3 - 2 * t)
        blob = pickle.dumps(_atom_model(fn))
        register_function("test_pickle_swap", lambda t: t**3 + 5 * t, replace=True)
        assert _tagged(pickle.loads(blob)) == 0
        clear_registered("test_pickle_swap")
        assert _tagged(pickle.loads(blob)) == 0


@pytest.mark.parametrize("n", [500, 5000])
def test_a_loop_built_expression_copies_and_pickles(n):
    """Before: ``clone()`` raised ``RecursionError`` at 500 terms, pickle at 5000."""
    from discopt.serialize import dumps

    m = dm.Model("deep")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    e = x[0]
    for i in range(1, n):
        e = e + (i % 7) * x[i]
    m.minimize(e)
    m.subject_to(e >= 3)
    ref = dumps(m, provenance=False)
    assert dumps(m.clone(), provenance=False) == ref
    assert dumps(_roundtrip(m), provenance=False) == ref


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_every_pickle_protocol(protocol):
    m = _minlp()
    m.solve(time_limit=30)
    c = pickle.loads(pickle.dumps(m, protocol=protocol))
    assert dt.model_fingerprint(c) == dt.model_fingerprint(m)


def test_array_bounds_rebuilt_by_numpy_unpickling_are_refrozen():
    """Before: numpy rebuilt a large bound as a view over ``bytes``, and
    ``Variable.__setstate__`` raised ``AttributeError`` reading its ``.flags``."""
    m = dm.Model("wide")
    x = m.continuous("x", shape=(600,), lb=0, ub=np.arange(600, dtype=float) + 1)
    m.minimize(dm.sum(x))
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        c = pickle.loads(pickle.dumps(m, protocol=protocol))
        v = c._variables[0]
        np.testing.assert_array_equal(v.ub, x.ub)
        assert not v.lb.flags.writeable and not v.ub.flags.writeable
