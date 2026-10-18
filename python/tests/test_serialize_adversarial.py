"""``discopt.serialize`` against the defects an adversarial pass found.

Three classes, each a way a document could reload as a model other than the one
that was saved -- or a document that is not one could load at all:

* **Attributes attached after ``Model.__init__`` were dropped without a word.**
  ``test_every_model_attribute_is_accounted_for`` (in ``test_serialize_roundtrip``)
  only sees what ``__init__`` creates; ``from_nl``, a solve and a reformulation pass
  all attach more. Each known one is now carried, deliberately not carried, or
  refused, and any other private attribute is refused by name.
* **``from_nl`` provenance.** The parse's presence decides how conservative a
  certificate guard is, and the source path decides which file the native NLP
  path reads -- whatever file sits there now.
* **Malformed documents loaded.** Most dangerously an indicator whose
  ``active_value`` the variable can never take, which dropped the row and
  certified a false optimum; also negative / ``true`` references that Python
  indexing resolved to a different node, and indices past the end of a variable.
"""

from __future__ import annotations

import copy
import json
import pathlib
import shutil
import time

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt.modeling.core import _DetachedNlRepr, from_nl
from discopt.serialize import SerializationError, dumps, loads
from discopt.solvers.nlp_native import _source_still_matches

DATA = pathlib.Path(__file__).parent / "data"
CORPUS = sorted((DATA / "minlplib_nl").glob("*.nl"))


def _small():
    m = dm.Model("small")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=4.0)
    b = m.binary("b")
    m.minimize(x[0] + 2 * x[1])
    m.subject_to(x[0] + x[1] >= 1, name="cover")
    m.if_then(b, [x[0] >= 0.5], name="ind")
    m.sos1([x[0], x[1]], name="s")
    return m


# ── from_nl provenance ────────────────────────────────────────────────────


def test_a_reloaded_from_nl_model_keeps_the_conservative_guard():
    """Before: the reloaded model walked its expressions where the original said
    "could not tell", so the certificate guard fired less after a round trip."""
    checked = 0
    for path in CORPUS:
        m = from_nl(str(path))
        back = loads(dumps(m))
        assert m._referenced_variable_names() is None
        assert isinstance(back._nl_repr, _DetachedNlRepr), path.name
        assert back._referenced_variable_names() is None, path.name
        checked += 1
    assert checked == len(CORPUS)


def test_the_source_path_travels_only_with_its_digest(tmp_path):
    src = tmp_path / "copy.nl"
    shutil.copy(CORPUS[0], src)
    m = from_nl(str(src))
    back = loads(dumps(m))
    assert back._source_nl_path == m._source_nl_path
    assert back._source_nl_sha256 == m._source_nl_sha256
    assert _source_still_matches(back, back._source_nl_path)

    # The same path now holds different bytes: neither the original nor the
    # reloaded model may hand that file to the native NLP path.
    src.write_bytes(src.read_bytes() + b"\n")
    assert not _source_still_matches(m, m._source_nl_path)
    assert not _source_still_matches(back, back._source_nl_path)


def test_a_path_without_a_digest_is_never_trusted(tmp_path):
    src = tmp_path / "copy.nl"
    shutil.copy(CORPUS[0], src)
    m = from_nl(str(src))
    del m._source_nl_sha256
    assert not _source_still_matches(m, m._source_nl_path)
    assert "path" not in json.loads(dumps(m))["state"]["nl_source"]


def test_a_transformation_clears_the_digest_with_the_path():
    m = from_nl(str(CORPUS[0]))
    assert getattr(m, "_source_nl_sha256", None)
    dt._invalidate_source_records(m)
    assert not hasattr(m, "_source_nl_path")
    assert not hasattr(m, "_source_nl_sha256")


def test_the_detached_marker_refuses_use():
    back = loads(dumps(from_nl(str(CORPUS[0]))))
    with pytest.raises(AttributeError, match="does not survive"):
        back._nl_repr.n_vars  # noqa: B018


# ── attributes attached after __init__ ────────────────────────────────────


def test_a_solved_model_saves_without_its_per_solve_state():
    # Nonconvex and mixed-integer, so the solve reaches the stage of ``solve_model``
    # that stamps its budgets.
    m = dm.Model("nc")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.integer("y", lb=0, ub=3)
    m.minimize(x * y - x + dm.sin(3 * x))
    m.subject_to(x + y >= 1.5)
    m.solve(time_limit=30)
    # ``Model.solve`` deletes the deadline on the way out; a direct ``solve_model``
    # call leaves it behind.
    m._solve_deadline = time.perf_counter() + 5.0
    assert {"_solve_deadline", "_convexity_time_budget", "_gauss_newton_hessian"} <= set(vars(m))
    back = loads(dumps(m))
    for attr in ("_solve_deadline", "_convexity_time_budget", "_gauss_newton_hessian"):
        assert not hasattr(back, attr)


def test_an_unclassified_private_attribute_is_refused_by_name():
    m = _small()
    m._some_future_state = {"anything": 1}
    with pytest.raises(SerializationError, match="_some_future_state"):
        dumps(m)


def test_a_public_attribute_the_user_attached_is_not_discopt_state():
    m = _small()
    m.notes = "mine"
    assert not hasattr(loads(dumps(m)), "notes")


@pytest.mark.parametrize(
    "name, build",
    [
        (
            "binary.multilinear",
            lambda: _bin_trilinear(),
        ),
        (
            "nonsmooth.epigraph",
            lambda: _nonsmooth(),
        ),
    ],
)
def test_a_reformulation_output_is_refused_with_the_remedy(name, build):
    out = dt.create_using(name, build())
    with pytest.raises(SerializationError, match=name.replace(".", r"\.") + ".*source model"):
        dumps(out)


def _bin_trilinear():
    m = dm.Model("bt")
    b = m.binary("b", shape=(3,))
    m.minimize(-(b[0] * b[1] * b[2]) + b[0])
    return m


def _nonsmooth():
    m = dm.Model("ns")
    x = m.continuous("x", shape=(2,), lb=-3, ub=3)
    m.minimize(dm.abs(x[0] - 0.3) + dm.maximum(x[1], -x[0]))
    m.subject_to(x[0] + x[1] == 1.0)
    return m


def test_recentre_marker_round_trips():
    m = _small()
    m._recentre_inner = True
    assert loads(dumps(m))._recentre_inner is True
    assert not hasattr(loads(dumps(_small())), "_recentre_inner")


def test_gams_initial_values_round_trip():
    m = _small()
    m._gams_initial_values = {"b": 1.0, "x": {0: 0.25, 1: 3.5}}
    back = loads(dumps(m))
    assert back._gams_initial_values == m._gams_initial_values


@pytest.mark.parametrize(
    "rows",
    [
        [["nope", 1.0]],  # undeclared variable
        [["x", [[2, 1.0]]]],  # element out of range
        [["x", [[True, 1.0]]]],  # a bool is not an element index
        [["b", 1.0], ["b", 0.0]],  # listed twice
    ],
)
def test_malformed_gams_initial_values_are_refused(rows):
    m = _small()
    m._gams_initial_values = {"b": 1.0}
    doc = json.loads(dumps(m))
    doc["state"]["_gams_initial_values"] = rows
    with pytest.raises(SerializationError):
        loads(json.dumps(doc))


# ── malformed documents ───────────────────────────────────────────────────


def _doc():
    return json.loads(dumps(_small()))


def _row(doc, kind):
    return next(r for r in doc["rows"] if r["kind"] == kind)


def _first(doc, op):
    return next(i for i, n in enumerate(doc["nodes"]) if n["op"] == op)


def _set_active_value(v):
    def f(doc):
        _row(doc, "indicator")["active_value"] = v

    return f


def _set_ref(field_value):
    def f(doc):
        _row(doc, "algebraic")["body"] = field_value

    return f


def _index_node(doc):
    return next(n for n in doc["nodes"] if n["op"] == "index")


MALFORMED = {
    # The false certificate: an active value the binary can never take dropped the
    # row, and the solve reported the relaxed optimum as optimal.
    "active_value 7": _set_active_value(7),
    "active_value -1": _set_active_value(-1),
    "active_value true": _set_active_value(True),
    # References: Python indexing read these as a different node.
    "negative row body ref": _set_ref(-1),
    "bool row body ref": _set_ref(True),
    "row body ref past the end": _set_ref(10_000),
    "negative var ref": lambda d: d["nodes"][_first(d, "var")].update(ref=-1),
    "forward node ref": lambda d: _index_node(d).update(base=len(d["nodes"]) - 1),
    "negative indicator ref": lambda d: _row(d, "indicator").update(indicator=-2),
    "negative sos member": lambda d: _row(d, "sos").update(variables=[-1, 0]),
    "objective ref negative": lambda d: d["objective"].update(expr=-1),
    # Indices the modeling API refuses.
    "index past the end": lambda d: _index_node(d).update(index={"k": "int", "v": 7}),
    "index as string": lambda d: _index_node(d).update(index={"k": "int", "v": "1"}),
    "slice bound as string": lambda d: _index_node(d).update(
        index={"k": "slice", "start": "a", "stop": None, "step": None}
    ),
    "slice step zero": lambda d: _index_node(d).update(
        index={"k": "slice", "start": None, "stop": None, "step": 0}
    ),
    # Fields that used to load and fail only inside a solve.
    "sense <": lambda d: _row(d, "algebraic").update(sense="<"),
    "sense null": lambda d: _row(d, "algebraic").update(sense=None),
    "sos_type 3": lambda d: _row(d, "sos").update(sos_type=3),
    "sos_type '1'": lambda d: _row(d, "sos").update(sos_type="1"),
    # Initial point.
    "initial point NaN": lambda d: d.update(initial_point=[["x", 0, "nan"]]),
    "initial point twice": lambda d: d.update(initial_point=[["x", 0, 1.0], ["x", 0, 2.0]]),
    "initial point bool element": lambda d: d.update(initial_point=[["x", True, 1.0]]),
    # Decomposition and bookkeeping state.
    "coupling row -1": lambda d: d["state"].update(_coupling_keys={"row_indices": [-1]}),
    "block label row -1": lambda d: d["state"].update(
        _block_labels_con={"rows": [{"row": -1, "labels": [0]}]}
    ),
    "aux counter string": lambda d: d["state"].update(_aux_counter="7"),
    "aux counter negative": lambda d: d["state"].update(_aux_counter=-1),
    "decomp stage not int": lambda d: d["state"].update(_decomp_stages={"x": "first"}),
    "atan2 sign junk": lambda d: d["state"].update(
        _atan2_preconditions=[{"denominator": 0, "sign": "sideways", "label": "l"}]
    ),
    # Shapes and numbers.
    "variable negative dimension": lambda d: d["variables"][0].update(shape=[-2]),
    "const holds a bool": lambda d: (
        d["nodes"].append({"op": "const", "v": True})
        or _row(d, "algebraic").update(body=len(d["nodes"]) - 1)
    ),
    "live box outside declared domain": lambda d: d["variables"][0].update(
        bound_stack=[[[1.0, 1.0], [2.0, 2.0]]]
    ),
    # Structure.
    "missing rows": lambda d: d.pop("rows"),
    "variables not a list": lambda d: d.update(variables=5),
}


def _fast_doc():
    """A fast-API model: a CSR constraint block and a quadratic builder objective."""
    import scipy.sparse as sp

    m = dm.Model("fast")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=4.0)
    m.add_linear_constraints(
        sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])), x, "<=", np.array([3.0, 1.0])
    )
    m.add_quadratic_objective(sp.csr_matrix(np.eye(3)), np.array([1.0, 0.0, -1.0]), x)
    return json.loads(dumps(m))


def _atom_doc():
    return json.loads((DATA / "serialize_fixtures" / "v1.5" / "registered_atom.json").read_text())


def _atom_node(d):
    return next(n for n in d["nodes"] if "atom" in n)


def _block_A(d):
    return d["builder_blocks"][0]["A"]


def _Q(d):
    return d["builder_objective"]["Q"]


#: Cases the WP-4 fuzzer found (``test_serialize_fuzz.py``, ``slow`` profile), one per
#: falsifying example. Before the fix: an out-of-range or negative CSR column index
#: LOADED, the negative one wrapping onto another column; a broken ``indptr`` made the
#: Rust builder PANIC (``pyo3_runtime.PanicException``, not even an ``Exception``); a
#: scalar bound written as ``[0.0]`` loaded as if it were ``0.0``.
FUZZ_FOUND = {
    "A column index past the last column": (
        _fast_doc,
        lambda d: _block_A(d)["indices"].__setitem__(0, 3),
    ),
    "A column index negative (wraps)": (
        _fast_doc,
        lambda d: _block_A(d)["indices"].__setitem__(0, -1),
    ),
    "A indptr past the entries (Rust panic)": (
        _fast_doc,
        lambda d: _block_A(d)["indptr"].__setitem__(1, 99),
    ),
    "Q indptr past the entries (Rust panic)": (
        _fast_doc,
        lambda d: _Q(d)["indptr"].__setitem__(1, 99),
    ),
    "Q column index past the last column": (
        _fast_doc,
        lambda d: _Q(d)["indices"].__setitem__(0, 3),
    ),
    "A column index overflows int32": (
        _fast_doc,
        lambda d: _block_A(d)["indices"].__setitem__(0, 10**12),
    ),
    "A values and indices differ in length": (
        _fast_doc,
        lambda d: _block_A(d)["data"].append(1.0),
    ),
    "A shape not a pair": (_fast_doc, lambda d: _block_A(d).update(shape=[2])),
    "scalar bound written as a list": (
        _doc,
        lambda d: next(v for v in d["variables"] if v["shape"] == []).update(lb=[0.0]),
    ),
    "array bound nested one level too deep": (
        _doc,
        lambda d: next(v for v in d["variables"] if v["shape"] == [2]).update(lb=[[0.0], [0.0]]),
    ),
    # The atom tag of a registered composite (#1248 A): a malformed one was skipped
    # as if it were an unregistered (dropped-by-design) tag.
    "atom tag not an object": (_atom_doc, lambda d: _atom_node(d).update(atom=5)),
    "atom tag name not a string": (
        _atom_doc,
        lambda d: _atom_node(d)["atom"].update(n=3),
    ),
    "atom tag argument a bool": (
        _atom_doc,
        lambda d: _atom_node(d)["atom"].update(arg=True),
    ),
    "atom tag argument a later node": (
        _atom_doc,
        lambda d: _atom_node(d)["atom"].update(arg=len(d["nodes"])),
    ),
    "array bound ragged": (
        _doc,
        lambda d: next(v for v in d["variables"] if v["shape"] == [2]).update(lb=[0.0]),
    ),
}


@pytest.mark.parametrize("case", sorted(FUZZ_FOUND))
def test_a_fuzzer_found_document_is_refused(case):
    make, corrupt = FUZZ_FOUND[case]
    doc = make()
    loads(json.dumps(doc))  # the base loads
    corrupt(doc)
    with pytest.raises(SerializationError):
        loads(json.dumps(doc))


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_a_malformed_document_is_refused(case):
    doc = _doc()
    MALFORMED[case](doc)
    with pytest.raises(SerializationError):
        loads(json.dumps(doc))


def test_the_malformed_probe_fired():
    """§6: each case above must start from a document that loads unmodified."""
    assert len(MALFORMED) >= 30
    loads(json.dumps(_doc()))


def test_unknown_disjunction_method_is_refused():
    m = dm.Model("d")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.either_or([[x >= 4], [x <= 1]])
    doc = json.loads(dumps(m))
    _row(doc, "disjunction")["method"] = "bigggm"
    with pytest.raises(SerializationError, match="bigggm"):
        loads(json.dumps(doc))


def test_sum_axis_past_the_end_is_refused():
    m = dm.Model("s")
    x = m.continuous("x", shape=(2, 3), lb=0, ub=1)
    m.minimize(dm.sum(dm.sum(x, axis=1)))
    doc = json.loads(dumps(m))
    node = next(n for n in doc["nodes"] if n["op"] == "sum" and n["axis"] == 1)
    node["axis"] = 5
    with pytest.raises(SerializationError, match="axis"):
        loads(json.dumps(doc))


def test_not_json_and_not_an_object_are_refused():
    with pytest.raises(SerializationError):
        loads("{not json")
    with pytest.raises(SerializationError):
        loads("[1, 2]")


def test_a_valid_document_still_loads_after_hardening():
    """The hardening must not refuse what the writer produces: deep-copied,
    fixed, decomposed and relation-carrying models all still round-trip."""
    m = _small()
    m.mark_coupling(m._constraints[0])
    with m.fixed({m._variables[0]: [1.0, 1.0]}):
        d1 = dumps(m, provenance=False)
        assert dumps(loads(d1), provenance=False) == d1
    c = copy.deepcopy(m)
    assert dumps(c, provenance=False) == dumps(m, provenance=False)
    np.testing.assert_array_equal(loads(d1)._variables[0].lb, [1.0, 1.0])
