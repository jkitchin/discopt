"""Unknown keys inside nested records are refused, not dropped (#1780).

WP-2 (#1776) refused an unknown key at a document's top level and in ``state``;
inside a variable, node, row, builder block, relation or the embedded result an
unknown key was still ignored. Probe on the pre-fix reader: adding ``"zzz"`` to
``nodes[0]`` or ``variables[0]`` of the 1.0 ``scalar_nlp.json`` fixture loaded with
no error and no warning, and the key was gone on re-save (2/2).

The reader now enforces it two ways (``discopt.serialize._Record``):

* every object outside the node table records which keys the decoder read, and
  any key it never read is refused -- derived from the decoder, so it covers
  every record kind without a hand-kept list;
* node records and their index objects are checked against ``_NODE_KEYS`` /
  ``_INDEX_KEYS`` (tracking every node was measured at +130 % on ``loads``), and
  the test below proves those tables are exactly what the decoder reads.

Also refused now: a key repeated within one JSON object (``json`` keeps the last
and drops the others unread).
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import serialize
from discopt.serialize import (
    SCHEMA_MAJOR,
    SCHEMA_MINOR,
    SchemaVersionWarning,
    SerializationError,
    dumps,
    loads,
)

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures"
FIXTURE_PATHS = sorted(FIXTURES.glob("v*/*.json"))
FUTURE = f"discopt.model/{SCHEMA_MAJOR}.{SCHEMA_MINOR + 1}"
INJECTED = "zzz_injected_key"


def _every_op_model():
    """One model exercising every node op and every index kind."""
    from discopt.operators import register_function

    cube = register_function("nested_keys_cube", lambda t: t**3 + t, replace=True)
    m = dm.Model("every_op")
    x = m.continuous("x", shape=(3, 2), lb=-1.0, ub=1.0)
    y = m.continuous("y", shape=(2,), lb=0.0, ub=2.0)
    p = m.parameter("p", value=np.array([1.0, 2.0]))
    c = m.parameter("c", value=0.5)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    body = (
        dm.sum(x[0:2, 1])  # slice + tuple index, sum (axis None)
        + dm.sum(x, axis=0)[1]  # sum with an axis, int index
        + (A @ y)[0]  # matmul
        - dm.exp(y[0]) * c  # call, param
        + abs(y[1])  # unop abs
        + (-y[0]) ** 2  # unop neg, binop **
        + y[True][0]  # bool index
        + cube(y[1])  # registered atom
        + dm.sum(p * y)  # array param, array const via broadcasting
    )
    m.minimize(body)
    m.subject_to(dm.sum([y[0], y[1], x[2, 0]]) <= 3.0, name="so")  # sum_over
    return m


# ── the node tables are what the decoder reads ─────────────────────────────


def _decode_with_tracked_nodes(text: str) -> list:
    doc = json.loads(text, object_pairs_hook=serialize._unique_pairs)
    doc["nodes"] = [serialize._track(nd) for nd in doc["nodes"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        serialize._loads_document(doc)
    return doc["nodes"]


def test_node_key_tables_are_what_the_decoder_reads():
    texts = [p.read_text(encoding="utf-8") for p in FIXTURE_PATHS]
    texts.append(dumps(_every_op_model(), provenance=False))
    read: dict[str, set] = {}
    written: dict[str, set] = {}
    index_read: dict[str, set] = {}
    atom_read: set = set()
    for text in texts:
        for nd in _decode_with_tracked_nodes(text):
            op = dict.__getitem__(nd, "op")
            read.setdefault(op, set()).update(nd._read)
            written.setdefault(op, set()).update(dict.keys(nd))
            atom = dict.get(nd, "atom")
            if atom is not None:
                atom_read |= atom._read
            stack = [dict.get(nd, "index")] if op == "index" else []
            while stack:
                ix = stack.pop()
                kind = dict.__getitem__(ix, "k")
                index_read.setdefault(kind, set()).update(ix._read)
                if kind == "tuple":
                    stack.extend(dict.__getitem__(ix, "v"))
    # Every op and index kind was exercised, or the comparison below is vacuous.
    assert set(read) == set(serialize._NODE_KEYS), set(serialize._NODE_KEYS) - set(read)
    assert set(index_read) == set(serialize._INDEX_KEYS)
    for op, allowed in serialize._NODE_KEYS.items():
        # `atom` is read by `_retag_atoms` only once some node carries one.
        assert read[op] | {"atom"} == allowed, (op, sorted(read[op]), sorted(allowed))
        assert written[op] <= allowed, (op, written[op] - allowed)
    for kind, allowed in serialize._INDEX_KEYS.items():
        assert index_read[kind] == allowed, (kind, index_read[kind], allowed)
    assert atom_read == serialize._ATOM_KEYS


# ── an injected key anywhere is refused or carried, never dropped ──────────


def _record_sites(doc) -> list[tuple]:
    """One path per kind of JSON object in *doc*: the first of each normalised path
    (indices -> ``*``), distinguished by node ``op``, row ``kind`` and index ``k``."""
    sites: dict[str, tuple] = {}

    def walk(obj, path):
        if isinstance(obj, dict):
            if path:
                tag = obj.get("op") or obj.get("kind") or obj.get("k") or ""
                norm = re.sub(r"\d+", "*", "/".join(map(str, path))) + f"#{tag}"
                sites.setdefault(norm, tuple(path))
            for k, v in obj.items():
                if not path and (k == "provenance" or k.startswith("x-")):
                    continue
                walk(v, [*path, k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, [*path, i])

    walk(doc, [])
    return list(sites.values())


def _inject(doc, path):
    target = doc
    for step in path:
        target = target[step]
    target[INJECTED] = 1


def _injection_cases():
    for p in FIXTURE_PATHS:
        doc = json.loads(p.read_text(encoding="utf-8"))
        for site in _record_sites(doc):
            yield p, site


INJECTION_CASES = list(_injection_cases())


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda p: f"{p.parent.name}-{p.stem}")
def test_an_injected_nested_key_is_refused_or_carried(path):
    """The #1780 invariant over every record kind of every golden fixture.

    Refused is the expected outcome for a record. Carried is right for a map whose
    keys are data (a free-form ``solve_options`` tree): there the key is not
    dropped, it is in the model. Dropped -- loaded, and gone on re-save -- is the
    one forbidden outcome.
    """
    base = json.loads(path.read_text(encoding="utf-8"))
    sites = [s for p, s in INJECTION_CASES if p == path]
    assert sites
    refused = carried = 0
    for site in sites:
        doc = copy.deepcopy(base)
        _inject(doc, site)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = loads(json.dumps(doc))
        except SerializationError:
            refused += 1
            continue
        resaved = dumps(model, result=model.saved_result, provenance=False)
        assert INJECTED in resaved, f"{path.name}: key injected at {site} was dropped"
        carried += 1
    assert refused + carried == len(sites)
    assert refused >= 3  # variables, nodes, rows at least


def test_injection_probe_fired():
    """§6: the parametrized injection above had sites to inject into."""
    assert len(INJECTION_CASES) >= 10 * len(FIXTURE_PATHS)


@pytest.mark.parametrize("section", ["nodes", "variables"])
def test_the_issue_probe_is_refused(section):
    """The exact probe from #1780."""
    p = FIXTURES / "v1.0" / "scalar_nlp.json"
    doc = json.loads(p.read_text())
    doc[section][0]["zzz"] = 1
    with pytest.raises(SerializationError, match="zzz"):
        loads(json.dumps(doc))


# ── the WP-2 version policy applies to nested keys too ─────────────────────


def _doc() -> dict:
    m = dm.Model("nested")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    m.minimize(x[0] + x[1])
    m.subject_to(x[0] + x[1] >= 0.5, name="c")
    return json.loads(dumps(m))


def test_unknown_row_key_at_a_known_minor_is_malformed():
    doc = _doc()
    doc["rows"][0]["weight"] = 2.0
    with pytest.raises(SerializationError, match=r"no writer of that format.*/rows\[0\]/weight"):
        loads(json.dumps(doc))


def test_unknown_nested_key_at_a_future_minor_names_the_upgrade():
    doc = _doc()
    doc["schema"] = FUTURE
    doc["variables"][0]["domain"] = "semicontinuous"
    with pytest.raises(
        SerializationError, match=r"newer than this reader.*/variables\[0\]/domain.*Upgrade"
    ):
        loads(json.dumps(doc))
    doc = _doc()
    doc["schema"] = FUTURE
    doc["nodes"][0]["unit"] = "kg"
    with pytest.raises(SerializationError, match=r"newer than this reader.*unit.*upgrade"):
        loads(json.dumps(doc))


def test_a_future_minor_with_known_nested_keys_still_loads_with_a_warning():
    doc = _doc()
    doc["schema"] = FUTURE
    with pytest.warns(SchemaVersionWarning):
        assert isinstance(loads(json.dumps(doc)), dm.Model)


def test_unknown_index_and_atom_keys_are_refused():
    text = dumps(_every_op_model(), provenance=False)
    doc = json.loads(text)
    i = next(i for i, nd in enumerate(doc["nodes"]) if nd["op"] == "index")
    doc["nodes"][i]["index"]["wrap"] = True
    with pytest.raises(SerializationError, match="wrap"):
        loads(json.dumps(doc))
    doc = json.loads(text)
    i = next(i for i, nd in enumerate(doc["nodes"]) if "atom" in nd)
    doc["nodes"][i]["atom"]["version"] = 2
    with pytest.raises(SerializationError, match="atom/version"):
        loads(json.dumps(doc))


def test_a_repeated_key_is_refused():
    text = dumps(dm.Model("dup"), provenance=False)
    # Repeat "name" inside the top-level object: json would keep the second.
    text = text.replace('"name": "dup"', '"name": "dup", "name": "other"', 1)
    assert text.count('"name"') >= 2
    with pytest.raises(SerializationError, match=r"repeats key\(s\) \['name'\]"):
        loads(text)


def test_builder_objective_with_a_real_objective_is_refused():
    """Found by the unread-key check: beside a builder objective the writer records
    the zero placeholder, which the reader never looked at -- so a document carrying
    a real objective there would have lost it without a word."""
    import scipy.sparse as sp

    m = dm.Model("fast")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    m.add_linear_constraints(sp.csr_matrix(np.ones((1, 2))), x, "<=", np.array([1.0]))
    m.add_linear_objective(np.array([1.0, -1.0]), x)
    doc = json.loads(dumps(m))
    assert loads(json.dumps(doc)) is not None
    doc["objective"]["placeholder"] = False
    with pytest.raises(SerializationError, match="non-placeholder"):
        loads(json.dumps(doc))
    doc["objective"]["placeholder"] = True
    doc["objective"]["sense"] = "maximize"
    with pytest.raises(SerializationError, match="disagrees"):
        loads(json.dumps(doc))


def test_a_loaded_model_holds_no_tracking_dicts():
    """The tracking class is the reader's instrument; nothing it builds keeps one."""
    from discopt.result_io import deserialize_result, serialize_result

    m = dm.Model("solved")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
    m.minimize(dm.sum((x - 1.0) ** 2))
    # A result carrying the subtrees the reader keeps verbatim or rebuilds as trees.
    r = deserialize_result(
        serialize_result(m.solve(time_limit=30), provenance=True, options={"tuning": {"a": 1}})
    )
    loaded = loads(dumps(m, result=r))
    assert loaded.saved_result._provenance and loaded.saved_result._solve_options
    loaded_every_op = loads(dumps(_every_op_model()))
    found = []

    def scan(obj, seen, depth=0):
        if id(obj) in seen or depth > 30:
            return
        seen.add(id(obj))
        if isinstance(obj, serialize._Record):
            found.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                scan(v, seen, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                scan(v, seen, depth + 1)
        elif type(obj).__module__.startswith("discopt") and hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                scan(v, seen, depth + 1)

    scan(loaded, set())
    scan(loaded_every_op, set())
    # And every golden fixture: a 1.0/1.1 result is re-tracked after `_dec_tree`,
    # and a pre-version-3 trace is kept verbatim, so both paths must hand back
    # plain objects.
    for path in FIXTURE_PATHS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scan(loads(path.read_text(encoding="utf-8")), set())
    assert found == []


# ── the embedded result's sub-records, at every minor (review of #1783) ────
#
# Two holes the first version missed, neither reachable from the fixtures (none
# holds a validation report or an infeasibility certificate): (1) the report
# decoder iterated its dict and FILTERED to the dataclass fields, so an unknown
# key counted as read and was then discarded; (2) for a 1.0/1.1 document the
# solution subtree was rebuilt by `_dec_tree` -- iterating, so every key read --
# and the unread-key check never saw below the result's top level.


def _result_doc(schema: str) -> dict:
    import numpy as np
    from discopt.solvers import InfeasibilityCertificate
    from discopt.validation.examiner import CheckResult, ExaminerReport

    m = dm.Model("vr")
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.minimize((x - 1.0) ** 2)
    r = m.solve(time_limit=30)
    r.validation_report = ExaminerReport(
        checks=[CheckResult(name="c", passed=True, tolerance=1e-6, violators=[("x", 0.0)])]
    )
    r.infeasibility_certificate = InfeasibilityCertificate(
        total_violation=1.0, ineq_violations=np.array([1.0]), eq_violations=np.array([])
    )
    doc = json.loads(dumps(m, result=r))
    doc["schema"] = schema
    return doc


RESULT_SITES = {
    "validation_report": lambda d: d["solution"]["validation_report"],
    "validation_report check": lambda d: d["solution"]["validation_report"]["checks"][0],
    "infeasibility_certificate": lambda d: d["solution"]["infeasibility_certificate"],
}
MINORS = [
    "discopt.model/1" if minor == 0 else f"discopt.model/{SCHEMA_MAJOR}.{minor}"
    for minor in range(SCHEMA_MINOR + 1)
]


@pytest.mark.parametrize("schema", MINORS)
@pytest.mark.parametrize("site", sorted(RESULT_SITES))
def test_an_unknown_key_in_a_result_sub_record_is_refused_at_every_minor(site, schema):
    doc = _result_doc(schema)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = loads(json.dumps(doc))  # unmodified: loads, with both sub-records
    assert loaded.saved_result.validation_report.checks[0].name == "c"
    assert loaded.saved_result.infeasibility_certificate.total_violation == 1.0
    RESULT_SITES[site](doc)[INJECTED] = 1
    with pytest.raises(SerializationError, match=INJECTED):
        loads(json.dumps(doc))


def test_the_review_probe_on_the_1_0_and_1_1_fixtures():
    for version in ("v1.0", "v1.1"):
        doc = json.loads((FIXTURES / version / "with_result.json").read_text())
        doc["solution"]["infeasibility_certificate"] = {
            "total_violation": 1.0,
            "ineq_violations": [1.0],
            "eq_violations": [],
            "zzz": 1,
        }
        with pytest.raises(SerializationError, match="zzz"):
            loads(json.dumps(doc))
