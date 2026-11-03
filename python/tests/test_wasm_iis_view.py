"""The browser page's IIS view: ``crates/discopt-wasm/web/iis_view.py``.

The page draws an IIS as a table and a constraint--variable graph, from the
dict ``iis_view.py`` builds. That file is served beside the page, not shipped
in the wheel, so nothing else imports it; this module loads it from its path
and checks the dict against the ``IISResult`` it describes.

What is checked here is the *data* the page draws. The drawing itself
(``iis_render.js``) is exercised by ``crates/discopt-wasm/tests/``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import discopt.modeling as dm
import pytest
from test_wasm_examples import EXAMPLES, browser_environment  # noqa: F401  (fixture)

pytestmark = pytest.mark.smoke

IIS_VIEW = Path(__file__).resolve().parents[2] / "crates" / "discopt-wasm" / "web" / "iis_view.py"


def _load_iis_view():
    spec = importlib.util.spec_from_file_location("discopt_iis_view", IIS_VIEW)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iis_view = _load_iis_view()


def _check_consistent(p: dict) -> int:
    """Invariants every payload must satisfy; returns how many were checked.

    The page trusts these: an edge to a variable id that is not a node would
    be silently dropped by the graph and still listed by the table.
    """
    checks = 0
    var_ids = [v["id"] for v in p["variables"]]
    assert len(var_ids) == len(set(var_ids)), var_ids
    used = set()
    for m in p["members"] + p["context"]:
        assert m["vars"], m
        for vid in m["vars"]:
            assert vid in var_ids, (m["name"], vid)
            checks += 1
        if m["kind"] != "context":
            used.update(m["vars"])
    # Every variable node is touched by a member; context never adds one.
    assert used == set(var_ids), (used, var_ids)
    ids = [m["id"] for m in p["members"] + p["context"]]
    assert len(ids) == len(set(ids)), ids
    assert json.loads(json.dumps(p)) == p
    return checks + 3


def test_iis_example_payload(capsys: pytest.CaptureFixture[str], browser_environment) -> None:  # noqa: F811
    """The page's IIS example, run as the page runs it, then collected."""
    example = next(e for e in EXAMPLES if e["id"] == "IIS")
    ns: dict[str, object] = {"__name__": "__main__"}
    exec(compile(example["code"], "<IIS example>", "exec"), ns)
    assert "infeasible" in capsys.readouterr().out

    payloads = iis_view.collect(ns)
    assert len(payloads) == 1
    p = payloads[0]
    iis, model = ns["iis"], ns["m"]
    assert p["schema"] == iis_view.SCHEMA
    assert p["label"] == "iis"

    # Members are exactly the IIS: its constraints, then its bound elements.
    con_names = [m["name"] for m in p["members"] if m["kind"] == "constraint"]
    assert con_names == [c.name for c in iis.constraints]
    bounds = [m for m in p["members"] if m["kind"] == "bound"]
    assert len(bounds) == len(iis.bound_elements)
    for m, (var, side, idx) in zip(bounds, iis.bound_elements):
        assert m["variable"] == var.name and m["side"] == side
        assert m["vars"] == [iis_view.element_id(var, tuple(idx))]
    assert p["proven_irreducible"] is iis.proven_irreducible is True
    assert p["n_solves"] == iis.n_solves > 0

    # Element granularity: the shared-supply row touches one feed of each mill.
    soy = next(m for m in p["members"] if m["name"] == "soy_supply")
    assert soy["vars"] == ["a[1]", "b[1]"]

    # Context is every model constraint outside the IIS that shares a variable.
    in_iis = {c.name for c in iis.constraints}
    assert {c["name"] for c in p["context"]} == {
        c.name for c in model._constraints if c.name not in in_iis
    }
    assert p["context_total"] == len(p["context"])
    assert p["model"] == {"name": "two mills", "n_constraints": 8, "n_variables": 6}

    checks = _check_consistent(p)
    assert checks > 10, checks


def test_variable_refs_are_element_level() -> None:
    m = dm.Model("refs")
    s = m.continuous("s", lb=0, ub=1)
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    z = m.continuous("z", shape=(2, 2), lb=0, ub=1)

    def ids(expr):
        return [(v.name, idx) for v, idx in iis_view.variable_refs(expr)]

    assert ids(s + x[0] + x[0] * x[2]) == [("s", ()), ("x", (0,)), ("x", (2,))]
    assert ids(x[-1] + z[1, 0]) == [("x", (2,)), ("z", (1, 0))]
    # A vector used whole is one reference with no element.
    assert ids(dm.sum(x) + s) == [("x", None), ("s", ())]
    assert iis_view.element_id(z, (1, 0)) == "z[1,0]"
    assert iis_view.element_id(s) == "s"


def test_whole_vector_use_links_to_the_elements_in_the_iis() -> None:
    """``sum(x) >= 25`` with ``x <= [1, 10, 10]``: the bound elements are the
    nodes, and the constraint that uses ``x`` whole links to all three."""
    m = dm.Model("whole")
    x = m.continuous("x", shape=(3,), lb=0, ub=[1, 10, 10])
    m.subject_to(dm.sum(x) >= 25, name="need")
    iis = m.compute_iis()

    p = iis_view.iis_payload(iis, m)
    need = next(mm for mm in p["members"] if mm["name"] == "need")
    assert need["vars"] == ["x[0]", "x[1]", "x[2]"]
    assert [v["id"] for v in p["variables"]] == ["x[0]", "x[1]", "x[2]"]
    assert not any(v["whole"] for v in p["variables"])
    _check_consistent(p)


def test_one_element_vector_used_bare_is_one_node() -> None:
    """``w`` of shape ``(1,)`` used bare and its bound ``w[0] <= 1`` are one node."""
    m = dm.Model("one")
    w = m.continuous("w", shape=(1,), lb=0, ub=1)
    m.subject_to(dm.sum(w) >= 3, name="need")
    iis = m.compute_iis()

    p = iis_view.iis_payload(iis, m)
    assert [v["id"] for v in p["variables"]] == ["w[0]"]
    assert [mm["vars"] for mm in p["members"]] == [["w[0]"], ["w[0]"]]
    _check_consistent(p)


def test_whole_vector_with_no_element_in_the_iis_is_one_node() -> None:
    m = dm.Model("conflict")
    y = m.continuous("y", shape=(4,), lb=-10, ub=10)
    m.subject_to(dm.sum(y) >= 5, name="at_least")
    m.subject_to(dm.sum(y) <= 3, name="at_most")
    iis = m.compute_iis()

    p = iis_view.iis_payload(iis, m)
    assert [mm["name"] for mm in p["members"]] == ["at_least", "at_most"]
    assert p["variables"] == [{"id": "y", "base": "y", "index": [], "whole": True}]
    assert all(mm["vars"] == ["y"] for mm in p["members"])
    assert p["context"] == [] and p["context_total"] == 0
    _check_consistent(p)


def test_context_is_capped_and_counted() -> None:
    m = dm.Model("many")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y >= 30, name="too_much")
    for k in range(5):
        m.subject_to(x - y <= k, name=f"side{k}")
    iis = m.compute_iis()

    p = iis_view.iis_payload(iis, m, max_context=3)
    assert p["context_total"] == 5
    assert [c["name"] for c in p["context"]] == ["side0", "side1", "side2"]
    _check_consistent(p)


def test_collect_finds_the_model_and_skips_private_and_aliases() -> None:
    m = dm.Model("bound conflict")
    x = m.continuous("x", lb=0, ub=2)
    m.subject_to(x >= 5, name="need")  # the IIS: this row and x's upper bound
    m.subject_to(x >= -100, name="slack")
    iis = m.compute_iis()

    ns = {"m": m, "x": x, "iis": iis, "same": iis, "_hidden": iis}
    payloads = iis_view.collect(ns)
    assert [p["label"] for p in payloads] == ["iis"]
    p = payloads[0]
    assert p["model"] is not None
    assert [(mm["kind"], mm["name"]) for mm in p["members"]] == [
        ("constraint", "need"),
        ("bound", "x <= 2"),
    ]
    assert [c["name"] for c in p["context"]] == ["slack"]
    _check_consistent(p)

    # Without the model in the namespace there is nothing to draw context from,
    # and the payload says so rather than inventing one.
    lone = iis_view.collect({"iis": iis})[0]
    assert lone["model"] is None and lone["context"] == []
    assert json.loads(iis_view.collect_json({"iis": iis})) == [lone]


def test_collect_with_no_iis_is_empty() -> None:
    assert iis_view.collect({"m": dm.Model("empty"), "n": 3}) == []
    assert iis_view.collect_json({}) == "[]"
