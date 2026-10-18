"""Schema 1.5: ``indicators`` on a disjunction row (#1782, #1775 G1.1).

#1775 G1.1 lets a user read ``either_or(...).indicators``; the document then has
to say which binary columns are that disjunction's selectors, or a reload leaves
them in the model tied to nothing. That is a new nested key, so the format minor
goes to 1.5 (the policy in ``discopt.serialize``'s *Versioning*):

* it is written only once the indicators were read, so a model that never reads
  them writes the 1.4 content under the 1.5 identifier;
* it is read only from a 1.5+ document, so in a <= 1.4 document it is refused as
  a key no writer of that minor produced;
* a 1.4 reader refuses it in a 1.5 document as *newer* content -- "upgrade" --
  not as a malformed document, and not by dropping it. The real 1.4 readers
  were measured by the fixture generator (``MANIFEST.json`` forward cells); the
  simulation here pins the message for the decoder logic that produces it.
"""

from __future__ import annotations

import json
import pathlib

import discopt.modeling as dm
import pytest
from discopt import serialize
from discopt.modeling.core import _DisjunctiveConstraint
from discopt.serialize import SchemaVersionWarning, SerializationError, dumps, loads

pytestmark = pytest.mark.smoke

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures"


def _model():
    """``mode`` high forces ``pump`` high; the third pump disjunct costs 3.

    Optimum 45 (x=15, w=10). With ``mode`` fixed low: 50 (x=5, w=3). Without the
    link between the indicators the optimum would be 10 -- what a reader that
    drops ``indicators`` solves to.
    """
    m = dm.Model("ind")
    x = m.continuous("x", lb=0.0, ub=20.0)
    w = m.continuous("w", lb=0.0, ub=20.0)
    mode = m.either_or([[x <= 5.0], [x >= 15.0]], name="mode")
    pump = m.either_or([[w <= 3.0], [w >= 10.0], [w >= 6.0, w <= 7.0]], name="pump")
    m.either_or([[x + w <= 30.0], [x - w >= 0.0]], name="unread")
    m.subject_to(mode.indicators[1].variable <= pump.indicators[1].variable)
    m.minimize((x - 12.0) ** 2 + (w - 4.0) ** 2 + 3.0 * pump.indicators[2].variable)
    return m


def _disjunctions(model):
    return [c for c in model._constraints if type(c) is _DisjunctiveConstraint]


def _rows(doc):
    return [r for r in doc["rows"] if r["kind"] == "disjunction"]


def test_schema_is_1_5():
    assert serialize.SCHEMA == "discopt.model/1.5"
    assert serialize._SCHEMA_MINOR_LOGIC_LOWERING_INDICATORS == serialize.SCHEMA_MINOR == 5


def test_indicators_written_only_once_read():
    doc = json.loads(dumps(_model(), provenance=False))
    mode, pump, unread = _rows(doc)
    assert len(mode["indicators"]) == 2 and len(pump["indicators"]) == 3
    assert "indicators" not in unread


def test_unread_model_writes_the_1_4_content():
    """The v1.5 fixture of a case with unread disjunctions is the v1.4 one but for
    the identifier."""
    new = json.loads((FIXTURES / "v1.5" / "mip_indicator_sos_disjunction.json").read_text())
    old = json.loads((FIXTURES / "v1.4" / "mip_indicator_sos_disjunction.json").read_text())
    assert (new.pop("schema"), old.pop("schema")) == ("discopt.model/1.5", "discopt.model/1.4")
    assert new == old


def test_round_trip_then_fix_an_indicator_solves_right():
    m2 = loads(dumps(_model(), provenance=False))
    mode, pump, unread = _disjunctions(m2)
    assert unread._indicator_vars is None
    assert [v.name for v in mode._indicator_vars] == ["mode_0_active", "mode_1_active"]
    assert all(any(v is w for w in m2._variables) for v in pump._indicator_vars)
    # .indicators returns the loaded columns; it allocates nothing new.
    n_vars = len(m2._variables)
    assert [b.variable for b in mode.indicators] == list(mode._indicator_vars)
    assert len(m2._variables) == n_vars

    r = m2.solve(time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(45.0, abs=1e-5)
    with mode._indicator_vars[1].fixed(0.0):
        r0 = m2.solve(time_limit=60)
    assert r0.status == "optimal" and r0.objective == pytest.approx(50.0, abs=1e-5)
    assert float(r0.x["x"]) == pytest.approx(5.0, abs=1e-5)
    assert float(r0.x["w"]) == pytest.approx(3.0, abs=1e-5)


def test_the_gdp_indicators_fixture_solves_to_45():
    m = loads((FIXTURES / "v1.5" / "gdp_indicators.json").read_text())
    r = m.solve(time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(45.0, abs=1e-5)


@pytest.mark.parametrize("schema", ["discopt.model/1.4", "discopt.model/1.2", "discopt.model/1"])
def test_indicators_in_an_older_document_are_malformed(schema):
    doc = json.loads(dumps(_model(), provenance=False))
    doc["schema"] = schema
    with pytest.raises(SerializationError, match=r"no writer of that format.*indicators"):
        loads(json.dumps(doc))
    # Without them the same document loads (the columns stay, unlinked).
    for row in _rows(doc):
        row.pop("indicators", None)
    assert all(d._indicator_vars is None for d in _disjunctions(loads(json.dumps(doc))))


def test_a_1_4_reader_says_upgrade_not_malformed(monkeypatch):
    """A reader at 1.4 does not read ``indicators``: simulate it by moving the key's
    minor past the reader's, which is all that separates the two decoders."""
    monkeypatch.setattr(serialize, "SCHEMA", "discopt.model/1.4")
    monkeypatch.setattr(serialize, "SCHEMA_MINOR", 4)
    monkeypatch.setattr(serialize, "_SCHEMA_MINOR_LOGIC_LOWERING_INDICATORS", 6)
    text = (FIXTURES / "v1.5" / "gdp_indicators.json").read_text()
    with pytest.raises(SerializationError) as info:
        loads(text)
    msg = str(info.value)
    assert "newer than this reader (discopt.model/1.4)" in msg
    assert "Upgrade discopt" in msg and "/rows[0]/indicators" in msg
    assert "malformed" not in msg
    # A 1.5 document with no indicators read still loads there, with the warning.
    with pytest.warns(SchemaVersionWarning):
        loads((FIXTURES / "v1.5" / "mip_indicator_sos_disjunction.json").read_text())


def _bad(mutate):
    doc = json.loads(dumps(_model(), provenance=False))
    mutate(_rows(doc)[0], doc)
    return json.dumps(doc)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r, d: r.__setitem__("indicators", None), "must be a list"),
        (lambda r, d: r.__setitem__("indicators", r["indicators"][:1]), "1 indicators for 2"),
        (lambda r, d: r.__setitem__("indicators", [r["indicators"][0]] * 2), "repeats"),
        (lambda r, d: r.__setitem__("indicators", [r["indicators"][0], 10**6]), "out of range"),
        (lambda r, d: r.__setitem__("indicators", [r["indicators"][0], -1]), "out of range|neg"),
        (lambda r, d: r.__setitem__("indicators", [r["indicators"][0], True]), "reference"),
        # node 0 is the first variable used, ``x``: continuous.
        (lambda r, d: r.__setitem__("indicators", [r["indicators"][0], 0]), "scalar binary"),
    ],
)
def test_malformed_indicators_are_refused(mutate, match):
    with pytest.raises(SerializationError, match=match):
        loads(_bad(mutate))


# ── logic over indicators (G1.1's point) round-trips with WP-3's logical rows ──


def _logic_model(force=None):
    """Two read ``either_or`` disjunctions; ``min (x-12)^2 + (w-4)^2``.

    Hand-computed optima, by enumerating the four disjunct selections
    (x<=5 | x>=15) x (w<=3 | w>=10): the per-selection optima are 49+1=50 (x=5,
    w=3), 49+36=85, 9+1=10 (x=15, w=3) and 9+36=45 (x=15, w=10).

    * ``force=None``: no logic, optimum 10 (x=15, w=3);
    * ``force="implies"``: ``mode_1 -> pump_1`` rules out (15, 3): 45 (x=15, w=10);
    * ``force="low"``: ``mode_0`` (and ``~pump_1``) forces x<=5, w<=3: 50 (x=5, w=3).
    """
    m = dm.Model("logic_ind")
    x = m.continuous("x", lb=0.0, ub=20.0)
    w = m.continuous("w", lb=0.0, ub=20.0)
    mode = m.either_or([[x <= 5.0], [x >= 15.0]], name="mode")
    pump = m.either_or([[w <= 3.0], [w >= 10.0]], name="pump")
    if force == "implies":
        m.logical(mode.indicators[1].implies(pump.indicators[1]), name="mode_implies_pump")
    elif force == "low":
        m.logical(mode.indicators[0] & ~pump.indicators[1], name="low")
    else:
        # Read, so the indicators are columns, but nothing constrains them.
        assert len(mode.indicators) == 2 and len(pump.indicators) == 2
    m.minimize((x - 12.0) ** 2 + (w - 4.0) ** 2)
    return m


_LOGIC_EXPECTED = {None: (10.0, 15.0, 3.0), "implies": (45.0, 15.0, 10.0), "low": (50.0, 5.0, 3.0)}


@pytest.mark.parametrize("force", [None, "implies", "low"])
def test_logic_over_indicators_round_trips(force):
    """Saved, reloaded and solved: the same status, objective and node count as the
    original, and the hand-computed optimum. The reloaded logic leaves are the
    reloaded disjunction's indicator columns, not copies."""
    m = _logic_model(force)
    text = dumps(m, provenance=False)
    m2 = loads(text)
    assert dumps(m2, provenance=False) == text

    mode, pump = _disjunctions(m2)
    cols = {id(v) for d in (mode, pump) for v in d._indicator_vars}
    from discopt.modeling.core import BooleanVar, _LogicalConstraint

    def leaves(node):
        if isinstance(node, BooleanVar):
            yield node.variable
            return
        for attr in ("left", "right", "antecedent", "consequent", "operand"):
            if hasattr(node, attr):
                yield from leaves(getattr(node, attr))
        for op in getattr(node, "operands", ()):
            yield from leaves(op)

    logic = [c for c in m2._constraints if type(c) is _LogicalConstraint]
    assert len(logic) == (0 if force is None else 1)
    seen = [v for c in logic for v in leaves(c.expression)]
    assert all(id(v) in cols for v in seen)
    assert len(seen) == (0 if force is None else 2)

    obj, xs, ws = _LOGIC_EXPECTED[force]
    before = m.solve(time_limit=60)
    after = m2.solve(time_limit=60)
    assert before.status == after.status == "optimal"
    assert before.objective == pytest.approx(obj, abs=1e-5)
    assert after.objective == pytest.approx(before.objective, abs=1e-9)
    assert after.node_count == before.node_count
    assert float(after.x["x"]) == pytest.approx(xs, abs=1e-5)
    assert float(after.x["w"]) == pytest.approx(ws, abs=1e-5)


def test_the_gdp_indicators_logic_fixture_solves_to_45():
    m = loads((FIXTURES / "v1.5" / "gdp_indicators_logic.json").read_text())
    r = m.solve(time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(45.0, abs=1e-5)
    assert float(r.x["w"]) == pytest.approx(10.0, abs=1e-5)
