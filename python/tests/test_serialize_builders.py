"""Every model-producing builder round-trips, or is a tested refusal (WP-6, D12).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-6. The cases live in
``serialize_builder_cases.py`` (shared with ``scripts/serialize_builder_sweep.py``,
which prints the table): one small model per builder -- DAE, frozen and trainable
ML, GDP, stochastic, robust, bilevel, GP and every registered transformation.

* A round-trip case must be a ``dumps`` fixed point with an equal
  ``model_fingerprint``, through ``loads`` and through ``pickle``.
* A refused case must raise ``SerializationError`` with the message the case
  names, and the remedy that message gives must work: ``pickle`` keeps the model;
  "save the source and re-apply" gives a fingerprint-equal model; an unattached
  GDP block, attached, round-trips.
* Where the solve finishes (not by a time limit) in under 5 s, the reload solves
  to the identical ``(status, node_count, objective)`` (slow).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import discopt.modeling as dm
import discopt.transformations as dt
import pytest
import serialize_builder_cases as cases
from discopt.modeling.core import Variable, _DisjunctiveConstraint
from discopt.serialize import SerializationError, dumps, loads
from discopt.transformations import model_fingerprint


def _build(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _loads(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return loads(text)


@pytest.mark.parametrize("key", sorted(cases.ROUND_TRIPS))
def test_builder_round_trips(key):
    model = _build(cases.ROUND_TRIPS[key])
    text = dumps(model, provenance=False)
    reloaded = _loads(text)
    assert dumps(reloaded, provenance=False) == text
    fp = model_fingerprint(model)
    assert model_fingerprint(reloaded) == fp
    assert model_fingerprint(pickle.loads(pickle.dumps(model))) == fp


def _round_trip_checks(model) -> int:
    text = dumps(model, provenance=False)
    reloaded = _loads(text)
    assert dumps(reloaded, provenance=False) == text
    fp = model_fingerprint(model)
    assert model_fingerprint(reloaded) == fp
    assert model_fingerprint(pickle.loads(pickle.dumps(model))) == fp
    return 3


#: torch and cyipopt each link their own OpenMP runtime; importing torch into a
#: process that already loaded cyipopt aborts it ("OMP: Error #15", measured with
#: torch 2.7.1 / cyipopt 1.7.0 on macOS), which takes an xdist worker down with
#: it. A torch case therefore runs in a fresh interpreter that imports torch first.
_FRESH_TORCH = """
import sys, warnings
import torch  # first: before anything that loads cyipopt's OpenMP
sys.path.insert(0, sys.argv[1])
import serialize_builder_cases as cases, test_serialize_builders as t
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = cases.READERS[sys.argv[2]][1]()
print("checks", t._round_trip_checks(model))
"""


@pytest.mark.parametrize("key", sorted(cases.READERS))
def test_reader_round_trips(key):
    modules, build = cases.READERS[key]
    for mod in modules:
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} not installed")
    if "torch" not in modules:
        assert _round_trip_checks(_build(build)) == 3
        return
    here = str(Path(__file__).resolve().parent)
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH_TORCH, here, key], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.split()[-2:] == ["checks", "3"], proc.stdout


@pytest.mark.parametrize("key", sorted(cases.REFUSED))
def test_builder_is_refused_with_remedy(key):
    case = cases.REFUSED[key]
    model = _build(case.build)
    with pytest.raises(SerializationError, match=case.match):
        dumps(model)
    if case.remedy == "pickle":
        # "Use pickle or Model.clone() for same-version transport."
        fp = model_fingerprint(model)
        assert model_fingerprint(pickle.loads(pickle.dumps(model))) == fp
        assert model_fingerprint(model.clone()) == fp
    elif isinstance(case.remedy, cases.Remedy):
        # "Save the source model instead, and re-apply the transformation after loading it."
        source = _build(case.remedy.source)
        restored = _build(lambda: case.remedy.apply(_loads(dumps(source))))
        direct = _build(lambda: case.remedy.apply(case.remedy.source()))
        assert model_fingerprint(restored) == model_fingerprint(direct)
    else:
        # #1430: the block attached to the model (what the message asks for) writes.
        attached = _build(case.remedy)
        assert dumps(_loads(dumps(attached)), provenance=False) == dumps(attached, provenance=False)


def test_builder_sweep_probe_fired():
    """§6: the parametrized tests above measure nothing over an empty table."""
    # Pinned, not a floor: a case that disappears must be noticed (#1797 review N3).
    assert (len(cases.ROUND_TRIPS), len(cases.REFUSED), len(cases.READERS)) == (50, 7, 4)
    areas = {key.split(":")[0] for key in (*cases.ROUND_TRIPS, *cases.REFUSED, *cases.READERS)}
    assert areas == {"dae", "ml", "gdp", "stochastic", "ro", "bilevel", "gp", "transform"}


def test_every_transformation_is_swept():
    """A pass added to the registry without a sweep case fails here."""
    assert set(dt.available()) <= cases.SWEPT_TRANSFORMATIONS, sorted(
        set(dt.available()) - cases.SWEPT_TRANSFORMATIONS
    )


# ── the crashes the sweep found (each fails on the pre-WP-6 tree) ──────────


def _leaf_variables(expr, out):
    if isinstance(expr, Variable):
        out.add(id(expr))
        return
    for attr in ("left", "right", "operand", "base", "args"):
        value = getattr(expr, attr, None)
        for child in value if isinstance(value, (list, tuple)) else [value]:
            if child is not None:
                _leaf_variables(child, out)


@pytest.mark.parametrize(
    "copier",
    [lambda m: pickle.loads(pickle.dumps(m)), copy.deepcopy, lambda m: m.clone()],
    ids=["pickle", "deepcopy", "clone"],
)
def test_a_nested_disjunction_copies(copier):
    """A ``Model.disjunction()`` nested in ``either_or`` crashed pickle, deepcopy
    and clone: the expression-root walk read ``.body`` off the nested disjunction.
    The copy must also own its nested rows (point at its own variable)."""
    model = cases.gdp_nested_disjunction()
    copied = copier(model)
    assert model_fingerprint(copied) == model_fingerprint(model)
    inner = copied._constraints[0].disjuncts[0][0]
    assert isinstance(inner, _DisjunctiveConstraint)
    leaves: set = set()
    for disjunct in inner.disjuncts:
        for row in disjunct:
            _leaf_variables(row.body, leaves)
    assert leaves == {id(copied._variables[0])} != {id(model._variables[0])}


def test_a_nested_disjunction_keeps_its_or_through_a_round_trip():
    """1.6 writes a ``Model.disjunction()`` nested in ``either_or`` as a member
    (before: refused, WP-6). The outer alternative is infeasible (x <= 20), so the
    optimum is the inner OR's best arm, x = 7; flattened into an AND of the inner
    rows the model would be infeasible."""
    m = dm.Model("nested_or")
    x = m.continuous("x", lb=0.0, ub=20.0)
    m.maximize(x)
    inner = m.disjunction([[x >= 1, x <= 3], [x >= 5, x <= 7]], name="inner")
    m.either_or([[inner], [x >= 30]], name="outer")
    text = dumps(m, provenance=False)
    member = json.loads(text)["rows"][0]["disjuncts"][0][0]
    assert set(member) == {"disjunction"}
    reloaded = loads(text)
    assert dumps(reloaded, provenance=False) == text
    r = reloaded.solve(time_limit=30)
    assert r.status == "optimal" and abs(r.objective - 7.0) < 1e-6


@pytest.mark.parametrize("build", [cases.gdp_unattached_disjunction, cases.gdp_unattached_disjunct])
def test_an_unattached_gdp_block_is_a_serialization_error(build):
    """#1430's guard raised a bare ``ValueError`` from ``dumps``; it is now this
    module's error (a ``ValueError`` subclass, so ``except ValueError`` still works)."""
    with pytest.raises(SerializationError, match="never added to it"):
        dumps(build())


def test_transform_current_disjunctive_state_names_its_remedy():
    """Its undo tokens were an unclassified attribute: the refusal told the user to
    edit discopt. They are the pass's record of its source model (plan §3.6)."""
    model = cases.REFUSED["transform:gdp.transform_current_disjunctive_state"].build()
    with pytest.raises(SerializationError, match="Save the source model instead") as err:
        dumps(model)
    assert "Classify each" not in str(err.value)


# ── the solve is bound-neutral across the round trip (CLAUDE.md §5) ────────

#: Rule: every round-trip case whose solve finishes on its own (not by a time
#: limit) in under 5 s. Measured by ``scripts/serialize_builder_sweep.py --solve``
#: at WP-6 entry; excluded, with the reason:
NOT_SOLVED_HERE = {
    "dae:fit_trajectories": "stops at the time limit (30 s)",
    "ml:full_space": "16.6 s",
    "ml:reduced_space (sigmoid)": "19.4 s",
    "ml:frozen network": "ends 'feasible', not a terminal status",
    "bilevel:strong_duality": "34 s, ends 'feasible'",
    "ml:trainable network": "stops at the time limit (30 s), ends 'feasible'",
}


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(set(cases.ROUND_TRIPS) - set(NOT_SOLVED_HERE)))
def test_builder_round_trip_is_bound_neutral(key):
    model = _build(cases.ROUND_TRIPS[key])
    reloaded = _loads(dumps(model))

    def outcome(m):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = m.solve(time_limit=30)
        except NotImplementedError as exc:  # a semantics the solver declines (#1124)
            return ("raised", str(exc))
        return (r.status, r.node_count, r.objective)

    first = outcome(model)
    assert first[0] in {"optimal", "infeasible", "raised"}, first
    assert outcome(reloaded) == first


def test_excluded_solves_are_real_cases():
    assert set(NOT_SOLVED_HERE) <= set(cases.ROUND_TRIPS)


def test_the_bound_neutral_panel_covers_every_area():
    solved = set(cases.ROUND_TRIPS) - set(NOT_SOLVED_HERE)
    assert {k.split(":")[0] for k in solved} == {
        "dae", "ml", "gdp", "stochastic", "ro", "bilevel", "gp", "transform",
    }  # fmt: skip


# ── #1797 review follow-ups ────────────────────────────────────────────────


def test_if_then_refuses_a_disjunction_member_when_it_is_built():
    """S1: ``if_then(y, [m.disjunction(...)])`` stored the disjunction as the row;
    pickle, deepcopy and clone then crashed reading its ``.body`` (the C1 crash
    class), and solve reported the block as never added. Refused at the call."""
    import discopt.modeling as dm

    m = dm.Model("if_then_nested")
    x = m.continuous("x", lb=0, ub=20)
    y = m.binary("y")
    m.maximize(x)
    inner = m.disjunction([[x >= 1, x <= 3], [x >= 5, x <= 7]], name="inner")
    n_before = len(m._constraints)
    with pytest.raises(TypeError, match=r"then_constraints\[1\] is a _DisjunctiveConstraint"):
        m.if_then(y, [x <= 10, inner])
    assert len(m._constraints) == n_before  # nothing recorded by the refused call
    m.either_or([[inner], [x >= 15, x <= 16]], name="outer")  # the block is still usable
    pickle.loads(pickle.dumps(m))


def test_a_reversed_disjunctive_state_serializes_again():
    """N1: after the last ``reverse_current_disjunctive_state`` the model is its
    source again, but the empty token list left behind kept ``dumps`` refusing it."""
    from discopt.gdp.state import reverse_current_disjunctive_state

    source = cases.gdp_disjunct_blocks()
    text = dumps(source, provenance=False)
    model = cases.gdp_disjunct_blocks()
    dt.apply_to(
        "gdp.transform_current_disjunctive_state",
        model,
        selected=cases._selected_second_disjunct(model),
    )
    with pytest.raises(SerializationError, match="Save the source model instead"):
        dumps(model)
    reverse_current_disjunctive_state(model)
    assert "_gdp_state_tokens" not in vars(model)
    assert dumps(model, provenance=False) == text
