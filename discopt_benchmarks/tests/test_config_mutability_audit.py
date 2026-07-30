"""The mutability audit must never clear a name it cannot actually clear.

This instrument decides which of ``solve_model``'s config crossers may enter a
``frozen=True`` holder. Its failure mode is not "wrong number" — it is a
plausible-looking ``CLEAN`` on a name that some callee mutates, which becomes a
guarantee the code does not make. The 2026-07-30 design review found exactly that
shape one level up, in the census; the audit exists so the *transitive* level
cannot repeat it.

Three real defects found while building it are pinned here, because each one
produced a confident wrong answer rather than an error:

* ``HeuristicGovernor.record`` mutates through ``st = self._get(source)``, so a
  probe looking only for ``self.x = …`` cleared the governor.
* A first taint rule propagated through *any* expression mentioning a name, so
  ``gap_tolerance`` — a ``float`` — came back mutated.
* The next rule propagated through ``d[k] = tainted``, which is the tainted object
  escaping *into* ``d``, not ``d`` becoming an alias of it.

Every corpus-level test asserts a non-zero executed count (CLAUDE.md §6).
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent

if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from scripts.solve_model_config_mutability_audit import (  # noqa: E402
    HAND_ADJUDICATED,
    IMMUTABLE_TYPE_NAMES,
    MUTATING_CALLABLES,
    NON_MUTATING_BUILTINS,
    V_CLEAN,
    V_DIRECT,
    V_IMMUTABLE,
    V_METHOD,
    V_TRANSITIVE,
    _alias_roots,
    _annotation_is_immutable,
    audit,
    immutable_locals,
    taint_closure,
)

_SOURCE = _REPO_ROOT / "python" / "discopt" / "solver" / "__init__.py"


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(src)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


# --------------------------------------------------------------------------- #
# aliasing: what may and may not carry a mutation back to the caller
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.unit
def test_alias_roots_follows_references_but_not_computation() -> None:
    assert _alias_roots(_expr("obj")) == {"obj"}
    assert _alias_roots(_expr("obj.rows")) == {"obj"}
    assert _alias_roots(_expr("obj[3].col")) == {"obj"}
    assert _alias_roots(_expr("obj.method()")) == {"obj"}, "a method may return internal state"
    # arithmetic builds a NEW object — this is why `time_limit` is not "mutated"
    assert _alias_roots(_expr("t_start + time_limit")) == set()
    assert _alias_roots(_expr("f(x)")) == set(), "a free call is the transitive channel's job"
    assert _alias_roots(_expr("3.0")) == set()


@pytest.mark.smoke
@pytest.mark.unit
def test_taint_reaches_through_a_getter_but_not_into_a_container() -> None:
    """Both halves of the rule, on the two real shapes that motivated it."""
    getter = _fn("""
        def record(self, source):
            st = self._get(source)
            st.calls += 1
    """)
    assert "st" in taint_closure(getter, "self"), (
        "`st = self._get(...)` aliases part of self; missing this cleared the "
        "heuristic governor, which throttles primal heuristics per node"
    )

    escape = _fn("""
        def build(gap_tolerance):
            amp_kwargs = {}
            amp_kwargs["rel_gap"] = gap_tolerance
            return amp_kwargs
    """)
    assert "amp_kwargs" not in taint_closure(escape, "gap_tolerance"), (
        "storing a value INTO a dict does not make the dict an alias of the value"
    )


@pytest.mark.smoke
@pytest.mark.unit
def test_taint_closure_always_contains_its_root() -> None:
    assert taint_closure(_fn("def f(a):\n    return a"), "a") == {"a"}


# --------------------------------------------------------------------------- #
# the immutable-type channel
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.unit
def test_scalar_annotations_are_immutable_and_containers_are_not() -> None:
    assert _annotation_is_immutable(_expr("float"))
    assert _annotation_is_immutable(_expr("int"))
    assert _annotation_is_immutable(_expr("Optional[str]"))
    assert _annotation_is_immutable(_expr("Union[bool, str]"))
    assert _annotation_is_immutable(_expr("str | None"))
    assert not _annotation_is_immutable(_expr("Optional[dict]"))
    assert not _annotation_is_immutable(_expr("np.ndarray"))
    assert not _annotation_is_immutable(_expr("list[int]"))


@pytest.mark.smoke
@pytest.mark.unit
def test_a_bare_None_annotation_is_not_treated_as_immutable() -> None:  # noqa: N802
    """Four of ``solve_model``'s parameters are annotated ``None`` while holding
    callables and lists. Reading that as "NoneType, therefore immutable" would be
    a false clear, so the audit refuses it."""
    assert not _annotation_is_immutable(_expr("None"))
    assert "None" not in IMMUTABLE_TYPE_NAMES
    assert "NoneType" not in IMMUTABLE_TYPE_NAMES


@pytest.mark.smoke
@pytest.mark.unit
def test_derived_scalars_are_proved_and_containers_are_not() -> None:
    fn = _fn("""
        def f(time_limit: float, opts: dict):
            t_start = time.perf_counter()
            n_vars = len(cols)
            deadline = t_start + time_limit
            rows = opts["rows"]
            return deadline, n_vars, rows
    """)
    known = immutable_locals(fn)
    assert {"time_limit", "t_start", "n_vars", "deadline"} <= known
    assert "opts" not in known
    assert "rows" not in known


# --------------------------------------------------------------------------- #
# the curated sets must not overlap or drift into guessing
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.unit
def test_mutating_and_non_mutating_callable_sets_are_disjoint() -> None:
    overlap = MUTATING_CALLABLES & NON_MUTATING_BUILTINS
    assert not overlap, f"a callable cannot be both: {overlap}"
    assert "setattr" in MUTATING_CALLABLES, "setattr mutates; it must never be filtered as noise"
    assert "setattr" not in NON_MUTATING_BUILTINS


# --------------------------------------------------------------------------- #
# corpus level — the real source
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.unit
def test_the_reviewed_mutables_are_all_caught_and_none_come_back_clean() -> None:
    reviewed = {
        "tree",
        "evaluator",
        "opts",
        "kwargs",
        "_adaptive_nlp_state",
        "_heuristic_governor",
        "_reduce_timers",
    }
    rep = audit(_SOURCE, "solve_model")
    rows = {r["name"]: r for r in rep["rows"]}
    missing = reviewed - set(rows)
    assert not missing, f"names left the audit population; re-audit rather than assume: {missing}"
    checked = 0
    for name in sorted(reviewed):
        verdict = rows[name]["verdict"]
        assert verdict in (V_DIRECT, V_METHOD, V_TRANSITIVE), (
            f"{name} audited as {verdict}; the design review established it is mutated"
        )
        checked += 1
    assert checked == len(reviewed)
    print(f"executed verdict assertions: {checked}")


@pytest.mark.smoke
@pytest.mark.unit
def test_evaluator_is_caught_only_by_transitive_callee_analysis() -> None:
    """The headline finding: the one name no call-site inspection could catch.

    ``solve_model`` only ever calls ``evaluator.evaluate_objective(...)``. The
    write is ``evaluator._structural_linear_mask_cache = …`` inside a module-level
    helper the evaluator is *passed to*. This is the case the review named as
    unchecked, so it gets its own test rather than riding on a count.
    """
    rep = audit(_SOURCE, "solve_model")
    rows = {r["name"]: r for r in rep["rows"]}
    assert "evaluator" in rows
    assert rows["evaluator"]["verdict"] == V_TRANSITIVE
    assert rows["evaluator"]["caught_only_transitively"] is True
    chains = rows["evaluator"]["transitive"]
    assert chains, "no transitive evidence recorded — the channel did not fire"
    assert any("_structural_linear_mask_cache" in m["detail"] for m in chains), chains
    print(f"executed transitive-chain assertions: {len(chains)}")


@pytest.mark.smoke
@pytest.mark.unit
def test_every_audit_channel_fired_and_the_population_is_fully_classified() -> None:
    rep = audit(_SOURCE, "solve_model")
    zero = [k for k, v in rep["executed"].items() if v == 0]
    assert not zero, f"these channels executed nothing: {zero}"
    assert rep["population"] > 0
    assert sum(rep["counts"].values()) == rep["population"], "a name got no verdict"
    print(f"executed: {rep['executed']}")


@pytest.mark.smoke
@pytest.mark.unit
def test_hand_adjudications_do_not_override_the_machine_verdict() -> None:
    """A human claim must stay visibly a human claim.

    Folding ``HAND_ADJUDICATED`` into the JSON verdict would launder an argument
    into a measurement — and the next reader would have no way to tell which is
    which. The names stay ``UNRESOLVED`` in the data; the reasoning is published
    beside it.
    """
    rep = audit(_SOURCE, "solve_model")
    rows = {r["name"]: r for r in rep["rows"]}
    checked = 0
    for name in HAND_ADJUDICATED:
        assert name in rows, f"{name} is hand-adjudicated but not in the population"
        assert rows[name]["verdict"] not in (V_CLEAN, V_IMMUTABLE), (
            f"{name} is recorded as hand-adjudicated, but the automated verdict is "
            f"{rows[name]['verdict']} — remove the hand entry, it is now redundant"
        )
        checked += 1
    assert checked == len(HAND_ADJUDICATED)
    print(f"executed hand-adjudication assertions: {checked}")
