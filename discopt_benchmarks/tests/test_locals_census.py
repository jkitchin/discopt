"""The locals census must not label a mutable value read-only.

The census (``solve_model_locals_census.py``) is the instrument every item-11
increment reads to decide which locals may enter a *frozen* holder. Its first
revision had three defects, and the worst of them — ``CONFIG`` meaning "the name is
never rebound" rather than "the value is immutable" — propagated a wrong premise
into ``docs/dev/consolidation-plan-2026-07-28.md``, where it survived until a
design review caught it by hand. A holder built on that label would have wrapped
the live B&B tree in a ``frozen=True`` dataclass and offered a guarantee it does
not make (``frozen`` blocks field rebinding, not mutation of the pointed-to value).

So these tests are written against **synthetic sources with known answers**, not
against ``solve_model``: a test that only asserted properties of the real 7,600-line
function would pass just as happily if the visitor stopped firing. Each test names
the defect it pins. The two corpus-level tests at the end are the ones that would
catch a *regression* of the fix on the real source, and both assert a non-zero
executed count so that a traversal of nothing cannot read as a pass (CLAUDE.md §6).
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

from scripts.solve_model_locals_census import (  # noqa: E402
    CARVE_RELEVANT,
    CONFIG,
    EXCEPT_TARGET,
    KNOWN_MUTATORS,
    MUTATED,
    NEEDS_AUDIT,
    STATE,
    Binding,
    Region,
    _ScopeCollector,
    classify,
    crosses,
    run,
)

_SOURCE = _REPO_ROOT / "python" / "discopt" / "solver" / "__init__.py"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _collect(body: str) -> dict[str, Binding]:
    """Run the collector over a function body given as source text."""
    fn = ast.parse(textwrap.dedent(body)).body[0]
    assert isinstance(fn, ast.FunctionDef)
    c = _ScopeCollector()
    for stmt in fn.body:
        c.visit(stmt)
    return c.bindings


def _classify(body: str, name: str) -> str:
    """Classify one name of a synthetic function whose whole body is one region."""
    bindings = _collect(body)
    assert name in bindings, f"{name!r} was never collected — the visitor did not fire"
    region = Region("only", 1, 10_000)
    # A loop region that holds nothing, so `bound_in_loop` cannot fire spuriously.
    empty_loop = Region("loop", 10_001, 10_002)
    return classify(bindings[name], [region], empty_loop)


# --------------------------------------------------------------------------- #
# defect 1 — CONFIG meant "never rebound", not "immutable"
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.unit
def test_subscript_store_is_a_mutation_not_a_load() -> None:
    """``opts["max_wall_time"] = …`` is the real site that was mis-filed.

    The pre-fix ``_bind_target`` walked the Store-context ``Subscript``, found the
    inner ``Name`` in Load context, and recorded a *load* of ``opts``. The name was
    then "bound once, never rebound" and came out ``CONFIG``.
    """
    body = """
    def f(opts):
        opts = dict(opts)
        opts["max_wall_time"] = 1.0
        return opts
    """
    b = _collect(body)["opts"]
    kinds = {s["kind"] for s in b.mutation_sites}
    assert "subscript_store" in kinds, b.mutation_sites
    assert b.proven_mutation_lines(), "a subscript store must be tier-1 evidence"
    assert _classify(body, "opts") == MUTATED


@pytest.mark.smoke
@pytest.mark.unit
def test_attribute_store_is_a_mutation() -> None:
    body = """
    def f():
        holder = Thing()
        holder.spent = 3.0
        return holder.spent
    """
    assert _classify(body, "holder") == MUTATED


@pytest.mark.smoke
@pytest.mark.unit
def test_augmented_subscript_store_is_a_mutation() -> None:
    """``_reduce_timers['fbbt'] += …`` — an AugAssign to a subscript."""
    body = """
    def f():
        timers = {}
        timers["fbbt"] += 1.0
        return timers
    """
    assert _classify(body, "timers") == MUTATED


@pytest.mark.smoke
@pytest.mark.unit
def test_del_of_a_subscript_is_a_mutation_but_del_of_the_name_is_not() -> None:
    body = """
    def f():
        d = {}
        del d["k"]
        return d
    """
    assert _classify(body, "d") == MUTATED

    plain = """
    def f():
        d = {}
        use(d)
        del d
    """
    # `del d` unbinds the name — that is STATE (lifetime ends), not a value mutation.
    assert _classify(plain, "d") == STATE


@pytest.mark.smoke
@pytest.mark.unit
def test_nested_subscript_chain_attributes_to_the_root_name() -> None:
    """``a["k"].b[0] = v`` mutates the object bound to ``a``, not some temporary."""
    body = """
    def f():
        a = build()
        a["k"].b[0] = 7
        return a
    """
    b = _collect(body)["a"]
    assert b.proven_mutation_lines(), b.mutation_sites
    assert _classify(body, "a") == MUTATED


@pytest.mark.smoke
@pytest.mark.unit
def test_known_mutator_call_is_proven_but_an_unknown_call_only_needs_audit() -> None:
    """The instrument must not *guess* in either direction.

    ``.pop()`` is unambiguous, so it is proof. ``.export_batch()`` could be anything,
    so the census refuses to call it CONFIG **and** refuses to call it MUTATED —
    that verdict needs the callee, which is a different instrument's job.
    """
    known = """
    def f(kwargs):
        opt = kwargs.pop("x", None)
        return opt, kwargs
    """
    assert _classify(known, "kwargs") == MUTATED

    unknown = """
    def f():
        tree = make_tree()
        tree.export_batch(8)
        return tree
    """
    assert _classify(unknown, "tree") == NEEDS_AUDIT
    assert "export_batch" not in KNOWN_MUTATORS


@pytest.mark.smoke
@pytest.mark.unit
def test_a_name_only_read_and_passed_is_still_CONFIG() -> None:  # noqa: N802
    """The fix must not collapse into "everything is mutable"."""
    body = """
    def f():
        n_vars = len(cols)
        helper(n_vars)
        other(n_vars + 1)
        return n_vars
    """
    assert _classify(body, "n_vars") == CONFIG


# --------------------------------------------------------------------------- #
# defect 2 — except targets cannot cross a region
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.unit
def test_except_target_is_its_own_class_and_binds_no_store() -> None:
    """PEP 3110 deletes the target at handler exit, so it cannot be live later."""
    body = """
    def f():
        try:
            go()
        except ValueError as exc:
            log(exc)
    """
    b = _collect(body)["exc"]
    assert b.except_lines, "the except binding was not recorded at all"
    assert not b.store_lines, "an except target must not count as an ordinary store"
    assert _classify(body, "exc") == EXCEPT_TARGET


@pytest.mark.smoke
@pytest.mark.unit
def test_a_name_that_is_both_assigned_and_an_except_target_is_not_downgraded() -> None:
    """Only *purely* except-bound names get the exemption."""
    body = """
    def f():
        exc = None
        try:
            go()
        except ValueError as exc:
            pass
        return exc
    """
    assert _classify(body, "exc") != EXCEPT_TARGET


@pytest.mark.smoke
@pytest.mark.unit
def test_except_target_of_a_nested_scope_is_not_a_free_variable() -> None:
    """A nested ``def``'s own except target must not be attributed to the parent."""
    body = """
    def f():
        def inner():
            try:
                go()
            except ValueError as e:
                return e
            return None
        return inner
    """
    bindings = _collect(body)
    # `e` is bound by `inner`, so it is not a local of `f` at all.
    assert "e" not in bindings or not bindings["e"].store_lines


# --------------------------------------------------------------------------- #
# defect 3 — one crossing predicate, not two
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.unit
def test_crosses_is_symmetric_in_its_inputs_and_ignores_ordering() -> None:
    """The pre-fix field compared an alphabetically sorted set against an
    ordered list, so ``['root','loop']`` vs ``['loop','root']`` decided the answer.
    """
    assert crosses(["root"], ["root"], []) is False
    assert crosses(["root"], ["loop"], []) is True
    assert crosses(["root", "loop"], ["loop", "root"], []) is True
    # a never-stored parameter read in exactly one region does not "cross"
    assert crosses([], ["setup"], []) is False
    # a mutation in another region does cross, even with no rebinding anywhere
    assert crosses(["root"], ["root"], ["loop"]) is True


@pytest.mark.smoke
@pytest.mark.unit
def test_json_field_and_printed_predicate_agree_on_the_real_source() -> None:
    """The defect was a *disagreement*, so the regression test is agreement."""
    census = run(_SOURCE, "solve_model")
    checked = 0
    for row in census["rows"]:
        expected = crosses(row["store_regions"], row["load_regions"], row["mutation_regions"])
        assert row["crosses_regions"] is expected, row["name"]
        checked += 1
    assert checked > 0, "the census produced no rows — it traversed nothing"
    print(f"executed crossing-predicate comparisons: {checked}")


# --------------------------------------------------------------------------- #
# corpus-level: the fix must still hold on the real source
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.unit
def test_the_seven_reviewed_mutables_are_never_CONFIG_on_the_real_source() -> None:  # noqa: N802
    """The design review named these by hand; the instrument must now find them.

    If a later edit removes one of these locals the test says so loudly rather than
    passing vacuously on an empty set — that is the ``if x is None: continue``
    failure mode CLAUDE.md §6 exists to prevent.
    """
    reviewed = {
        "tree",
        "evaluator",
        "opts",
        "kwargs",
        "_adaptive_nlp_state",
        "_heuristic_governor",
        "_reduce_timers",
    }
    census = run(_SOURCE, "solve_model")
    rows = {r["name"]: r for r in census["rows"]}
    missing = reviewed - set(rows)
    assert not missing, f"locals vanished from solve_model; re-audit rather than delete: {missing}"
    checked = 0
    for name in sorted(reviewed):
        row = rows[name]
        assert row["kind"] in (MUTATED, NEEDS_AUDIT), (
            f"{name} is classified {row['kind']}; the design review established it is "
            "mutated in place and it must never be admitted to a frozen holder"
        )
        assert row["mutation_sites"], f"{name} carries no mutation evidence"
        checked += 1
    assert checked == len(reviewed)
    print(f"executed mutability assertions: {checked}")


@pytest.mark.smoke
@pytest.mark.unit
def test_census_records_mutation_evidence_and_excludes_except_targets() -> None:
    census = run(_SOURCE, "solve_model")
    assert census["classified"] > 0
    assert census["mutation_evidence_sites"] > 0, "the mutation visitor never fired"
    rows = census["rows"]
    excepts = [r for r in rows if r["kind"] == EXCEPT_TARGET]
    assert excepts, "solve_model has `except ... as` handlers; none were classified"
    for r in excepts:
        assert r["stores"] == 0, r["name"]
        assert r["excepts"] > 0, r["name"]
        # `crosses_regions` may still be True for such a row — a name reused as the
        # target of handlers in two regions has loads in both. What must hold is
        # that it never enters the carve-relevant population, because each binding
        # dies at its own handler's exit and so nothing has to be passed across.
        assert r["kind"] not in CARVE_RELEVANT, r["name"]
    print(
        f"executed: {census['classified']} classifications, "
        f"{census['mutation_evidence_sites']} mutation sites, {len(excepts)} except targets"
    )
