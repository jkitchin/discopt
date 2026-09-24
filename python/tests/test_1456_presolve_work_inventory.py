"""#1456: the enforced inventory of pre-solve passes that nothing bounds.

The defect, and why it is a third thing
---------------------------------------

A ``time_limit`` cannot reach a pass that runs *before* branch and bound starts.
``solve_model`` runs a long sequence of optional structure detection and
reformulation passes first, and a pass with no bound of its own runs to
completion however long that takes. Measured: ``johnall`` returned **44+ minutes**
after being given ``time_limit=20``, having never explored a node.

#912 established the discipline for this layer, and named two roles for a clock:

===== ============================================== ==========================
role   what decides how much work runs                status
===== ============================================== ==========================
1      the caller's ``time_limit`` — *when to stop*   correct by contract
2      a component-local **clock** — *how much work*  #912: converted, or
                                                      recorded in
                                                      ``test_912_wall_budget_inventory.py``
**3**  **nothing at all**                             **this file**
===== ============================================== ==========================

Role 3 is strictly worse than role 2: a role-2 gate at least terminates.

Why this is a work count and not the deadline #1456 asked for
--------------------------------------------------------------

#1456's "what done means" asked for a wall-clock deadline threaded into these
passes. That prescription was retracted (see the issue's second revision). A
deadline decides *how much structure a pass recognises*, so under it the same
model on a busy machine gets a different relaxation, a different bound and a
different node count — which makes CLAUDE.md §5's bound-neutral regime
(``node_count`` and certified ``objective`` **exactly** unchanged) permanently
unenforceable on exactly the instances this issue is about.

That is not a judgement invented here. The sibling pass in the very same
function already follows the rule: ``dependent_vars.find_functionally_dependent_names``
carries ``_SCAN_WORK_BUDGET``, an operation count, and ``_relax/dependent_vars.py``
says why in as many words — *"bounding it with a clock would make the search tree
a function of machine speed, the exact defect #912 exists to prevent"*.
``discopt._work_budget.WorkBudget`` is the shared primitive #912 built for it.

What this file does
-------------------

Leaf-fixing is what #1456 itself calls whack-a-mole. #1455 and #1458 bound the
three *measured* leaves (``johnall``, ``hadamard_9``, ``glider400``) and that
still left the class open — measured, not assumed: four optional structure
passes in this region contain **zero** occurrences of ``budget``, ``BUDGET``,
``WorkCounter``, ``WorkBudget`` or ``deadline``.

So: :func:`_scan` parses ``solver.py``, finds ``solve_model``, derives the end of
the pre-solve region as the first call into a terminal solve route, and collects
every ``discopt`` function called before it. :data:`INVENTORY` must list exactly
that set, each row carrying a category. A newly added unbounded pre-solve pass
fails :func:`test_no_unrecorded_presolve_pass` — which is the whole point, because
the alternative way to discover one is a user waiting 44 minutes for a 20-second
solve.

Categories
----------

``terminal``
    This callee **is** the solve. It receives ``time_limit`` and answers "when do
    we stop" — role 1, correct by definition, deliberately not bounded here.
``bounded``
    Carries a deterministic work budget. The row names the symbol, and
    :func:`test_bounded_rows_name_a_live_budget` imports the module and asserts it
    still exists — so deleting a budget breaks this file rather than going quiet.
``wall``
    Carries a wall-clock gate. Not role 3, so not this issue's business; it is
    role 2 and belongs to the #912 inventory, which is where it is recorded.
    Listed here only so the row is accounted for.
``trivial``
    Cost is linear (or near-linear) in model size by construction — no symbolic
    expansion, no per-node eigenproblem, no unmemoized DAG re-walk. A model big
    enough to make one of these slow is a model too big to build. Cache clears,
    predicate reads, and one-pass rewrites live here.
``residual``
    A genuine role-3 pass: it does real structural work and nothing bounds it.
    **This is a backlog, not a resting point** — the opposite of #912's
    ``residual``, which records a decision. :func:`test_residual_count_is_visible`
    publishes the count so that shrinking it is a visible act.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SOLVER = Path(__file__).resolve().parents[1] / "discopt" / "solver.py"

# The first call to one of these ends the pre-solve region: they ARE the solve
# and they receive ``time_limit``. Derived rather than hardcoded as a line
# number so the region tracks edits to ``solve_model`` instead of silently
# drifting to cover the wrong code.
_TERMINAL_ENTRIES = ("_solve_continuous", "_solve_milp_bb", "_solve_spatial_bb")


def _called_name(call: ast.Call) -> str | None:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def _scan() -> tuple[int, int, set[tuple[str, str]]]:
    """Return ``(start, end, {(module, callee)})`` for ``solve_model``'s pre-solve
    region.

    Resolution is via the ``from discopt.X import y`` statements inside the
    region itself — these passes are imported locally, at the point of use, so
    the import site is the authority on which module a bare name came from.
    """
    tree = ast.parse(_SOLVER.read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solve_model"
    )
    ends = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and _called_name(n) in _TERMINAL_ENTRIES
    ]
    assert ends, (
        "no terminal solve entry found in solve_model — the region boundary "
        "cannot be derived, so this scanner is measuring nothing (rule 6). "
        f"Looked for {_TERMINAL_ENTRIES}."
    )
    end = min(ends)

    origin: dict[str, str] = {}
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.ImportFrom)
            and n.lineno < end
            and (n.module or "").startswith("discopt")
        ):
            for a in n.names:
                origin.setdefault(a.asname or a.name, n.module)

    found: set[tuple[str, str]] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and n.lineno < end:
            nm = _called_name(n)
            if nm in origin:
                found.add((origin[nm], nm))
    return fn.lineno, end, found


# Deterministic work budgets a ``bounded`` row may name. Keyed by the symbol as
# written in the row; the value is the module that must still define it.
_BUDGET_HOME = {
    "_SCAN_WORK_BUDGET": "discopt._relax.dependent_vars",
    "_MAX_EXPAND_OPS": "discopt._relax.binary_multilinear_reform",
    "_MAX_MONOMIALS": "discopt._relax.binary_multilinear_reform",
    "_DISTRIBUTE_TERM_BUDGET": "discopt._relax.term_classifier",
    "_FBBT_MAX_ITER": "discopt._relax.disjunctive_config_bound",
}

_CATEGORIES = ("terminal", "bounded", "wall", "trivial", "residual")

# (module, callee, category, note). ``note`` names the budget for ``bounded``
# rows and carries the justification otherwise.
INVENTORY: tuple[tuple[str, str, str, str], ...] = (
    # --- terminal: IS the solve, receives time_limit (role 1) ----------------
    ("discopt._relax.convexity.signomial_global", "solve_signomial_global", "terminal", ""),
    ("discopt._relax.lp_spatial_bb", "solve_lp_spatial_bb", "terminal", ""),
    ("discopt.decomposition.benders", "solve_benders", "terminal", ""),
    ("discopt.decomposition.lagrangian", "solve_lagrangian", "terminal", ""),
    ("discopt.gp", "solve_gp", "terminal", ""),
    ("discopt.gp", "solve_gp_minlp", "terminal", ""),
    ("discopt.solvers.amp", "solve_amp", "terminal", ""),
    ("discopt.solvers.direct", "solve_direct", "terminal", ""),
    ("discopt.solvers.gdpopt_loa", "solve_gdpopt_loa", "terminal", ""),
    ("discopt.solvers.mip_nlp", "solve_mip_nlp", "terminal", ""),
    ("discopt.solvers.surrogate", "solve_surrogate", "terminal", ""),
    # --- bounded: carries a deterministic work budget -------------------------
    (
        "discopt._relax.binary_multilinear_reform",
        "has_binary_multilinear_work",
        "bounded",
        "_MAX_EXPAND_OPS,_MAX_MONOMIALS",
    ),
    (
        "discopt._relax.binary_multilinear_reform",
        "reformulate_binary_multilinear",
        "bounded",
        "_MAX_EXPAND_OPS,_MAX_MONOMIALS",
    ),
    (
        "discopt._relax.dependent_vars",
        "find_functionally_dependent_names",
        "bounded",
        "_SCAN_WORK_BUDGET",
    ),
    (
        "discopt._relax.disjunctive_config_bound",
        "compute_disjunctive_config_bound",
        "bounded",
        "_FBBT_MAX_ITER",
    ),
    (
        "discopt._relax.factorable_reform",
        "factorable_reformulate",
        "bounded",
        "_DISTRIBUTE_TERM_BUDGET",
    ),
    (
        "discopt._relax.factorable_reform",
        "has_factorable_work",
        "bounded",
        "_DISTRIBUTE_TERM_BUDGET",
    ),
    (
        "discopt._relax.term_classifier",
        "classify_nonlinear_terms",
        "bounded",
        "_DISTRIBUTE_TERM_BUDGET",
    ),
    # --- wall: role 2, recorded in the #912 inventory -------------------------
    ("discopt._relax.nonlinear_bound_tightening", "tighten_nonlinear_bounds", "wall", "NBT"),
    ("discopt._relax.presolve_pipeline", "run_reverse_ad_tightening", "wall", "reverse-AD"),
    ("discopt._relax.presolve_pipeline", "run_root_presolve", "wall", "OBBT"),
    ("discopt.solvers._root_presolve", "tighten_root_bounds_with_fbbt", "wall", "root FBBT"),
    # --- trivial: linear in model size by construction ------------------------
    (
        "discopt._relax.binary_multilinear_reform",
        "extend_initial_point",
        "trivial",
        "one pass over x0",
    ),
    (
        "discopt._relax.binary_multilinear_reform",
        "heuristic_incumbent",
        "trivial",
        "one bit assignment",
    ),
    ("discopt._relax.convexity.eigenvalue_arith", "QuadraticForm", "trivial", "constructor"),
    ("discopt._relax.convexity.g_convex_inject", "g_convex_cuts_enabled", "trivial", "env read"),
    ("discopt._relax.convexity.patterns", "clear_declared_box_cache", "trivial", "cache clear"),
    ("discopt._relax.integer_product_reform", "_iml_extend", "trivial", "one pass over x0"),
    ("discopt._relax.integer_product_reform", "_ipx_extend", "trivial", "one pass over x0"),
    ("discopt._relax.learned_relaxations", "load_pretrained_registry", "trivial", "file load"),
    ("discopt._relax.model_utils", "_dcb_flat", "trivial", "flat index map"),
    (
        "discopt._relax.nonlinear_bound_tightening",
        "FunctionDomainBoundRule",
        "trivial",
        "constructor",
    ),
    (
        "discopt._relax.nonlinear_bound_tightening",
        "PeriodicVariableBoundRule",
        "trivial",
        "constructor",
    ),
    ("discopt._relax.problem_classifier", "extract_qp_data", "trivial", "memory-budgeted, #863"),
    ("discopt._relax.uniform_relax", "clear_analysis_cache", "trivial", "cache clear"),
    ("discopt._rust", "model_to_repr", "trivial", "marshal, linear"),
    ("discopt.decomposition.learning", "record_outcome", "trivial", "record append"),
    ("discopt.decomposition.learning.store", "RecordStore", "trivial", "constructor"),
    ("discopt.mpec", "require_all_relations_lowered", "trivial", "predicate over rows"),
    ("discopt.solvers._root_presolve", "coef_tighten_enabled", "trivial", "env read"),
    ("discopt.solvers.amp", "_apply_flat_bounds_to_model", "trivial", "one pass over bounds"),
    ("discopt.warm_start", "_prepare_warm_start", "trivial", "one pass over x0"),
    ("discopt.warm_start", "prepare_warm_start", "trivial", "one pass over x0"),
    # --- residual: role 3, real structural work, nothing bounds it ------------
    ("discopt._relax.convexity.eigenvalue_arith", "quadratic_form_bound", "residual", ""),
    ("discopt._relax.convexity.g_convex_inject", "inject_g_convex_cuts", "residual", ""),
    ("discopt._relax.convexity.signomial_global", "classify_signomial_global", "residual", ""),
    ("discopt._relax.factorable_reform", "canonicalize_entropy", "residual", ""),
    ("discopt._relax.gdp_reformulate", "reformulate_gdp", "residual", ""),
    (
        "discopt._relax.integer_product_reform",
        "has_integer_multilinear_reformulation_work",
        "residual",
        "",
    ),
    ("discopt._relax.integer_product_reform", "has_nonconvex_integer_bilinear", "residual", ""),
    ("discopt._relax.integer_product_reform", "reformulate_integer_bilinear", "residual", ""),
    ("discopt._relax.integer_product_reform", "reformulate_integer_multilinear", "residual", ""),
    ("discopt._relax.objective_epigraph", "relax_objective_defining_equality", "residual", ""),
    ("discopt._relax.presolve_pipeline", "propagate_bounds_to_model", "residual", ""),
    ("discopt._relax.problem_classifier", "classify_problem", "residual", ""),
    ("discopt._relax.symbolic.cut_recognizer", "recognize_and_inject", "residual", ""),
    ("discopt.decomposition", "analyze_decomposition", "residual", ""),
    ("discopt.gp", "classify_gp", "residual", ""),
    ("discopt.gp", "classify_gp_minlp", "residual", ""),
    ("discopt.solvers._root_presolve", "tighten_bigm_coefficients", "residual", ""),
)

_KEYS = {(m, n) for m, n, _c, _t in INVENTORY}
_CATEGORY = {(m, n): c for m, n, c, _t in INVENTORY}
_NOTE = {(m, n): t for m, n, _c, t in INVENTORY}


def test_the_scanner_still_finds_the_region():
    """Rule 6: a scanner that matches nothing reports a clean inventory. Every
    assertion below is vacuous if this one does not hold."""
    start, end, found = _scan()
    assert end > start, f"empty pre-solve region ({start}..{end})"
    assert len(found) > 40, (
        f"the scanner found only {len(found)} pre-solve callees — it has stopped "
        "working, or solve_model was restructured. It is not reporting a clean bill."
    )


def test_no_unrecorded_presolve_pass():
    """THE RATCHET. A new pass in the pre-solve region must be categorised."""
    _start, _end, found = _scan()
    new = sorted(found - _KEYS)
    assert not new, (
        "unrecorded pre-solve pass(es) — #1456.\n"
        "These run BEFORE branch and bound arms `time_limit`, so the user's time\n"
        "limit cannot reach them: an unbounded one returns when it is finished and\n"
        "not a second sooner (`johnall`: 44+ minutes against a 20 s limit).\n"
        "Either give it a deterministic work budget (see\n"
        "discopt._work_budget.WorkBudget, and _relax/dependent_vars.py for the\n"
        "worked example) and record it `bounded`, or add it to INVENTORY with a\n"
        "category and the measurement behind it. Do NOT reach for a wall clock:\n"
        "that is #912's role 2, and it makes structure recognition a function of\n"
        "machine speed.\n\n" + "\n".join(f"  {m}.{n}" for m, n in new)
    )


def test_the_ratchet_actually_fires(tmp_path, monkeypatch):
    """Rule 6 applied to the ratchet itself.

    ``test_no_unrecorded_presolve_pass`` passing means one of two things: the
    inventory is complete, or the scanner cannot see new passes. Those look
    identical from the outside, and the second is the failure mode that would let
    this whole file certify a defect it exists to catch. So inject a pass into a
    synthetic ``solve_model`` and require the scanner to report it.
    """
    fake = tmp_path / "solver.py"
    fake.write_text(
        "def solve_model(model, time_limit=None):\n"
        "    from discopt._relax.brand_new_pass import scan_everything\n"
        "    scan_everything(model)\n"
        "    return _solve_continuous(model)\n"
    )
    monkeypatch.setitem(globals(), "_SOLVER", fake)
    _start, _end, found = _scan()
    assert ("discopt._relax.brand_new_pass", "scan_everything") in found, (
        "the scanner did not see an injected pre-solve pass — it cannot detect "
        f"new ones, so every other test here is vacuous. Saw: {sorted(found)}"
    )
    assert sorted(found - _KEYS) == [("discopt._relax.brand_new_pass", "scan_everything")]


def test_recorded_passes_still_exist():
    """The ratchet must not rot: a row whose call is gone is stale bookkeeping
    that makes the inventory look more complete than it is."""
    _start, _end, found = _scan()
    stale = sorted(_KEYS - found)
    assert not stale, (
        "INVENTORY lists pre-solve pass(es) that are no longer called there — "
        "remove the rows:\n" + "\n".join(f"  {m}.{n}" for m, n in stale)
    )


def test_every_row_has_a_known_category():
    checked = 0
    for mod, name, cat, _note in INVENTORY:
        assert cat in _CATEGORIES, f"{mod}.{name}: unknown category {cat!r}"
        checked += 1
    assert checked == len(INVENTORY)


def test_bounded_rows_name_a_live_budget():
    """A ``bounded`` row is a claim about code, so verify the code.

    Without this, deleting a budget downgrades a pass to role 3 silently and the
    inventory keeps asserting it is bounded — the inventory would then be the
    thing hiding the defect it exists to expose.
    """
    checked = 0
    for mod, name, cat, note in INVENTORY:
        if cat != "bounded":
            continue
        assert note, f"{mod}.{name} is `bounded` but names no budget symbol"
        for sym in note.split(","):
            sym = sym.strip()
            home = _BUDGET_HOME.get(sym)
            assert home, f"{mod}.{name}: budget {sym!r} is not in _BUDGET_HOME"
            module = importlib.import_module(home)
            assert hasattr(module, sym), (
                f"{mod}.{name} is recorded `bounded` on {sym}, but {home} no "
                f"longer defines it. The pass is now role 3 (nothing bounds it). "
                f"Restore the budget or re-categorise the row `residual`."
            )
            checked += 1
    assert checked >= 7, f"only checked {checked} budget symbols"


def test_terminal_rows_are_really_terminal():
    """A ``terminal`` row claims the callee receives the time limit. If that is
    wrong the row excuses an unbounded pass, so spot-check the claim rather than
    trusting the name."""
    import inspect

    checked = 0
    for mod, name, cat, _note in INVENTORY:
        if cat != "terminal":
            continue
        module = importlib.import_module(mod)
        fn = getattr(module, name, None)
        assert fn is not None, f"{mod}.{name} no longer exists"
        params = inspect.signature(fn).parameters
        # A ``**solve_kwargs`` passthrough does receive the limit — it forwards
        # it to the nested solve — so it satisfies the claim just as a named
        # parameter does (``gp.solve_gp`` is spelled that way).
        takes_limit = "time_limit" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        assert takes_limit, (
            f"{mod}.{name} is recorded `terminal` — i.e. it IS the solve and "
            "answers 'when do we stop' — but it takes neither a `time_limit` "
            "parameter nor a **kwargs to forward one through. Either it is not "
            "terminal, or it is an unbounded pre-solve pass wearing a solver's "
            "name."
        )
        checked += 1
    assert checked >= 10, f"only checked {checked} terminal rows"


def test_residual_count_is_visible():
    """Publish the role-3 backlog so shrinking it is a deliberate, reviewable act.

    Unlike #912's residual count — a recorded decision not to convert — this one
    is a genuine backlog: every entry is a pass that can run for an unbounded
    time before the user's `time_limit` is armed. Lower this number by bounding a
    pass, never by reclassifying one without a measurement.
    """
    residual = sorted(k for k, c in _CATEGORY.items() if c == "residual")
    assert len(residual) == 17, (
        f"the #1456 role-3 backlog changed ({len(residual)} entries, expected 17).\n"
        "Bounding one: drop this number and re-categorise the row `bounded` with "
        "its budget symbol.\nAdding one: bound it instead.\n"
        + "\n".join(f"  {m}.{n}" for m, n in residual)
    )


def test_the_four_budget_free_modules_are_all_recorded_residual():
    """The four modules measured to contain no bound of any kind must not be
    sitting in the inventory under a category that excuses them."""
    budget_free = {
        "discopt._relax.integer_product_reform",
        "discopt._relax.gdp_reformulate",
        "discopt._relax.symbolic.cut_recognizer",
        "discopt._relax.convexity.signomial_global",
    }
    checked = 0
    for (mod, name), cat in _CATEGORY.items():
        if mod not in budget_free:
            continue
        assert cat in ("residual", "trivial", "terminal"), (
            f"{mod}.{name} is recorded {cat!r}, but {mod} contains no work budget "
            "at all. A `bounded` or `wall` row there is a false claim."
        )
        if cat == "bounded":  # pragma: no cover - guarded by the assert above
            pytest.fail("unreachable")
        checked += 1
    assert checked >= 6, f"only checked {checked} rows from the budget-free modules"
