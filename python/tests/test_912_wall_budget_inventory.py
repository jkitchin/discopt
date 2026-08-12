"""#912: the enforced inventory of wall-clock work gates.

#912's finding is that the solver reads a clock to decide *how much work to do*,
not merely *when to stop*, at scores of sites — and that this makes the search
tree a function of machine speed. The root integer local search (the mechanism
the issue names as decisive) and its three siblings in the primal-heuristic
layer are converted to deterministic operation counts; see
``test_912_work_budget.py`` and ``docs/dev/work-budget-calibration-2026-08-01.md``.

The rest of the inventory is still wall-clock, and this test is what stops that
fact from quietly drifting. It scans for the two constructions that create a
component-local wall budget —

    <clock>() + <budget>              (a locally invented deadline)
    <clock>() - <origin> > <budget>   (elapsed against a budget)

— and asserts the set is exactly the recorded one below. A new one fails the
test, and the author must either convert it to a deterministic budget or record
it here with a category. Without this, "the class is fixed" would mean "the
sites that existed on 2026-08-01 were fixed".

Both patterns above are matched a line at a time, and that had a hole: a gate
spelled across *two* lines — pass ``time.perf_counter()`` into a helper, do the
``+ <budget>`` arithmetic on the parameter inside it — matches neither, because
the call site has no ``+`` after the clock read and the body has no clock read at
all. ``solver.py``'s ``_gdp_config_deadline`` was exactly that and sat unrecorded
for its whole life, while the inline sibling gate in the same feature was caught
on the day it was written; the difference was only that one author had factored
the arithmetic into a helper. ``_scan_via_argument`` below does the one level of
dataflow that closes it, and its findings join the line-based ones. The hole was
sized before it was closed: across the package, five call sites hand a clock read
to a function defined here, and exactly one of them built a budget from it.

Categories:

``contract``
    The value being spent is the caller's own ``time_limit`` (or a sub-solver's
    slice of it), i.e. the clock is answering "when do we stop?". Correct by
    definition — the role #912 explicitly leaves alone.
``legacy``
    The escape-hatch arm of an already-converted gate, reachable only with the
    deterministic budget disabled.
``residual``
    A genuine component-local budget that still decides *how much* work runs and
    is **not** converted. This is a **decision, not a backlog** — #912 was closed
    as not-planned for these, on the evidence below. Converting one needs a
    deterministic work metric natural to that layer, re-tuned against measured
    consumption, plus (for the bound-changing ones — OBBT, NBT, root cuts, PSD
    separation, convexity classification) its own differential-bound panel.

Why the residuals were left, and when to revisit
------------------------------------------------

Two measurements decided it. First, after the primal-heuristic layer was
converted the corpus-wide clock-scale panel returned **18 in-scope comparisons,
0 mismatches** at 1x vs 2x, and across the whole investigation *every* extent
gate ever observed cutting a search short was the root ILS. No residual gate was
ever caught moving a tree. Second, the cheap way to convert them all at once
does not exist (§9 of the calibration doc): per-operation cost varies 55x across
instances, so a seconds-valued budget cannot be re-denominated in deterministic
units without being re-tuned, one gate at a time.

The honest limit on that first measurement: these budgets bind mainly on *large*
models, and the in-repo corpus is 66 small ones. A `watercontamination0202`-scale
instance is where a root pass actually reaches a 30 s budget. So "no residual
gate was seen moving a tree" is bounded by corpus coverage, not proven in
general — **that is the trigger to revisit**. If a large-instance panel ever
shows one of these gates moving a tree while the solve is comfortably inside its
`time_limit`, convert that gate and lower the count below.

Why the residuals were not simply switched to one global deterministic clock:
that design was built and **falsified** — see the calibration doc §9. A work
clock instrumented over the Python-side primitives (evaluations, NLP solves, LP
relaxation solves, DAG visits) advances at 0.01-0.65x wall across the corpus
(fac2: 0.09 deterministic seconds against 15 s of real work), because the
dominant cost — Rust B&B nodes, presolve, JIT compilation — is invisible to it.
Re-denominating these budgets in that clock would have silently stopped them
firing, turning a #875-style 27 s root pass back on while reporting success.
That is a worse failure than the nondeterminism it would have fixed, so the
residuals stay wall-clock and stay listed here.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1] / "discopt"

# A clock read. ``_now()`` is ``primal_heuristics``' module-level seam over
# ``time.perf_counter`` (#950) — it is still a wall-clock read, so the scanner
# must see through it, or routing a new gate through the seam would be a way to
# leave this inventory silently.
_CLOCK = r"(?:(?:time|_time)\.(?:perf_counter|monotonic)\(\)|(?<![\w.])_now\(\))"
_MAKE = re.compile(_CLOCK + r"\s*\+")
_ELAPSED = re.compile(_CLOCK + r"\s*-\s*\w+[^<>]*[<>]=?")

_CLOCK_FUNCS = {"perf_counter", "monotonic"}


def _is_clock_read(node: ast.AST) -> bool:
    """``time.perf_counter()`` / ``time.monotonic()`` / the ``_now()`` seam."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in _CLOCK_FUNCS:
        return True
    return isinstance(f, ast.Name) and f.id == "_now"


def _defs_by_name(trees: dict[Path, ast.Module]) -> dict[str, list[tuple[Path, ast.AST]]]:
    out: dict[str, list[tuple[Path, ast.AST]]] = {}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.setdefault(node.name, []).append((path, node))
    return out


def _params_receiving_clock(call: ast.Call, defs: dict) -> tuple[set[str], list]:
    """Parameter names of ``call``'s callee that are handed a clock read here."""
    callee = None
    if isinstance(call.func, ast.Name):
        callee = call.func.id
    elif isinstance(call.func, ast.Attribute):
        callee = call.func.attr
    if callee is None or callee not in defs:
        return set(), []
    names: set[str] = set()
    for pos, arg in enumerate(call.args):
        if not _is_clock_read(arg):
            continue
        for _dpath, dnode in defs[callee]:
            formal = dnode.args.posonlyargs + dnode.args.args
            # ``obj.method(clock)`` binds argument 0 to formal 1: the receiver has
            # already consumed ``self``. Without this the mapping is off by one on
            # every method, which reads as "no budget built from it" — a silent
            # false negative exactly where a gate is easiest to hide.
            bound = 1 if isinstance(call.func, ast.Attribute) and formal else 0
            if bound and formal[0].arg not in ("self", "cls"):
                bound = 0
            if pos + bound < len(formal):
                names.add(formal[pos + bound].arg)
    for kw in call.keywords:
        if kw.arg is not None and _is_clock_read(kw.value):
            names.add(kw.arg)
    return names, defs[callee]


def _scan_via_argument() -> set[tuple[str, str]]:
    """Budgets built from a clock read that arrived as an ARGUMENT.

    The line-local patterns above see one line at a time, so they are blind to the
    construction that spells the same gate across two: pass ``time.perf_counter()``
    into a helper, and do the ``+ budget`` arithmetic on the *parameter* inside it.
    Neither line matches — the call site has no ``+`` after the clock read, and the
    body has no clock read at all. Measured on the tree that closed this hole, the
    package had exactly one such gate (``solver.py``'s ``_gdp_config_deadline``,
    added by #993 and invisible for its whole life), while the inline sibling in
    the same feature *was* caught — so coverage depended on nothing more than
    whether an author had refactored the arithmetic into a helper.

    This does the one level of dataflow that closes it: find call sites handing a
    clock read to a function defined in this package, map argument position to
    parameter name, and look for the same two budget constructions on that
    parameter inside the callee's body.
    """
    trees: dict[Path, ast.Module] = {}
    for root, dirs, files in os.walk(_PKG):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            path = Path(root) / f
            # Deliberately not wrapped: a file this cannot parse must fail the
            # test loudly, not be skipped into an "all clear" (rule 7).
            trees[path] = ast.parse(path.read_text(), filename=str(path))

    defs = _defs_by_name(trees)
    src: dict[Path, list[str]] = {p: p.read_text().splitlines() for p in trees}
    found: set[tuple[str, str]] = set()

    for path, tree in trees.items():
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            params, targets = _params_receiving_clock(call, defs)
            if not params:
                continue
            for dpath, dnode in targets:
                rel = str(dpath.relative_to(_PKG))
                for inner in ast.walk(dnode):
                    hit = False
                    # ``param + <budget>`` — a locally invented deadline.
                    if (
                        isinstance(inner, ast.BinOp)
                        and isinstance(inner.op, ast.Add)
                        and isinstance(inner.left, ast.Name)
                        and inner.left.id in params
                    ):
                        hit = True
                    # ``<clock> - param > <budget>`` — elapsed against a budget.
                    if isinstance(inner, ast.Compare) and isinstance(inner.left, ast.BinOp):
                        b = inner.left
                        if (
                            isinstance(b.op, ast.Sub)
                            and isinstance(b.right, ast.Name)
                            and b.right.id in params
                        ):
                            hit = True
                    if hit:
                        found.add((rel, src[dpath][inner.lineno - 1].strip()))
    return found


# (module-relative path, source line, category). See the module docstring.
KNOWN: tuple[tuple[str, str, str], ...] = (
    (
        "_daemon_core.py",
        "if self.max_lifetime > 0 and time.monotonic() - started >= self.max_lifetime:",
        "contract",
    ),
    (
        "_daemon_core.py",
        "deadline = time.monotonic() + 1.0",
        "contract",
    ),
    (
        "_daemon_core.py",
        "deadline = time.monotonic() + wait",
        "contract",
    ),
    (
        "solver.py",
        "deadline = time.perf_counter() + float(_NATIVE_SEED_HEURISTIC_S)",
        "residual",
    ),
    (
        "solver.py",
        "if time.perf_counter() - t_start > time_limit:",
        "contract",
    ),
    (
        "solver.py",
        "deadline = (time.perf_counter() + budget) if budget else None",
        "residual",
    ),
    # The #138 root-relaxation fallback's rule-2 stop. This was ``residual`` and is
    # now ``contract``, because the thing it spends changed: the fallback used to
    # receive a fresh ``_ROOT_FALLBACK_FLOOR_S`` grant handed out *past* an already
    # spent deadline, so the gate genuinely decided how much EXTRA work ran beyond
    # the caller's budget. It is now ``_ROOT_FALLBACK_RESERVE_S``, withheld from the
    # search up front and spent inside ``time_limit``, so the gate answers "have I
    # used up my slice of the caller's budget?" — the ``contract`` role by the
    # definition above. The arithmetic moved into the ``_fb_left()`` helper, which is
    # why the recorded line text changed with it: the grant is now materialised once
    # as an absolute deadline, which ``_fb_left()``/``_fb_stop`` and the two
    # relaxation build deadlines all read instead of each recomputing it.
    (
        "solver.py",
        "_fb_deadline = time.perf_counter() + max(0.0, float(time_limit))",
        "contract",
    ),
    (
        "solver.py",
        "deadline=time.perf_counter() + _per_budget_s,",
        "residual",
    ),
    (
        "solver.py",
        "deadline=time.perf_counter() + _dcb_budget,",
        "residual",
    ),
    (
        "solver.py",
        "model, deadline=time.perf_counter() + _nbt_budget_s",
        "residual",
    ),
    (
        "solver.py",
        "deadline=time.perf_counter() + _obbt_budget,",
        "residual",
    ),
    (
        "solver.py",
        "time.perf_counter() + max(2.0, min(15.0, 0.2 * float(time_limit))),",
        "residual",
    ),
    # The #823/#993 GDP configuration constructor's slice: 15% of what is left,
    # capped at 15 s. Same shape as the entry above it — a capped fraction of the
    # caller's remaining budget, spent by a root component that decides how many
    # fixed-integer sub-NLPs get solved — so it takes the same category. It is not
    # ``contract``: never exceeding ``outer_deadline`` makes it *sound*, but the
    # number of configuration plans it gets through is still set by machine speed.
    # Found only once this scanner learned to follow a clock read through an
    # argument (see ``_scan_via_argument``); it was invisible for its whole life.
    (
        "solver.py",
        "return now + min(remaining, share)",
        "residual",
    ),
    (
        "solver.py",
        "deadline=time.perf_counter() + 5.0,",
        "residual",
    ),
    (
        "solver.py",
        "deadline=time.perf_counter() + _rf_budget,",
        "residual",
    ),
    (
        "solver.py",
        "if not _root_incumbent and (time.perf_counter() - t_start) < time_limit:",
        "contract",
    ),
    (
        "solver.py",
        "if time.perf_counter() - t_start >= time_limit:",
        "contract",
    ),
    (
        "solver.py",
        "deadline=time.perf_counter() + budget,",
        "residual",
    ),
    (
        "decomposition/lagrangian/node_bounder.py",
        "if time.perf_counter() - t0 > time_budget:",
        "residual",
    ),
    (
        "solvers/_root_cuts.py",
        "if _time.perf_counter() - t0 > time_budget_s:",
        "residual",
    ),
    (
        "solvers/amp.py",
        "local_deadline = time.perf_counter() + time_limit if time_limit is not None else None",
        "contract",
    ),
    (
        "solvers/amp.py",
        "deadline = time.perf_counter() + total_time_limit",
        "contract",
    ),
    (
        "solvers/amp.py",
        "deadline = time.perf_counter() + total_budget",
        "residual",
    ),
    (
        "solvers/amp.py",
        "deadline=time.perf_counter() + remaining,",
        "contract",
    ),
    (
        "solvers/amp.py",
        "model, flat_lb, flat_ub, deadline=time.perf_counter() + _nbt_budget_s",
        "residual",
    ),
    (
        "solvers/milp_simplex.py",
        "_deadline = None if time_limit is None else time.perf_counter() + max(0.0, time_limit)",
        "contract",
    ),
    (
        "solvers/mip_nlp_rootsearch.py",
        "deadline = None if time_limit is None else time.perf_counter() + "
        "max(0.0, float(time_limit))",
        "contract",
    ),
    (
        "solvers/oa.py",
        "if time.perf_counter() - t_start >= time_limit:",
        "contract",
    ),
    (
        "solvers/oa.py",
        "return (time.perf_counter() - t_start) >= float(time_limit)",
        "contract",
    ),
    (
        "_relax/deadline.py",
        "_deadline_monotonic = time.monotonic() + max(0.0, float(seconds_from_now))",
        "contract",
    ),
    (
        "_relax/lp_spatial_bb.py",
        "deadline=time.perf_counter() + _obbt_budget,",
        "residual",
    ),
    (
        "_relax/lp_spatial_bb.py",
        "if (time.perf_counter() - t0) >= time_limit:",
        "contract",
    ),
    (
        "_relax/lp_spatial_bb.py",
        "if (time.perf_counter() - t0) >= time_limit or nodes >= max_nodes:",
        "contract",
    ),
    (
        "_relax/mccormick_lp.py",
        "deadline = time.perf_counter() + _INTEGER_RATIO_DIVE_BUDGET_S",
        "residual",
    ),
    (
        "_relax/mccormick_lp.py",
        "_deadline = None if time_limit is None else time.perf_counter() + time_limit",
        "contract",
    ),
    (
        "_relax/mccormick_lp.py",
        "and (time.perf_counter() - _psd_t0) > _gate_budget * _base_solve_wall",
        "residual",
    ),
    (
        "_relax/obbt.py",
        "deadline = time.perf_counter() + total_time_limit "
        "if total_time_limit is not None else None",
        "contract",
    ),
    # These three read the clock through ``primal_heuristics._now()`` since #950
    # (one seam, monkeypatchable, so a deadline-edge test pins the schedule it
    # means to test instead of racing the machine). Same three gates, same
    # categories — only the spelling of the clock read changed.
    (
        "_relax/primal_heuristics.py",
        "_wall = _now() + max(0.0, time_budget)",
        "legacy",
    ),
    (
        "_relax/primal_heuristics.py",
        "slice_deadline = _now() + max(0.0, float(submip_time_limit))",
        "contract",
    ),
    (
        "_relax/primal_heuristics.py",
        "t_end = _now() + max(0.0, float(time_budget))",
        "legacy",
    ),
    (
        "_relax/root_reduce.py",
        "obbt_deadline = None if obbt_budget is None else time.perf_counter() + obbt_budget",
        "residual",
    ),
    (
        "_relax/presolve/orchestrator.py",
        "if time_limit_ms > 0 and (time.monotonic() - started) * 1000.0 >= time_limit_ms:",
        "contract",
    ),
    (
        "_relax/convexity/signomial_global.py",
        "if time_limit is not None and (time.perf_counter() - t0) > time_limit:",
        "contract",
    ),
    (
        "mo/scalarization.py",
        "if time.perf_counter() - self._t0 >= self.total:",
        "residual",
    ),
)

_KNOWN_KEYS = {(p, s) for p, s, _ in KNOWN}
_CATEGORY = {(p, s): c for p, s, c in KNOWN}


def _scan() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for root, dirs, files in os.walk(_PKG):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            path = Path(root) / f
            rel = str(path.relative_to(_PKG))
            for line in path.read_text().splitlines():
                s = line.strip()
                if s.startswith("#"):
                    continue
                if _MAKE.search(s) or _ELAPSED.search(s):
                    found.add((rel, s))
    return found | _scan_via_argument()


@pytest.mark.unit
def test_no_unrecorded_wall_clock_work_gate():
    """A new component-local wall budget must be converted or justified."""
    found = _scan()
    assert found, "the scanner matched nothing — it has stopped working (rule 6)"
    new = sorted(found - _KNOWN_KEYS)
    assert not new, (
        "unrecorded wall-clock work gate(s) — #912.\n"
        "A clock may decide WHEN TO STOP (the caller's time_limit); it must not\n"
        "decide HOW MUCH WORK to do, or the search tree becomes a function of\n"
        "machine speed. Either bound this loop with a deterministic operation\n"
        "count (see discopt._work_budget.WorkBudget and integer_local_search),\n"
        "or add it to KNOWN in this file with a category.\n\n"
        + "\n".join(f"  {p}: {s}" for p, s in new)
    )


@pytest.mark.unit
def test_recorded_gates_still_exist():
    """The ratchet must not rot: an entry whose line is gone is stale bookkeeping
    that hides the fact the inventory is no longer an inventory."""
    stale = sorted(_KNOWN_KEYS - _scan())
    assert not stale, "KNOWN lists gate(s) that no longer exist — remove them:\n" + "\n".join(
        f"  {p}: {s}" for p, s in stale
    )


@pytest.mark.unit
def test_the_converted_layer_stays_converted():
    """The primal-heuristic extent gates #912 converted must stay converted: the
    only wall budgets left in that module are the documented legacy arms and the
    caller's own slice."""
    offenders = sorted(
        (p, s)
        for (p, s), c in _CATEGORY.items()
        if p == "_relax/primal_heuristics.py" and c == "residual"
    )
    assert not offenders, (
        "a converted primal heuristic grew a component-local wall budget again:\n"
        + "\n".join(f"  {p}: {s}" for p, s in offenders)
    )


@pytest.mark.unit
def test_residual_count_is_visible():
    """Publish the residual count so shrinking it is a visible, reviewable act
    rather than a silent one."""
    residual = sorted(k for k, c in _CATEGORY.items() if c == "residual")
    # 20 -> 19: the #138 fallback's rule-2 stop was reclassified ``residual`` ->
    # ``contract`` when its grant became a pre-deadline reserve rather than a
    # post-deadline hand-out. Not a conversion to a deterministic budget — a change
    # in what the gate spends. See the comment on its KNOWN entry.
    # 19 -> 20: ``_gdp_config_deadline`` — not a new gate, a newly *visible* one.
    # It predates this bump; the scanner could not see it until it learned to
    # follow a clock read through a function argument (#993).
    assert len(residual) == 20, (
        f"the #912 residual inventory changed ({len(residual)} entries, expected 20). "
        "If you converted one, drop it from KNOWN and lower this number; if you added "
        "one, convert it instead. This count is a deliberate resting point, not a "
        "backlog — see the module docstring for the evidence and the condition that "
        "would justify shrinking it.\n" + "\n".join(f"  {p}: {s}" for p, s in residual)
    )
