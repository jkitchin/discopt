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

import os
import re
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1] / "discopt"

_MAKE = re.compile(r"(time|_time)\.(perf_counter|monotonic)\(\)\s*\+")
_ELAPSED = re.compile(r"(time|_time)\.(perf_counter|monotonic)\(\)\s*-\s*\w+[^<>]*[<>]=?")

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
    (
        "solver.py",
        "return bool(have) and (time.perf_counter() - _fb_t0) >= max(0.0, float(time_limit))",
        "residual",
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
        "_jax/deadline.py",
        "_deadline_monotonic = time.monotonic() + max(0.0, float(seconds_from_now))",
        "contract",
    ),
    (
        "_jax/lp_spatial_bb.py",
        "deadline=time.perf_counter() + _obbt_budget,",
        "residual",
    ),
    (
        "_jax/lp_spatial_bb.py",
        "if (time.perf_counter() - t0) >= time_limit:",
        "contract",
    ),
    (
        "_jax/lp_spatial_bb.py",
        "if (time.perf_counter() - t0) >= time_limit or nodes >= max_nodes:",
        "contract",
    ),
    (
        "_jax/mccormick_lp.py",
        "deadline = time.perf_counter() + _INTEGER_RATIO_DIVE_BUDGET_S",
        "residual",
    ),
    (
        "_jax/mccormick_lp.py",
        "_deadline = None if time_limit is None else time.perf_counter() + time_limit",
        "contract",
    ),
    (
        "_jax/mccormick_lp.py",
        "and (time.perf_counter() - _psd_t0) > _gate_budget * _base_solve_wall",
        "residual",
    ),
    (
        "_jax/obbt.py",
        "deadline = time.perf_counter() + total_time_limit "
        "if total_time_limit is not None else None",
        "contract",
    ),
    (
        "_jax/primal_heuristics.py",
        "_wall = time.perf_counter() + max(0.0, time_budget)",
        "legacy",
    ),
    (
        "_jax/primal_heuristics.py",
        "slice_deadline = time.perf_counter() + max(0.0, float(submip_time_limit))",
        "contract",
    ),
    (
        "_jax/primal_heuristics.py",
        "t_end = time.perf_counter() + max(0.0, float(time_budget))",
        "legacy",
    ),
    (
        "_jax/root_reduce.py",
        "obbt_deadline = None if obbt_budget is None else time.perf_counter() + obbt_budget",
        "residual",
    ),
    (
        "_jax/presolve/orchestrator.py",
        "if time_limit_ms > 0 and (time.monotonic() - started) * 1000.0 >= time_limit_ms:",
        "contract",
    ),
    (
        "_jax/convexity/signomial_global.py",
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
    return found


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
        if p == "_jax/primal_heuristics.py" and c == "residual"
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
    assert len(residual) == 20, (
        f"the #912 residual inventory changed ({len(residual)} entries, expected 20). "
        "If you converted one, drop it from KNOWN and lower this number; if you added "
        "one, convert it instead. This count is a deliberate resting point, not a "
        "backlog — see the module docstring for the evidence and the condition that "
        "would justify shrinking it.\n" + "\n".join(f"  {p}: {s}" for p, s in residual)
    )
