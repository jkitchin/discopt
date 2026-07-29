"""The declared solve-path routing table (consolidation plan Card 4a).

``solver.solve_model`` decides *which engine solves the model* with roughly
thirty sequential gates spread over 2,900 lines.  Before this module that
dispatch tree existed only as the physical order of those ``if`` statements:
there was no artefact to read, no way to ask "why did this model take the
spatial loop?", and nothing that would fail when a gate moved, was deleted, or
— the case that matters most — when one of the #740/#748 soundness
*fall-throughs* was quietly turned back into an early ``return``.

This module is that artefact.  It declares, as data:

* :data:`ROUTE_TABLE` — every dispatch gate in ``solve_model``, **in the order
  the function evaluates them**, with the gate in the source's own terms, the
  handler it dispatches to, and why the route exists.

## What this module is, and what it deliberately is not

Exactly as Card 3a(a)'s :mod:`discopt._jax.tightening_schedule`, this is a
**declaration plus a conformance test plus a run recorder**.  It is *not* a
re-implementation of the dispatch tree and it does **not** evaluate any
predicate itself: the predicates stay where they are, in ``solve_model``, and
each route site calls :func:`entered` (or :func:`fell_through`) as its first
statement.  Recording is a dict write with no control-flow effect, so every
call site is bound-neutral (Regime N).

Three things make the declaration authoritative rather than decorative, all in
``python/tests/test_routing.py``:

1. the ``entered``/``fell_through`` markers must appear inside ``solve_model``
   **exactly once each and in the declared order** (a moved, renamed, deleted
   or newly-inserted gate fails);
2. each route's declared ``handler`` must actually be called inside that
   route's region (the span between its marker and the next route's);
3. each route declared ``fallthrough=True`` must still contain its declared
   ``fallthrough_guard`` source text — this is the #740/#748 guard, and turning
   a fall-through back into an early ``return`` deletes it.

Why fall-throughs are the reason this file exists
-------------------------------------------------

Three routes match their gate and then deliberately **decline** to dispatch,
continuing into the generic spatial branch-and-bound loop, because the
specialized engine they would otherwise call cannot honour a
``lazy_constraints`` / ``incumbent_callback`` rejection:

* ``class_milp``  — ``_solve_milp_simplex`` / ``_solve_milp_bb`` (#748)
* ``class_miqp``  — ``_solve_miqp_bb`` (#748)
* ``nlp_bb_auto`` — ``_solve_nlp_bb`` (#740 / INT-1 #413)

For a lazy constraint the callback *defines the feasible set*, so an engine
that never consults it would accept a point outside it and certify it — a
false-optimal.  Losing one of these fall-throughs is a soundness regression,
not a performance one, which is why (3) above is a hard test.

Usage
-----

``discopt.explain_routing(model, **solve_kwargs)`` solves and returns the route
walk with each gate's real verdict; ``Model.solve(explain_routing=True)`` prints
it.  Both compose ``discopt._jax.tightening_schedule.explain()`` so one call
shows the engine that was chosen *and* the tightening stages that ran inside it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "Route",
    "ROUTE_TABLE",
    "ROUTE_NAMES",
    "RouteRun",
    "entered",
    "fell_through",
    "reset_run",
    "enter_solve",
    "exit_solve",
    "current_runs",
    "explain",
]


@dataclass(frozen=True)
class Route:
    """One declared dispatch gate of ``solve_model``.

    Attributes:
        name: stable identifier; the literal passed to :func:`entered` /
            :func:`fell_through` at the route's site, and what the conformance
            test greps the AST for.
        gate: the route's enabling condition, **verbatim in the source's own
            terms**.  Card 4a is a mechanical extraction: this string is copied
            from the ``if`` statement, never redesigned.
        handler: the callable this route dispatches to.  Empty when the route
            builds its ``SolveResult`` inline (the infeasible / time-limit
            terminations) or when it is the terminal loop.
        reason: why the route exists — what it decides and why that decision is
            safe.  A gate nobody can explain is a gate nobody can audit.
        fallthrough: ``True`` when the route may match its gate and still
            decline to dispatch, continuing to the generic spatial loop.
        fallthrough_reason: what makes it decline, and why declining is the
            *correct* behaviour (never a performance choice).
        fallthrough_guard: exact source text that must remain inside the
            route's region for the fall-through to still exist.  Checked by the
            conformance test; empty means "no single guard string to pin".
        handler_precedes_marker: ``True`` when the handler is invoked *before*
            the route's marker (the native-kernel route calls its handler, then
            records only if the kernel returned a complete result).  The
            conformance test then looks for the handler in the preceding region
            instead.
        terminal: the final route — reached when no earlier gate dispatched.
    """

    name: str
    gate: str
    handler: str
    reason: str
    fallthrough: bool = False
    fallthrough_reason: str = ""
    fallthrough_guard: str = ""
    handler_precedes_marker: bool = False
    terminal: bool = False


# --------------------------------------------------------------------------
# The declaration.  Order is load-bearing: it is exactly the order
# ``solve_model`` evaluates these gates, and the conformance test asserts it
# against the function's AST.
# --------------------------------------------------------------------------

ROUTE_TABLE: tuple[Route, ...] = (
    Route(
        name="substitution_presolve",
        gate="_sub is not None  (DISCOPT_PRESOLVE_SUBSTITUTE built a reduced model)",
        handler="_sub_lift_result",
        reason=(
            "Doubleton/singleton substitution presolve produced a strictly smaller "
            "model; solve that recursively and lift the solution back. Blocked "
            "whenever a warm start, a callback or a decomposition structure is "
            "present, because the reduced column space is not the caller's. If the "
            "lift cannot be verified the route declines and the original model is "
            "solved instead — never a silent lift."
        ),
        fallthrough=True,
        fallthrough_reason=(
            "an unverifiable lift falls back to solving the original model rather "
            "than returning a solution in the wrong space"
        ),
    ),
    Route(
        name="lp_spatial_engine",
        gate='kwargs.get("lp_spatial", False)',
        handler="solve_lp_spatial_bb",
        reason=(
            "Explicit opt-in to the LP-relaxation spatial B&B engine. Declines "
            "(falls through) whenever the engine raises or returns None, so an "
            "unavailable engine costs a slower solve, never a wrong answer."
        ),
        fallthrough=True,
        fallthrough_reason="engine unavailable or declined (_lps is None)",
    ),
    Route(
        name="solver_mip_nlp",
        gate='_solver == "mip-nlp"',
        handler="solve_mip_nlp",
        reason=(
            "Explicit MIP-NLP decomposition (OA / ECP / GOA / SHOT / feasibility "
            "pump). Warns loudly about every solve_model option the backend "
            "ignores rather than silently dropping it."
        ),
    ),
    Route(
        name="solver_amp",
        gate='_solver == "amp"',
        handler="solve_amp",
        reason="Explicit Adaptive Multivariate Partitioning global solver.",
    ),
    Route(
        name="solver_gp",
        gate='_solver == "gp"',
        handler="solve_gp",
        reason=(
            "Explicit geometric-programming path. Raises rather than falling back "
            "when the model is not a GP: an explicit solver request that cannot be "
            "honoured is an error, not a silent re-route."
        ),
    ),
    Route(
        name="solver_gp_minlp",
        gate='_solver == "gp-minlp"',
        handler="solve_gp_minlp",
        reason=(
            "Explicit GP-MINLP path (y-space node relaxations + integer B&B). "
            "Raises rather than falling back, as with solver='gp'."
        ),
    ),
    Route(
        name="auto_gp",
        gate=(
            "_solver is None and not _has_bb_callbacks and not skip_convex_check "
            "and classify_gp(model) is not None"
        ),
        handler="solve_gp",
        reason=(
            "A recognised geometric program solves exactly through its log-space "
            "convex reformulation, which is strictly better than spatial B&B. "
            "Skipped when B&B streaming callbacks are attached (the GP path cannot "
            "fire them) and when the caller opted out with solver='bb'."
        ),
        fallthrough=True,
        fallthrough_reason="classify_gp declined, or solve_gp returned None",
    ),
    Route(
        name="auto_gp_minlp",
        gate=(
            "_solver is None and not _has_bb_callbacks and not skip_convex_check "
            'and env_bool("DISCOPT_GP_MINLP", False) '
            "and classify_gp_minlp(model) is not None"
        ),
        handler="solve_gp_minlp",
        reason=(
            "Default-OFF flag: every node bound is a rigorous convex-GP bound so a "
            "closed tree certifies, but it changes default-solve behaviour for a "
            "class of models and therefore waits on a differential panel."
        ),
        fallthrough=True,
        fallthrough_reason="classify_gp_minlp declined, or solve_gp_minlp returned None",
    ),
    Route(
        name="auto_signomial_global",
        gate=(
            "_solver is None and not _has_bb_callbacks and not skip_convex_check "
            'and env_bool("DISCOPT_SGO", False) '
            "and classify_signomial_global(model) is not None"
        ),
        handler="solve_signomial_global",
        reason=(
            "Default-OFF flag: certified log-domain DC envelope + spatial B&B for "
            "the mixed-sign signomial class the GP path soundly abstains from "
            "(#114 / #741). Same graduation discipline as auto_gp_minlp."
        ),
        fallthrough=True,
        fallthrough_reason="classify_signomial_global declined, or the solve returned None",
    ),
    Route(
        name="decomposition",
        gate="decomposition is not None",
        handler="solve_benders",
        reason=(
            "Explicit opt-in structure-exploiting decomposition (benders / "
            "lagrangian / an auto-detected structure)."
        ),
    ),
    Route(
        name="gdp_oa",
        gate='gdp_method == "oa"',
        handler="solve_mip_nlp",
        reason=(
            "Generalized-disjunctive OA: big-M reformulate, then the MIP-NLP "
            "outer-approximation driver. Intercepted before the generic GDP "
            "reformulation below."
        ),
    ),
    Route(
        name="gdp_loa",
        gate='gdp_method == "loa"',
        handler="solve_gdpopt_loa",
        reason="Logic-based outer approximation, intercepted before GDP reformulation.",
    ),
    Route(
        name="custom_call_reduced_space",
        gate="_model_contains_custom_call(model)",
        handler="_solve_continuous",
        reason=(
            "A dm.custom(...) node either traces soundly through MCBox — in which "
            "case the reduced-space engine solves it globally — or it does not, in "
            "which case the result is returned with gap_certified=False. The "
            "certificate is downgraded, never fabricated."
        ),
    ),
    Route(
        name="nonlinear_bound_infeasible",
        gate="nonlinear_infeasibility is not None",
        handler="",
        reason=(
            "The declared-box nonlinear tightening proved the model infeasible. "
            "Terminal, gap_certified=True — an FBBT-style emptiness proof is a "
            "rigorous certificate."
        ),
    ),
    Route(
        name="nlp_bb_forced",
        gate="nlp_bb is True and not _pure_continuous",
        handler="_solve_nlp_bb",
        reason=(
            "Caller explicitly forced NLP branch-and-bound. Warns when a rejecting "
            "callback is also supplied, since this path cannot enforce one — the "
            "auto-select route below declines instead of warning."
        ),
    ),
    Route(
        name="solver_gurobi",
        gate='_solver == "gurobi"',
        handler="_solve_lp_gurobi",
        reason=(
            "Explicit Gurobi backend for the classified LP/MILP/QP/MIQP/QCP family. "
            "Raises NotImplementedError outside it rather than re-routing, so an "
            "explicit backend request is never silently substituted."
        ),
    ),
    Route(
        name="class_lp",
        gate="problem_class == ProblemClass.LP",
        handler="_solve_lp",
        reason="Pure LP — one simplex/IPM solve, globally optimal by construction.",
    ),
    Route(
        name="class_qp",
        gate="problem_class == ProblemClass.QP",
        handler="_solve_qp",
        reason=(
            "QP. A pure-continuous QP is dispatched only when convexity is KNOWN; "
            "an indefinite one sets _pure_continuous_force_spatial and falls "
            "through, because a convex QP solver would certify a local stationary "
            "point as global."
        ),
        fallthrough=True,
        fallthrough_reason=(
            "pure-continuous and convexity not known-convex: forces the spatial "
            "path instead of certifying a local stationary point"
        ),
    ),
    Route(
        name="class_milp",
        gate="problem_class == ProblemClass.MILP",
        handler="_solve_milp_bb",
        reason=(
            "MILP via the self-hosted Rust B&B (warm-started simplex node LPs), or "
            "the monolithic Rust engine under nlp_solver='simplex'."
        ),
        fallthrough=True,
        fallthrough_reason=(
            "#748: _solve_milp_simplex / _solve_milp_bb neither receive nor consult "
            "lazy_constraints / incumbent_callback, so a callback-rejected or "
            "cut-off integer point would be accepted as the incumbent — a soundness "
            "bug for lazy constraints (the callback defines the feasible set) and an "
            "API violation for an incumbent rejection. With either callback present "
            "the route declines and the spatial loop, which honours both (#740), "
            "solves it. Trades the specialized engine's speed for correctness "
            "(CLAUDE.md §1). DO NOT convert this back into a return."
        ),
        fallthrough_guard="if lazy_constraints is None and incumbent_callback is None:",
    ),
    Route(
        name="class_miqp",
        gate="problem_class == ProblemClass.MIQP",
        handler="_solve_miqp_bb",
        reason=(
            "MIQP. Dispatched to the convex MIQP B&B only when convexity is KNOWN: "
            "_solve_miqp_bb assumes a convex node QP, so on an indefinite or "
            "concave-maximize objective it would return a local stationary point "
            "and certify it as global (max x**2 over integer [-3,3] returned 0, not 9)."
        ),
        fallthrough=True,
        fallthrough_reason=(
            "two independent declines, both mandatory. (a) NONCONVEX MIQP: the "
            "convex B&B would certify a local stationary point, so it falls through "
            "to the spatial loop. (b) #748: even a convex MIQP falls through when "
            "lazy_constraints / incumbent_callback is present, because "
            "_solve_miqp_bb neither receives nor consults them (#740). "
            "DO NOT convert either back into a return."
        ),
        fallthrough_guard="if lazy_constraints is None and incumbent_callback is None:",
    ),
    Route(
        name="pure_continuous_convex_nlp",
        gate="_pure_continuous and _pure_continuous_convexity_known and _pure_continuous_is_convex",
        handler="_solve_continuous",
        reason=(
            "A convex NLP is solved globally by a single NLP solve. Returns only on "
            "status == 'optimal'; anything else is handled below rather than "
            "certified."
        ),
        fallthrough=True,
        fallthrough_reason=(
            "the convex NLP failed to certify — either a clearable denominator "
            "(reformulate and take the spatial path) or a nonsmooth node; in both "
            "cases the route declines rather than returning an uncertified 'optimal'"
        ),
    ),
    Route(
        name="pure_continuous_unclassified",
        gate=(
            "_pure_continuous and not _pure_continuous_force_spatial "
            "and (skip_convex_check or not _pure_continuous_convexity_known)"
        ),
        handler="_solve_continuous",
        reason=(
            "Convexity unknown (or the check was skipped) on a continuous model: a "
            "local NLP is run, and its result is returned with the dual bound and "
            "gap STRIPPED (gap_certified=False) unless it proved infeasibility. A "
            "local solve never yields a global certificate."
        ),
        fallthrough=True,
        fallthrough_reason="the local NLP errored — fall through to spatial B&B (#266)",
    ),
    Route(
        name="nlp_bb_auto",
        gate="nlp_bb is None and lazy_constraints is None and incumbent_callback is None",
        handler="_solve_nlp_bb",
        reason=(
            "Auto-select NLP-BB for a genuinely nonlinear CONVEX MINLP: every node "
            "relaxation is a convex NLP solved to global optimality, so the tree "
            "bound is valid."
        ),
        fallthrough=True,
        fallthrough_reason=(
            "#740 / INT-1 (#413): the callback guard is IN THE GATE ITSELF. The "
            "NLP-BB path cannot honour a lazy constraint or an incumbent-callback "
            "rejection — it has no per-node cut application and its primal "
            "heuristics inject incumbents without consulting the callback — so a "
            "solve carrying either callback must fall through to the spatial loop, "
            "which enforces both. Also falls through whenever convexity is not "
            "KNOWN-convex. DO NOT relax the callback conjuncts out of this gate."
        ),
        fallthrough_guard="lazy_constraints is None and incumbent_callback is None",
    ),
    Route(
        name="root_fbbt_infeasible",
        gate="root_infeasible  (from tighten_root_bounds_with_fbbt)",
        handler="",
        reason=(
            "The Rust root FBBT emptied the box. Terminal 'infeasible' — an FBBT "
            "emptiness proof removes only infeasible regions, so it is rigorous."
        ),
    ),
    Route(
        name="nonlinear_root_infeasible",
        gate="_nl_root_infeasible  (Python forward-substitution FBBT, or lb > ub + 1e-9)",
        handler="",
        reason=(
            "The Python nonlinear forward-substitution FBBT (which defines "
            "unbounded division/sqrt auxiliaries the Rust pass leaves open) emptied "
            "the box. Terminal 'infeasible'."
        ),
    ),
    Route(
        name="root_obbt_infeasible",
        gate="_obbt_res.infeasible  (inside the root-OBBT gate)",
        handler="",
        reason=(
            "Root optimality-based bound tightening emptied the box. Terminal "
            "'infeasible'; OBBT solves relaxations, so an empty box is a rigorous "
            "proof."
        ),
    ),
    Route(
        name="native_spatial_kernel",
        gate="_native_kernel_feature_safe(...) and _native_result is not None",
        handler="_try_native_spatial_kernel",
        reason=(
            "#789: the native Rust spatial kernel runs only when the solve does not "
            "exercise a Python-engine contract (callbacks, lazy constraints, "
            "solution pool, warm start, non-default McCormick mode, explicit "
            "tuning). It returns a result only on a complete, certified exit."
        ),
        fallthrough=True,
        fallthrough_reason=(
            "any incomplete kernel exit returns None and falls through to the "
            "trusted Python search — the fallback is load-bearing, not legacy "
            "(review §2.5.2, Card 3c measured 0 kernel-served solves on its corpus)"
        ),
        handler_precedes_marker=True,
    ),
    Route(
        name="deadline_exhausted",
        gate="_deadline_exhausted()",
        handler="",
        reason=(
            "#654: everything below is optional search apparatus (XLA compile, "
            "McCormick relaxer build, node loop) whose fixed multi-second cost is "
            "uninterruptible. With the budget already spent, short-circuit to "
            "'time_limit' after a bounded root-relaxation attempt at recovering a "
            "dual bound."
        ),
    ),
    Route(
        name="spatial_branch_and_bound",
        gate="(no gate — reached when no earlier route dispatched)",
        handler="",
        reason=(
            "The generic McCormick spatial branch-and-bound loop: the engine every "
            "fall-through above lands in, and the only one that honours "
            "lazy_constraints and incumbent_callback."
        ),
        terminal=True,
    ),
)

ROUTE_NAMES: tuple[str, ...] = tuple(r.name for r in ROUTE_TABLE)
_ROUTE_BY_NAME = {r.name: r for r in ROUTE_TABLE}
_FALLTHROUGH_NAMES = frozenset(r.name for r in ROUTE_TABLE if r.fallthrough)

if len(ROUTE_NAMES) != len(set(ROUTE_NAMES)):  # pragma: no cover - declaration guard
    raise RuntimeError("duplicate route name in ROUTE_TABLE")


@dataclass
class RouteRun:
    """Last-run record for a single route (populated by :func:`entered`)."""

    verdict: str = "not reached"
    detail: str = ""
    hits: int = 0
    order: int = -1


# --------------------------------------------------------------------------
# Run recording.  Per-thread, so concurrent solves (the daemon, the benchmark
# harness's process pool) cannot interleave their walks.  Nested solves
# (the substitution-presolve route re-enters solve_model) share the outer
# run: only the outermost entry resets.
# --------------------------------------------------------------------------


class _State(threading.local):
    def __init__(self) -> None:
        self.runs: dict[str, RouteRun] = {}
        self.depth: int = 0
        self.seq: int = 0


_STATE = _State()


def _state() -> _State:
    if not hasattr(_STATE, "runs"):
        _STATE.runs = {}
        _STATE.depth = 0
        _STATE.seq = 0
    return _STATE


def reset_run() -> None:
    """Clear this thread's route records."""
    st = _state()
    st.runs = {}
    st.seq = 0


def enter_solve() -> int:
    """Mark entry into ``solve_model``; resets the walk only at the outermost call."""
    st = _state()
    if st.depth == 0:
        reset_run()
    st.depth += 1
    return st.depth


def exit_solve() -> None:
    """Mark exit from ``solve_model``."""
    st = _state()
    st.depth = max(0, st.depth - 1)


def _record(name: str, verdict: str, detail: str) -> None:
    if name not in _ROUTE_BY_NAME:
        raise KeyError(
            f"unknown route {name!r}; declare it in discopt.routing.ROUTE_TABLE before recording it"
        )
    st = _state()
    rec = st.runs.get(name)
    if rec is None:
        rec = RouteRun()
        st.runs[name] = rec
    rec.hits += 1
    rec.verdict = verdict
    if detail:
        rec.detail = detail
    if rec.order < 0:
        rec.order = st.seq
        st.seq += 1


def entered(name: str, detail: str = "") -> None:
    """Record that ``name``'s gate matched and the route's branch was entered.

    Entering is **not** the same as dispatching: most routes run a second, finer
    classification inside the branch (``classify_gp``, a convexity check, an
    engine that may return ``None``) and continue on when it declines.  Which
    route actually dispatched is derived in :func:`explain` from the record
    order — a dispatching branch returns, so nothing after it can record.
    Claiming "taken" here would be the kind of instrument that reports a result
    it never measured.

    Pure bookkeeping: a dict write returning ``None``.  It must never raise into
    the solve path and must never influence a decision — that is what keeps
    every call site Regime N.  An undeclared route name raises
    :class:`KeyError` deliberately (CLAUDE.md §6): a typo would otherwise make
    :func:`explain` report ``not reached`` forever.
    """
    _record(name, "ENTERED", detail)


def fell_through(name: str, detail: str = "") -> None:
    """Record that ``name``'s gate matched but the route declined to dispatch.

    Used at the #740/#748 soundness fall-throughs, where the specialized engine
    cannot honour a user callback and the generic loop must solve the model
    instead.  See this module's docstring.
    """
    _record(name, "FELL THROUGH", detail)


def current_runs() -> dict[str, RouteRun]:
    """This thread's route records (for tests and for ``explain_routing``)."""
    return dict(_state().runs)


def explain(runs: Optional[dict[str, RouteRun]] = None, *, with_schedule: bool = True) -> str:
    """Render the route walk: every gate, in order, with its verdict.

    With no ``runs`` argument the current thread's recorded walk is used, so
    calling this immediately after a solve describes that solve.  With no
    recorded walk every route prints ``not reached``, which is a truthful
    statement about an un-run table rather than a fabricated one.
    """
    if runs is None:
        runs = current_runs()
    reached = {n for n, r in runs.items() if r.hits}
    # The route that dispatched is the LAST one to record: a branch that
    # dispatches returns, so no later gate can be evaluated.  Deriving it this
    # way rather than asserting it at the call site is what keeps every marker a
    # single line at branch entry (Regime N) while still answering "which engine
    # actually solved this model?".
    dispatched = None
    if reached:
        dispatched = max(reached, key=lambda n: runs[n].order)
    lines = [
        f"routing walk  [discopt.solver:solve_model]  {len(ROUTE_TABLE)} gates, "
        f"{len(reached)} entered"
    ]
    passed_dispatch = False
    for i, route in enumerate(ROUTE_TABLE, 1):
        run = runs.get(route.name, RouteRun())
        if route.name == dispatched:
            verdict = "DISPATCHED"
        elif run.verdict == "FELL THROUGH":
            verdict = "FELL THROUGH"
        elif run.hits:
            verdict = "entered, declined"
        elif passed_dispatch:
            verdict = "not reached"
        else:
            verdict = "gate false"
        bits = [verdict]
        if run.hits > 1:
            bits.append(f"hits={run.hits}")
        if run.detail:
            bits.append(run.detail)
        if route.fallthrough:
            bits.append("may fall through")
        mark = "->" if route.name == dispatched else "  "
        lines.append(f" {mark} {i:2d}. {route.name:<30} [{', '.join(bits)}]")
        lines.append(f"        gate:    {route.gate}")
        if route.handler:
            lines.append(f"        handler: {route.handler}()")
        if route.fallthrough and verdict in ("FELL THROUGH", "entered, declined"):
            lines.append(f"        declined: {route.fallthrough_reason}")
        if route.name == dispatched:
            passed_dispatch = True
    if dispatched is None:
        lines.append("  (no route recorded — the walk was never run)")
    if with_schedule:
        try:
            from discopt._jax import tightening_schedule as _ts

            lines.append("")
            lines.append(_ts.explain("all"))
        except Exception as exc:  # pragma: no cover - explain must never break a solve
            lines.append(f"  (tightening schedule unavailable: {exc!r})")
    return "\n".join(lines)
