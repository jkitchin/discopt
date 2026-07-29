"""The declared bound-tightening schedule (consolidation plan Card 3a).

Before this module the ordering of discopt's bound-tightening mechanisms existed
**only** as the physical line order of ~20 hand-placed blocks inside
``solver.solve_model`` and ``solver._solve_nlp_bb``.  The architecture review's
§2.3 finding was not that any individual mechanism is wrong — they are sound and
individually well-evidenced — but that *nothing declares the pipeline*, so there
is no artefact to read, to diff, or to test against.  Adding a stage meant
inserting an ``if`` at the right index and hoping the index was right.

This module is that artefact.  It declares, as data:

* :data:`ROOT_SCHEDULE` — the root-stage sequence in ``solve_model``.
* :data:`NLP_BB_ROOT_SCHEDULE` — the root sequence on the convex NLP-BB path.
* :data:`SPATIAL_NODE_SCHEDULE` — the per-node sequence in the spatial B&B loop.
* :data:`NLP_BB_NODE_SCHEDULE` — the per-node sequence on the NLP-BB loop.

Each :class:`TighteningStage` carries the stage name, the *anchor* (the callable
actually invoked at that stage, which is what the conformance test greps for),
the gate in the source's own terms, and the soundness note that justifies the
stage being skippable.

## What this module is, and what it deliberately is not

It is a **declaration plus a conformance test plus a run recorder**.  It is *not*
a re-implementation of the pipeline and it does **not** move the stage bodies out
of ``solve_model``: carving those blocks into modules is Card 4b's job, and doing
it here could not be verified bound-neutral in one step.  Instead:

1. ``python/tests/test_tightening_schedule.py`` walks the AST of the host
   functions and asserts the declared stage anchors occur **in the declared
   order**.  A stage inserted out of order, renamed, or deleted fails that test,
   which is what makes this declaration authoritative rather than decorative.
2. :func:`record` is called from each stage site with that stage's outcome, so
   :meth:`TighteningSchedule.explain` prints the real last-run verdict and stats
   rather than a static listing.  Recording is a dict write with no control-flow
   effect — the schedule cannot change what the solver decides (Regime N).

Card 4a's ``explain_routing`` consumes :meth:`TighteningSchedule.explain`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "TighteningStage",
    "TighteningSchedule",
    "ROOT_SCHEDULE",
    "NLP_BB_ROOT_SCHEDULE",
    "SPATIAL_NODE_SCHEDULE",
    "NLP_BB_NODE_SCHEDULE",
    "ALL_SCHEDULES",
    "record",
    "reset_run",
    "explain",
]


@dataclass(frozen=True)
class TighteningStage:
    """One declared stage of a tightening schedule.

    Attributes:
        name: stable identifier used by :func:`record` and by the tests.
        anchor: the name of the callable invoked at this stage.  The
            conformance test looks for a call to *this* name in the host
            function's AST; it is the link between the declaration and the code.
        gate: the stage's enabling condition, in the source's own terms.  A
            stage whose gate is ``"always"`` runs unconditionally once reached.
        soundness: why skipping the stage (budget exhaustion, an exception, a
            declined gate) cannot invalidate a certificate.  Every stage here
            only ever *tightens* a valid outer box, so declining one leaves a
            looser-but-valid box — that invariant is the reason the whole
            pipeline may be budget-truncated at any point.
        optional: ``True`` when the anchor may legitimately be absent from the
            host function's AST (e.g. it is reached through a helper).  Ordering
            is still asserted for the anchors that are present.
    """

    name: str
    anchor: str
    gate: str
    soundness: str
    optional: bool = False


@dataclass
class StageRun:
    """Last-run record for a single stage (populated by :func:`record`)."""

    ran: bool = False
    gate_verdict: str = "not reached"
    n_tightened: Optional[int] = None
    infeasible: bool = False
    wall_s: Optional[float] = None
    detail: str = ""
    calls: int = 0


# --------------------------------------------------------------------------
# The declarations.  Order is load-bearing: it is what the conformance test
# asserts against the host functions' ASTs.
# --------------------------------------------------------------------------

ROOT_SCHEDULE: tuple[TighteningStage, ...] = (
    TighteningStage(
        name="pre_factorable_fbbt",
        anchor="tighten_root_bounds_with_fbbt",
        gate="has_factorable_work(model)",
        soundness=(
            "FBBT run before the factorable reform so the reform's interval checks "
            "see finite bounds (#138: a fractional-power-of-product lift only fires "
            "when the lifted base has a finite interval). Sound — FBBT removes only "
            "infeasible regions — and the later root presolve re-tightens, so this "
            "can never leave the box looser than it would otherwise have been."
        ),
    ),
    TighteningStage(
        name="rust_root_presolve",
        anchor="run_root_presolve",
        gate="_model_repr is not None and presolve and not _deadline_exhausted()",
        soundness=(
            "Rust orchestrator (eliminate/simplify/fbbt/probing/...). Only its "
            "BOUNDS reach the Python DAG, via propagate_bounds_to_model; its "
            "model rewrites do not (measured: Card 2c.2, filed as Card 3d)."
        ),
    ),
    TighteningStage(
        name="propagate_bounds_to_model",
        anchor="propagate_bounds_to_model",
        gate="same block as rust_root_presolve",
        soundness="Copies tightened per-element bounds into the Python Model; intersect-only.",
    ),
    TighteningStage(
        name="bigm_coefficient_tightening",
        anchor="tighten_bigm_coefficients",
        gate="presolve and not _deadline_exhausted() and coef_tighten_enabled() "
        "[DISCOPT_COEF_TIGHTEN, default OFF]",
        soundness=(
            "Rewrites big-M coefficients toward the FBBT activity slack. Strictly "
            "tightens the LP relaxation without removing an integer-feasible "
            "point. Bound-changing, hence flag-gated default-OFF (CLAUDE.md §5)."
        ),
    ),
    TighteningStage(
        name="reverse_ad_tightening",
        anchor="run_reverse_ad_tightening",
        gate="presolve and presolve_reverse_ad and not _deadline_exhausted() [opt-in]",
        soundness="Gauss-Seidel reverse-mode interval AD to a fixpoint; tightening-only.",
    ),
    TighteningStage(
        name="declared_box_tightening",
        anchor="_declared_box_tightening",
        gate="always (pre-dispatch); budgeted at 15% of the time limit (#863)",
        soundness=(
            "17-rule interval/structural pass over the DECLARED box, run before "
            "LP/QP/convex dispatch because the paths that return early depend on "
            "its infeasibility proof. Its box is intersected back in at "
            "`declared_box_intersect` below (Card 2c.1)."
        ),
    ),
    TighteningStage(
        name="root_fbbt",
        anchor="tighten_root_bounds_with_fbbt",
        gate="always (last step before tree creation); capped at the remaining budget (#863)",
        soundness=(
            "Rust DAG FBBT + integer rounding. Anytime: a truncated sweep returns "
            "a looser-but-valid box. An empty box is a rigorous infeasibility proof."
        ),
    ),
    TighteningStage(
        name="declared_box_intersect",
        anchor="_apply_nonlinear_tightening_with_status",
        gate="_declared_tightening is not None and _declared_tightening_model is model "
        "and shapes match",
        soundness=(
            "Card 2c.1: intersects the pre-dispatch declared-box result into the "
            "root working box. Sound both ways — every mutation between the two "
            "points only ADDS rows (the feasible set shrinks, so a valid box stays "
            "valid) and an intersection can only tighten. Guarded by model identity "
            "and array shape. The anchor is the nonlinear pass that immediately "
            "follows it in the same block."
        ),
    ),
    TighteningStage(
        name="root_obbt",
        anchor="obbt_tighten_root",
        gate=(
            "obbt_at_root and model._objective is not None and not known-convex "
            "and not _deadline_exhausted() and ((continuous and n_vars<=500) or "
            "(integer-nonlinear and n_vars<=_AUTO_RLT_LEVEL1_MAX_VARS))"
        ),
        soundness=(
            "min/max of each variable over the McCormick LP polytope, a valid outer "
            "approximation, so every tightening is sound; integers rounded inward. "
            "cascade_aux=False here EXPLICITLY by measurement (Card 2a)."
        ),
    ),
)

NLP_BB_ROOT_SCHEDULE: tuple[TighteningStage, ...] = (
    TighteningStage(
        name="nlp_bb_root_fbbt",
        anchor="tighten_root_bounds_with_fbbt",
        gate="always, at _solve_nlp_bb entry",
        soundness="Same Rust DAG FBBT as the spatial root; tightening-only, anytime.",
    ),
)

SPATIAL_NODE_SCHEDULE: tuple[TighteningStage, ...] = (
    TighteningStage(
        name="node_global_box_intersect",
        anchor="node_infeasible_mask",
        gate="always, per exported batch",
        soundness=(
            "Intersects each exported node box with the current global box, which "
            "the incumbent-cutoff phases shrink. Sound: a region cut away holds "
            "only solutions no better than the incumbent."
        ),
    ),
    TighteningStage(
        name="node_jacobian_fbbt",
        anchor="_tighten_node_bounds_with_status",
        gate="cl_list (i.e. the node has constraint rows)",
        soundness=(
            "Pure-Python Jacobian-sampled linear-row FBBT plus the 17 structural / "
            "interval nonlinear rules. NOT subsumed by the Rust kernel that follows "
            "it: Card 2b measured Python-only inferences on 278 of 1,495 decided "
            "nodes (18.6%), 147 from the Jacobian half and 83 from the nonlinear "
            "half. Both call sites stay until Phase 5 ports the two missing "
            "inference classes into the kernel."
        ),
    ),
    TighteningStage(
        name="node_in_tree_presolve",
        anchor="in_tree_presolve",
        gate="in_tree_presolve_stride and _model_repr is not None; strided by node depth",
        soundness=(
            "Cutoff-aware Rust FBBT (+optional probing). Every contraction is "
            "outward-rounded interval propagation applied as an INTERSECTION; a "
            "proven-empty box fathoms the node."
        ),
    ),
    TighteningStage(
        name="node_obbt",
        anchor="obbt_tighten_root",
        gate="_per_node_obbt_enabled and _pn_obbt_spent < _pn_obbt_budget_total; "
        "skipped on the root batch (iteration 0, #287)",
        soundness=(
            "OBBT over the node's own McCormick relaxation, so the bounds are valid "
            "for that node's subtree. cascade_aux=False EXPLICITLY (Card 2a)."
        ),
    ),
    TighteningStage(
        name="incumbent_cutoff_obbt",
        anchor="obbt_tighten_root",
        gate="_incumbent_improved and n_vars <= 200 and the incumbent is finite",
        soundness=(
            "Phase C: DBBT (one objective LP whose reduced costs bound every "
            "variable) then OBBT, both against the McCormick relaxation with the "
            "incumbent as a cutoff. Rigorous tightenings of an outer approximation, "
            "but cutoff-conditioned — the resulting box is NOT valid for a global "
            "dual bound, which is why _root_lb_snapshot is taken before the loop. "
            "cascade_aux=False EXPLICITLY (Card 2a: this site owns ex1252's "
            "measured 52% incumbent regression)."
        ),
    ),
    TighteningStage(
        name="incumbent_cutoff_fbbt",
        anchor="fbbt_with_cutoff",
        gate="_model_repr is not None and _incumbent_improved and the incumbent is finite",
        soundness=(
            "Phase C3: Rust FBBT with the incumbent as an objective cutoff — no LP "
            "solves. Applied only when the repr's variable layout provably aligns "
            "1:1 with the flat B&B columns (C-40); a length mismatch would otherwise "
            "write a misaligned, crossed box and corrupt the global box."
        ),
    ),
)

NLP_BB_NODE_SCHEDULE: tuple[TighteningStage, ...] = (
    TighteningStage(
        name="nlp_bb_node_jacobian_fbbt",
        anchor="_tighten_node_bounds_with_status",
        gate="cl_list",
        soundness=(
            "The same Python Jacobian + nonlinear pass as node_jacobian_fbbt, on "
            "the convex NLP-BB loop. Card 2b's counterfactual covered both loops."
        ),
    ),
    TighteningStage(
        name="nlp_bb_node_in_tree_presolve",
        anchor="in_tree_presolve",
        gate="in_tree_presolve_stride and in_tree_presolve_repr is not None",
        soundness=(
            "The same cutoff-aware Rust kernel as node_in_tree_presolve: "
            "intersect-only interval propagation, empty box fathoms the node."
        ),
    ),
)


class TighteningSchedule:
    """A named, ordered sequence of :class:`TighteningStage` with run records."""

    def __init__(self, name: str, host: str, stages: tuple[TighteningStage, ...]) -> None:
        self.name = name
        #: ``module:function`` the stages are inlined in — what the AST
        #: conformance test parses.
        self.host = host
        self.stages = stages
        self._by_name = {s.name: s for s in stages}

    def __iter__(self):
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def stage(self, name: str) -> TighteningStage:
        return self._by_name[name]

    @property
    def anchors(self) -> tuple[str, ...]:
        return tuple(s.anchor for s in self.stages)

    def explain(self, runs: Optional[dict[str, StageRun]] = None) -> str:
        """Render the schedule: stage, gate, and last-run stats.

        With no ``runs`` argument the current thread's recorded run is used, so
        ``explain()`` immediately after a solve describes that solve.  With no
        recorded run the stages print as ``not reached``, which is a truthful
        statement about an un-run schedule rather than a fabricated one.
        """
        if runs is None:
            runs = _thread_state().runs
        lines = [f"{self.name}  [{self.host}]  {len(self.stages)} stages"]
        for i, st in enumerate(self.stages, 1):
            run = runs.get(st.name, StageRun())
            bits = [run.gate_verdict if not run.ran else "RAN"]
            if run.calls > 1:
                bits.append(f"calls={run.calls}")
            if run.n_tightened is not None:
                bits.append(f"tightened={run.n_tightened}")
            if run.infeasible:
                bits.append("INFEASIBLE")
            if run.wall_s is not None:
                bits.append(f"{run.wall_s:.3f}s")
            if run.detail:
                bits.append(run.detail)
            lines.append(f"  {i}. {st.name:<32} [{', '.join(bits)}]")
            lines.append(f"       anchor: {st.anchor}()")
            lines.append(f"       gate:   {st.gate}")
        return "\n".join(lines)


ROOT = TighteningSchedule("root", "discopt.solver:solve_model", ROOT_SCHEDULE)
NLP_BB_ROOT = TighteningSchedule(
    "nlp_bb_root", "discopt.solver:_solve_nlp_bb", NLP_BB_ROOT_SCHEDULE
)
SPATIAL_NODE = TighteningSchedule(
    "spatial_node", "discopt.solver:solve_model", SPATIAL_NODE_SCHEDULE
)
NLP_BB_NODE = TighteningSchedule(
    "nlp_bb_node", "discopt.solver:_solve_nlp_bb", NLP_BB_NODE_SCHEDULE
)

ALL_SCHEDULES: tuple[TighteningSchedule, ...] = (
    ROOT,
    NLP_BB_ROOT,
    SPATIAL_NODE,
    NLP_BB_NODE,
)

_KNOWN_STAGE_NAMES = frozenset(s.name for sched in ALL_SCHEDULES for s in sched)


# --------------------------------------------------------------------------
# Run recording.  Per-thread so concurrent solves (the daemon, the benchmark
# harness's process pool workers) cannot interleave their records.
# --------------------------------------------------------------------------


class _State(threading.local):
    def __init__(self) -> None:
        self.runs: dict[str, StageRun] = {}


_STATE = _State()


def _thread_state() -> _State:
    # ``threading.local`` runs ``__init__`` per thread, but only for threads
    # created after the object; be explicit rather than rely on that.
    if not hasattr(_STATE, "runs"):
        _STATE.runs = {}
    return _STATE


def reset_run() -> None:
    """Clear this thread's stage records (called at the top of a solve)."""
    _thread_state().runs = {}


def record(
    stage: str,
    *,
    ran: bool = True,
    gate_verdict: str = "",
    n_tightened: Optional[int] = None,
    infeasible: bool = False,
    wall_s: Optional[float] = None,
    detail: str = "",
) -> None:
    """Record one stage's outcome.

    Pure bookkeeping: it writes to a per-thread dict and returns ``None``.  It
    must never raise into the solve path and must never influence a decision —
    that is what keeps every ``record`` call site Regime N.

    An unknown ``stage`` name raises :class:`KeyError`.  That is deliberate
    (CLAUDE.md §3, and §6's rule that an instrument which silently measures
    nothing is worse than none): a typo'd stage name would otherwise make
    ``explain()`` quietly report ``not reached`` forever.
    """
    if stage not in _KNOWN_STAGE_NAMES:
        raise KeyError(
            f"unknown tightening stage {stage!r}; declare it in "
            f"discopt._jax.tightening_schedule before recording it"
        )
    runs = _thread_state().runs
    rec = runs.get(stage)
    if rec is None:
        rec = StageRun()
        runs[stage] = rec
    rec.calls += 1
    rec.ran = rec.ran or ran
    if gate_verdict:
        rec.gate_verdict = gate_verdict
    elif ran:
        rec.gate_verdict = "RAN"
    if n_tightened is not None:
        rec.n_tightened = (rec.n_tightened or 0) + int(n_tightened)
    rec.infeasible = rec.infeasible or bool(infeasible)
    if wall_s is not None:
        rec.wall_s = (rec.wall_s or 0.0) + float(wall_s)
    if detail:
        rec.detail = detail


def declined(stage: str, reason: str) -> None:
    """Record that a stage's gate declined, with the reason."""
    record(stage, ran=False, gate_verdict=f"skipped: {reason}")


def explain(schedule: str = "all") -> str:
    """Render one schedule, or every schedule, with this thread's run records."""
    if schedule == "all":
        return "\n\n".join(s.explain() for s in ALL_SCHEDULES)
    for s in ALL_SCHEDULES:
        if s.name == schedule:
            return s.explain()
    raise KeyError(f"unknown schedule {schedule!r}; known: {[s.name for s in ALL_SCHEDULES]}")


def current_runs() -> dict[str, StageRun]:
    """This thread's stage records (for tests and for Card 4a)."""
    return dict(_thread_state().runs)
