"""Candidate generation: map discovered structure to decomposition candidates.

Each :class:`CandidateGenerator` inspects the :class:`StructureReport` and, when
applicable, proposes one or more :class:`Candidate` decompositions. Generators
are the extensibility seam of Phase 2: a new method is a new generator plus (in
a later phase) a scoring rule, touching no analysis code.

The generators run cheap-first (the cascade of design doc §5.1) and their output
is de-duplicated. Ranking is *not* done here — every applicable candidate is
returned, including the monolithic ``NONE`` baseline, for scoring/selection to
order later.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from discopt.decomposition.advisor.analyzer import StructureReport
from discopt.decomposition.advisor.types import Candidate, ElementView, MethodKind, Soundness
from discopt.decomposition.structure import DecompositionStructure, detect_decomposition


@runtime_checkable
class CandidateGenerator(Protocol):
    """Proposes decomposition candidates from a model and its structure report."""

    name: str

    def applicable(self, report: StructureReport) -> bool:
        """Cheap gate: could this generator produce anything for *report*?"""
        ...

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        """Produce candidates (possibly empty) for *model*."""
        ...


class MonolithicGenerator:
    """Always proposes the no-decomposition baseline.

    Keeping ``NONE`` in the candidate set guarantees the ranker always has the
    honest competitor to beat (design doc §6.3): a decomposition must be *better*
    than solving the monolith, not merely feasible.
    """

    name = "monolithic"

    def applicable(self, report: StructureReport) -> bool:
        return True

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        return [
            Candidate(
                method=MethodKind.NONE,
                structure=None,
                provenance="baseline",
                est_soundness=Soundness.PROVEN_EQUIVALENT,
                notes=("solve the model as written",),
            )
        ]


class IndependentBlockGenerator:
    """Block-diagonal models: solve the independent blocks with no coordination."""

    name = "independent-blocks"

    def applicable(self, report: StructureReport) -> bool:
        return report.is_block_diagonal and not report.coupling_constraints

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        structure = detect_decomposition(model)
        return [
            Candidate(
                method=MethodKind.INDEPENDENT_BLOCKS,
                structure=structure,
                provenance=f"connected components ({structure.num_blocks} blocks)",
                est_soundness=Soundness.PROVEN_EQUIVALENT,
                notes=(
                    f"{structure.num_blocks} independent blocks, no coupling",
                    "exact if the objective is separable across blocks",
                ),
            )
        ]


class IntegerProjectionGenerator:
    """Fixing the integer variables exposes independent recourse → Benders.

    This generalizes the shipping default (complicating vars = integer/binary
    vars): it only fires when *removing* those variables actually disconnects the
    model into ≥2 recourse blocks, which is the structural precondition for
    Benders paying off. Nonlinear recourse selects Generalized Benders.
    """

    name = "integer-projection"

    def applicable(self, report: StructureReport) -> bool:
        return report.integer_localizes

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        structure = detect_decomposition(model)
        if not structure.complicating_vars:
            return []
        if report.model_is_nonlinear:
            method = MethodKind.GENERALIZED_BENDERS
            soundness = Soundness.UNKNOWN
            notes = (
                f"fixing {len(structure.complicating_vars)} integer vars exposes "
                f"{report.blocks_after_integer_projection} recourse blocks",
                "nonlinear recourse → GBD; exact only if subproblems are convex "
                "(checked during scoring)",
            )
        else:
            method = MethodKind.BENDERS
            soundness = Soundness.PROVEN_EQUIVALENT
            notes = (
                f"fixing {len(structure.complicating_vars)} integer vars exposes "
                f"{report.blocks_after_integer_projection} recourse blocks",
                "linear recourse → classical Benders (exact)",
            )
        return [
            Candidate(
                method=method,
                structure=structure,
                provenance="integer-variable projection",
                est_soundness=soundness,
                notes=notes,
            )
        ]


class OuterApproximationGenerator:
    """A convex MINLP → Outer Approximation (Duran & Grossmann 1986).

    On a *convex* MINLP, OA's per-constraint linearizations dominate GBD's
    aggregated single cut, so OA typically needs fewer major iterations. This
    generator fires on a nonlinear model with integer variables whose recourse
    the convexity classifier accepts as convex, and proposes OA as a
    proven-equivalent alternative that outranks GBD in scoring. Nonconvex models
    are left to GBD's heuristic mode.
    """

    name = "outer-approximation"

    def applicable(self, report: StructureReport) -> bool:
        return report.model_is_nonlinear and report.num_integer > 0

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        from discopt._relax.convexity import classify_oa_cut_convexity

        try:
            conv = classify_oa_cut_convexity(model)
            is_convex = conv.objective_is_convex and all(conv.constraint_mask)
        except Exception:  # noqa: BLE001 - a classifier failure just skips OA
            return []
        if not is_convex:
            return []
        # Reuse the integer-projection structure so scoring sees the same block
        # structure as GBD; dispatch routes OA to the whole-model solver.
        structure = detect_decomposition(model)
        return [
            Candidate(
                method=MethodKind.OUTER_APPROXIMATION,
                structure=structure,
                provenance="convex-MINLP outer approximation",
                est_soundness=Soundness.PROVEN_EQUIVALENT,
                notes=(
                    "convex MINLP: OA per-constraint cuts dominate GBD's aggregated "
                    "cut (Duran & Grossmann 1986), usually fewer iterations",
                    "exact for convex models (recourse verified convex)",
                ),
            )
        ]


class CouplingGenerator:
    """A few coupling constraints link otherwise-independent blocks → Lagrangian.

    Dualizing the coupling rows separates the model into blocks solvable in
    parallel and yields a valid dual bound. Fires on annotated or bridge-detected
    coupling.
    """

    name = "coupling"

    def applicable(self, report: StructureReport) -> bool:
        return bool(report.coupling_constraints)

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        structure = detect_decomposition(model)
        if not structure.coupling_constraints:
            return []
        provenance = "user annotations" if report.has_annotations else "bridge-constraint scan"
        return [
            Candidate(
                method=MethodKind.LAGRANGIAN,
                structure=structure,
                provenance=provenance,
                est_soundness=Soundness.RELAXATION,
                notes=(
                    f"{len(structure.coupling_constraints)} coupling constraint(s) "
                    f"dualized; {structure.num_blocks} blocks",
                    "yields a valid dual bound; primal recovery may need extra work",
                ),
            )
        ]


# ── element-level generators ───────────────────────────────────
#
# Each fires only where its array-level counterpart is blind for reasons of
# *granularity*: the array view treats each array variable as one vertex, so
# scenarios sharing an array read as one block. Where the array view already
# sees the structure, these stay silent, so every candidate the array-level
# generators produce -- and every recommendation built on them -- is unchanged.


def _element_view(
    model,
    *,
    drop_cols=(),
    drop_model_constraints=(),
    separator_fraction: float = 0.0,
    coupling_density: float = 0.0,
) -> ElementView:
    """The constraint-bearing blocks of the element graph after removals."""
    import numpy as np

    from discopt.decomposition.graph.element import ElementGraph

    graph = ElementGraph.from_model(model)
    drop_cols = np.asarray(drop_cols, dtype=np.int64)
    dropped_cons = set(int(i) for i in drop_model_constraints)
    drop_rows = [
        i
        for i in range(graph.num_rows)
        if graph.con_model_index[int(graph.row_con[i])] in dropped_cons
    ]
    block_of, _ = graph.components(drop_cols=drop_cols, drop_rows=drop_rows)
    dropped_rows = set(drop_rows)
    bearing: dict[int, None] = {}
    for i, clique in enumerate(graph.row_cliques):
        if i in dropped_rows:
            continue
        kept = [b for b in block_of[clique].tolist() if b >= 0]
        if kept:
            bearing.setdefault(kept[0], None)
    cols_of: dict[int, list[int]] = {b: [] for b in bearing}
    for j, b in enumerate(block_of.tolist()):
        if b in cols_of:
            cols_of[b].append(j)
    blocks = [tuple(cols_of[b]) for b in sorted(cols_of, key=lambda b: cols_of[b][0])]
    return ElementView(
        block_columns=tuple(blocks),
        block_labels=tuple(tuple(graph.col_label(j) for j in blk) for blk in blocks),
        separator_fraction=separator_fraction,
        coupling_density=coupling_density,
    )


class ElementIndependentBlockGenerator:
    """Independent blocks the array view merges (they share an array variable)."""

    name = "independent-blocks-element"

    def applicable(self, report: StructureReport) -> bool:
        return (
            not report.is_block_diagonal
            and not report.coupling_constraints
            and (report.element_blocks or 0) >= 2
        )

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        view = _element_view(model)
        if view.num_blocks < 2:
            return []
        return [
            Candidate(
                method=MethodKind.INDEPENDENT_BLOCKS,
                structure=detect_decomposition(model),
                provenance=f"element-level connected components ({view.num_blocks} blocks)",
                est_soundness=Soundness.PROVEN_EQUIVALENT,
                notes=(
                    f"{view.num_blocks} independent blocks that share array variables",
                    "solved as one model (the per-block parallel solve is Phase 6)",
                ),
                element_view=view,
            )
        ]


class ElementIntegerProjectionGenerator:
    """Fixing the integers splits the recourse element by element → Benders.

    The array-level generator asks the same question on array vertices, where
    a recourse array shared by every scenario never splits. Benders itself
    already partitions its recourse into element-level blocks (multicut), so
    the candidate needs no new driver: the structure is the shipping default
    (complicating = integer variables).
    """

    name = "integer-projection-element"

    def applicable(self, report: StructureReport) -> bool:
        return (
            report.num_integer > 0
            and not report.integer_localizes
            and (report.element_blocks_after_integer_projection or 0) >= 2
        )

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        from discopt.decomposition.advisor.analyzer import integer_columns
        from discopt.decomposition.graph.element import ElementGraph

        structure = detect_decomposition(model)
        if not structure.complicating_vars:
            return []
        graph = ElementGraph.from_model(model)
        int_cols = integer_columns(model, graph)
        view = _element_view(
            model,
            drop_cols=int_cols,
            separator_fraction=(int_cols.size / graph.num_cols) if graph.num_cols else 0.0,
            coupling_density=report.coupling_density,
        )
        if view.num_blocks < 2:
            return []
        if report.model_is_nonlinear:
            method, soundness = MethodKind.GENERALIZED_BENDERS, Soundness.UNKNOWN
            kind = "nonlinear recourse → GBD; exact only if subproblems are convex"
        else:
            method, soundness = MethodKind.BENDERS, Soundness.PROVEN_EQUIVALENT
            kind = "linear recourse → classical Benders (exact)"
        return [
            Candidate(
                method=method,
                structure=structure,
                provenance="integer-variable projection (element level)",
                est_soundness=soundness,
                notes=(
                    f"fixing {int_cols.size} integer column(s) exposes {view.num_blocks} "
                    "recourse blocks that share array variables",
                    kind,
                ),
                element_view=view,
            )
        ]


class ElementCouplingGenerator:
    """Rows that couple *elements* of shared arrays → Lagrangian.

    Fires only when the array-level bridge scan ran and found nothing; where it
    was truncated by its budget the element scan's extra finds are a budget
    effect, not hidden structure, and adopting them would change routing on
    models the array view handles today. Linear models only: the Lagrangian
    driver extracts a linear model and refuses anything else.
    """

    name = "coupling-element"

    def applicable(self, report: StructureReport) -> bool:
        return (
            not report.coupling_constraints
            and not report.coupling_scan_truncated
            and not report.model_is_nonlinear
            and bool(report.element_coupling_constraints)
        )

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        coupling = list(report.element_coupling_constraints)
        structure = detect_decomposition(model, coupling=coupling)
        m = report.num_constraints
        view = _element_view(
            model,
            drop_model_constraints=coupling,
            coupling_density=(len(coupling) / m) if m else 0.0,
        )
        if view.num_blocks < 2:
            return []
        return [
            Candidate(
                method=MethodKind.LAGRANGIAN,
                structure=structure,
                provenance="element-level bridge-constraint scan",
                est_soundness=Soundness.RELAXATION,
                notes=(
                    f"{len(coupling)} coupling constraint(s) dualized; "
                    f"{view.num_blocks} blocks that share array variables",
                    "yields a valid dual bound; primal recovery may need extra work",
                ),
                element_view=view,
            )
        ]


class SeparatorGenerator:
    """Complicating features chosen by an integer program → Benders / Lagrangian.

    The other generators find a decomposition only where the integers, a single
    constraint, or the model as written already split it. This one solves for
    the separator (``graph.separator``, after Kalamaris & Maravelias 2026 §4):
    variable separators become Benders candidates with a column-level master
    (``master_columns``; the integer columns are forced in, as the Benders
    driver requires), constraint separators become Lagrangian candidates whose
    coupling rows are rounded up to whole constraints and re-verified.

    **Opt-in**: not in :data:`DEFAULT_GENERATORS`, so neither the advisor's
    default analysis nor ``solve(decomposition="auto")`` runs it; request it with
    ``analyze_decomposition(model, separator_search=True)``. *Why not default:*
    its search costs about 0.3x the monolithic solve where it helps, and on LPs
    the decomposed solve does not recover that (break-even to slower beyond
    ~12k columns). On a mixed-integer capacity-expansion family it is 1.7-2.1x
    faster end to end from ~3.5k columns. *What would change that:* a corpus
    panel (not one synthetic family) showing it at least neutral on models
    where it fires, including the advisor's cost. Fires on a **linear** model
    that is not already exactly decomposable (block-diagonal, or split by fixing
    its integers); unbounded variables are kept out of the master. The search is a
    deterministic node budget (the recommendation must not depend on machine
    load; a wall-clock limit made it do so) with a wall-clock backstop. Its
    reach is measured, not guessed: on a 150-column capacity-expansion LP each
    program costs ~1 s up to ``|C| = 3`` (variables) and ``|C| = 2``
    (constraints), and 4-8 s beyond, in HiGHS's root processing even at 50
    nodes -- hence the defaults. Constraint separators are **off** by default
    (``max_components_constraints=0``; set it to 2 or more to search them):
    coarsening merges a period-linking row into the chain of ordinary rows of
    the same type, and measured on the 774-column LP the ``|C| = 2`` program
    spent 3.9 s to return a 302-row separator where about 11 rows suffice.
    Classical Benders is exact on any linear partition (``PROVEN_EQUIVALENT``);
    the Lagrangian bound is a relaxation. Nonlinear models are left out: GBD
    takes whole arrays only and is exact only on convex recourse.
    """

    name = "separator-program"

    def __init__(
        self,
        max_components_variables: int = 3,
        max_components_constraints: int = 0,
        node_limit: int = 200,
        time_limit: float = 30.0,
        max_columns: int = 20_000,
        skip_covered: bool = True,
        epsilon: float = 2.0,
        var_axes: Mapping[str, Sequence[int]] | None = None,
        con_axes: Mapping[str, Sequence[int]] | None = None,
    ):
        self.max_components_variables = max_components_variables
        self.max_components_constraints = max_components_constraints
        self.node_limit = node_limit
        self.time_limit = time_limit
        self.max_columns = max_columns
        self.skip_covered = skip_covered  # see find_separators
        # Separator weight vs component imbalance (Kalamaris & Maravelias eq. 40):
        # larger favours small separators, smaller favours balanced blocks.
        self.epsilon = epsilon
        # The paper's index coarsening: axes of each array (variables) or
        # constraint (rows) to merge, e.g. {"Q": (3,)} for "over time". Given
        # either, the program runs on ``graph.coarsen(var_axes, con_axes)``
        # instead of the automatic matching.
        self.var_axes = dict(var_axes) if var_axes else None
        self.con_axes = dict(con_axes) if con_axes else None

    def applicable(self, report: StructureReport) -> bool:
        # Skipped only where an EXACT decomposition already exists (the model is
        # block-diagonal, or fixing the integers splits it). It is not gated on
        # coupling constraints: the bridge scans flag pendant rows -- a farmer
        # scenario's beet row cuts off a two-variable piece -- on most real
        # models, and that gate left the two-stage farmer and newsvendor
        # families with no candidate at all. (The gate only ever served the
        # default advisor's invariance, which opt-in no longer needs.)
        return (
            not report.model_is_nonlinear
            and report.element_blocks is not None
            and report.element_blocks < 2
            and not report.is_block_diagonal
            and not report.integer_localizes
            and (report.element_blocks_after_integer_projection or 0) < 2
        )

    def generate(self, model, report: StructureReport) -> list[Candidate]:
        import dataclasses

        import numpy as np

        from discopt.decomposition.advisor.analyzer import integer_columns
        from discopt.decomposition.graph.element import ElementGraph
        from discopt.decomposition.graph.separator import find_separators

        graph = ElementGraph.from_model(model)
        if graph.num_cols > self.max_columns:
            return []
        out: list[Candidate] = []
        base = detect_decomposition(model)
        n = graph.num_cols
        coarse = (
            graph.coarsen(self.var_axes, self.con_axes)
            if (self.var_axes or self.con_axes)
            else None
        )
        int_cols = integer_columns(model, graph)
        # A Benders master needs bounded variables: an unbounded one (the CVaR
        # value-at-risk variable, bounds +-1e20) leaves the first master
        # unbounded and the driver can only report an error. Bar them.
        from discopt.decomposition.structure import flat_bounds

        lb, ub = flat_bounds(model)
        unbounded = np.flatnonzero((np.abs(lb) >= 1e19) | (np.abs(ub) >= 1e19))
        for sep in find_separators(
            graph,
            "variables",
            forced_columns=int_cols.tolist(),
            excluded_columns=unbounded.tolist(),
            max_components=self.max_components_variables,
            node_limit=self.node_limit,
            time_limit=self.time_limit,
            skip_covered=self.skip_covered,
            epsilon=self.epsilon,
            coarse=coarse,
        ):
            arrays = sorted({graph.var_names[int(graph.col_var[j])] for j in sep.columns})
            structure = dataclasses.replace(
                base, complicating_vars=arrays, master_columns=sep.columns
            )
            view = ElementView(
                block_columns=sep.block_columns,
                block_labels=tuple(
                    tuple(graph.col_label(j) for j in blk) for blk in sep.block_columns
                ),
                separator_fraction=sep.weight / n if n else 0.0,
            )
            out.append(
                Candidate(
                    method=MethodKind.BENDERS,
                    structure=structure,
                    provenance=f"separator program, variables (|C|={sep.num_components})",
                    est_soundness=Soundness.PROVEN_EQUIVALENT,
                    notes=(
                        f"fixing {sep.weight} column(s) of {', '.join(arrays)} exposes "
                        f"{sep.blocks} recourse blocks",
                        f"separator {sep.status} in {sep.seconds:.2f}s; split verified on "
                        "the element graph",
                    ),
                    element_view=view,
                )
            )
        m = report.num_constraints
        if self.max_components_constraints < 2:
            return out  # constraint separators are off unless asked for
        for sep in find_separators(
            graph,
            "constraints",
            max_components=self.max_components_constraints,
            node_limit=self.node_limit,
            time_limit=self.time_limit,
            skip_covered=self.skip_covered,
            epsilon=self.epsilon,
            coarse=coarse,
        ):
            owners = {graph.con_model_index[int(graph.row_con[r])] for r in sep.rows}
            if None in owners or not all(
                graph.con_is_algebraic[int(graph.row_con[r])] for r in sep.rows
            ):
                continue  # a row with no dualizable model constraint behind it
            coupling = sorted(int(i) for i in owners if i is not None)
            # Rounding rows up to whole constraints only removes more; re-measure.
            view = _element_view(
                model,
                drop_model_constraints=coupling,
                coupling_density=(len(coupling) / m) if m else 0.0,
            )
            if view.num_blocks < 2:
                continue
            out.append(
                Candidate(
                    method=MethodKind.LAGRANGIAN,
                    structure=detect_decomposition(model, coupling=coupling),
                    provenance=f"separator program, constraints (|C|={sep.num_components})",
                    est_soundness=Soundness.RELAXATION,
                    notes=(
                        f"{len(coupling)} coupling constraint(s) dualized; "
                        f"{view.num_blocks} blocks",
                        "yields a valid dual bound; primal recovery may need extra work",
                    ),
                    element_view=view,
                )
            )
        return out


#: Default generator cascade, cheap-first (design doc §5.1).
DEFAULT_GENERATORS: tuple[CandidateGenerator, ...] = (
    MonolithicGenerator(),
    IndependentBlockGenerator(),
    IntegerProjectionGenerator(),
    OuterApproximationGenerator(),
    CouplingGenerator(),
    ElementIndependentBlockGenerator(),
    ElementIntegerProjectionGenerator(),
    ElementCouplingGenerator(),
)
"""The cascade every advisor runs. ``SeparatorGenerator`` is not in it: its
partitioning program is opt-in (``analyze_decomposition(model,
separator_search=True)``), because measured on a 774-column LP it spent 39 s
against a 0.8 s monolithic solve, which ``solve(decomposition="auto")`` would
pay before solving."""


def _dedup_key(cand: Candidate) -> tuple:
    """Identity of a candidate for de-duplication: method + partition signature."""
    s: DecompositionStructure | None = cand.structure
    if s is None:
        return (cand.method,)
    return (
        cand.method,
        tuple(sorted(s.complicating_vars)),
        tuple(sorted(s.coupling_constraints)),
        cand.element_view is not None,
        s.master_columns,
    )


def generate_candidates(
    model,
    report: StructureReport,
    generators: tuple[CandidateGenerator, ...] = DEFAULT_GENERATORS,
) -> list[Candidate]:
    """Run the generator cascade and return de-duplicated candidates.

    Order follows the cascade (monolithic baseline first, then cheap structural
    detectors); duplicates from different generators collapse to the
    first-produced one. No ranking is applied.
    """
    out: list[Candidate] = []
    seen: set[tuple] = set()
    for gen in generators:
        if not gen.applicable(report):
            continue
        for cand in gen.generate(model, report):
            key = _dedup_key(cand)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


__all__ = [
    "DEFAULT_GENERATORS",
    "CandidateGenerator",
    "CouplingGenerator",
    "ElementCouplingGenerator",
    "ElementIndependentBlockGenerator",
    "ElementIntegerProjectionGenerator",
    "IndependentBlockGenerator",
    "IntegerProjectionGenerator",
    "MonolithicGenerator",
    "SeparatorGenerator",
    "generate_candidates",
]
