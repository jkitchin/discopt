"""Structure analysis: turn a model into a :class:`StructureReport`.

The :class:`StructureAnalyzer` runs the cheap, near-linear detectors from the
Phase 1 graph layer and packages the results into a single immutable
:class:`StructureReport`. Every downstream stage (candidate generation, scoring,
selection, explanation) reads from this report rather than re-analyzing the
model, so the model's expression DAG is scanned once.

The report deliberately stays *descriptive* — it records what structure exists,
not which method to use. Mapping structure to methods is candidate generation
(:mod:`discopt.decomposition.advisor.candidates`); ranking is scoring
(later phase).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from discopt.decomposition.graph import kernels
from discopt.decomposition.graph.base import ModelGraph
from discopt.decomposition.structure import detect_decomposition
from discopt.modeling.core import VarType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructureReport:
    """Immutable summary of the exploitable structure a model exhibits.

    Attributes
    ----------
    num_vars, num_constraints, num_incidences : int
        Sizes of the variable–constraint incidence graph.
    num_blocks : int
        Connected components of the model as written (before any projection).
    is_block_diagonal : bool
        True iff the model already splits into ≥2 independent blocks.
    integer_vars : tuple[str, ...]
        Names of the integer/binary variables — the default complicating set.
    integer_fraction : float
        Fraction of variables that are integer/binary.
    blocks_after_integer_projection : int
        Number of constraint-bearing components once the integer variables are
        removed — i.e. how many independent recourse subproblems a Benders split
        would expose. ``≥2`` is the green light for Benders.
    coupling_constraints : tuple[int, ...]
        Constraint indices detected as coupling (from ``detect_decomposition``).
    coupling_density : float
        Coupling constraints as a fraction of all constraints; the single best
        predictor of whether decomposition pays (design doc §6).
    bridge_constraints : tuple[int, ...]
        Constraints whose sole removal disconnects the model.
    articulation_vars : tuple[str, ...]
        Variables whose removal disconnects the dependency graph — prime
        complicating-variable candidates.
    has_annotations : bool
        Whether the user supplied ``first_stage`` / ``mark_coupling`` / block hints.
    model_is_nonlinear : bool
        Whether any constraint body is nonlinear (selects GBD over classical
        Benders, and gates convexity-dependent soundness).
    element_blocks : int | None
        Constraint-bearing components of the scalar-level graph
        (:class:`~discopt.decomposition.graph.element.ElementGraph`). The fields
        above treat each array as one vertex, so a model whose scenarios share an
        array reads as one block there; this count sees through it. Only blocks
        that carry a constraint are counted, so the elements of an array no
        constraint touches are not mistaken for independent blocks. ``None``
        when the element graph could not be built (see ``element_graph_error``).
    element_blocks_after_integer_projection : int | None
        ``blocks_after_integer_projection`` on the scalar-level graph.
    array_level_hides_blocks : bool
        True iff the scalar-level graph has more constraint-bearing blocks than
        the array-level one, either as written or with the integers fixed:
        structure the array-level fields, and the candidates built from them,
        cannot see.
    element_graph_error : str | None
        Why the element graph is unavailable (no evaluator for the model), or
        ``None`` when it was built.
    """

    num_vars: int
    num_constraints: int
    num_incidences: int
    num_blocks: int
    is_block_diagonal: bool
    integer_vars: tuple[str, ...]
    integer_fraction: float
    blocks_after_integer_projection: int
    coupling_constraints: tuple[int, ...]
    coupling_density: float
    bridge_constraints: tuple[int, ...]
    articulation_vars: tuple[str, ...]
    has_annotations: bool
    model_is_nonlinear: bool
    element_blocks: int | None = None
    element_blocks_after_integer_projection: int | None = None
    element_graph_error: str | None = None
    array_level_hides_blocks: bool = False

    @property
    def num_integer(self) -> int:
        """Number of integer/binary variables."""
        return len(self.integer_vars)

    @property
    def integer_localizes(self) -> bool:
        """True iff fixing the integer variables exposes ≥2 recourse blocks."""
        return self.num_integer > 0 and self.blocks_after_integer_projection >= 2

    def summary(self) -> str:
        """Human-readable multi-line summary of the report."""
        return "\n".join(
            [
                "StructureReport",
                f"  size:        {self.num_vars} vars, {self.num_constraints} cons, "
                f"{self.num_incidences} incidences",
                f"  blocks:      {self.num_blocks} (block-diagonal={self.is_block_diagonal})",
                f"  integer:     {self.num_integer} "
                f"({self.integer_fraction:.0%}); projection blocks="
                f"{self.blocks_after_integer_projection}",
                f"  coupling:    {len(self.coupling_constraints)} constraints "
                f"(density={self.coupling_density:.1%})",
                f"  bridges:     {len(self.bridge_constraints)}; "
                f"articulation vars={len(self.articulation_vars)}",
                f"  nonlinear:   {self.model_is_nonlinear}; annotated={self.has_annotations}",
                (
                    f"  elements:    unavailable ({self.element_graph_error})"
                    if self.element_blocks is None
                    else f"  elements:    blocks={self.element_blocks}; projection blocks="
                    f"{self.element_blocks_after_integer_projection}"
                    + (" (array level hides blocks)" if self.array_level_hides_blocks else "")
                ),
            ]
        )


def _model_is_nonlinear(model) -> bool:
    """Does any algebraic constraint body or the objective contain a nonlinear term?

    #1657: this used the GDP ``_is_linear`` predicate, which has no arm for a
    full array reduction (``dm.sum(x[i, :])``) and so reported a linear MILP
    nonlinear — and disagreed with the Benders dispatcher, which also looks at
    the objective. Both now ask the same witness,
    :func:`discopt.decomposition._linear.first_nonlinear_reason`, which is the
    classical-Benders extractor itself. Non-algebraic rows (SOS, indicator,
    disjunctive) carry no body to be nonlinear and are not counted here.
    """
    from discopt.decomposition._linear import first_nonlinear_reason

    return first_nonlinear_reason(model, algebraic_only=True) is not None


class StructureAnalyzer:
    """Analyze a model's structure into a :class:`StructureReport` (one DAG scan)."""

    def analyze(self, model) -> StructureReport:
        """Compute the :class:`StructureReport` for *model*."""
        graph = ModelGraph.from_model(model)
        n = graph.num_vars
        m = graph.num_constraints

        block_of, num_blocks = graph.variable_components()

        # Integer / binary variables and the "fix them, does it split?" test.
        integer_idx = {
            i
            for i, v in enumerate(model._variables)
            if v.var_type in (VarType.BINARY, VarType.INTEGER)
        }
        integer_vars = tuple(model._variables[i].name for i in sorted(integer_idx))
        projected_cliques = [
            [j for j in clique if j not in integer_idx] for clique in graph.constraint_cliques
        ]
        blocks_after_projection = (
            kernels.bearing_blocks(n, projected_cliques) if integer_idx else num_blocks
        )

        # Coupling structure via the existing resolver (annotations + bridge scan).
        decomp = detect_decomposition(model)
        coupling = tuple(decomp.coupling_constraints)
        coupling_density = (len(coupling) / m) if m else 0.0

        bridges = tuple(graph.bridge_constraints())
        artic_idx = graph.articulation_variables()
        articulation_vars = tuple(graph.var_names[i] for i in artic_idx)

        has_annotations = bool(
            getattr(model, "_decomp_stages", None)
            or getattr(model, "_decomp_blocks", None)
            or getattr(model, "_coupling_keys", None)
        )

        element_blocks, element_projected, element_error = _element_counts(model)
        array_bearing = kernels.bearing_blocks(n, graph.constraint_cliques)
        array_projected_bearing = (
            kernels.bearing_blocks(n, projected_cliques) if integer_idx else array_bearing
        )
        hides = (
            element_blocks is not None
            and element_projected is not None
            and (
                element_blocks > array_bearing
                or (bool(integer_idx) and element_projected > array_projected_bearing)
            )
        )

        return StructureReport(
            num_vars=n,
            num_constraints=m,
            num_incidences=graph.num_incidences,
            num_blocks=num_blocks,
            is_block_diagonal=num_blocks >= 2,
            integer_vars=integer_vars,
            integer_fraction=(len(integer_idx) / n) if n else 0.0,
            blocks_after_integer_projection=blocks_after_projection,
            coupling_constraints=coupling,
            coupling_density=coupling_density,
            bridge_constraints=bridges,
            articulation_vars=articulation_vars,
            has_annotations=has_annotations,
            model_is_nonlinear=_model_is_nonlinear(model),
            element_blocks=element_blocks,
            element_blocks_after_integer_projection=element_projected,
            element_graph_error=element_error,
            array_level_hides_blocks=hides,
        )


def _element_counts(model) -> tuple[int | None, int | None, str | None]:
    """``(blocks, blocks after integer projection, error)`` on the scalar graph.

    The element graph is built from the model's evaluator. A model no evaluator
    can be built for (no objective yet, or the tape declines it and the JAX extra
    is absent) gets ``None`` counts and the reason, recorded in the report and logged, so
    "unavailable" is never read as "no further structure". Any other failure,
    including an evaluator whose layout disagrees with the model, propagates.
    """
    from discopt.decomposition.graph.element import ElementGraph

    if getattr(model, "_objective", None) is None:
        # The evaluator the graph is read from requires an objective; the
        # array-level fields do not, so this is reported, not raised.
        return None, None, "the model has no objective, and the evaluator needs one"
    try:
        graph = ElementGraph.from_model(model)
    except ImportError as exc:
        logger.info("element graph unavailable: %s", exc)
        return None, None, str(exc)
    blocks = graph.bearing_blocks()
    int_vars = [
        p for p, v in enumerate(model._variables) if v.var_type in (VarType.BINARY, VarType.INTEGER)
    ]
    if int_vars:
        int_cols = np.flatnonzero(np.isin(graph.col_var, int_vars))
        projected = graph.bearing_blocks(drop_cols=int_cols)
    else:
        projected = blocks
    return blocks, projected, None


__all__ = ["StructureAnalyzer", "StructureReport"]
