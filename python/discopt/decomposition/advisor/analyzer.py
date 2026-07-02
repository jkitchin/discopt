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

from dataclasses import dataclass

from discopt.decomposition.graph import kernels
from discopt.decomposition.graph.base import ModelGraph
from discopt.decomposition.structure import detect_decomposition
from discopt.modeling.core import VarType


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
            ]
        )


def _model_is_nonlinear(model) -> bool:
    """Best-effort: does any constraint body contain a nonlinear term?

    Uses the modeling layer's ``_is_linear`` predicate; on any import/inspection
    failure it conservatively reports ``False`` (classical Benders is the safe
    default and GBD is only *offered*, never forced).
    """
    try:
        from discopt._jax.gdp_reformulate import _is_linear
    except Exception:
        return False
    for c in model._constraints:
        body = getattr(c, "body", None)
        try:
            if body is not None and not _is_linear(body):
                return True
        except Exception:
            continue
    return False


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
        )


__all__ = ["StructureAnalyzer", "StructureReport"]
