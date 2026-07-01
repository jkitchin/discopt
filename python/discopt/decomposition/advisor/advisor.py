"""The :class:`DecompositionAdvisor` façade.

This is the user-facing object of the Decomposition Advisor. It exposes the
*analysis* surface — structure report, discovered candidates, scored/ranked
candidates (:meth:`scores`), blocks, and the Phase 1 graph views/exports.
Recommendation- and reformulation-facing methods (``recommendation``,
``explain``, ``decompose``) are stubbed to raise a clear "not yet" error until
their phases land, so the public API shape is stable from the start.

Entry point::

    from discopt.decomposition import analyze_decomposition
    adv = analyze_decomposition(model)      # or: model.analyze_decomposition()
    print(adv.summary())
    for c in adv.candidates():
        print(c.summary())
"""

from __future__ import annotations

from discopt.decomposition.advisor.analyzer import StructureAnalyzer, StructureReport
from discopt.decomposition.advisor.candidates import generate_candidates
from discopt.decomposition.advisor.scoring import Scorer, ScoreVector
from discopt.decomposition.advisor.types import Candidate
from discopt.decomposition.graph.base import GraphKind, ModelGraph, build_graph
from discopt.decomposition.graph.export import export_graph as _export_graph
from discopt.decomposition.structure import DecompositionStructure, detect_decomposition


class DecompositionAdvisor:
    """Analyze a model's decomposition structure and enumerate candidates.

    Construct via :func:`analyze_decomposition` or ``model.analyze_decomposition()``.
    Analysis is lazy and memoized: the structure report and candidate list are
    computed on first access and cached for the advisor's lifetime (the model is
    assumed unchanged while an advisor is held).
    """

    def __init__(
        self,
        model,
        *,
        analyzer: StructureAnalyzer | None = None,
        scorer: Scorer | None = None,
    ) -> None:
        self._model = model
        self._analyzer = analyzer or StructureAnalyzer()
        self._scorer = scorer or Scorer()
        self._report: StructureReport | None = None
        self._candidates: list[Candidate] | None = None
        self._scores: list[tuple[Candidate, ScoreVector]] | None = None

    # ── analysis surface ────────────────────────────────────────

    def structure(self) -> StructureReport:
        """Return the :class:`StructureReport` (analyzing on first call)."""
        if self._report is None:
            self._report = self._analyzer.analyze(self._model)
        return self._report

    def candidates(self) -> list[Candidate]:
        """Return the discovered decomposition candidates (unranked)."""
        if self._candidates is None:
            self._candidates = generate_candidates(self._model, self.structure())
        return list(self._candidates)

    def scores(self) -> list[tuple[Candidate, ScoreVector]]:
        """Return candidates paired with their score, ranked best-first.

        The monolithic ``NONE`` baseline is included so its rank shows whether any
        decomposition is estimated to beat solving the model as written.
        """
        if self._scores is None:
            graph = build_graph(self._model)
            self._scores = self._scorer.score_all(
                self._model, self.candidates(), self.structure(), graph
            )
        return list(self._scores)

    def blocks(self) -> DecompositionStructure:
        """Return the resolved block/coupling partition (the existing contract)."""
        return detect_decomposition(self._model)

    def graph(self, kind: GraphKind | str = GraphKind.VARIABLE_CONSTRAINT) -> ModelGraph:
        """Return a Phase 1 :class:`ModelGraph` view of the requested kind."""
        if isinstance(kind, str):
            kind = GraphKind(kind)
        return build_graph(self._model, kind)

    def export_graph(
        self,
        kind: GraphKind | str = GraphKind.VARIABLE_CONSTRAINT,
        fmt: str = "json",
    ) -> str:
        """Serialize a model-induced graph (see :func:`graph.export.export_graph`)."""
        return _export_graph(self.graph(kind), fmt)

    def summary(self) -> str:
        """Human-readable summary of the structure and ranked candidates."""
        scored = self.scores()
        lines = [self.structure().summary(), "", f"Ranked candidates ({len(scored)}):"]
        for cand, sv in scored:
            lines.append(f"  • {cand.summary()}")
            lines.append(f"        {sv.summary()}")
        return "\n".join(lines)

    # ── later-phase surface (stable shape, not yet implemented) ──

    def recommendation(self):
        """Ranked recommendation with rationale. Requires the selection stage.

        Scoring (Phase 3) is available via :meth:`scores`; the explained
        single-pick recommendation lands with the selection/explanation engine
        (Phase 4).
        """
        raise NotImplementedError(
            "recommendation() needs the selection + explanation stage (Phase 4); "
            "today use scores() for the ranked candidate list."
        )

    def explain(self, *args, **kwargs):
        """Explain the recommendation. Requires the explanation stage (Phase 4)."""
        raise NotImplementedError("explain() lands with the recommendation engine (Phase 4).")

    def decompose(self, *args, **kwargs):
        """Build a solvable decomposed model. Requires reformulation (Phase 5)."""
        raise NotImplementedError(
            "decompose() lands with automatic reformulation (Phase 5); today the "
            "shipping solve_benders / solve_lagrangian drivers consume a candidate's "
            ".structure directly."
        )


def analyze_decomposition(model) -> DecompositionAdvisor:
    """Analyze *model* and return a :class:`DecompositionAdvisor`.

    The public entry point to the advisor's analysis surface. Does not solve or
    modify the model.
    """
    return DecompositionAdvisor(model)


__all__ = ["DecompositionAdvisor", "analyze_decomposition"]
