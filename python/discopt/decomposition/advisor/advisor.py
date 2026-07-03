"""The :class:`DecompositionAdvisor` façade.

This is the user-facing object of the Decomposition Advisor. It exposes the
*analysis* surface — structure report, discovered candidates, scored/ranked
candidates (:meth:`scores`), blocks, and the Phase 1 graph views/exports — plus
the *recommendation* surface: :meth:`recommendation` (an explained single pick)
and :meth:`explain` (rendered rationale / counterfactual "why not X?") — and the
*reformulation* surface :meth:`decompose`, which builds a solvable
:class:`~discopt.decomposition.ir.reformulation.DecomposedModel`.

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
from discopt.decomposition.advisor.explain import Explainer, Explanation
from discopt.decomposition.advisor.scoring import Scorer, ScoreVector
from discopt.decomposition.advisor.selection import (
    Policy,
    Ranked,
    RuleBasedPolicy,
    SelectionContext,
)
from discopt.decomposition.advisor.types import Candidate, MethodKind
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
        policy: Policy | None = None,
        store=None,
    ) -> None:
        self._model = model
        self._analyzer = analyzer or StructureAnalyzer()
        self._scorer = scorer or Scorer()
        # When a telemetry store is supplied (and no explicit policy), wrap the
        # rule-based policy with the instance-based learned policy so past solves
        # inform the recommendation (W2). It defers to the rules until the store
        # holds enough near-neighbours.
        if policy is None and store is not None:
            from discopt.decomposition.learning.policies import InstanceBasedPolicy

            self._policy: Policy = InstanceBasedPolicy(store, base=RuleBasedPolicy())
        else:
            self._policy = policy or RuleBasedPolicy()
        self._explainer = Explainer(self._policy)
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

    def features(self):
        """Return the instance's learning feature vector (design §10.3).

        A :class:`~discopt.decomposition.learning.record.InstanceFeatures`
        derived from the structure report — the input to instance-based
        recommendation and the telemetry store.
        """
        from discopt.decomposition.learning.features import extract_features

        return extract_features(self.structure())

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

    def ranked(self) -> list[Ranked]:
        """Return the policy's ranking (score + rank + recommended flag)."""
        return self._policy.rank(self.scores(), SelectionContext(self.structure()))

    def recommendation(self) -> Explanation:
        """Return the explained single-pick recommendation (design §9).

        Applies the selection policy to the scored candidates and builds an
        evidence-backed :class:`Explanation` (reasons, concerns, alternatives,
        performance estimate). Recommends no-decomposition when nothing beats the
        monolith.
        """
        return self._explainer.explain(self.scores(), self.structure())

    def explain(self, fmt: str = "text", *, method: MethodKind | str | None = None) -> str:
        """Render the recommendation, or answer a counterfactual 'why (not) X?'.

        With *method* set, explains why that specific method was (not) chosen
        (design §9.3); otherwise renders the recommendation in *fmt*
        (``"text"`` | ``"markdown"`` | ``"json"``).
        """
        if method is not None:
            if isinstance(method, str):
                method = MethodKind(method)
            return self._explainer.explain_counterfactual(method, self.scores(), self.structure())
        return self.recommendation().render(fmt)

    def decompose(self, method: MethodKind | str | None = None):
        """Build a solvable :class:`DecomposedModel` (design §8, Phase 5).

        With *method* omitted, decomposes using the recommended method; otherwise
        uses the named method's candidate (raising if it was not generated for
        this model). The returned object's ``solve()`` runs the coordinated solve
        via the shipping drivers. Does not solve here.
        """
        from discopt.decomposition.ir.reformulation import build_decomposition

        if method is not None and isinstance(method, str):
            method = MethodKind(method)

        cand = self._candidate_for(method)
        if cand is None:
            raise ValueError(
                f"no {method.label if method else 'recommended'} candidate for this "
                "model; call candidates() to see what is available"
            )
        return build_decomposition(self._model, cand)

    def _candidate_for(self, method: MethodKind | None) -> Candidate | None:
        """The candidate to reformulate: a named method's, or the recommendation's."""
        cands = self.candidates()
        if method is None:
            method = self.recommendation().recommendation
        return next((c for c in cands if c.method is method), None)


def analyze_decomposition(model, *, store=None) -> DecompositionAdvisor:
    """Analyze *model* and return a :class:`DecompositionAdvisor`.

    The public entry point to the advisor's analysis surface. Does not solve or
    modify the model. When *store* (a
    :class:`~discopt.decomposition.learning.store.RecordStore`) is given, the
    advisor consults it via the instance-based learned policy (W2).
    """
    return DecompositionAdvisor(model, store=store)


__all__ = ["DecompositionAdvisor", "analyze_decomposition"]
