"""The reformulation IR: turn a chosen candidate into a solvable decomposition.

This is Phase 5 (design §8). A :class:`DecomposedModel` bundles the partition
(master + subproblems), a soundness certificate, a variable mapping, and a
``solve()`` that dispatches to the shipping decomposition driver
(``solve_benders`` / ``solve_gbd`` / ``solve_lagrangian``) for the chosen method.

Per the design (§8.4) we *wrap* the existing drivers behind one uniform interface
rather than reimplementing their coordination loops: the IR contributes the
common structure — master/subproblem descriptors, the correctness certificate,
the original-space variable mapping, and a single dispatch point — so a new
method is a new ``build_*`` mapping, and nesting/parallelism (Phase 6) have a
stable object to hang off. The drivers all share the signature
``(model, *, structure=..., **kwargs) -> SolveResult``, which is what makes the
dispatch uniform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from discopt.decomposition.advisor.types import Candidate, MethodKind, Soundness

# Import drivers at module scope so they dispatch uniformly (and so tests can
# monkeypatch the dispatch without a built solver backend).
from discopt.decomposition.benders import solve_benders, solve_gbd
from discopt.decomposition.graph import kernels
from discopt.decomposition.graph.base import ModelGraph
from discopt.decomposition.ir.certificate import SoundnessCertificate, VariableMapping
from discopt.decomposition.ir.models import MasterModel, SubproblemModel
from discopt.decomposition.lagrangian import solve_lagrangian
from discopt.decomposition.structure import DecompositionStructure, detect_decomposition

# Certificate rationale per method.
_CERTIFICATE = {
    MethodKind.NONE: (
        Soundness.PROVEN_EQUIVALENT,
        "solves the model as written",
        (),
    ),
    MethodKind.INDEPENDENT_BLOCKS: (
        Soundness.PROVEN_EQUIVALENT,
        "independent blocks share no variables or constraints",
        ("exact if the objective is separable across blocks",),
    ),
    MethodKind.BENDERS: (
        Soundness.PROVEN_EQUIVALENT,
        "classical Benders reproduces the monolithic optimum for linear recourse",
        (),
    ),
    MethodKind.GENERALIZED_BENDERS: (
        Soundness.UNKNOWN,
        "GBD reproduces the optimum only when the recourse is convex",
        ("verify convexity of the recourse subproblems",),
    ),
    MethodKind.OUTER_APPROXIMATION: (
        Soundness.PROVEN_EQUIVALENT,
        "outer approximation reproduces the optimum on a convex MINLP "
        "(Duran & Grossmann 1986); OA cuts dominate GBD's aggregated cut",
        ("exact only for convex models; recourse convexity is checked",),
    ),
    MethodKind.LAGRANGIAN: (
        Soundness.RELAXATION,
        "Lagrangian relaxation yields a valid dual bound",
        ("primal recovery may require additional work",),
    ),
}


@dataclass
class DecomposedModel:
    """A solvable decomposition of a model (the Phase 5 IR product).

    Holds the partition (``master`` + ``subproblems``), the correctness
    ``certificate``, the original-space ``var_map``, and the ``structure`` handed
    to the driver. ``solve()`` runs the coordinated solve and returns a
    ``SolveResult`` in the original variable space.
    """

    method: MethodKind
    structure: DecompositionStructure
    master: MasterModel | None
    subproblems: list[SubproblemModel]
    var_map: VariableMapping
    certificate: SoundnessCertificate
    source_model: object = field(repr=False, default=None)
    # Set by ``solve()`` to the driver's ``SolveResult`` so ``summary()`` can
    # surface the *authoritative* soundness (the driver may resolve an UNKNOWN
    # certificate to exact — #391 item 1). ``None`` until solved.
    last_result: object = field(repr=False, default=None, compare=False)

    @property
    def num_blocks(self) -> int:
        """Number of subproblem blocks."""
        return len(self.subproblems)

    def _dispatch(self, **config):
        """Run the coordinated solve and return the driver's ``SolveResult``."""
        model = self.source_model
        if self.method in (MethodKind.NONE, MethodKind.INDEPENDENT_BLOCKS):
            return model.solve(**config)
        if self.method is MethodKind.BENDERS:
            return solve_benders(model, structure=self.structure, **config)
        if self.method is MethodKind.GENERALIZED_BENDERS:
            return solve_gbd(model, structure=self.structure, **config)
        if self.method is MethodKind.OUTER_APPROXIMATION:
            # OA operates on the whole model (no structure needed).
            from discopt.solvers.oa import solve_oa

            return solve_oa(model, **config)
        if self.method is MethodKind.LAGRANGIAN:
            return solve_lagrangian(model, structure=self.structure, **config)
        raise NotImplementedError(f"no reformulation driver for {self.method.label}")

    def solve(self, **config):
        """Run the coordinated solve, dispatching to the shipping driver.

        Returns a ``SolveResult`` in the original variable space. ``NONE`` and
        ``INDEPENDENT_BLOCKS`` fall back to the monolithic ``Model.solve`` (the
        per-block *parallel* solve is Phase 6); Benders/GBD/Lagrangian dispatch to
        their drivers with this decomposition's ``structure``.

        The soundness certificate is enforced as a **post-condition** (#391 item
        1): after the driver returns, :meth:`SoundnessCertificate.check_result`
        raises if a merely-``RELAXATION``/``UNKNOWN`` certificate came back
        ``gap_certified=True`` without the finite dual bound a real proof leaves —
        the failure mode of a future driver that did not self-police. The driver's
        ``gap_certified`` is then the authoritative signal (see :meth:`summary`).
        """
        self.certificate.assert_sound()
        result = self._dispatch(**config)
        # Contract check: the advisory certificate must not be out-claimed by the
        # driver. Raises on an unearned certified gap.
        self.certificate.check_result(result)
        self.last_result = result
        return result

    def schedule(self):
        """Return the :class:`SchedulingGraph` over this decomposition's blocks."""
        from discopt.decomposition.parallel.schedule import build_schedule

        return build_schedule(self)

    def map_subproblems(self, solve_block, backend="sequential"):
        """Run *solve_block* over every subproblem on a parallel *backend*.

        Blocks execute biggest-first (straggler avoidance) on the chosen
        :class:`~discopt.decomposition.parallel.comm.CommunicationLayer`
        (``"sequential"`` | ``"threads"`` | an instance), but results are reduced
        back into **block order** — deterministic regardless of backend or
        schedule (design §13.4). Returns ``[solve_block(sp) for sp in
        subproblems]`` in block order.
        """
        from discopt.decomposition.parallel.comm import select_backend

        comm = select_backend(backend)
        # ``execution_order()`` returns *block ids* (``ScheduledTask.block_id``),
        # which are not guaranteed to equal list positions — only
        # ``build_decomposition`` happens to assign ``block_id == enumerate
        # index``. Resolve block id → position explicitly so a non-contiguous or
        # reordered block-id set maps correctly instead of silently mis-indexing
        # (or raising ``IndexError``). See #391 item 2.
        pos_of_block = {sp.block_id: k for k, sp in enumerate(self.subproblems)}
        if len(pos_of_block) != len(self.subproblems):
            raise ValueError(
                "map_subproblems: subproblem block_ids are not unique "
                f"({[sp.block_id for sp in self.subproblems]}); cannot reduce "
                "results back into block order."
            )
        order = self.schedule().execution_order()
        reordered = [self.subproblems[pos_of_block[b]] for b in order]
        exec_results = comm.map(reordered, solve_block)
        results: list = [None] * len(self.subproblems)
        for k, b in enumerate(order):
            results[pos_of_block[b]] = exec_results[k]
        return results

    def summary(self) -> str:
        """Human-readable multi-line description of the decomposition.

        Before a solve, the certificate's *static* level is shown. After a solve,
        the driver's ``gap_certified`` is authoritative (#391 item 1): a GBD run
        whose convexity gate resolved an ``UNKNOWN`` certificate to an exact solve
        is reported as ``proven equivalent`` here, so the certificate no longer
        disagrees with the run it certified.
        """
        cert_line = self.certificate.summary()
        if self.last_result is not None:
            effective = self.certificate.effective_level(self.last_result)
            # effective_level only differs from the static level when the driver
            # certified optimality, so this branch is exactly the driver-upgrade.
            if effective is not self.certificate.level:
                cert_line += f" [driver certified optimality → {effective.value}]"
        lines = [
            f"DecomposedModel: {self.method.label}",
            f"  {cert_line}",
        ]
        if self.master is not None:
            lines.append(f"  {self.master.summary()}")
        lines.append(f"  {self.num_blocks} subproblem block(s)")
        for sp in self.subproblems[:8]:
            lines.append(f"    {sp.summary()}")
        if self.num_blocks > 8:
            lines.append(f"    … and {self.num_blocks - 8} more")
        return "\n".join(lines)


def _recourse_blocks(model, complicating: list[str]) -> tuple[list[list[str]], dict[str, int]]:
    """Independent recourse blocks left after removing the complicating vars.

    Mirrors the effective-parallelism computation in scoring: components of the
    variable graph once the master (complicating) variables are fixed.
    """
    graph = ModelGraph.from_model(model)
    name_to_idx = {nm: i for i, nm in enumerate(graph.var_names)}
    remove = {name_to_idx[nm] for nm in complicating if nm in name_to_idx}
    projected = [[j for j in clique if j not in remove] for clique in graph.constraint_cliques]
    block_of, _ = kernels.connected_components(graph.num_vars, projected)
    bearing = {block_of[c[0]] for c in projected if c}
    relabel: dict[int, int] = {}
    blocks: list[list[str]] = []
    block_of_var: dict[str, int] = {}
    for j, nm in enumerate(graph.var_names):
        if j in remove:
            block_of_var[nm] = -1
            continue
        b = block_of[j]
        if b not in bearing:
            block_of_var[nm] = -1
            continue
        if b not in relabel:
            relabel[b] = len(blocks)
            blocks.append([])
        blocks[relabel[b]].append(nm)
        block_of_var[nm] = relabel[b]
    return blocks, block_of_var


def build_decomposition(model, candidate: Candidate) -> DecomposedModel:
    """Build a :class:`DecomposedModel` from a chosen :class:`Candidate`.

    Assembles the master/subproblem partition, the variable mapping, and the
    soundness certificate appropriate to the candidate's method. Does not solve.
    """
    method = candidate.method
    structure = candidate.structure or detect_decomposition(model)
    level, rationale, caveats = _CERTIFICATE.get(
        method, (Soundness.HEURISTIC, "no certified reformulation", ())
    )
    # T5.3: for the convexity-dependent methods, actually *run* the convexity
    # classifier so the certificate records a check that happened rather than a
    # static string. GBD stays UNKNOWN when the recourse is not verified convex
    # (its driver then withholds the bound); OA is only proven-equivalent when
    # convex (else it is not applied as an exact method).
    if method in (MethodKind.GENERALIZED_BENDERS, MethodKind.OUTER_APPROXIMATION):
        try:
            from discopt._jax.convexity import classify_oa_cut_convexity

            conv = classify_oa_cut_convexity(model)
            is_convex = conv.objective_is_convex and all(conv.constraint_mask)
        except Exception:  # noqa: BLE001 - a classifier failure is treated as unknown
            is_convex = False
        if is_convex:
            level = Soundness.PROVEN_EQUIVALENT
            rationale = f"recourse verified convex (classifier); {rationale}"
            caveats = ()
        else:
            level = Soundness.UNKNOWN
            rationale = "recourse not verified convex; " + rationale
            caveats = ("bound withheld unless the recourse is convex",)
    certificate = SoundnessCertificate(method, level, rationale, caveats)

    if method in (
        MethodKind.BENDERS,
        MethodKind.GENERALIZED_BENDERS,
        MethodKind.OUTER_APPROXIMATION,
    ):
        complicating = list(structure.complicating_vars)
        blocks, block_of_var = _recourse_blocks(model, complicating)
        master = MasterModel(method, tuple(complicating), tuple(structure.coupling_constraints))
        subproblems = [
            SubproblemModel(block_id=i, variables=tuple(vs)) for i, vs in enumerate(blocks)
        ]
        var_map = VariableMapping(tuple(complicating), block_of_var, len(blocks))
    elif method is MethodKind.LAGRANGIAN:
        blocks = structure.blocks
        master = MasterModel(method, (), tuple(structure.coupling_constraints))
        subproblems = [
            SubproblemModel(block_id=i, variables=tuple(vs)) for i, vs in enumerate(blocks)
        ]
        var_map = VariableMapping((), dict(structure.block_of_var), len(blocks))
    elif method is MethodKind.INDEPENDENT_BLOCKS:
        blocks = structure.blocks
        master = None
        subproblems = [
            SubproblemModel(block_id=i, variables=tuple(vs)) for i, vs in enumerate(blocks)
        ]
        var_map = VariableMapping((), dict(structure.block_of_var), len(blocks))
    else:  # NONE
        master = None
        subproblems = []
        var_map = VariableMapping((), {}, 0)

    return DecomposedModel(
        method=method,
        structure=structure,
        master=master,
        subproblems=subproblems,
        var_map=var_map,
        certificate=certificate,
        source_model=model,
    )


__all__ = ["DecomposedModel", "build_decomposition"]
