"""Soundness certificates and variable mappings for reformulations.

The correctness spine of the reformulation IR (design §8.2). A
:class:`SoundnessCertificate` records *why* a decomposition represents the
original model — exactly, as a relaxation, or property-dependent — and refuses
outright to run a method that carries **no** guarantee (``HEURISTIC``). A
:class:`VariableMapping` records which original variables live in the master
versus which subproblem block, so results come back in the user's variable space
and the partition is inspectable.

**The certificate is advisory, not the contract.** ``is_sound()`` admits
``RELAXATION`` and ``UNKNOWN`` (only ``HEURISTIC`` is refused), so the *authority*
on whether a run actually certified optimality is the **driver's** returned
``SolveResult`` (``status`` / ``bound`` / ``gap_certified``), which self-polices:
GBD withholds its bound unless a convexity classifier *proves* the recourse
convex, and Lagrangian certifies only when a recovered primal meets the dual
bound. :meth:`SoundnessCertificate.check_result` turns that division of labour
into a checked **post-condition** — after ``DecomposedModel.solve()`` it raises
if a merely-``RELAXATION``/``UNKNOWN`` certificate returns ``gap_certified=True``
*without* the finite dual ``bound`` that a real proof leaves behind (the only way
a non-self-policing future driver could smuggle an unearned certificate past the
gate), and it reports the driver's ``gap_certified`` as the authoritative signal
so a GBD ``UNKNOWN`` certificate that the driver resolved to *exact* no longer
disagrees with the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from discopt.decomposition.advisor.types import MethodKind, Soundness


@dataclass(frozen=True)
class VariableMapping:
    """Original-variable ↔ (master | subproblem-block) mapping.

    Attributes
    ----------
    master_vars : tuple[str, ...]
        Variables held in the coordinating master (complicating vars for Benders;
        empty for Lagrangian / independent-block decompositions).
    block_of_var : dict[str, int]
        Subproblem block id per variable; ``-1`` for a master variable.
    num_blocks : int
        Number of subproblem blocks.
    """

    master_vars: tuple[str, ...]
    block_of_var: dict[str, int]
    num_blocks: int

    def role(self, var: str) -> str:
        """Human label for a variable's role: ``"master"`` or ``"block{i}"``."""
        if var in self.master_vars:
            return "master"
        b = self.block_of_var.get(var, -1)
        return f"block{b}" if b >= 0 else "master"


@dataclass(frozen=True)
class SoundnessCertificate:
    """Why a reformulation is (or is not) a faithful stand-in for the monolith.

    Attributes
    ----------
    method : MethodKind
        The decomposition method certified.
    level : Soundness
        ``PROVEN_EQUIVALENT`` (same optimum), ``RELAXATION`` (valid bound),
        ``UNKNOWN`` (property-dependent, e.g. GBD needs convexity), or
        ``HEURISTIC`` (no guarantee).
    rationale : str
        One-line justification.
    caveats : tuple[str, ...]
        Conditions the guarantee depends on.
    """

    method: MethodKind
    level: Soundness
    rationale: str
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def is_sound(self) -> bool:
        """True unless the certificate is merely heuristic.

        A relaxation (valid bound) and an unknown-but-not-refuted certificate are
        both allowed to run; only a purely heuristic reformulation is refused by
        default (correctness-first, design goal #1).
        """
        return self.level is not Soundness.HEURISTIC

    def assert_sound(self) -> None:
        """Raise :class:`ValueError` if the reformulation is unsound to run."""
        if not self.is_sound():
            raise ValueError(
                f"refusing to build a {self.method.label} reformulation: {self.rationale}"
            )

    def check_result(self, result) -> None:
        """Post-condition: the driver's ``result`` must not out-claim this cert.

        ``is_sound()`` is deliberately permissive (``RELAXATION`` / ``UNKNOWN``
        are allowed to run), so the *driver* — not this certificate — is the
        authority on whether a run certified optimality. This method enforces the
        contract that ties the two together: a static ``RELAXATION`` / ``UNKNOWN``
        certificate is allowed to return ``gap_certified=True`` **only** when the
        driver actually proved it, and the proof always leaves a **finite dual
        ``bound``** behind (a valid lower bound for a min-sense problem). A
        ``gap_certified=True`` with no finite bound from such a certificate would
        be an *unearned* certificate — exactly what a future driver that failed to
        self-police would produce — so we raise rather than let it pass.

        ``SolveResult.__post_init__`` already downgrades ``gap_certified`` when the
        bound is absent/non-finite for a *non-infeasible* status; this is the
        decomposition-layer backstop that also refuses the case
        (``status="infeasible"``) that guard exempts, keeping the certificate a
        real checked post-condition rather than documentation.

        Raises
        ------
        AssertionError
            If the result claims a certified gap the certificate cannot justify.
        """
        # PROVEN_EQUIVALENT may certify freely; HEURISTIC never reaches solve()
        # (assert_sound refuses it). Only the permissive middle needs a backstop.
        if self.level in (Soundness.PROVEN_EQUIVALENT, Soundness.HEURISTIC):
            return
        if not getattr(result, "gap_certified", False):
            return
        # An infeasibility certificate legitimately carries bound=None; a certified
        # *optimality* gap from a relaxation/unknown method must show the finite
        # dual bound the driver's proof produced.
        if getattr(result, "status", None) == "infeasible":
            return
        bound = getattr(result, "bound", None)
        try:
            b = float(bound) if bound is not None else None
            finite = b is not None and b == b and abs(b) < 1e300  # b == b filters NaN
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise AssertionError(
                f"{self.method.label}: driver returned gap_certified=True from a "
                f"{self.level.value} certificate without a finite dual bound — the "
                "certificate cannot justify a certified optimality gap. The driver "
                "must withhold gap_certified (report a valid bound) when it did not "
                "prove optimality (see #391 item 1)."
            )

    def effective_level(self, result) -> Soundness:
        """Authoritative soundness for *result*, driver-first (#391 item 1).

        The stored ``level`` is the *static* estimate made before the solve; the
        driver may resolve it. In particular GBD attaches an ``UNKNOWN``
        certificate but, when its convexity gate proves the recourse convex,
        returns ``gap_certified=True`` — a genuinely *exact* run. Surfacing the
        driver's verdict avoids the user-facing disagreement where the run
        certified optimality yet the attached certificate still reads ``UNKNOWN``.
        Never *upgrades* past what the driver proved: a certified gap upgrades a
        permissive certificate to ``PROVEN_EQUIVALENT``; otherwise the static
        level stands.
        """
        if self.level in (Soundness.RELAXATION, Soundness.UNKNOWN) and getattr(
            result, "gap_certified", False
        ):
            return Soundness.PROVEN_EQUIVALENT
        return self.level

    def summary(self) -> str:
        """One-line human-readable certificate."""
        caveat = f" (caveats: {'; '.join(self.caveats)})" if self.caveats else ""
        return f"{self.method.label}: {self.level.value} — {self.rationale}{caveat}"


__all__ = ["SoundnessCertificate", "VariableMapping"]
