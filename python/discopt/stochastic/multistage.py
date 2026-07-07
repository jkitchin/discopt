"""Multistage and integer-recourse stochastic programming (reserved).

These methods are solver-heavy and not yet implemented; rather than ship an
untested solve, they refuse loudly and point at the design. See
``docs/dev/stochastic-module-plan.md`` §1, §5 (Phase 3).

* **Multistage** (>2 stages, scenario *trees*) → nested Benders / SDDP. The
  decomposition engine reserves ``MethodKind.NESTED_BENDERS`` and a ``SCENARIO``
  graph kind for this.
* **Integer L-shaped** (Laporte–Louveaux) for *integer recourse*, where the recourse
  value function is nonconvex/discontinuous so ordinary L-shaped optimality cuts are
  invalid. Two-stage integer-recourse problems should currently be solved by the
  **extensive-form MINLP** (`build_extensive_form`, still exact).
"""

from __future__ import annotations

__all__ = ["solve_multistage", "integer_lshaped"]


def solve_multistage(*args, **kwargs):
    """Not implemented — multistage (nested Benders / SDDP) is reserved.

    Use :func:`~discopt.stochastic.extensive_form.build_extensive_form` for a
    two-stage problem, or model a small scenario *tree* directly as its
    deterministic equivalent. A nested-Benders driver over the reserved
    ``MethodKind.NESTED_BENDERS`` slot is future work.
    """
    raise NotImplementedError(
        "multistage stochastic programming (nested Benders / SDDP) is not yet "
        "implemented. For two stages use build_extensive_form / solve_lshaped; the "
        "NESTED_BENDERS engine slot is reserved for the multistage driver."
    )


def integer_lshaped(*args, **kwargs):
    """Not implemented — integer-recourse L-shaped (Laporte–Louveaux) is reserved.

    L-shaped optimality cuts are invalid when the recourse has integer variables
    (the recourse value function is nonconvex/discontinuous). Solve integer-recourse
    two-stage problems with the extensive-form MINLP
    (:func:`~discopt.stochastic.extensive_form.build_extensive_form`), which is exact.
    """
    raise NotImplementedError(
        "integer-recourse L-shaped (Laporte–Louveaux) is not yet implemented. Integer "
        "recourse makes the recourse value function nonconvex, so ordinary optimality "
        "cuts are invalid; solve via build_extensive_form (exact) for now."
    )
