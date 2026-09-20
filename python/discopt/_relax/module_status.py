"""Declared status of every ``_relax`` module with no production importer.

Why this file exists
--------------------
``_relax`` contains modules that nothing in the package imports, yet which are
tested, documented, and in some cases user-facing. That made "does this module
have a consumer?" unanswerable as a maintenance question: the import graph says
no, the test suite says yes, and the docs say it is a feature. Issue #1231
answered it with a ``grep`` and got it wrong -- it listed ``chebyshev_model``,
``taylor_model`` and ``ellipsoidal_arith`` as dead when all three are reachable
from ``polyhedral_oa.py``, and attributed to ``differentiable_solve.py`` and
``embedding.py`` documentation that actually belongs to a same-named *function*
and to the unrelated ML-embedding docs. Issue #1347 is the correction.

The rule (binding)
------------------
A ``_relax`` module may exist without a production importer **only if it is
declared here**. Every module under ``discopt._relax`` must satisfy exactly one
of:

1. it is imported (directly or transitively) by another package module, or
2. it has an entry in :data:`MODULE_STATUS` giving its :class:`Status` and a
   one-line reason.

``python/tests/test_relax_module_status.py`` enforces this from the real AST
import graph and fails on an undeclared module *and* on a stale entry, so a
module's status never has to be re-derived from a ``grep`` again.

The statuses
------------
``public``
    An optional, user-facing entry point. Shipped as a feature, imported by the
    user rather than by the solver. Requires a documented entry point and at
    least one test importer.
``tooling``
    Design-time, offline, or CI instrumentation that is deliberately *not* on
    the solve path -- envelope derivation, certification catalogues, training
    pipelines, audit probes. May legitimately have no test importer, but must
    say so in its reason.
``incubating``
    An implemented and tested capability with no production call site yet. The
    reason must name what would wire it in. Requires at least one test importer:
    a capability claimed to work is a capability under test.

Adding a module here is a declaration, not a parking space. ``incubating`` in
particular is a promise that the module is finished enough to be wired in; if
that stops being true, retire the module (delete it with its tests and doc
references) rather than leaving the entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

Status = Literal["public", "tooling", "incubating"]

__all__ = ["MODULE_STATUS", "ModuleStatus", "Status"]


@dataclass(frozen=True)
class ModuleStatus:
    """The declared status of one ``_relax`` module without a production importer.

    Attributes:
        status: Which of the three declared kinds this module is.
        reason: One line saying why it exists with no production importer. For
            ``incubating``, what would wire it in; for ``tooling``, what
            consumes it (or that nothing automated does).
        entry_point: For ``public`` modules, the documented entry point users
            reach it through. Empty for every other status.
    """

    status: Status
    reason: str
    entry_point: str = ""


#: Module name relative to ``discopt._relax`` -> its declared status.
#:
#: Keys are dotted paths below ``discopt._relax`` (``"symbolic.patterns"``, not
#: ``"discopt._relax.symbolic.patterns"``). Anything imported by production code
#: must NOT appear here -- the guard test rejects stale entries.
MODULE_STATUS: Final[dict[str, ModuleStatus]] = {
    # -- public: optional user-facing entry points -------------------------
    "differentiable_solve": ModuleStatus(
        status="public",
        reason=(
            "Unified differentiable LP/QP/MILP/MIQP solve, an optional JAX "
            "feature users call directly; the solver never routes through it. "
            "Note the name collision with the *function* "
            "``discopt._relax.differentiable.differentiable_solve``, which is "
            "production code -- #1231 conflated the two."
        ),
        entry_point="discopt._relax.differentiable_solve.differentiable_solve",
    ),
    "pounce_layer": ModuleStatus(
        status="public",
        reason=(
            "Differentiable JAX layers over the POUNCE solver, for composing a "
            "solve into jax.grad/jit/vmap pipelines. User-facing and optional; "
            "documented by docs/notebooks/differentiable_pounce_layer.ipynb."
        ),
        entry_point="discopt._relax.pounce_layer.make_nlp_layer",
    ),
    # -- tooling: design-time / offline / CI instrumentation ---------------
    "claim_audit": ModuleStatus(
        status="tooling",
        reason=(
            "Read-only claim-boundary audit instrumentation (#632) behind the "
            "claim differential gate; consumed by tests/support/"
            "claim_differential.py, never by the solver."
        ),
    ),
    "icnn_trainer": ModuleStatus(
        status="tooling",
        reason=(
            "Offline training pipeline that *produces* the pretrained ICNN "
            "registry that learned_relaxations.py consumes at solve time. "
            "Requires the equinox/optax extras; importing it during a solve "
            "would violate the JAX-free solve path."
        ),
    ),
    "symbolic.certified_learned": ModuleStatus(
        status="tooling",
        reason=(
            "Design-time certification of guaranteed-convex networks as sound "
            "relaxations (Phase 8 prototype), in the design-time symbolic "
            "toolkit rather than on the solve path."
        ),
    ),
    "symbolic.patterns": ModuleStatus(
        status="tooling",
        reason=(
            "Design-time catalogue of certified relaxation/cut patterns and the "
            "sole importer of symbolic.gp_hull and symbolic.signed_signomial. "
            "No automated consumer: it is read and code-generated from by hand "
            "(design/relaxation-patterns.md)."
        ),
    ),
    "symbolic.registry": ModuleStatus(
        status="tooling",
        reason=(
            "Design-time certification catalogue for the symbolic domain packs "
            "and their sole importer (domains.chemeng/gas/power). No automated "
            "consumer: certify_all is run by hand when a pack changes."
        ),
    ),
    # -- incubating: tested, no production call site yet -------------------
    "embedding": ModuleStatus(
        status="incubating",
        reason=(
            "Gray-code SOS2 embedding helper owned by the AMP lane (#44/#86) "
            "and exercised through bilinear_lambda's kwargs in test_amp*. Wired "
            "in when AMP's piecewise convex-hull builder adopts the logarithmic "
            "encoding in place of one binary per interval."
        ),
    ),
    "monotonicity": ModuleStatus(
        status="incubating",
        reason=(
            "SUSPECT-style monotonicity proofs via interval AD, the counterpart "
            "to _relax/convexity. Wired in when a dispatcher consumes the "
            "verdict for DCP composition (docs/design/relaxation-catalog.md)."
        ),
    ),
    "operator_relaxations": ModuleStatus(
        status="incubating",
        reason=(
            "Operator-specific helpers (branch-safe tan ranges, periodic "
            "critical points) for the AMP relaxation builders; wired in when "
            "AMP's builder dispatches trigonometric atoms through them."
        ),
    ),
    "soc_cuts": ModuleStatus(
        status="incubating",
        reason=(
            "Second-order-cone outer-approximation cut family, the convex-cone "
            "counterpart to psd_cuts. Wired in when a separator for it is added "
            "to the cut loop in cutting_planes.py."
        ),
    ),
}
