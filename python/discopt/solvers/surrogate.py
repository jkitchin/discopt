"""Surrogate-based (model-based) derivative-free optimization for expensive objectives.

Selected with ``solve_surrogate(model, ...)``. Minimizes a bounded model by
fitting a cheap *response surface* to the points evaluated so far, maximizing an
acquisition function over that surface to decide where to look next, evaluating
there, and repeating. Like DIRECT (:mod:`discopt.solvers.direct`) it needs only a
finite box and the ability to evaluate the model at a point — no derivatives, no
relaxation, no convexity — and it serves the same class of model: an objective
whose body is an opaque ``dm.custom`` (``CustomCall``) outside the reduced-space
MCBox scope, which has no algebraic relaxation and would otherwise degrade to a
single local NLP (or be refused outright when integers are present).

The difference is the *regime*. DIRECT spends its budget on geometry and is happy
to take thousands of samples. This backend exists for the case where a single
evaluation is expensive — a CFD run, a process simulation, a lab experiment — and
you can afford tens of them. It spends real computation (a linear solve, a
maximum-likelihood fit, and a global optimization of the acquisition) between
evaluations in order to make each one count.

**It returns no certificate, and must never claim one.** ``bound`` and ``gap`` are
``None``, ``gap_certified`` is ``False``, and the status is never ``"optimal"``
(nor ``"infeasible"`` — a sampling method cannot prove infeasibility). A surrogate
is a *model of* the objective, not a bound on it: an interpolant says nothing
rigorous about the function between its data points, so reporting the incumbent as
a bound would be a false certificate (CLAUDE.md §1).

Note the distinction that the tests pin, because it is the whole point of the
design: the *acquisition subproblem* is an ordinary algebraic model and is
routinely solved to **certified global optimality** by discopt's own spatial
branch-and-bound (``gap_certified=True`` on the subproblem), while the *outer*
result reports no certificate at all. Certifying where to sample next is not
certifying the answer.

Cost model: what you are paying for
-----------------------------------
Nearly all the wall clock is the acquisition solve, not the objective. Measured
on branin with a free objective, ``max_evals=30``: the 6-point initial design
costs 0.29 s in total, and every subsequent evaluation costs almost exactly
``acquisition_time_limit`` (19.0 s mean against a 20 s limit). That is the
intended trade — the backend exists for objectives where one evaluation dwarfs
20 s of solver time — but it means ``solver="surrogate"`` on a cheap objective is far slower than
``solver="direct"`` for a worse answer, and the choice between them is a
statement about *your* evaluation cost.

Do not shorten ``acquisition_time_limit`` to make it feel faster. It is tempting:
with the default cubic kernel the acquisition never certifies (see the kernel
table below), so the budget looks wasted. It is not wasted — it is buying primal
solution quality. Measured, ``max_evals=30``, relative error at 20 s vs 2 s
(**taken under the pre-#1036 initial-design rule** — the conclusion is about
``acquisition_time_limit``, which #1036 did not touch, but the absolute numbers
were produced with a 15-point rather than a 6-point design and have not been
re-taken):

=================  ==========  =========
function           20 s        2 s
=================  ==========  =========
branin             0.2156      0.2029
six_hump_camel     **0.0164**  0.8063
hartman_3          **0.0098**  0.0103
=================  ==========  =========

branin alone says 2 s is free money; six_hump_camel is 49x worse for it. The
default stays at 20 s. This is recorded because the branin-only reading was
briefly acted on during development and was wrong — a single-instance tuning of
exactly the kind CLAUDE.md §2 forbids.

Two surrogate families
----------------------
``surrogate="rbf"`` (**default**) — a radial basis function interpolant

.. math::  s(x) = \\sum_i \\lambda_i \\varphi(\\lVert x - x_i \\rVert) + p(x)

with a cubic (``\\varphi(r) = r^3``), thin-plate (``r^2 \\log r``) or linear
(``r``) kernel and a linear tail ``p``. Fitting is one symmetric linear solve; the
tail is what makes the system nonsingular for a conditionally positive definite
kernel, and what lets the surrogate extrapolate a trend rather than flatten to the
mean.

RBF is the default rather than a Gaussian process for three reasons, in order of
weight here:

1. **Integer variables work natively.** discopt is a MINLP solver; a black-box
   method that cannot take integers is half a feature. The RBF interpolant is
   defined on any point set, the acquisition subproblem below carries the
   integrality straight into discopt's B&B, and nothing in the fit assumes a
   continuous domain.
2. **Fitting is cheap and has no failure mode of its own.** One ``(m+n+1)``
   symmetric solve, no likelihood to maximize, so no local optimum in the
   hyperparameters to land in and no silent "the fit did not converge".
3. **It degrades better with dimension.** Kriging's MLE has one length scale per
   dimension to estimate from ``m`` points; the RBF has none.

Reference: Costa & Nannicini, *RBFOpt: an open-source library for black-box
optimization with costly function evaluations*, Math. Prog. Comp. 10(4), 2018.

``surrogate="kriging"`` — DACE/EGO kriging with expected improvement, for smooth,
low-dimensional, genuinely expensive objectives. Correlation
``R_ij = exp(-Σ_h θ_h |x_h^i - x_h^j|^{p_h})``, hyperparameters by maximum
likelihood (Cholesky + :mod:`scipy.optimize`), predictor ``ŷ(x)`` and standard
error ``s(x)``. Reference: Jones, Schonlau & Welch, *Efficient global optimization
of expensive black-box functions*, J. Global Optim. 13(4), 1998.

**The nugget is not optional.** Vanilla EGO interpolates: it forces the surface
through every observation, which on a noisy objective means forcing it through the
measurement error, producing a wildly wiggly surface and a variance that is
identically zero at points you do not actually know. ``nugget`` adds ``η`` to the
diagonal of ``R``; the default ``1e-8`` is a conditioning jitter (the model still
interpolates to ~1e-4 of the observation spread), and ``nugget="auto"`` estimates
it by MLE alongside ``θ``, turning the interpolant into a regressor. This is the
single most dated assumption in the 1998 paper and it is cheap to fix.

Acquisition
-----------
**Kriging → expected improvement**, the closed form

.. math::  E[I] = (f_{\\min} - \\hat y)\\Phi(z) + s\\,\\phi(z), \\quad z = (f_{\\min} - \\hat y)/s,

which is exactly ``E[max(f_min - Y, 0)]`` for ``Y ~ N(ŷ, s²)`` and is checked
against numerical integration of that definition in the unit tests.

**RBF → CORS**, "constrained optimization using response surfaces" (Regis &
Shoemaker, *J. Global Optim.* 31, 2005): minimize the surrogate subject to being
at least ``δ_k`` away from every point sampled so far,

.. math::  \\min_x s(x) \\quad \\text{s.t.} \\quad \\lVert x - x_i \\rVert^2 \\ge \\delta_k^2,

with ``δ_k = β_k Δ_k`` cycling ``β`` through ``distance_cycle`` (the paper's own
schedule ``0.9, 0.75, 0.25, 0.05, 0.03, 0``) and ``Δ_k`` the maximin distance
achievable in the box. A large ``β`` forces exploration into unvisited regions; a
small one lets the search refine near the surrogate's minimum. Chosen over
Gutmann's bumpiness measure for two reasons: it needs no target value to be
guessed, and — decisively for this implementation — the constraint form is what
makes the subproblem *algebraic and certifiable*.

The distance constraint is also what makes "propose an already-sampled point"
structurally impossible rather than merely unlikely: ``δ_k`` is floored at
``min_distance > 0``, so a sampled point is infeasible for the subproblem. A
distance *penalty* would only have made it unattractive, and an acquisition that
can re-propose a sampled point stalls the whole method — the surrogate does not
change, so the next iteration proposes the same point again.

Certified acquisition, and what was actually measured
-----------------------------------------------------
Both acquisitions are expressible as ordinary discopt algebraic models — ``erf``
is a discopt intrinsic with a tight envelope (``_relax/envelopes.py::relax_erf``),
``exp``/``sqrt``/``log`` likewise — so the acquisition subproblem can be handed to
discopt's own spatial branch-and-bound and solved to *certified global
optimality*. That is the thing Jones et al. built a bespoke branch-and-bound for
in 1998 and that modern BO libraries approximate with multistart gradient ascent.

Two reformulations make the EI subproblem expressible at all, and both are worth
stating because the obvious formulation does not work:

* **The division is removed by lifting.** ``z = (f_min - ŷ)/s`` is a division by a
  quantity that goes to zero at every sampled point, so ``z`` is unbounded and the
  constraint ``z·s = d`` has no usable relaxation. But
  ``d Φ(u) + s φ(u)`` has derivative ``φ(u)(d - s u)`` in ``u``, so its maximum
  over ``u`` is at ``u = d/s`` — that is, ``EI(x) = max_u [d(x)Φ(u) + s(x)φ(u)]``.
  Maximizing jointly over ``(x, u)`` therefore *is* maximizing EI, with no
  division anywhere and with ``u`` confined to a tight ``[-8, 8]`` box (outside
  it the bracket is ``d`` or ``0`` to 1e-15).
* **The dense bilinear form is removed by whitening.** ``s²`` contains
  ``rᵀR⁻¹r``, an ``m²``-term indefinite quadratic in the correlation vector with
  entries as large as ``‖R⁻¹‖``. Writing ``v = L⁻¹r`` (``L`` the Cholesky factor,
  so ``Lv = r`` is a triangular *linear* constraint) turns it into ``Σ v_i²`` —
  separable, and every cross term gone.

**Measured result, and it does not support the ambition (CLAUDE.md §4 — the
measurement wins).** On Branin with a Gaussian correlation and a 60 s subproblem
budget, the lifted+whitened EI model finds the true acquisition maximum to 5
significant figures at every design size tried — but the *bound* does not close:

=====  ===========  ==============  ==============  ==========
``m``  B&B EI       dense-grid EI   B&B dual bound  certified?
=====  ===========  ==============  ==============  ==========
8      0.602102     0.602089        3.82            no
12     0.263069     0.263060        23.1            no
20     0.415208     0.415191        597             no
30     1.50919      1.50917         4871            no
=====  ===========  ==============  ==============  ==========

So for kriging, discopt's B&B is an excellent *primal* acquisition optimizer and
not (yet) a certifying one, and the backend says which happened rather than
implying the good case.

**The CORS subproblem is a different story, and the kernel decides it.** Branin,
a genuinely fitted surrogate, ``δ = 0.1``, 20 s subproblem budget, relative gap
achieved:

==========  ==============  =============  =============  =============
kernel      m = 6           m = 8          m = 12         m = 20
==========  ==============  =============  =============  =============
``linear``  **6.5e-5 ✓**    **4.2e-5 ✓**   **6.7e-6 ✓**   **2.6e-8 ✓**
cubic       **6.6e-8 ✓**    0.19           6.0            27.7
thin_plate  0.019           33.1           2.2            39.3
==========  ==============  =============  =============  =============

The mechanism is visible in the fit rather than in the solver: ``max |λ|`` for
the cubic kernel runs 10 → 23 → 61 → 164 across those four design sizes, while
for the linear kernel it stays at 2.9 → 4.7 → 3.4 → 4.3. A cubic kernel's
coefficients grow as the design fills (``r³`` is a very flat basis near the data,
so interpolation buys accuracy with large opposing coefficients), and a sum of
large opposing terms is exactly what a McCormick-style relaxation cannot bound
tightly. ``φ(r) = r`` is also a *simpler* expression — one ``sqrt`` of the
squared distance rather than ``q√q`` — so each term relaxes better in the first
place.

That is why ``rbf_kernel`` matters more than a kernel choice usually does here,
and why ``"linear"`` is the kernel to pick when the certified path is the point.
It is not the default (see ``rbf_kernel``), because the choice of *default* rests
on optimization quality rather than on subproblem tractability.

``acquisition_optimizer`` controls this explicitly:

* ``"auto"`` (default) — hand the subproblem to discopt's B&B; use the point it
  returns and **log whether it came back certified**; fall back to multistart only
  if the model cannot be built or the solve raises.
* ``"certified"`` — require ``gap_certified=True`` from the subproblem and raise
  otherwise. Refuses loudly rather than quietly degrading (CLAUDE.md §3).
* ``"multistart"`` — skip B&B entirely; stratified multistart plus an L-BFGS-B
  polish on the analytic acquisition.

Every path increments a distinct counter reported in ``solver_stats``
(``surrogate/acq_certified``, ``surrogate/acq_bb_uncertified``,
``surrogate/acq_multistart``, ``surrogate/acq_failures``), so which optimizer ran
is a measurement rather than a claim.

Sample efficiency, measured against DIRECT
------------------------------------------
The premise is that each evaluation is expensive, so the only figure of merit
that matters is *evaluations to a target accuracy*. Measured head to head against
``solvers.direct``'s engine on the same functions, the same boxes and the same
target — 1e-2 relative error, 60-evaluation budget, median of seeds 0-2 for this
backend (DIRECT is deterministic):

================  ==========  ======  ==============
function          surrogate   DIRECT  factor
================  ==========  ======  ==============
six_hump_camel    23          137     6.0x
branin            36          69      1.9x
hartman_3         50          79      1.6x
ackley_2          44.5 (2/3)  67      1.5x
goldstein_price   never 0/3   75      **loss**
================  ==========  ======  ==============

Re-measured 2026-08-29, after issue #1036 resized the initial design from
``max(n+2, min(10n, max_evals // 2))`` to ``2(n+1)``; DIRECT, being
deterministic, did not move. Read this table as the 3-seed slice it is rather
than as the verdict on that change: ``six_hump_camel`` (32 → 23) and ``branin``
(38 → 36) improved, ``goldstein_price`` — already the loss — got worse, and the
other two moved in both directions at once. ``hartman_3``'s median rose 46 → 50
because all three seeds now reach the tolerance where only two did; ``ackley_2``
fell 48 → 44.5 because one of the three stopped reaching it inside 60
evaluations. Three seeds cannot separate those from noise. The verdict is the
8-function, 12-seed panel in
``docs/dev/surrogate-initial-design-2026-08-29.md``, whose mean evaluations to
1e-2 fall 67.8 → 60.0.

The old rule also made the design, and therefore the whole trajectory, a
function of ``max_evals``. Two runs at different budgets were different searches
rather than one search and its continuation, which invalidated every "reached
the tolerance at evaluation ``k``, so budget ``B > k`` has headroom" statement
made about this backend. The default design size is now a function of the
dimension alone; ``test_a_larger_budget_continues_the_same_search`` pins it.

Two things to take from that rather than from an adjective:

* the advantage on smooth low-dimensional problems is a factor of ~2, not the
  order of magnitude a surrogate wins by in harder settings. DIRECT is a strong
  baseline here and saying otherwise would be inventing a result;
* **Goldstein-Price is a genuine loss, and the reason is objective scaling.** Its
  values span 3 to ~10⁶ on the box, so an interpolant fitted to raw values spends
  its resolution on the 10⁶ region and is nearly flat where the optimum is. The
  known remedy is a monotone transformation of the objective before fitting
  (RBFOpt's "objective function transformation", Costa & Nannicini §3.3), which is
  a real follow-up rather than a tuning knob. It is **not** implemented here, and
  ``rbf_ridge``/``rbf_kernel`` do not substitute for it.

Constraints
-----------
Constrained models are handled by fitting the surrogate to the same GLce-style
merit the DIRECT backend ranks by: total violation while nothing feasible is
known, then the objective for near-feasible points and ``f + viol + |f - f_min|``
for the rest. That penalty needs no weight to be tuned and denies an infeasible
point credit for a low objective. It is a deliberate v0: the acquisition sees one
scalar and the model's own constraints are not carried into the subproblem.

Follow-ups this module is shaped for but does NOT implement
-----------------------------------------------------------
Two extensions are left as clean seams rather than built, because each is a
project in itself and neither is needed for the single-point serial case:

* **Trust-region restriction of the acquisition domain** (TuRBO-style, Eriksson et
  al., NeurIPS 2019). :func:`_acquisition_domain` is the seam: it returns the box
  the subproblem is posed over and today always returns the full box. Restricting
  it to a trust region around the incumbent — with the usual expand-on-success /
  shrink-on-failure counters — is a local change there plus the counters.
* **Batch / q-point proposals.** :meth:`_SurrogateSearch.propose` already returns a
  *list* of candidate points and the driver loop evaluates the whole list, so a
  q-EI or a CORS-with-q-distance-constraints rule slots in without disturbing the
  loop or the budget accounting. The evaluation of a batch would then parallelize
  the way ``solve_direct``'s ``n_jobs`` does.

References
----------
Jones, Schonlau & Welch, *J. Global Optim.* 13(4), 1998 — EGO, kriging + EI.
Regis & Shoemaker, *J. Global Optim.* 31, 2005 — CORS-RBF.
Gutmann, *J. Global Optim.* 19, 2001 — the RBF method and bumpiness.
Costa & Nannicini, *Math. Prog. Comp.* 10(4), 2018 — RBFOpt.
Sacks, Welch, Mitchell & Wynn, *Statist. Sci.* 4(4), 1989 — DACE.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from discopt.modeling.core import Constant, Model, SolveResult
from discopt.solvers._dfo_common import build_oracle, glce_merit

logger = logging.getLogger(__name__)

__all__ = [
    "solve_surrogate",
    "expected_improvement",
    "RBFSurrogate",
    "KrigingSurrogate",
    "SurrogateStats",
]

#: RBF kernels. All three are conditionally positive definite of order <= 2, so a
#: linear tail makes the interpolation system nonsingular for any affinely
#: independent design — one code path covers all three.
_RBF_KERNELS = ("cubic", "thin_plate", "linear")

#: Bound on the lifted variable ``u`` in the certified EI model. ``Φ(-8) ≈ 6e-16``
#: and ``φ(8) ≈ 5e-15``, so restricting ``u`` to ``[-8, 8]`` changes the value of
#: ``max_u [dΦ(u) + sφ(u)]`` by less than 1e-14 relative — far below any tolerance
#: the acquisition is compared at, while giving the relaxation a tight box.
_U_LIMIT = 8.0

#: Guard inside ``sqrt``/``log`` in the algebraic RBF acquisition so the model is
#: defined on the closed box even where the distance-to-a-design-point vanishes.
#: The distance constraint keeps the argument at ``>= δ² >= min_distance²`` at any
#: feasible point, so this only protects the relaxation's evaluation of infeasible
#: corners.
_DOMAIN_EPS = 1e-12

#: Relative residual above which an RBF "solution" is rejected and refit by least
#: squares. Applied as ``||A x - rhs|| <= tol * (1 + ||rhs||)`` on the standardized
#: system, so it is scale-free.
#:
#: Set from the measured distribution, not from a rule of thumb. Over the 310 fits
#: the five-function convergence panel performs (branin, six_hump_camel, ackley_2,
#: hartman_3, goldstein_price), the relative residual of a healthy ``solve`` has
#: median 2.8e-13, 99th percentile 5.7e-6 and maximum 8.4e-6 — routinely far worse
#: than the ~1e-15 an "eps-sized" intuition suggests, because these systems are
#: genuinely ill-conditioned by 60 points. A rank-deficient design, by contrast,
#: measures 5.3e+16. That leaves 21 orders of magnitude of daylight, so 1e-3 sits
#: ~120x above anything healthy and ~5e19x below the failure it exists to catch.
#: (``sqrt(eps)`` was tried first and was wrong in the dangerous direction: it fired
#: on ordinary 49-to-69-point fits, turning a rare guard into the common path.)
#:
#: Falling back can never make the residual worse — ``lstsq`` returns the
#: minimum-norm least-squares solution, which minimizes exactly this quantity.
_RBF_RESIDUAL_TOL = 1e-3

#: Above this dimension a surrogate method degrades sharply — the design needed to
#: pin down a response surface grows with dimension, and the acquisition's own
#: global optimization gets harder at the same time. Warn, never refuse.
_DIMENSION_WARN_THRESHOLD = 15


def _default_design_size(n_vars: int) -> int:
    """Default initial-design size: ``2 (n + 1)``, a function of dimension ONLY.

    Two properties, and the first is the one that matters:

    **It does not depend on ``max_evals``.** The previous rule was
    ``max(n+2, min(10n, max_evals // 2))``, which made the design — and therefore
    the entire trajectory — a function of the budget. Two runs at different
    budgets were then two different searches rather than one search and its
    continuation, so "the incumbent first reached the tolerance at evaluation
    ``k``, therefore a budget of ``B > k`` has headroom" was not a valid argument
    about this backend. Measured on the ``on_evaluation`` traces at seed 0
    (``scratchpad/i1036/nesting_probe.py``): 17 of 30 budget pairs over
    ``{40, 46, 60, 80, 100}`` diverged at **evaluation 1**. With a
    dimension-only size the same probe finds 0 of 30 — a larger budget is now
    literally the smaller-budget run, continued.

    **It is smaller than the ``10n`` it replaces, and that is measured.** ``10n``
    is the sizing rule for fitting a response surface *once*, over a design chosen
    in advance; it is not a budget-allocation rule for a serial adaptive search,
    where every design point is a point not spent on the acquisition. Evaluations
    to 1e-2 relative error, 12 seeds per cell, ``max_evals=100``, non-reached
    counted as the full budget (``docs/dev/surrogate-initial-design-2026-08-29.md``
    has the per-function table):

    ==============================  ========  ============  ========  ========
    panel mean, evals to 1e-2       ``10n``   ``2(n+1)``    ``n+2``   ``5n``
    ==============================  ========  ============  ========  ========
    RBF, 8 functions                68.1      **60.0**      59.8      64.7
    kriging, 6 functions            65.6      **61.1**      —         62.2
    ==============================  ========  ============  ========  ========

    ``n+2`` edges out ``2(n+1)`` on the RBF mean but collapses on
    ``goldstein_price`` (0 of 12 seeds reach the tolerance, against 4 of 12 for
    ``10n``); ``2(n+1)`` is within 0.4% of it on the mean while matching ``10n``
    there. It also stays at or above the ``n+1`` an RBF with a linear tail needs
    to be fitted at all, for every ``n``.

    Not tuned per family: the two tables above point the same way, so one rule
    covers both. ``n_initial`` overrides it, and a caller who knows their
    objective is sharply scaled or densely multimodal should raise it — those are
    the two shapes on this panel that preferred a larger design.
    """
    return 2 * (int(n_vars) + 1)


@dataclass
class SurrogateStats:
    """Counters reported through ``SolveResult.solver_stats``.

    The acquisition counters are the ones to read first: they are how "the
    acquisition was solved to certified global optimality" stops being a claim
    and becomes a measurement.
    """

    evals: int = 0
    cache_hits: int = 0
    iterations: int = 0
    initial_design: int = 0
    fits: int = 0
    fit_fallbacks: int = 0
    ill_conditioned_fits: int = 0
    undefined_points: int = 0
    acq_certified: int = 0
    acq_bb_uncertified: int = 0
    acq_multistart: int = 0
    acq_failures: int = 0
    duplicate_escapes: int = 0
    improvements: int = 0
    feasible_found: int = 0
    local_solves: int = 0
    local_improvements: int = 0
    local_failures: int = 0

    def as_dict(self) -> dict[str, float]:
        return {f"surrogate/{k}": float(v) for k, v in vars(self).items()}


# ══════════════════════════════════════════════════════════════════════════════
# Expected improvement — pure numerics, unit-testable without a Model
# ══════════════════════════════════════════════════════════════════════════════


def expected_improvement(
    y_hat: np.ndarray,
    s: np.ndarray,
    f_min: float,
    xi: float = 0.0,
) -> np.ndarray:
    r"""``E[max(f_min - xi - Y, 0)]`` for ``Y ~ N(y_hat, s^2)`` (Jones et al. 1998).

    The closed form is ``d Φ(z) + s φ(z)`` with ``d = f_min - xi - y_hat`` and
    ``z = d/s``. Two properties it must have, both pinned in the unit tests
    against their definitions rather than against a second implementation:

    * it decreases in ``y_hat`` (``∂EI/∂ŷ = -Φ(z) < 0``) and increases in ``s``
      (``∂EI/∂s = φ(z) > 0``) — the two derivative identities the certified
      maximization rests on; and
    * it agrees with numerical integration of ``∫ max(f_min - y, 0) N(y; ŷ, s²) dy``.

    ``s = 0`` is the deterministic limit and is handled exactly rather than by a
    fudge: there is no uncertainty, so the improvement is ``max(d, 0)``. Dividing
    instead would produce a NaN that silently propagates into the acquisition
    ranking.

    Parameters
    ----------
    y_hat, s
        Predicted mean and standard error, broadcastable to a common shape.
    f_min
        Best objective value observed so far.
    xi
        Optional exploration margin subtracted from ``f_min``; ``0`` is the
        textbook EI. A positive ``xi`` demands a larger improvement before a
        point looks attractive, biasing toward exploration.
    """
    from scipy.special import ndtr

    y_hat = np.asarray(y_hat, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    d = float(f_min) - float(xi) - y_hat
    positive = s > 0.0
    z = np.divide(d, s, out=np.zeros_like(d * s), where=positive)
    closed = d * ndtr(z) + s * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return np.where(positive, closed, np.maximum(d, 0.0))


def _pairwise_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between rows of ``a`` and rows of ``b``."""
    diff = a[:, None, :] - b[None, :, :]
    out: np.ndarray = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# RBF surrogate
# ══════════════════════════════════════════════════════════════════════════════


def _rbf_phi(r: np.ndarray, kernel: str) -> np.ndarray:
    """``φ(r)`` for the supported kernels, with the ``r = 0`` limit taken exactly."""
    if kernel == "cubic":
        return r**3
    if kernel == "thin_plate":
        # r² log r -> 0 as r -> 0; np.where alone would still evaluate log(0).
        safe = np.where(r > 0.0, r, 1.0)
        return np.where(r > 0.0, safe * safe * np.log(safe), 0.0)
    if kernel == "linear":
        return r
    raise ValueError(f"unknown rbf_kernel {kernel!r}; expected one of {_RBF_KERNELS}")


@dataclass
class RBFSurrogate:
    """Radial basis function interpolant with a linear tail.

    Solves the saddle-point system

    ::

        [ Φ + γI   P ] [ λ ]   [ y ]
        [   Pᵀ     0 ] [ c ] = [ 0 ]

    where ``Φ_ij = φ(‖z_i - z_j‖)``, ``P = [Z, 1]``, and the second block row is
    the orthogonality condition that makes the system nonsingular for a
    conditionally positive definite kernel. ``γ`` (:attr:`ridge`) is the RBF
    analogue of kriging's nugget: ``0`` interpolates, ``> 0`` smooths.

    The fit is done on ``z`` in the **unit box** and on standardized ``y``, which
    is what keeps ``Φ`` conditioned when the variables have wildly different
    ranges. :meth:`predict` undoes the ``y`` standardization, so it speaks in the
    caller's units; :attr:`lam` and :attr:`tail` remain in standardized units,
    which is all the acquisition needs (an increasing affine map does not move an
    argmin).
    """

    kernel: str = "cubic"
    ridge: float = 0.0

    Z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    lam: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tail: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y_mean: float = 0.0
    y_std: float = 1.0
    used_least_squares: bool = False
    ill_conditioned: bool = False

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "RBFSurrogate":
        """Fit to design ``Z`` (m x n, in the unit box) and values ``y`` (m)."""
        from scipy.linalg import LinAlgError, LinAlgWarning, lstsq, solve

        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if Z.ndim != 2 or Z.shape[0] != y.size:
            raise ValueError(f"design/value shape mismatch: {Z.shape} vs {y.shape}")
        m, n = Z.shape
        if m < n + 1:
            raise ValueError(
                f"an RBF with a linear tail needs at least n+1 = {n + 1} design points, got {m}"
            )
        if not np.all(np.isfinite(Z)) or not np.all(np.isfinite(y)):
            raise ValueError("RBF fit received a non-finite design or value")

        self.y_mean = float(y.mean())
        spread = float(y.std())
        self.y_std = spread if spread > 0.0 else 1.0
        yn = (y - self.y_mean) / self.y_std

        phi = _rbf_phi(_pairwise_distance(Z, Z), self.kernel)
        if self.ridge:
            phi = phi + float(self.ridge) * np.eye(m)
        P = np.hstack([Z, np.ones((m, 1))])
        A = np.zeros((m + n + 1, m + n + 1), dtype=np.float64)
        A[:m, :m] = phi
        A[:m, m:] = P
        A[m:, :m] = P.T
        rhs = np.concatenate([yn, np.zeros(n + 1)])

        self.used_least_squares = False
        self.ill_conditioned = False
        try:
            # ``scipy.linalg.solve`` reports an ill-conditioned system as a
            # *warning* and returns a number anyway. Left alone that is a silent
            # loss of accuracy in the object the whole search then trusts, so the
            # warning is captured and re-reported as a fact about this fit
            # (CLAUDE.md §7) rather than left to whatever warning filter happens
            # to be installed.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sol = solve(A, rhs, assume_a="sym")
            for record in caught:
                if issubclass(record.category, LinAlgWarning):
                    self.ill_conditioned = True
                    logger.warning(
                        "surrogate: the RBF interpolation system on %d points is "
                        "ill-conditioned (%s); the surrogate's accuracy near the "
                        "design is degraded. Consider rbf_ridge > 0, or "
                        "rbf_kernel='linear'.",
                        m,
                        record.message,
                    )
            if not np.all(np.isfinite(sol)):
                raise LinAlgError("non-finite RBF coefficients")
            # Finite is not the same as solved. On a rank-deficient system LAPACK's
            # symmetric factorization can return a finite vector that does not
            # satisfy ``A x = rhs`` at all, so the only trustworthy test is the
            # residual itself -- not whether an exception was raised, and not a
            # conditioning threshold. Measured on a design with one duplicated row
            # (rank 10 of 11): ``solve`` returned a finite solution with
            # ||x|| = 1.5e33 and residual 2.0e17, and the resulting surrogate
            # mispredicted its *own* design points by 2.6e17. ``lstsq`` on the same
            # system gave ||x|| = 38, residual 0.31, and reproduced the design
            # values to 0.5 -- the best achievable, since the duplicated row
            # carries a different y. This check is what makes the fallback below
            # reachable on a LAPACK that declines to raise; an earlier version
            # keyed the fallback on the exception alone and so was silently
            # platform-dependent.
            resid = float(np.linalg.norm(A @ sol - rhs))
            if resid > _RBF_RESIDUAL_TOL * (1.0 + float(np.linalg.norm(rhs))):
                raise LinAlgError(
                    f"RBF solution does not satisfy its own system (residual {resid:.3e})"
                )
        except (LinAlgError, ValueError) as exc:
            # Never swallowed: a degenerate design (duplicate or collinear points)
            # is a real event and the caller must be able to see it happened.
            # Least squares gives the minimum-norm solution, which is the honest
            # answer for a rank-deficient interpolation system.
            logger.warning(
                "surrogate: RBF interpolation system is singular (%s); "
                "falling back to a least-squares fit on %d points",
                exc,
                m,
            )
            sol = lstsq(A, rhs)[0]
            self.used_least_squares = True

        self.Z = Z.copy()
        self.lam = sol[:m]
        self.tail = sol[m:]
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Surrogate value at each row of ``Z``, in the caller's ``y`` units."""
        return self.y_mean + self.y_std * self.predict_standardized(Z)

    def predict_standardized(self, Z: np.ndarray) -> np.ndarray:
        """Surrogate value in standardized units — what the acquisition minimizes."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        phi = _rbf_phi(_pairwise_distance(Z, self.Z), self.kernel)
        n = self.Z.shape[1]
        out: np.ndarray = phi @ self.lam + Z @ self.tail[:n] + self.tail[n]
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Kriging / DACE surrogate
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class KrigingSurrogate:
    """Ordinary kriging with an anisotropic exponential correlation (DACE/EGO).

    ``R_ij = exp(-Σ_h θ_h |z_h^i - z_h^j|^{p_h}) + η δ_ij``. With the constant
    trend profiled out, the concentrated negative log-likelihood is
    ``m ln σ̂² + ln|R|``, which is what :meth:`fit` minimizes over ``log10 θ``
    (and optionally ``p`` and ``log10 η``) with L-BFGS-B from several
    deterministic starts.

    Everything is computed through the Cholesky factor ``L`` of ``R``: ``ln|R|``
    is ``2 Σ ln L_ii``, and the predictor and variance are written in terms of
    ``v = L⁻¹r`` rather than ``R⁻¹``. That is not only numerically better, it is
    what makes the certified acquisition model tractable — see the module
    docstring.
    """

    nugget: float = 1e-8
    estimate_nugget: bool = False
    power: float = 2.0
    estimate_power: bool = False
    #: Measured, not guessed. On normalized inputs a ``θ`` much below ``0.1``
    #: makes every correlation ~1 across the whole box, ``R`` numerically
    #: singular, and the predictive error a flat function of the nugget rather
    #: than of the design — i.e. the model degenerates without ever failing. On a
    #: 16-point 2-D probe, lowering the floor from ``0.1`` to ``0.01`` moved the
    #: interpolation error from 2.0e-5 to 1.4e-4 and cut the "uncertain between
    #: samples" contrast from 105x to 32x.
    theta_bounds: tuple[float, float] = (1e-1, 1e3)
    nugget_bounds: tuple[float, float] = (1e-10, 1e-1)

    Z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    L: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mu: float = 0.0
    sigma2: float = 1.0
    fitted_nugget: float = 1e-8
    fitted_power: float = 2.0
    y_mean: float = 0.0
    y_std: float = 1.0
    neg_log_lik: float = np.inf

    # -- fitting -----------------------------------------------------------
    def fit(self, Z: np.ndarray, y: np.ndarray) -> "KrigingSurrogate":
        """Fit to design ``Z`` (m x n, unit box) and values ``y`` (m) by MLE."""
        from scipy.optimize import minimize

        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if Z.ndim != 2 or Z.shape[0] != y.size:
            raise ValueError(f"design/value shape mismatch: {Z.shape} vs {y.shape}")
        m, n = Z.shape
        if m < 2:
            raise ValueError(f"kriging needs at least 2 design points, got {m}")

        self.y_mean = float(y.mean())
        spread = float(y.std())
        self.y_std = spread if spread > 0.0 else 1.0
        yn = (y - self.y_mean) / self.y_std

        # |z_i - z_j| per dimension, raised to p inside the objective so that an
        # estimated p does not force a recompute of the differences.
        absdiff = np.abs(Z[:, None, :] - Z[None, :, :])

        lo, hi = (math.log10(self.theta_bounds[0]), math.log10(self.theta_bounds[1]))
        bounds = [(lo, hi)] * n
        if self.estimate_power:
            bounds.append((1.0, 2.0))
        if self.estimate_nugget:
            bounds.append((math.log10(self.nugget_bounds[0]), math.log10(self.nugget_bounds[1])))

        def unpack(params: np.ndarray):
            theta = 10.0 ** np.asarray(params[:n], dtype=np.float64)
            k = n
            power = float(params[k]) if self.estimate_power else float(self.power)
            k += int(self.estimate_power)
            eta = 10.0 ** float(params[k]) if self.estimate_nugget else float(self.nugget)
            return theta, power, eta

        def objective(params: np.ndarray) -> float:
            theta, power, eta = unpack(params)
            out = self._concentrated(absdiff, yn, theta, power, eta)
            return np.inf if out is None else float(out[0])

        # Deterministic starts: a geometric ladder plus the classic
        # "1 / (2 * mean squared distance)" scale heuristic. No RNG, so a refit on
        # identical data gives identical hyperparameters and the whole backend
        # stays reproducible.
        mean_sq = float(np.mean(absdiff**2)) or 1.0
        heuristic = float(np.clip(1.0 / (2.0 * mean_sq), *self.theta_bounds))
        ladder = [0.1, 1.0, 10.0, 100.0, heuristic]
        starts = []
        for t0 in ladder:
            base = [math.log10(float(np.clip(t0, *self.theta_bounds)))] * n
            if self.estimate_power:
                base.append(2.0)
            if self.estimate_nugget:
                base.append(math.log10(float(np.clip(1e-6, *self.nugget_bounds))))
            starts.append(np.array(base, dtype=np.float64))

        best_params, best_val = None, np.inf
        for start in starts:
            try:
                res = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                logger.warning("surrogate: kriging MLE start failed (%s)", exc)
                continue
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_params = float(res.fun), np.asarray(res.x, dtype=np.float64)

        if best_params is None:
            # Refuse loudly rather than silently returning an unfitted model whose
            # predictions would look plausible and mean nothing (CLAUDE.md §3).
            raise RuntimeError(
                "kriging maximum-likelihood fit failed from every start: the "
                "correlation matrix was not positive definite anywhere in the "
                f"hyperparameter box for {m} design points. Increase `nugget`, or "
                'use surrogate="rbf".'
            )

        theta, power, eta = unpack(best_params)
        out = self._concentrated(absdiff, yn, theta, power, eta)
        if out is None:
            raise RuntimeError("kriging fit lost positive definiteness at its own optimum")
        self.neg_log_lik, self.L, self.w, self.a, self.mu, self.sigma2 = out
        self.Z, self.theta = Z.copy(), theta
        self.fitted_power, self.fitted_nugget = power, eta
        return self

    def _concentrated(self, absdiff, yn, theta, power, eta):
        """``(psi, L, w, a, mu, sigma2)`` at these hyperparameters, or ``None``.

        ``None`` means the correlation matrix was not positive definite, which the
        caller turns into ``+inf`` so L-BFGS-B walks away from it. Returning
        ``None`` rather than raising keeps the MLE loop free of exception control
        flow while still never *hiding* a failure — a fit that finds no positive
        definite point at all raises in :meth:`fit`.
        """
        from scipy.linalg import LinAlgError, cholesky, solve_triangular

        m = yn.size
        R = np.exp(-np.sum((absdiff**power) * theta, axis=-1)) + float(eta) * np.eye(m)
        try:
            L = cholesky(R, lower=True)
        except LinAlgError:
            return None
        one = np.ones(m)
        w = solve_triangular(L, one, lower=True)  # L w = 1
        Ly = solve_triangular(L, yn, lower=True)
        wtw = float(w @ w)
        if wtw <= 0.0:
            return None
        mu = float(w @ Ly) / wtw
        a = Ly - mu * w  # L a = y - mu*1
        sigma2 = float(a @ a) / m
        if not np.isfinite(sigma2) or sigma2 <= 0.0:
            return None
        log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
        return (m * math.log(sigma2) + log_det, L, w, a, mu, sigma2)

    # -- prediction --------------------------------------------------------
    def _whitened(self, Z: np.ndarray) -> np.ndarray:
        """``v = L⁻¹ r(z)`` for each row of ``Z``, shape ``(m, N)``."""
        from scipy.linalg import solve_triangular

        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        absdiff = np.abs(Z[:, None, :] - self.Z[None, :, :])
        r = np.exp(-np.sum((absdiff**self.fitted_power) * self.theta, axis=-1))  # (N, m)
        out: np.ndarray = solve_triangular(self.L, r.T, lower=True)
        return out

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Kriging predictor at each row of ``Z``, in the caller's ``y`` units."""
        return self.predict_with_error(Z)[0]

    def predict_with_error(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""``(ŷ, s)`` at each row of ``Z``, in the caller's ``y`` units.

        With ``v = L⁻¹r``:

        ``ŷ = μ + aᵀv``  and
        ``s² = σ̂² [1 - vᵀv + (1 - wᵀv)² / wᵀw]``,

        the last term being the correction for having estimated the trend ``μ``.
        Both are the standard DACE expressions with ``R⁻¹`` eliminated in favour
        of the Cholesky factor. ``s²`` is clipped at zero before the square root:
        it is nonnegative in exact arithmetic and a rounding-negative value
        must not become a NaN.
        """
        v = self._whitened(Z)
        y_hat = self.mu + self.a @ v
        wtw = float(self.w @ self.w)
        s2 = self.sigma2 * (1.0 - np.sum(v * v, axis=0) + (1.0 - self.w @ v) ** 2 / wtw)
        s = np.sqrt(np.maximum(s2, 0.0))
        return self.y_mean + self.y_std * y_hat, self.y_std * s


# ══════════════════════════════════════════════════════════════════════════════
# Acquisition subproblems as discopt algebraic models
# ══════════════════════════════════════════════════════════════════════════════


class AcquisitionNotExpressible(Exception):
    """The acquisition cannot be written as a discopt algebraic model.

    Raised (and caught by ``acquisition_optimizer="auto"``) rather than silently
    returning ``None``, so the reason a multistart ran instead of the B&B is
    always attributable.
    """


def _scalar_entries(var):
    """The scalar expressions of ``var`` in flat (C) order.

    ``np.ndindex`` walks C order, matching ``.reshape(-1)`` and therefore the flat
    column layout the rest of the solver uses (``_flat_var_box``).
    """
    if var.shape == ():
        return [var]
    return [var[idx] for idx in np.ndindex(var.shape)]


def _mirror_box(source: Model, target: Model) -> list:
    """Recreate ``source``'s variables (type, shape, bounds) in ``target``.

    Returns the flat list of scalar expressions in the same order as
    ``_flat_var_box``. Integer and binary variables are recreated as **integer**
    variables with the source bounds, so discopt's B&B branches on them natively
    and the proposed point is integral by construction rather than by rounding.
    A binary is exactly an integer on ``[0, 1]``, and going through ``integer``
    preserves a fixed (``lb == ub``) binary that ``binary()`` would reopen.
    """
    from discopt.modeling.core import VarType

    flat: list = []
    for v in source._variables:
        lb = np.asarray(v.lb, dtype=np.float64)
        ub = np.asarray(v.ub, dtype=np.float64)
        lb_arg: float | np.ndarray
        ub_arg: float | np.ndarray
        if v.shape == ():
            lb_arg, ub_arg = float(lb.reshape(-1)[0]), float(ub.reshape(-1)[0])
        else:
            lb_arg = np.broadcast_to(lb, v.shape).copy()
            ub_arg = np.broadcast_to(ub, v.shape).copy()
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            new = target.integer(v.name, shape=v.shape, lb=lb_arg, ub=ub_arg)
        else:
            new = target.continuous(v.name, shape=v.shape, lb=lb_arg, ub=ub_arg)
        flat.extend(_scalar_entries(new))
    return flat


def _normalized_coordinates(xs: list, lb: np.ndarray, ub: np.ndarray) -> list:
    """``z_h = (x_h - lb_h) / (ub_h - lb_h)`` as expressions, matching the fit."""
    width = ub - lb
    safe = np.where(width > 0.0, width, 1.0)
    return [(xs[h] - float(lb[h])) / float(safe[h]) for h in range(len(xs))]


def _rbf_phi_expr(q, kernel: str):
    """``φ(r)`` written in terms of the *squared* distance ``q = r²``.

    Keeping the squared distance as the primitive is what makes this relaxable:
    ``q`` is a plain sum of squares, and ``r³ = q√q``, ``r² log r = ½ q log q``,
    ``r = √q`` are each one intrinsic applied to it. Going through ``r = √q``
    first would add an extra ``sqrt`` node per design point with a bound that the
    relaxation then has to propagate through the cube.
    """
    import discopt.modeling as dm

    if kernel == "cubic":
        return q * dm.sqrt(q + _DOMAIN_EPS)
    if kernel == "thin_plate":
        return 0.5 * q * dm.log(q + _DOMAIN_EPS)
    if kernel == "linear":
        return dm.sqrt(q + _DOMAIN_EPS)
    raise ValueError(f"unknown rbf_kernel {kernel!r}; expected one of {_RBF_KERNELS}")


def build_cors_model(
    source: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    rbf: RBFSurrogate,
    delta: float,
    box: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> Model:
    """The CORS subproblem as a discopt model: min surrogate s.t. distance ≥ ``delta``.

    ``delta`` is in the **normalized** (unit-box) metric the surrogate was fitted
    in, so it means the same thing whatever the variables' units are.

    Everything here is algebraic: the objective is a sum of ``λ_i φ(q_i)`` over
    squared distances plus a linear tail, and each constraint is a convex
    quadratic reversed (``q_i ≥ δ²``). That is what spatial B&B certifies well.
    """
    import discopt.modeling as dm

    acq = dm.Model("surrogate_cors")
    xs = _mirror_box(source, acq)
    if box is not None:
        _restrict_box(acq, *box)
    zs = _normalized_coordinates(xs, lb, ub)

    n = rbf.Z.shape[1]
    # Seeded as a Constant, not a float: with no tail terms and no centres the
    # accumulation below never promotes it, and Model.minimize needs an
    # Expression. Same value, correct type.
    obj: Any = Constant(float(rbf.tail[n]))
    for h in range(n):
        obj = obj + float(rbf.tail[h]) * zs[h]
    for i in range(rbf.Z.shape[0]):
        q = sum((zs[h] - float(rbf.Z[i, h])) ** 2 for h in range(n))
        obj = obj + float(rbf.lam[i]) * _rbf_phi_expr(q, rbf.kernel)
        if delta > 0.0:
            acq.subject_to(q >= float(delta) ** 2)
    acq.minimize(obj)
    return acq


def build_ei_model(
    source: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    kr: KrigingSurrogate,
    f_min: float,
    xi: float = 0.0,
    box: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> Model:
    r"""The EI subproblem as a discopt model, in lifted + whitened form.

    Maximizes ``d(x) Φ(u) + s(x) φ(u)`` jointly over ``(x, u)``, which equals
    ``max_x EI(x)`` because the inner maximum over ``u`` is attained at
    ``u = d/s`` (the derivative in ``u`` is ``φ(u)(d - su)``). See the module
    docstring for why the lift and the whitening are both necessary.

    Auxiliary variables, all with tight bounds derived from the box:

    * ``r_i ∈ [0, 1]`` — the correlation to design point ``i``, tied to ``x`` by
      ``r_i = exp(-Σ_h θ_h (z_h - z_h^i)²)``;
    * ``v = L⁻¹ r`` — introduced by the *triangular linear* constraint ``Lv = r``;
    * ``s ≥ 0`` with ``s² ≤ σ̂²[1 - vᵀv + (1 - wᵀv)²/wᵀw]``. The inequality is
      exact at the optimum because ``φ(u) ≥ 0`` makes larger ``s`` strictly
      better, and an inequality relaxes far better than the equality would.

    Raises
    ------
    AcquisitionNotExpressible
        If the fitted correlation power is not exactly 2. ``|z|^p`` for
        ``1 ≤ p < 2`` is not differentiable at 0 and has no discopt intrinsic;
        writing it as ``(z²)^{p/2}`` would be a silent approximation of the model
        that was actually fitted.
    """
    import discopt.modeling as dm

    if kr.fitted_power != 2.0:
        raise AcquisitionNotExpressible(
            f"the certified EI model requires a squared-exponential correlation, "
            f"but the fitted power is {kr.fitted_power}; |z|^p has no discopt "
            "intrinsic for p != 2"
        )
    m, n = kr.Z.shape

    acq = dm.Model("surrogate_ei")
    xs = _mirror_box(source, acq)
    if box is not None:
        _restrict_box(acq, *box)
    zs = _normalized_coordinates(xs, lb, ub)

    r = acq.continuous("_r", shape=m, lb=0.0, ub=1.0)
    for i in range(m):
        q = sum(float(kr.theta[h]) * (zs[h] - float(kr.Z[i, h])) ** 2 for h in range(n))
        acq.subject_to(r[i] == dm.exp(-q))

    # v = L^-1 r, imposed as the triangular system L v = r. |v_i| <= |r|/min|L_ii|
    # is a valid bound and is far tighter than any generic box.
    diag = np.abs(np.diag(kr.L))
    v_bound = float(max(1.0, math.sqrt(m) / max(float(diag.min()), 1e-12)))
    v = acq.continuous("_v", shape=m, lb=-v_bound, ub=v_bound)
    for i in range(m):
        acq.subject_to(sum(float(kr.L[i, j]) * v[j] for j in range(i + 1)) == r[i])

    wtw = float(kr.w @ kr.w)
    y_hat = kr.mu + sum(float(kr.a[i]) * v[i] for i in range(m))
    trend = sum(float(kr.w[i]) * v[i] for i in range(m))
    s2 = kr.sigma2 * (1.0 - sum(v[i] ** 2 for i in range(m)) + (1.0 - trend) ** 2 / wtw)
    s_max = math.sqrt(max(kr.sigma2 * (1.0 + 1.0 / wtw), 0.0))
    s = acq.continuous("_s", lb=0.0, ub=s_max)
    acq.subject_to(s**2 <= s2)

    # f_min and xi arrive in the caller's y units; the model is standardized.
    f_min_std = (float(f_min) - kr.y_mean) / kr.y_std
    xi_std = float(xi) / kr.y_std
    d = f_min_std - xi_std - y_hat

    u = acq.continuous("_u", lb=-_U_LIMIT, ub=_U_LIMIT)
    cdf = 0.5 * (1.0 + dm.erf(u / math.sqrt(2.0)))
    pdf = dm.exp(-0.5 * u**2) / math.sqrt(2.0 * math.pi)
    acq.maximize(d * cdf + s * pdf)
    return acq


def _restrict_box(acq: Model, lo: np.ndarray, hi: np.ndarray) -> None:
    """Tighten the acquisition model's variable box to ``[lo, hi]``.

    The seam a trust-region rule (TuRBO-style) would drive; today
    :func:`_acquisition_domain` always hands back the full box, so this is only
    ever called with the model's own bounds.
    """
    offset = 0
    for v in acq._variables:
        size = int(v.size)
        sl = slice(offset, offset + size)
        v.lb = np.asarray(lo[sl], dtype=np.float64).reshape(v.shape).copy()
        v.ub = np.asarray(hi[sl], dtype=np.float64).reshape(v.shape).copy()
        offset += size


def _acquisition_domain(
    lb: np.ndarray, ub: np.ndarray, search: "_SurrogateSearch"
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """The box the acquisition subproblem is posed over.

    **Seam, deliberately inert.** Returning ``None`` means "the full model box",
    which is what a global surrogate method does. A trust-region variant (TuRBO,
    Eriksson et al. 2019) would return a shrinking box centred on the incumbent
    with expand-on-success / shrink-on-failure counters; that change lives here
    and in the counters, and nothing else in the loop needs to know.
    """
    del lb, ub, search
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Acquisition optimizers
# ══════════════════════════════════════════════════════════════════════════════


def _solve_acquisition_model(
    acq: Model,
    source: Model,
    time_limit: float,
    gap_tolerance: float,
) -> tuple[Optional[np.ndarray], bool]:
    """``(point, certified)`` from discopt's B&B on the acquisition subproblem.

    The point is returned in the *source* model's flat variable order. ``None``
    means the subproblem produced no usable point at all.
    """
    result = acq.solve(time_limit=float(time_limit), gap_tolerance=float(gap_tolerance))
    assert isinstance(result, SolveResult)  # stream=False, so never an iterator
    if result.x is None:
        return None, False
    flat = np.concatenate(
        [np.asarray(result.x[v.name], dtype=np.float64).reshape(-1) for v in source._variables]
    )
    return flat, bool(result.gap_certified)


def _multistart_maximize(
    score: Callable[[np.ndarray], np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
    integer_mask: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
    top_k: int = 5,
) -> tuple[np.ndarray, float]:
    """Maximize ``score`` (vectorized over rows) by stratified multistart + polish.

    The fallback path, and what a conventional BO library does as its *only*
    path. Stratified sampling comes from ``_relax.primal_heuristics._generate_starts``
    rather than a private reimplementation. Integer coordinates are rounded and
    then held fixed through the polish, so the polish only moves what it can
    legitimately move.
    """
    from scipy.optimize import minimize

    from discopt._relax.primal_heuristics import _generate_starts

    cands = _generate_starts(lb, ub, int(max(1, n_starts)), rng)
    cands = np.clip(cands, lb, ub)
    if integer_mask.any():
        cands[:, integer_mask] = np.clip(
            np.round(cands[:, integer_mask]),
            np.ceil(lb[integer_mask] - 1e-9),
            np.floor(ub[integer_mask] + 1e-9),
        )
    values = np.asarray(score(cands), dtype=np.float64)
    order = np.argsort(-values)[: max(1, int(top_k))]

    best_x = cands[order[0]].copy()
    best_v = float(values[order[0]])

    free = ~integer_mask
    if free.any():
        lo, hi = lb[free], ub[free]
        movable = hi > lo
        for idx in order:
            base = cands[idx].copy()

            def negative(z: np.ndarray, _base=base) -> float:
                x = _base.copy()
                x[free] = np.clip(z, lo, hi)
                return -float(np.asarray(score(x[None, :]))[0])

            if not movable.any():
                continue
            try:
                res = minimize(
                    negative,
                    base[free],
                    method="L-BFGS-B",
                    bounds=list(zip(lo, hi)),
                    options={"maxiter": 200},
                )
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                logger.warning("surrogate: acquisition polish raised (%s)", exc)
                continue
            cand = base.copy()
            cand[free] = np.clip(res.x, lo, hi)
            val = float(np.asarray(score(cand[None, :]))[0])
            if val > best_v:
                best_x, best_v = cand, val
    return best_x, best_v


# ══════════════════════════════════════════════════════════════════════════════
# The search
# ══════════════════════════════════════════════════════════════════════════════


class _SurrogateSearch:
    """Model-based search over a box, driven by a caller-supplied oracle.

    Deliberately independent of ``Model`` for its *numerics*: it takes a function
    from a model point to ``(objective, violation)``, keeps the design and the
    merit values, fits the surrogate and ranks candidates. Only
    :meth:`propose` needs a ``Model``, and only to build the algebraic acquisition
    subproblem — the multistart path works on plain arrays. That keeps the method
    unit-testable on plain functions.
    """

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        *,
        integer_mask: Optional[np.ndarray] = None,
        eps_cons: float = 1e-6,
    ) -> None:
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        # Checked here as well as in solve_surrogate: the engine is usable
        # directly on a plain callable, and an infinite side makes the normalized
        # coordinates NaN, which would silently corrupt every distance.
        if not (np.all(np.isfinite(self.lb)) and np.all(np.isfinite(self.ub))):
            raise ValueError(
                "the surrogate backend requires a finite box on every variable; "
                f"got lb={self.lb!r}, ub={self.ub!r}"
            )
        if np.any(self.ub < self.lb):
            raise ValueError("lower bound exceeds upper bound")
        self.width = self.ub - self.lb
        self.n = self.lb.size
        self.integer_mask = (
            np.zeros(self.n, dtype=bool)
            if integer_mask is None
            else np.asarray(integer_mask, dtype=bool)
        )
        self.eps_cons = float(eps_cons)
        self.stats = SurrogateStats()

        self.X: list[np.ndarray] = []  # evaluated model points
        self.f: list[float] = []  # raw objective values (may be +inf)
        self.viol: list[float] = []
        self._cache: dict[bytes, tuple[float, float]] = {}

        self.best_feasible_value: Optional[float] = None
        self.best_feasible_point: Optional[np.ndarray] = None
        self.best_violation = np.inf
        self.best_violation_point: Optional[np.ndarray] = None

    # -- coordinates -------------------------------------------------------
    def to_model_point(self, x: np.ndarray) -> np.ndarray:
        """Clip into the box and round integer coordinates.

        Rounding at the point of evaluation — rather than reshaping the design —
        is exactly what ``_DirectSearch.to_model_point`` does, and it is what lets
        the surrogate be fitted on a continuous relaxation of the domain while
        every value it is fitted to comes from a genuinely integral point.
        """
        x = np.clip(np.asarray(x, dtype=np.float64).reshape(-1), self.lb, self.ub)
        if self.integer_mask.any():
            x = x.copy()
            x[self.integer_mask] = np.clip(
                np.round(x[self.integer_mask]),
                np.ceil(self.lb[self.integer_mask] - 1e-9),
                np.floor(self.ub[self.integer_mask] + 1e-9),
            )
        return x

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Model points -> the unit box the surrogate is fitted in."""
        safe = np.where(self.width > 0.0, self.width, 1.0)
        return (np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.lb) / safe

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self, x: np.ndarray, oracle: Callable[[np.ndarray], tuple[float, float]]
    ) -> Optional[tuple[float, float]]:
        """Evaluate at model point ``x``; ``None`` when it repeats a design point.

        The cache is not an optimization here, it is a correctness guard: two
        identical design points make the RBF interpolation matrix and the kriging
        correlation matrix exactly singular. Returning ``None`` tells the driver
        that no new information was obtained.
        """
        x = self.to_model_point(x)
        key = x.tobytes()
        if key in self._cache:
            self.stats.cache_hits += 1
            return None
        fval, viol = oracle(x)
        self._cache[key] = (fval, viol)
        self.stats.evals += 1
        self.X.append(x.copy())
        self.f.append(float(fval))
        self.viol.append(float(viol))
        if not np.isfinite(fval):
            self.stats.undefined_points += 1
        self._offer(x, float(fval), float(viol))
        return float(fval), float(viol)

    def _offer(self, x: np.ndarray, fval: float, viol: float) -> None:
        if viol < self.best_violation:
            self.best_violation = viol
            self.best_violation_point = x.copy()
        if viol <= self.eps_cons and np.isfinite(fval):
            if self.best_feasible_value is None:
                self.stats.feasible_found += 1
            if self.best_feasible_value is None or fval < self.best_feasible_value:
                self.best_feasible_value = float(fval)
                self.best_feasible_point = x.copy()
                self.stats.improvements += 1

    # -- merit -------------------------------------------------------------
    def merits(self) -> np.ndarray:
        """The scalar the surrogate is fitted to, one per evaluated point.

        The rule itself lives in :func:`_dfo_common.glce_merit`, shared with
        ``_DirectSearch.rank_values``: the two backends must optimize the same
        thing to be comparable. For an unconstrained model it is just the
        objective.

        ``finite_fill=True`` because this value is *fitted*, not only compared: a
        non-finite objective (a black box undefined at that point) would make the
        interpolation system meaningless as ``+inf``, while dropping the point
        would throw away the one thing it does tell us, that the region is bad.
        """
        return glce_merit(
            np.asarray(self.f, dtype=np.float64),
            np.asarray(self.viol, dtype=np.float64),
            self.best_feasible_value,
            self.eps_cons,
            finite_fill=True,
        )

    # -- proposal ----------------------------------------------------------
    def propose(
        self,
        source: Model,
        model_kind: str,
        surrogate_obj,
        *,
        delta: float,
        xi: float,
        optimizer: str,
        acq_time_limit: float,
        acq_gap_tolerance: float,
        n_multistart: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Candidate point(s) to evaluate next.

        **Returns a list on purpose** — that is the batch/q-point seam. Today it
        always holds exactly one point; a q-EI or a CORS rule with mutual
        distance constraints would fill it with ``q`` points and the driver loop,
        which already iterates over the list and charges each element to the
        budget, would need no change.
        """
        box = _acquisition_domain(self.lb, self.ub, self)
        point: Optional[np.ndarray] = None

        if optimizer in ("auto", "certified"):
            try:
                if model_kind == "rbf":
                    acq = build_cors_model(source, self.lb, self.ub, surrogate_obj, delta, box)
                else:
                    f_min = self._acquisition_f_min()
                    acq = build_ei_model(source, self.lb, self.ub, surrogate_obj, f_min, xi, box)
                point, certified = _solve_acquisition_model(
                    acq, source, acq_time_limit, acq_gap_tolerance
                )
            except AcquisitionNotExpressible as exc:
                if optimizer == "certified":
                    raise
                self.stats.acq_failures += 1
                logger.info("surrogate: acquisition is not expressible algebraically (%s)", exc)
                point = None
            except Exception as exc:  # noqa: BLE001 - reported, never hidden
                if optimizer == "certified":
                    raise
                self.stats.acq_failures += 1
                logger.warning(
                    "surrogate: the algebraic acquisition subproblem raised (%s); "
                    "falling back to multistart",
                    exc,
                )
                point = None
            else:
                if point is None:
                    self.stats.acq_failures += 1
                    if optimizer == "certified":
                        raise RuntimeError(
                            "acquisition_optimizer='certified' but the acquisition "
                            "subproblem returned no point"
                        )
                    logger.warning(
                        "surrogate: the acquisition subproblem returned no point; "
                        "falling back to multistart"
                    )
                elif certified:
                    self.stats.acq_certified += 1
                    logger.info(
                        "surrogate: acquisition maximized to CERTIFIED global optimality "
                        "by discopt's spatial B&B (iteration %d)",
                        self.stats.iterations,
                    )
                else:
                    self.stats.acq_bb_uncertified += 1
                    if optimizer == "certified":
                        raise RuntimeError(
                            "acquisition_optimizer='certified' but the acquisition "
                            "subproblem finished without a certificate "
                            f"(acq_time_limit={acq_time_limit}s). Raise "
                            "acquisition_time_limit, or use "
                            "acquisition_optimizer='auto'."
                        )
                    logger.info(
                        "surrogate: acquisition solved by discopt's spatial B&B but "
                        "NOT certified within %.1fs (iteration %d); using the "
                        "incumbent point",
                        acq_time_limit,
                        self.stats.iterations,
                    )

        if point is None:
            self.stats.acq_multistart += 1
            logger.info(
                "surrogate: acquisition maximized by multistart (iteration %d)",
                self.stats.iterations,
            )
            score = self._analytic_acquisition(model_kind, surrogate_obj, delta=delta, xi=xi)
            point, _ = _multistart_maximize(
                score, self.lb, self.ub, self.integer_mask, n_multistart, rng
            )
        return [self.to_model_point(point)]

    def _acquisition_f_min(self) -> float:
        """``f_min`` for EI: the best merit observed, in merit units."""
        merit = self.merits()
        return float(merit.min()) if merit.size else 0.0

    def _analytic_acquisition(
        self, model_kind: str, surrogate_obj, *, delta: float, xi: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        """The acquisition as a plain numpy callable to MAXIMIZE over model points.

        Used by the multistart fallback, and it is deliberately the *same*
        mathematics the algebraic model expresses, so the two optimizers can be
        compared on one objective.
        """
        design = self.normalize(np.asarray(self.X, dtype=np.float64))

        if model_kind == "rbf":
            # CORS as an unconstrained score: minimize the surrogate, with the
            # distance constraint carried as an exact penalty. The penalty scale
            # is tied to the surrogate's own spread so it dominates whatever units
            # the objective happens to be in.
            def cors_score(X: np.ndarray) -> np.ndarray:
                Z = self.normalize(X)
                value = surrogate_obj.predict_standardized(Z)
                dist2 = np.sum((Z[:, None, :] - design[None, :, :]) ** 2, axis=-1)
                shortfall = np.maximum(0.0, delta**2 - dist2.min(axis=1))
                penalized: np.ndarray = -(value + 1e3 * (1.0 + np.abs(value)) * shortfall)
                return penalized

            return cors_score

        f_min = self._acquisition_f_min()

        def ei_score(X: np.ndarray) -> np.ndarray:
            Z = self.normalize(X)
            y_hat, s = surrogate_obj.predict_with_error(Z)
            ei: np.ndarray = expected_improvement(y_hat, s, f_min, xi)
            return ei

        return ei_score

    def maximin_distance(self, rng: np.random.Generator, n_candidates: int = 2000) -> float:
        """``Δ``: the largest distance-to-the-design achievable in the box.

        CORS scales its exclusion radius by this, which is what makes the
        ``β`` cycle mean the same thing early (design sparse, ``Δ`` large) and
        late (design dense, ``Δ`` small). Estimated on a stratified candidate
        pool: the estimate is a *lower* bound on the true maximin, so
        ``δ = β Δ̂ ≤ Δ`` keeps the subproblem feasible for ``β ≤ 1``.
        """
        from discopt._relax.primal_heuristics import _generate_starts

        if not self.X:
            return float(np.sqrt(self.n))
        cands = self.normalize(_generate_starts(self.lb, self.ub, int(n_candidates), rng))
        design = self.normalize(np.asarray(self.X, dtype=np.float64))
        d2 = np.sum((cands[:, None, :] - design[None, :, :]) ** 2, axis=-1)
        return float(np.sqrt(np.max(np.min(d2, axis=1))))

    def escape_point(self, rng: np.random.Generator, n_candidates: int = 2000) -> np.ndarray:
        """The most-distant candidate from the design — used when a proposal repeats.

        Not a random restart: it is the maximin point of a stratified pool, which
        is the best single space-filling addition available without another fit.
        """
        from discopt._relax.primal_heuristics import _generate_starts

        raw = _generate_starts(self.lb, self.ub, int(n_candidates), rng)
        cands = np.array([self.to_model_point(z) for z in raw])
        if not self.X:
            first: np.ndarray = cands[0]
            return first
        Zc = self.normalize(cands)
        design = self.normalize(np.asarray(self.X, dtype=np.float64))
        d2 = np.sum((Zc[:, None, :] - design[None, :, :]) ** 2, axis=-1)
        farthest: np.ndarray = cands[int(np.argmax(np.min(d2, axis=1)))]
        return farthest


# ══════════════════════════════════════════════════════════════════════════════
# Model-facing entry point
# ══════════════════════════════════════════════════════════════════════════════


def _refine_locally(
    model: Model,
    start: np.ndarray,
    integer_mask: np.ndarray,
    time_limit: float,
    nlp_solver: str,
    stats: SurrogateStats,
) -> Optional[tuple[float, np.ndarray]]:
    """Local NLP from ``start``; ``None`` when it yields nothing usable.

    Off by default here, unlike in ``solve_direct``. The premise of this backend
    is that one evaluation is expensive, and a local NLP spends an *uncounted and
    unbounded* number of them — turning it on makes the reported evaluation count
    stop describing the true cost. It is available for the case where the caller
    knows the objective is cheap enough to afford it.
    """
    import time as _time

    from discopt.solver import _solve_continuous
    from discopt.solvers.amp import _restore_variable_bounds, _snapshot_variable_bounds

    saved = _snapshot_variable_bounds(model)
    try:
        if integer_mask.any():
            offset = 0
            for var in model._variables:
                size = var.size
                sl = slice(offset, offset + size)
                if integer_mask[sl].any():
                    fixed = np.round(np.asarray(start[sl], dtype=np.float64))
                    var.lb = fixed.reshape(var.shape).copy()
                    var.ub = fixed.reshape(var.shape).copy()
                offset += size
        stats.local_solves += 1
        result = _solve_continuous(
            model,
            time_limit,
            None,
            _time.perf_counter(),
            nlp_solver,
            initial_point=np.asarray(start, dtype=np.float64),
        )
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        stats.local_failures += 1
        logger.warning(
            "surrogate: local refinement raised (%s); keeping the sampled incumbent", exc
        )
        return None
    finally:
        _restore_variable_bounds(saved)

    if result.objective is None or result.x is None:
        stats.local_failures += 1
        logger.info(
            "surrogate: local refinement returned no usable point (status=%s); "
            "keeping the sampled incumbent",
            result.status,
        )
        return None
    flat = np.concatenate(
        [np.asarray(result.x[v.name], dtype=np.float64).reshape(-1) for v in model._variables]
    )
    return float(result.objective), flat


def _offending_variable_names(model: Model, bad_mask: np.ndarray) -> list[str]:
    """Names of the variables covering the flagged flat positions."""
    names: list[str] = []
    offset = 0
    for var in model._variables:
        size = var.size
        if bad_mask[offset : offset + size].any():
            names.append(var.name)
        offset += size
    return names or ["<unknown>"]


def solve_surrogate(
    model: Model,
    *,
    time_limit: float = 3600.0,
    max_evals: int = 200,
    surrogate: str = "rbf",
    rbf_kernel: str = "cubic",
    rbf_ridge: float = 0.0,
    n_initial: Optional[int] = None,
    acquisition_optimizer: str = "auto",
    acquisition_time_limit: float = 20.0,
    acquisition_gap_tolerance: float = 1e-4,
    acquisition_multistarts: int = 128,
    distance_cycle: Sequence[float] = (0.9, 0.75, 0.25, 0.05, 0.03, 0.0),
    min_distance: float = 1e-3,
    nugget: float = 1e-8,
    estimate_nugget: bool = False,
    kriging_power: float = 2.0,
    estimate_kriging_power: bool = False,
    theta_bounds: tuple[float, float] = (1e-1, 1e3),
    xi: float = 0.0,
    local_refine: bool = False,
    local_refine_time_limit: float = 30.0,
    nlp_solver: str = "pounce",
    feasibility_tolerance: float = 1e-6,
    seed: int = 0,
    initial_point: Optional[np.ndarray] = None,
    on_evaluation: Optional[Callable[[int, Optional[float]], None]] = None,
) -> SolveResult:
    """Minimize ``model`` over its box with a surrogate. **Returns no certificate.**

    Parameters
    ----------
    max_evals
        Objective-evaluation budget, **including the initial design**. This is the
        primary control: the cost of this backend is counted in evaluations, not
        nodes, and the whole point is that the count is small. A point that
        repeats an already-evaluated point is served from a cache and does not
        consume budget.
    surrogate
        ``"rbf"`` (default) or ``"kriging"``. See the module docstring for why
        RBF is the default — integer support, a fit with no failure mode of its
        own, and better behaviour as dimension grows.
    rbf_kernel
        ``"cubic"`` (default), ``"thin_plate"`` or ``"linear"``. Cubic is RBFOpt's
        default and is kept as the default here on optimization quality.

        **It is, however, the wrong choice when the certified acquisition is the
        point** — and that is measured, not asserted. A cubic interpolant's
        coefficients grow as the design fills (``max |λ|`` runs 10 → 23 → 61 → 164
        over ``m = 6, 8, 12, 20`` on Branin), and a sum of large opposing terms is
        what a McCormick-style relaxation cannot bound. The CORS subproblem then
        stops certifying past ``m ≈ 6``. With ``"linear"`` the coefficients stay
        at 3-5 and every design size tried certified, in seconds. See the module
        docstring for the table.
    rbf_ridge
        Smoothing on the RBF's ``Φ`` block — the RBF analogue of ``nugget``.
        ``0`` (default) interpolates.
    n_initial
        Size of the initial space-filling design. Default ``2 (n + 1)``, capped at
        ``max_evals`` — **a function of the dimension only**, deliberately not of
        the budget. See :func:`_default_design_size` for the measurement behind
        both halves of that: the budget-independence is what makes a larger
        ``max_evals`` a *continuation* of a smaller one rather than a different
        search (it was not, before issue #1036), and ``2(n+1)`` reaches 1e-2
        relative error in fewer evaluations than the ``10n`` it replaces on 6 of 8
        panel functions with RBF and 4 of 6 with kriging.

        Raise it if you know the objective is sharply scaled or densely
        multimodal: ``goldstein_price`` and ``rastrigin_2`` are the two shapes on
        that panel that preferred a bigger design.
    acquisition_optimizer
        How the acquisition subproblem is maximized.

        * ``"auto"`` (default) — build it as a discopt algebraic model and hand it
          to spatial branch-and-bound; use the point it returns and **log whether
          it came back certified**. Falls back to multistart only if the model
          cannot be built or the solve raises.
        * ``"certified"`` — require ``gap_certified=True`` on the subproblem and
          raise otherwise, rather than quietly degrading.
        * ``"multistart"`` — skip the B&B; stratified multistart plus an L-BFGS-B
          polish on the analytic acquisition, which is what a conventional
          Bayesian-optimization library does.

        Which path ran is reported in ``solver_stats`` as
        ``surrogate/acq_certified``, ``surrogate/acq_bb_uncertified``,
        ``surrogate/acq_multistart`` and ``surrogate/acq_failures``. **Measured:**
        the CORS subproblem certifies readily; the EI subproblem's *primal* answer
        matches a dense grid to five figures but its bound does not close at
        realistic design sizes (module docstring). Do not assume ``"auto"`` means
        ``"certified"`` — read the counter.
    acquisition_time_limit, acquisition_gap_tolerance
        Budget for one acquisition subproblem. This is compute spent *between*
        evaluations, which is the trade this backend exists to make.
    distance_cycle, min_distance
        CORS's ``β`` cycle and the floor under the exclusion radius, both in the
        normalized (unit-box) metric. The default is Regis & Shoemaker's own
        published schedule ``(0.9, 0.75, 0.25, 0.05, 0.03, 0.0)`` — one global
        step, then progressively local ones — rather than a shorter cycle invented
        here; a cycle with a larger fraction of global steps spends a small budget
        on exploration it cannot afford to follow up. ``min_distance > 0`` is what
        makes proposing an already-sampled point *infeasible* rather than merely
        unattractive (it is also what the trailing ``0.0`` is floored to); a
        method that can re-propose a sampled point stalls, because the surrogate
        does not change and the next iteration proposes it again.
    nugget, estimate_nugget
        Kriging's ridge on the correlation diagonal. The default ``1e-8`` is a
        conditioning jitter and the model still interpolates. Set it larger (or
        pass ``estimate_nugget=True`` to estimate it by MLE) for a **noisy**
        objective, so the surface is not forced through its own measurement
        error — the assumption in vanilla EGO that has aged worst.
    kriging_power, estimate_kriging_power
        The correlation exponent ``p`` (Jones et al. estimate it in ``[1, 2]``).
        ``2.0`` (default) is the squared-exponential and is the **only** value for
        which the certified EI subproblem can be built — ``|z|^p`` has no discopt
        intrinsic otherwise, and approximating it would be a silent
        misrepresentation of the model that was fitted. With any other value the
        acquisition falls back to multistart, loudly.
    theta_bounds
        Box on the kriging correlation parameters, in the **normalized** input
        metric. The lower end matters: below ~``0.1`` every correlation is ~1
        across the whole box, ``R`` goes numerically singular, and the predictive
        error becomes a flat function of the nugget instead of of the design —
        the model degenerates without ever failing.
    xi
        Exploration margin in EI: an improvement must beat ``f_min - xi``.
    local_refine
        Off by default, unlike ``solve_direct``. A local NLP polish spends an
        uncounted and unbounded number of evaluations, which contradicts the
        premise that each one is expensive. Turn it on only when the objective is
        cheap enough that the count does not matter.
    seed
        Seeds the stratified designs and the multistart pools. Two runs with the
        same seed are identical.
    on_evaluation
        Called ``(evaluations_spent, best_feasible_value_or_None)`` after every
        evaluation that produced a new design point. Two reasons it exists rather
        than being left to a caller's wrapper: a run whose objective takes minutes
        needs *some* incremental output or it cannot be told apart from a hung one
        (CLAUDE.md §10), and evaluations-to-target — the only number in which this
        backend's advantage is actually expressed — is not recoverable from a
        ``SolveResult``. Exceptions raised inside it are not caught: a broken
        progress hook is the caller's bug, and swallowing it would make the
        instrument silently measure nothing.

    Raises
    ------
    ValueError
        If the model has no objective, any variable bound is non-finite, or an
        option value is not one of the documented choices. The finite-box
        requirement is not negotiable: a surrogate is fitted on distances, and an
        infinite side makes every normalized distance NaN. Substituting a big-M
        box would be a silent approximation, and it is the caller who knows the
        real range.
    """
    from discopt.solver import _flat_var_box, _unpack_solution

    t_start = time.perf_counter()

    # ---- preconditions, refused loudly ------------------------------------
    if model._objective is None:
        raise ValueError(
            "solver='surrogate' requires an objective; the model has none. "
            "A surrogate method is a minimization method, not a feasibility search."
        )
    if surrogate not in ("rbf", "kriging"):
        raise ValueError(f"surrogate must be 'rbf' or 'kriging', got {surrogate!r}")
    if rbf_kernel not in _RBF_KERNELS:
        raise ValueError(f"rbf_kernel must be one of {_RBF_KERNELS}, got {rbf_kernel!r}")
    if acquisition_optimizer not in ("auto", "certified", "multistart"):
        raise ValueError(
            "acquisition_optimizer must be 'auto', 'certified' or 'multistart', "
            f"got {acquisition_optimizer!r}"
        )
    if max_evals < 1:
        raise ValueError(f"max_evals must be >= 1, got {max_evals}")
    if n_initial is not None and n_initial < 1:
        raise ValueError(f"n_initial must be >= 1, got {n_initial}")
    if min_distance <= 0.0:
        raise ValueError(
            f"min_distance must be > 0 (it is what makes re-proposing a sampled "
            f"point infeasible), got {min_distance}"
        )

    lb, ub = _flat_var_box(model)
    bad = ~(np.isfinite(lb) & np.isfinite(ub))
    if bad.any():
        names = _offending_variable_names(model, bad)
        raise ValueError(
            "solver='surrogate' requires a finite box on every variable; "
            f"non-finite bounds on: {', '.join(names)}. The surrogate is fitted on "
            "normalized distances, so an infinite side makes every distance NaN. "
            "Add explicit bounds, e.g. m.continuous('x', lb=-10, ub=10)."
        )
    if np.any(ub < lb):
        names = _offending_variable_names(model, ub < lb)
        raise ValueError(f"lower bound exceeds upper bound on: {', '.join(names)}")

    n_vars = lb.size
    if n_vars > _DIMENSION_WARN_THRESHOLD:
        logger.warning(
            "surrogate backend on %d variables: a response surface needs a design "
            "that grows with dimension, and the acquisition's own global "
            "optimization gets harder at the same time. Expect to need a larger "
            "max_evals, and prefer an algebraic formulation if one exists.",
            n_vars,
        )

    oracle, n_from_model, integer_mask = build_oracle(model, log_prefix="surrogate")
    if n_from_model != n_vars:
        raise ValueError(f"variable-count mismatch: box has {n_vars}, evaluator has {n_from_model}")

    model_kind = "kriging" if surrogate == "kriging" else "rbf"
    search = _SurrogateSearch(
        lb, ub, integer_mask=integer_mask, eps_cons=float(feasibility_tolerance)
    )
    stats = search.stats
    rng = np.random.default_rng(int(seed))
    deadline = t_start + float(time_limit)

    # ---- initial design ---------------------------------------------------
    from discopt._relax.primal_heuristics import _generate_starts

    if n_initial is None:
        # Sized from the DIMENSION alone. The previous rule,
        # ``max(n+2, min(10n, max_evals // 2))``, made the design a function of
        # the budget, and that is not a tuning detail — it is what makes two runs
        # at different budgets two *different searches* rather than one search and
        # its continuation. See :func:`_default_design_size` and issue #1036.
        n_design = _default_design_size(n_vars)
    else:
        n_design = int(n_initial)
    n_design = max(1, min(n_design, max_evals))

    design_points: list[np.ndarray] = []
    if initial_point is not None:
        design_points.append(
            search.to_model_point(np.asarray(initial_point, dtype=np.float64).reshape(-1))
        )
    design_points.extend(_generate_starts(lb, ub, n_design, rng))

    for x in design_points:
        if stats.evals >= max_evals or time.perf_counter() >= deadline:
            break
        if search.evaluate(x, oracle) is not None and on_evaluation is not None:
            on_evaluation(stats.evals, search.best_feasible_value)
    stats.initial_design = stats.evals
    logger.info("surrogate: initial design of %d point(s) on %d variable(s)", stats.evals, n_vars)

    # ---- the model-based loop --------------------------------------------
    cycle = tuple(float(b) for b in distance_cycle) or (0.0,)
    min_points = (n_vars + 1) if model_kind == "rbf" else 2

    while stats.evals < max_evals:
        if time.perf_counter() >= deadline:
            logger.info("surrogate: time limit reached after %d evaluations", stats.evals)
            break
        if len(search.X) < min_points:
            logger.info(
                "surrogate: only %d design point(s) — a %s surrogate needs %d; "
                "adding a space-filling point",
                len(search.X),
                model_kind,
                min_points,
            )
            if search.evaluate(search.escape_point(rng), oracle) is None:
                logger.info("surrogate: the box is exhausted at the model's resolution; stopping")
                break
            if on_evaluation is not None:
                on_evaluation(stats.evals, search.best_feasible_value)
            continue

        merit = search.merits()
        Z = search.normalize(np.asarray(search.X, dtype=np.float64))
        try:
            if model_kind == "rbf":
                fitted = RBFSurrogate(kernel=rbf_kernel, ridge=float(rbf_ridge)).fit(Z, merit)
                if fitted.used_least_squares:
                    stats.fit_fallbacks += 1
                if fitted.ill_conditioned:
                    stats.ill_conditioned_fits += 1
            else:
                fitted = KrigingSurrogate(  # type: ignore[assignment]  # union by design
                    nugget=float(nugget),
                    estimate_nugget=bool(estimate_nugget),
                    power=float(kriging_power),
                    estimate_power=bool(estimate_kriging_power),
                    theta_bounds=(float(theta_bounds[0]), float(theta_bounds[1])),
                ).fit(Z, merit)
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            # A fit that cannot be produced ends the search with a stated reason;
            # continuing on a stale surrogate would keep proposing the same point.
            logger.warning(
                "surrogate: fitting the %s surrogate on %d points failed (%s); stopping",
                model_kind,
                len(search.X),
                exc,
            )
            break
        stats.fits += 1

        delta = 0.0
        if model_kind == "rbf":
            beta = cycle[stats.iterations % len(cycle)]
            delta = max(beta * search.maximin_distance(rng), float(min_distance))

        remaining = max(0.0, deadline - time.perf_counter())
        proposals = search.propose(
            model,
            model_kind,
            fitted,
            delta=delta,
            xi=float(xi),
            optimizer=acquisition_optimizer,
            acq_time_limit=min(float(acquisition_time_limit), remaining) or 1e-3,
            acq_gap_tolerance=float(acquisition_gap_tolerance),
            n_multistart=int(acquisition_multistarts),
            rng=rng,
        )
        stats.iterations += 1

        progressed = False
        for x in proposals:
            if stats.evals >= max_evals:
                break
            if search.evaluate(x, oracle) is not None:
                progressed = True
                if on_evaluation is not None:
                    on_evaluation(stats.evals, search.best_feasible_value)
                continue
            # The acquisition proposed a point already in the design. With CORS's
            # positive exclusion radius that should be structurally impossible, so
            # it means integer rounding collapsed the proposal onto a neighbour.
            # Take the maximin space-filling point instead of spinning: the
            # surrogate would not change, so the next iteration would propose the
            # same point again.
            stats.duplicate_escapes += 1
            logger.info(
                "surrogate: the proposal repeated an evaluated point (integer "
                "rounding); substituting the maximin space-filling point"
            )
            if search.evaluate(search.escape_point(rng), oracle) is not None:
                progressed = True
                if on_evaluation is not None:
                    on_evaluation(stats.evals, search.best_feasible_value)
        if not progressed:
            logger.info(
                "surrogate: no new point could be evaluated after %d evaluations "
                "(the box is exhausted at the model's resolution); stopping",
                stats.evals,
            )
            break

    # ---- optional local polish -------------------------------------------
    if local_refine and search.best_feasible_point is not None and time.perf_counter() < deadline:
        refined = _refine_locally(
            model,
            search.best_feasible_point,
            integer_mask,
            min(float(local_refine_time_limit), max(0.0, deadline - time.perf_counter())),
            nlp_solver,
            stats,
        )
        if refined is not None:
            _value, point = refined
            fval, viol = oracle(point)
            before = search.best_feasible_value
            search._offer(search.to_model_point(point), fval, viol)
            if (
                before is not None
                and search.best_feasible_value is not None
                and search.best_feasible_value < before - 1e-12
            ):
                stats.local_improvements += 1

    wall = time.perf_counter() - t_start
    logger.info(
        "surrogate: %d evaluations, %d fit(s); acquisition certified=%d "
        "bb-uncertified=%d multistart=%d failures=%d",
        stats.evals,
        stats.fits,
        stats.acq_certified,
        stats.acq_bb_uncertified,
        stats.acq_multistart,
        stats.acq_failures,
    )

    # ---- the result contract (CLAUDE.md §1) -------------------------------
    # A surrogate is a MODEL of the objective, not a bound on it: an interpolant
    # says nothing rigorous about the function between its data points. So there
    # is no bound, no gap, and the status must never read as a proof. In
    # particular a budget that ran out without a feasible point is a LIMIT, never
    # "infeasible" — this method cannot prove infeasibility. Note that the
    # acquisition subproblems above may legitimately have been solved WITH a
    # certificate; that certifies where to sample, not the answer.
    hit_deadline = time.perf_counter() >= deadline and stats.evals < max_evals
    if search.best_feasible_point is None:
        status = "time_limit" if hit_deadline else "iteration_limit"
        logger.info("surrogate: no feasible point found in %d evaluations", stats.evals)
        return SolveResult(
            status=status,
            objective=None,
            bound=None,
            gap=None,
            x=None,
            wall_time=wall,
            node_count=stats.evals,
            gap_certified=False,
            solver_stats=stats.as_dict(),
        )

    best_value = search.best_feasible_value
    assert best_value is not None  # the no-incumbent case returned above
    return SolveResult(
        status="time_limit" if hit_deadline else "feasible",
        objective=float(best_value),
        bound=None,
        gap=None,
        x=_unpack_solution(model, np.asarray(search.best_feasible_point, dtype=np.float64)),
        wall_time=wall,
        node_count=stats.evals,
        gap_certified=False,
        solver_stats=stats.as_dict(),
    )
