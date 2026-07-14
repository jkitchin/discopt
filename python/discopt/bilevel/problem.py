"""Bilevel problem front-end.

``BilevelProblem`` mirrors the :class:`~discopt.ro.counterpart.RobustCounterpart`
builder: the user builds a normal :class:`~discopt.modeling.core.Model` holding
the **leader's** objective and constraints, describes the follower separately, and
calls :meth:`~BilevelProblem.formulate`, which rewrites the model in place into a
single-level MPEC that ``model.solve()`` handles.

Scope (``docs/dev/bilevel-module-plan.md`` §1): **optimistic** bilevel with a lower
level that is **convex in the follower variables** — an LP, a convex-QP, or a
certified convex-NLP in ``y`` — via either the KKT reformulation
(:mod:`~discopt.bilevel.kkt`, ``method="kkt"``) or the strong-duality reformulation
(:mod:`~discopt.bilevel.strong_duality`, ``method="strong_duality"``). LP/QP curvature
is proved by a constant-Hessian PSD test; a nonlinear follower is proved convex in
``y`` by an interval-Hessian Gershgorin test over the box (leader variables enter as a
bounded interval, so the natural linearly-coupled form such as ``exp(y) - x*y`` is
accepted). Finite follower-variable bounds are folded into the KKT system
automatically. Everything the reduction is unsound for is refused loudly rather than
silently approximated:

* integer follower variables — KKT does not characterize integer optima;
* a lower level the certifier cannot prove **convex in** ``y`` (indefinite/concave
  curvature, or an unsupported nonlinear atom) — refused (the gate names the witness);
* pessimistic semantics.

Example
-------
>>> from discopt.modeling.core import Model
>>> from discopt.bilevel import BilevelProblem
>>> m = Model("toll")
>>> x = m.continuous("x", lb=0, ub=10)          # leader (toll)
>>> y = m.continuous("y", lb=0, ub=10)          # follower (flow)
>>> m.minimize(x - 4 * y)                        # leader objective
>>> bl = BilevelProblem(
...     m, upper_vars=[x], lower_vars=[y],
...     lower_objective=y,                        # follower minimizes y
...     lower_constraints=[x + y >= 3, y <= 2 * x],
... )
>>> bl.formulate(method="kkt", mpec_method="gdp")   # doctest: +SKIP
>>> result = m.solve()                              # doctest: +SKIP
"""

from __future__ import annotations

import warnings
from typing import TypeGuard

import numpy as np

from discopt.bilevel import kkt as _kkt
from discopt.bilevel import strong_duality as _sd
from discopt.bilevel.symbolic_diff import diff
from discopt.modeling.core import Constant, Constraint, Expression, Model, Variable, VarType
from discopt.mpec import reformulate_gdp, reformulate_sos1

__all__ = ["BilevelProblem"]


def _is_const(expr: Expression) -> TypeGuard[Constant]:
    return isinstance(expr, Constant) and expr.value.ndim == 0


def _hessian_in_y(expr: Expression, ys: list[Variable]):
    """Constant Hessian ``∂²expr/∂y_j∂y_k`` if ``expr`` is (at most) quadratic in ``ys``.

    Uses the symbolic differentiator twice. Returns ``(H, None)`` with ``H`` an
    ``n×n`` numpy matrix when every second partial is a *constant* (so ``expr`` is
    affine or quadratic in ``ys`` with data-constant curvature); returns
    ``(None, witness)`` when some second partial still depends on a variable — i.e.
    ``expr`` is non-quadratic in ``y`` (or has a variable-dependent curvature such as
    ``x·y²``) and needs the full convexity certifier rather than a constant-Hessian
    test.
    """
    n = len(ys)
    H = np.zeros((n, n))
    grads = [diff(expr, yk) for yk in ys]
    for k, gk in enumerate(grads):
        for j in range(k, n):
            hjk = diff(gk, ys[j])
            if not _is_const(hjk):
                return None, ys[j]
            val = float(hjk.value)
            H[k, j] = val
            H[j, k] = val
    return H, None


def _convexity_status(expr: Expression, ys: list[Variable]):
    """Classify curvature of ``expr`` in ``ys``: affine / convex / nonconvex / nonquadratic."""
    H, witness = _hessian_in_y(expr, ys)
    if H is None:
        return "nonquadratic", witness
    if np.allclose(H, 0.0, atol=1e-12):
        return "affine", None
    if np.linalg.eigvalsh(H).min() >= -1e-9:  # PSD -> convex
        return "convex", None
    return "nonconvex", None


class BilevelProblem:
    """Optimistic bilevel program with a convex lower level (LP, convex-QP, or convex-NLP)."""

    def __init__(
        self,
        model: Model,
        *,
        upper_vars: list[Variable],
        lower_vars: list[Variable],
        lower_objective: Expression,
        lower_constraints: list[Constraint],
        lower_sense: str = "min",
        prefix: str = "bl",
        include_follower_bounds: bool = True,
        multiplier_ub: float | None = None,
    ):
        self.model = model
        self.upper_vars = list(upper_vars)
        self.lower_vars = list(lower_vars)
        self.lower_objective = lower_objective
        self.lower_constraints = list(lower_constraints)
        self.lower_sense = lower_sense
        self.prefix = prefix
        self.include_follower_bounds = include_follower_bounds
        # A user-asserted valid finite upper bound on every follower KKT multiplier,
        # enabling the certified big-M (gdp/sos1) path. See _apply_multiplier_ub for
        # the soundness contract (a too-small bound silently cuts the true optimum).
        self.multiplier_ub = multiplier_ub
        self._formulated = False
        self.kkt: _kkt.KKTSystem | None = None
        self.strong_duality: _sd.StrongDualitySystem | None = None
        # The full lower constraint set actually reformulated (user constraints +
        # synthesized finite follower-variable bounds); populated in formulate().
        self.lower_constraints_full: list[Constraint] = list(lower_constraints)

        self._validate_inputs()

    # ── validation / gates ────────────────────────────────────────────

    def _validate_inputs(self) -> None:
        if self.lower_sense not in ("min", "max"):
            raise ValueError(f"lower_sense must be 'min' or 'max', got {self.lower_sense!r}")
        if not self.lower_vars:
            raise ValueError("bilevel problem needs at least one lower (follower) variable")

        for v in self.lower_vars:
            if not isinstance(v, Variable) or v.size != 1:
                raise NotImplementedError(
                    "Phase 1 requires scalar lower variables; pass array components "
                    "(e.g. y[i]) individually."
                )
            if v.var_type is not VarType.CONTINUOUS:
                # Integer follower: KKT / strong-duality do not characterize an
                # integer optimum (the follower value function is discontinuous).
                raise NotImplementedError(
                    f"lower variable '{v.name}' is {v.var_type.name}; the KKT "
                    f"reformulation is only valid for a continuous (convex) follower. "
                    f"Integer lower levels are out of scope (a value-function / "
                    f"branch-in-the-lower-level method is future work)."
                )

        upper_set = {id(v) for v in self.upper_vars}
        for v in self.lower_vars:
            if id(v) in upper_set:
                raise ValueError(f"variable '{v.name}' is in both upper_vars and lower_vars")

    def _gate_convexity(self) -> None:
        """Gate: the lower level must be **convex in y** (LP, convex-QP, or convex-NLP).

        KKT / strong-duality characterize follower optimality only when the lower
        problem is convex in the follower variables. We require the (sign-adjusted)
        objective and every inequality body to be convex in ``y`` — proved by a
        constant-Hessian PSD test for LP/QP curvature, falling through to an
        interval-Hessian Gershgorin test over the box for a nonlinear body — and every
        equality body to be affine in ``y`` (a nonlinear equality makes the lower
        feasible set nonconvex). Anything the certifier cannot prove convex in ``y`` is
        refused (rather than assuming convexity).
        """
        # Follower minimizes f (or maximizes f == minimizes -f): the *minimized*
        # objective must be convex in y.
        signed = self.lower_objective if self.lower_sense == "min" else -self.lower_objective
        self._require_convex(signed, "lower objective")
        for i, con in enumerate(self.lower_constraints):
            if con.sense == "==":
                self._require_affine(con.body, f"lower equality constraint {i}")
            else:
                self._require_convex(con.body, f"lower inequality constraint {i}")

    def _require_convex(self, expr: Expression, what: str) -> None:
        status, info = _convexity_status(expr, self.lower_vars)
        if status in ("affine", "convex"):
            return
        if status == "nonquadratic":
            # The constant-Hessian (LP/QP) test can't classify a nonlinear body.
            # Certify convexity in y directly: enclose the symbolic y-Hessian over the
            # box and run an interval-Gershgorin PSD test (sound sufficient condition).
            # This is convexity *in the follower variables* (leader vars enter as a
            # bounded interval), so it accepts the natural bilevel form where the
            # leader couples linearly, e.g. exp(y) − x·y (∂²/∂y² = exp(y) > 0).
            if self._y_convex_on_box(expr):
                return
            raise NotImplementedError(
                f"{what} is nonlinear in the follower variables and the convexity "
                f"certifier could not prove it convex in y over the box (witness "
                f"'{info.name}'); the KKT reduction needs a certified-convex lower "
                f"level. LP, convex-QP, and certified convex-NLP lower levels are "
                f"supported. Refusing rather than assuming convexity."
            )
        raise NotImplementedError(  # nonconvex
            f"{what} is nonconvex in the follower variables (indefinite Hessian in y). "
            f"The KKT reduction is unsound for a nonconvex lower level; refusing."
        )

    def _require_affine(self, expr: Expression, what: str) -> None:
        status, _ = _convexity_status(expr, self.lower_vars)
        if status != "affine":
            raise NotImplementedError(
                f"{what} must be affine in the follower variables (a nonlinear equality "
                f"makes the lower feasible set nonconvex); got curvature '{status}'."
            )

    def _y_convex_on_box(self, expr: Expression) -> bool:
        """Sound check that ``expr`` is convex in the follower variables over the box.

        Builds the symbolic follower Hessian ``H[j][k] = ∂²expr/∂y_j∂y_k`` (reusing the
        symbolic differentiator), encloses each entry in an interval over the declared
        leader×follower box (:func:`evaluate_interval`), then applies an interval
        Gershgorin test: a symmetric matrix whose every row has
        ``min(diag) − max(off-diagonal abs row sum) ≥ 0`` is PSD, hence the function is
        convex in ``y`` for every leader decision in the box. Sound (sufficient): it
        returns ``False`` (abstains) whenever it cannot prove PSD — including when an
        entry's enclosure is non-finite (unsupported atom) — so the caller refuses
        rather than assume convexity.
        """
        from discopt._jax.convexity.interval_eval import evaluate_interval

        ys = self.lower_vars
        n = len(ys)
        grads = [diff(expr, yk) for yk in ys]
        ent: dict[tuple[int, int], tuple[float, float]] = {}
        for k in range(n):
            for j in range(k, n):
                iv = evaluate_interval(diff(grads[k], ys[j]), self.model)
                lo, hi = float(np.asarray(iv.lo)), float(np.asarray(iv.hi))
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    return False  # unsupported atom -> cannot certify
                ent[(k, j)] = ent[(j, k)] = (lo, hi)
        tol = 1e-9
        for i in range(n):
            diag_min = ent[(i, i)][0]
            radius = sum(max(abs(ent[(i, j)][0]), abs(ent[(i, j)][1])) for j in range(n) if j != i)
            if diag_min - radius < -tol:
                return False
        return True

    # ── big-M feasibility gate for the KKT complementarity path ────────

    # A KKT multiplier left at discopt's ±sentinel bound is "unbounded above" for
    # the purpose of the big-M complementarity encoding: the GDP/SOS1 reformulation
    # needs a finite bound on μ to build the disjunction link, and a sentinel-sized
    # big-M is numerically vacuous (it would let the selector binary sit below the
    # integrality tolerance while μ > 0, silently defeating complementarity — a
    # false optimal). Match the GDP layer's threshold (gdp_reformulate._BIGM_SENTINEL).
    _MU_UNBOUNDED = 1e15

    def _require_bounded_multipliers(self, mpec_method: str) -> None:
        """Refuse loudly when a complementarity multiplier is unbounded above.

        KKT multipliers have no a-priori upper bound, so :func:`kkt.build_kkt`
        creates them unbounded. The big-M complementarity encodings (``gdp``,
        ``sos1``) cannot certify such a system — the linking big-M would be the
        ±sentinel and the disjunction becomes vacuous. Rather than emit an unsound
        (false-optimal) reformulation, refuse and point at the sound alternative.
        """
        unbounded = [
            pair.f.name
            for pair in self.kkt.comp_pairs
            if isinstance(pair.f, Variable) and float(pair.f.ub) >= self._MU_UNBOUNDED
        ]
        if unbounded:
            raise NotImplementedError(
                f"the KKT complementarity multiplier(s) {unbounded} are unbounded "
                f"above, so the '{mpec_method}' big-M reformulation cannot build a "
                f"finite, non-vacuous complementarity encoding — it would certify a "
                f"follower-infeasible point (a false optimum). KKT multipliers have "
                f"no a-priori bound. Use method='strong_duality' (a single bilinear "
                f"equality, no big-M; exact for a convex lower level, though its "
                f"nonconvex equality means the solve is not gap-certified), or pass "
                f"BilevelProblem(..., multiplier_ub=<M>) with a valid finite upper "
                f"bound on the follower's dual multipliers to get a gap-certified solve."
            )

    def _apply_multiplier_ub(self) -> None:
        """Apply the user-supplied ``multiplier_ub`` to the complementarity multipliers.

        ``multiplier_ub`` is a **user-asserted valid** finite upper bound on every
        follower KKT multiplier (exactly like a user-supplied big-M in any MILP). With
        it, the ``gdp`` complementarity encoding is finite and the solve is
        gap-certified. (``sos1`` additionally needs the *other* complementarity operand
        ``-g_i`` bounded, which :func:`discopt.mpec.reformulate_sos1` does not do, so
        it may still refuse — use ``mpec_method="gdp"`` for the certified path.)
        **Soundness contract:** the bound must be ``>=`` the largest
        multiplier at the true follower optimum for *every* leader decision in the
        upper box; a too-small bound silently excludes the true follower response and
        yields a wrong certificate. :meth:`solve` warns if a multiplier ends up at its
        supplied bound (a sign the bound may be cutting). Free equality multipliers
        (no complementarity) are untouched.
        """
        if self.multiplier_ub is None:
            return
        m = float(self.multiplier_ub)
        if not np.isfinite(m) or m <= 0.0:
            raise ValueError(
                f"multiplier_ub must be a finite positive number, got {self.multiplier_ub!r}"
            )
        for pair in self.kkt.comp_pairs:
            if isinstance(pair.f, Variable):
                pair.f.ub = np.asarray(m, dtype=float)

    # ── follower variable bounds are follower constraints ──────────────

    _BIG = 1e19  # discopt's "unbounded" sentinel is 9.999e19; treat |b| >= 1e19 as ∞

    def _follower_bound_constraints(self) -> list[Constraint]:
        """Synthesize KKT constraints for each *finite* follower-variable bound.

        A follower's variable bounds are part of *its* feasible region, so they must
        enter the KKT system — otherwise the reformulation is wrong whenever the
        follower optimum sits on a bound. Emits ``lb - y <= 0`` and/or ``y - ub <= 0``
        for finite bounds (skipping the ±sentinel that means "unbounded").
        """
        cons: list[Constraint] = []
        for v in self.lower_vars:
            lb = float(v.lb)
            ub = float(v.ub)
            if lb > -self._BIG:
                cons.append(
                    Constraint(
                        Constant(lb) - v, sense="<=", rhs=0.0, name=f"{self.prefix}_{v.name}_lb"
                    )
                )
            if ub < self._BIG:
                cons.append(
                    Constraint(
                        v - Constant(ub), sense="<=", rhs=0.0, name=f"{self.prefix}_{v.name}_ub"
                    )
                )
        return cons

    # ── build ─────────────────────────────────────────────────────────

    def build_kkt_system(self) -> _kkt.KKTSystem:
        """Emit the follower's KKT system onto the model and return it (sound math).

        Runs the convexity gate, folds finite follower-variable bounds into the
        lower constraint set, and builds the KKT stationarity + primal-feasibility
        constraints and multipliers via :func:`discopt.bilevel.kkt.build_kkt`. It
        does **not** encode the complementarity pairs (that is :meth:`formulate`'s
        big-M / strong-duality step) and does not mark the problem formulated, so it
        is the way to inspect or validate the KKT characterization independently of
        the encoding. Idempotent: repeated calls return the already-built system;
        :meth:`formulate` reuses it.
        """
        if self.kkt is None:
            self._gate_convexity()
            # Follower variable bounds are follower constraints: fold finite ones in
            # so the KKT system is complete (user constraints first, so
            # multipliers[0:k] stay aligned with the user's lower_constraints).
            extra = self._follower_bound_constraints() if self.include_follower_bounds else []
            self.lower_constraints_full = list(self.lower_constraints) + extra
            self.kkt = _kkt.build_kkt(
                self.model,
                lower_vars=self.lower_vars,
                lower_objective=self.lower_objective,
                lower_constraints=self.lower_constraints_full,
                lower_sense=self.lower_sense,
                prefix=self.prefix,
            )
        return self.kkt

    def formulate(self, *, method: str = "kkt", mpec_method: str = "gdp") -> None:
        """Rewrite the model in place into a single-level MPEC.

        Parameters
        ----------
        method : {"kkt", "strong_duality"}
            The single-level reduction. ``"kkt"`` replaces the follower by its KKT
            conditions with per-row complementarity handed to :mod:`discopt.mpec`.
            ``"strong_duality"`` keeps stationarity/primal/dual feasibility but uses
            the single aggregate equality ``Σ μ_i g_i == 0`` (one bilinear equality,
            no disjunctive branching). Both are exact for a convex lower level and
            agree on the optimum. ``"pessimistic"`` is out of scope.
        mpec_method : {"gdp", "sos1"}
            For ``method="kkt"``: how the complementarity conditions are encoded for
            the **global** solve. ``"gdp"`` (disjunction) typically branches least;
            ``"sos1"`` uses a Special Ordered Set. (The local ``"scholtes"`` homotopy
            is driven at solve time via :func:`discopt.mpec.solve_mpec`.) Ignored for
            ``method="strong_duality"``.
        """
        if self._formulated:
            raise RuntimeError("formulate() has already been called on this BilevelProblem")
        if method == "pessimistic":
            raise NotImplementedError(
                "pessimistic bilevel (leader hedges against adversarial follower ties) "
                "is out of scope; the reduction is optimistic."
            )
        if method not in ("kkt", "strong_duality"):
            raise ValueError(f"unknown method {method!r}; use 'kkt' or 'strong_duality'")

        if method == "kkt":
            if mpec_method not in ("gdp", "sos1"):
                raise ValueError(
                    f"mpec_method must be 'gdp' or 'sos1' for a global certificate, got "
                    f"{mpec_method!r}. (For the local Scholtes homotopy, call "
                    f"discopt.mpec.solve_mpec on the formulated pairs.)"
                )
            # Gate + fold follower bounds + build the KKT stationarity/primal system.
            self.build_kkt_system()
            if self.kkt.comp_pairs:
                # A user-supplied multiplier bound (if any) makes the big-M encoding
                # certifiable; apply it before the unbounded-multiplier gate.
                self._apply_multiplier_ub()
                # The big-M complementarity encodings cannot certify an unbounded
                # multiplier without a vacuous big-M — refuse before emitting it.
                self._require_bounded_multipliers(mpec_method)
                if mpec_method == "gdp":
                    reformulate_gdp(self.model, self.kkt.comp_pairs)
                else:
                    reformulate_sos1(self.model, self.kkt.comp_pairs)
        else:  # strong_duality
            self._gate_convexity()
            extra = self._follower_bound_constraints() if self.include_follower_bounds else []
            self.lower_constraints_full = list(self.lower_constraints) + extra
            self.strong_duality = _sd.build_strong_duality(
                self.model,
                lower_vars=self.lower_vars,
                lower_objective=self.lower_objective,
                lower_constraints=self.lower_constraints_full,
                lower_sense=self.lower_sense,
                prefix=self.prefix,
            )

        self._formulated = True

    def solve(self, **kwargs):
        """Formulate (if needed) and solve the resulting single-level MPEC.

        Thin wrapper over :meth:`Model.solve`; the global optimality certificate is
        the MPEC's (see :mod:`discopt.mpec`). When a ``multiplier_ub`` was supplied,
        this best-effort-warns if any complementarity multiplier ends up *at* its
        supplied bound in the returned solution — a sign the bound may be binding and
        thus cutting the true follower response (raise ``multiplier_ub`` and re-solve).
        """
        if not self._formulated:
            self.formulate()
        result = self.model.solve(**kwargs)
        self._warn_if_multiplier_bound_active(result)
        return result

    def _warn_if_multiplier_bound_active(self, result) -> None:
        """Warn if a comp-pair multiplier sits at its user-supplied ``multiplier_ub``."""
        if self.multiplier_ub is None or self.kkt is None or not self.kkt.comp_pairs:
            return
        m = float(self.multiplier_ub)
        tol = 1e-6 * max(1.0, m)
        for pair in self.kkt.comp_pairs:
            mu = pair.f
            if not isinstance(mu, Variable):
                continue
            try:
                val = float(np.max(np.abs(np.asarray(result.value(mu)))))
            except Exception:
                continue
            if val >= m - tol:
                warnings.warn(
                    f"bilevel multiplier '{mu.name}' is at its supplied multiplier_ub="
                    f"{m:g} in the returned solution; the bound may be binding and "
                    f"cutting the true follower response, so the certificate may be "
                    f"invalid. Raise multiplier_ub and re-solve to confirm.",
                    stacklevel=2,
                )
                return
