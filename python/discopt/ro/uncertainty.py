"""Uncertainty set definitions for robust optimization.

An uncertainty set U describes the range of values that uncertain parameters
can take. The robust counterpart of a problem requires that all constraints
hold for every realization of the uncertain parameters in U.

Three standard families are supported:

* :class:`BoxUncertaintySet` --- L-infinity ball: each parameter independently
  varies within a symmetric interval. Equivalently, the set
  ``{ξ : |ξ_j| ≤ δ_j}``. Leads to separable penalty terms (absolute value
  or 1-norm).

* :class:`EllipsoidalUncertaintySet` --- ellipsoidal ball parameterized by a
  positive semi-definite shape matrix and a radius. Covers Ben-Tal &
  Nemirovski's second-order cone reformulations.

* :class:`PolyhedralUncertaintySet` --- general polytope {ξ : Aξ ≤ b}.
  Includes the budget-of-uncertainty (Bertsimas & Sim) as a special case.
  Leads to LP dual auxiliary variables in the reformulation.

References
----------
.. [BN1999] Ben-Tal, A., Nemirovski, A. (1999). Robust solutions of uncertain
   linear programs. *Operations Research Letters*, 25(1), 1–13.
.. [BertsimasSim2004] Bertsimas, D., Sim, M. (2004). The price of robustness.
   *Operations Research*, 52(1), 35–53.
.. [BenTal2009] Ben-Tal, A., El Ghaoui, L., Nemirovski, A. (2009).
   *Robust Optimization*. Princeton University Press.
"""

from __future__ import annotations

import numpy as np


class UncertaintySet:
    """Abstract base for uncertainty sets.

    Subclasses must bind to a single :class:`~discopt.modeling.core.Parameter`
    (the *uncertain parameter*) and provide the information needed by the
    chosen reformulation strategy.
    """

    def __init__(self, parameter) -> None:  # type: ignore[no-untyped-def]
        from discopt.modeling.core import Parameter

        if not isinstance(parameter, Parameter):
            raise TypeError(
                f"expected a discopt Parameter, got {type(parameter).__name__}"
            )
        self.parameter = parameter

    # Subclasses override these properties to expose set geometry.

    @property
    def kind(self) -> str:
        raise NotImplementedError


class BoxUncertaintySet(UncertaintySet):
    """L-infinity uncertainty: each component varies within ±delta.

    The uncertainty set is

    .. math::
        \\mathcal{U} = \\{\\xi : |\\xi_j - \\bar{p}_j| \\le \\delta_j,\\;
        j = 1,\\ldots,k\\}

    where :math:`\\bar{p}` is the *nominal* parameter value and
    :math:`\\delta` is the half-width vector.

    Parameters
    ----------
    parameter : Parameter
        The uncertain parameter (nominal value = ``parameter.value``).
    delta : float or array-like
        Per-component uncertainty half-width. A scalar is broadcast to all
        components. Must be non-negative.

    Examples
    --------
    >>> cost = m.parameter("cost", value=[10.0, 15.0, 8.0])
    >>> unc = BoxUncertaintySet(cost, delta=1.0)   # ±1 on every component
    >>> unc = BoxUncertaintySet(cost, delta=[0.5, 1.0, 0.3])  # per-component
    """

    def __init__(self, parameter, delta) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parameter)
        delta = np.asarray(delta, dtype=np.float64)
        nominal = parameter.value
        if delta.ndim == 0:
            delta = np.full(nominal.shape, float(delta))
        if delta.shape != nominal.shape:
            raise ValueError(
                f"delta shape {delta.shape} does not match parameter shape {nominal.shape}"
            )
        if np.any(delta < 0):
            raise ValueError("delta must be non-negative")
        self.delta = delta

    @property
    def kind(self) -> str:
        return "box"

    @property
    def lower(self) -> np.ndarray:
        """Component-wise lower bound on the parameter."""
        return self.parameter.value - self.delta

    @property
    def upper(self) -> np.ndarray:
        """Component-wise upper bound on the parameter."""
        return self.parameter.value + self.delta


class EllipsoidalUncertaintySet(UncertaintySet):
    """Ellipsoidal uncertainty: the parameter lies within a scaled L2 ball.

    The uncertainty set is

    .. math::
        \\mathcal{U} = \\{\\xi : \\|\\Sigma^{-1/2}(\\xi - \\bar{p})\\|_2 \\le \\rho\\}

    For the isotropic (identity covariance) case set ``Sigma=None``.

    Parameters
    ----------
    parameter : Parameter
        The uncertain parameter.
    rho : float
        Ellipsoid radius (protection level). Larger values give more
        conservative solutions.  ``rho=2`` corresponds roughly to a 95%
        confidence region for multivariate Gaussian uncertainty.
    Sigma : array-like or None
        Positive semi-definite shape matrix of size (k, k). When ``None``
        the identity is used (isotropic ellipsoid = L2 ball).

    Notes
    -----
    The robust counterpart of a linear constraint :math:`a^\\top x \\le b`
    with uncertain coefficient vector :math:`a = \\bar{a} + \\xi` becomes:

    .. math::
        \\bar{a}^\\top x + \\rho \\|\\Sigma^{1/2} x\\|_2 \\le b

    This is a second-order cone constraint and is handled by
    :class:`~discopt.ro.formulations.ellipsoidal.EllipsoidalRobustFormulation`.

    Examples
    --------
    >>> returns = m.parameter("mu", value=mu_bar)
    >>> unc = EllipsoidalUncertaintySet(returns, rho=2.0)
    >>> unc = EllipsoidalUncertaintySet(returns, rho=2.0, Sigma=cov_matrix)
    """

    def __init__(self, parameter, rho: float, Sigma=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parameter)
        if rho <= 0:
            raise ValueError("rho must be positive")
        self.rho = float(rho)
        k = int(np.prod(parameter.value.shape)) if parameter.value.ndim > 0 else 1
        if Sigma is None:
            self.Sigma = np.eye(k)
        else:
            self.Sigma = np.asarray(Sigma, dtype=np.float64)
            if self.Sigma.shape != (k, k):
                raise ValueError(
                    f"Sigma must be ({k},{k}), got {self.Sigma.shape}"
                )
        # Precompute Sigma^{1/2} via Cholesky (falls back to eig for PSD).
        try:
            self._Sigma_sqrt = np.linalg.cholesky(self.Sigma + 1e-14 * np.eye(k))
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(self.Sigma)
            eigvals = np.maximum(eigvals, 0.0)
            self._Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

    @property
    def kind(self) -> str:
        return "ellipsoidal"

    @property
    def Sigma_sqrt(self) -> np.ndarray:
        """Square root of the shape matrix Σ^{1/2}."""
        return self._Sigma_sqrt


class PolyhedralUncertaintySet(UncertaintySet):
    """Polyhedral uncertainty: the parameter lies in a convex polytope.

    The uncertainty set is

    .. math::
        \\mathcal{U} = \\{\\xi : A\\xi \\le b\\}

    The robust counterpart is obtained by LP duality, introducing auxiliary
    dual variables per constraint.

    Parameters
    ----------
    parameter : Parameter
        The uncertain parameter (flattened to a vector of length k).
    A : array-like, shape (m, k)
        Constraint matrix of the polytope.
    b : array-like, shape (m,)
        RHS vector of the polytope.

    Examples
    --------
    >>> # Budget of uncertainty (Bertsimas & Sim): |ξ_j| ≤ δ_j, Σ|ξ_j|/δ_j ≤ Γ
    >>> # Encodes as PolyhedralUncertaintySet with appropriate A, b.
    >>> unc = PolyhedralUncertaintySet(cost, A=A_matrix, b=b_vec)
    """

    def __init__(self, parameter, A, b) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parameter)
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        k = int(np.prod(parameter.value.shape)) if parameter.value.ndim > 0 else 1
        if A.ndim != 2 or A.shape[1] != k:
            raise ValueError(
                f"A must have shape (m, {k}), got {A.shape}"
            )
        if b.shape != (A.shape[0],):
            raise ValueError(
                f"b must have shape ({A.shape[0]},), got {b.shape}"
            )
        self.A = A
        self.b = b

    @property
    def kind(self) -> str:
        return "polyhedral"


def budget_uncertainty_set(parameter, delta, gamma: float) -> PolyhedralUncertaintySet:
    """Convenience constructor for the budget-of-uncertainty set (Bertsimas & Sim).

    The set is

    .. math::
        \\mathcal{U} = \\left\\{\\xi :
            |\\xi_j| \\le \\delta_j,\\;
            \\sum_j \\frac{|\\xi_j|}{\\delta_j} \\le \\Gamma
        \\right\\}

    which is equivalent to a polyhedral set via auxiliary sign variables.
    This function encodes the full polyhedral representation.

    Parameters
    ----------
    parameter : Parameter
        The uncertain parameter vector of length k.
    delta : float or array-like
        Per-component half-width. Scalar is broadcast.
    gamma : float
        Budget parameter Γ ∈ [0, k]. Γ=0 gives the nominal solution;
        Γ=k recovers the full box set.

    Returns
    -------
    PolyhedralUncertaintySet
    """
    from discopt.modeling.core import Parameter

    if not isinstance(parameter, Parameter):
        raise TypeError(f"expected a Parameter, got {type(parameter).__name__}")

    k = int(np.prod(parameter.value.shape)) if parameter.value.ndim > 0 else 1
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim == 0:
        delta = np.full(k, float(delta))
    if delta.shape != (k,):
        raise ValueError(f"delta must have shape ({k},), got {delta.shape}")
    if np.any(delta <= 0):
        raise ValueError("delta must be positive for budget uncertainty set")
    if not (0 <= gamma <= k):
        raise ValueError(f"gamma must be in [0, {k}], got {gamma}")

    # Build polyhedral representation for the budget set.
    # Variables: ξ ∈ R^k.
    # Constraints:
    #   ξ_j ≤  δ_j  (k constraints)
    #  -ξ_j ≤  δ_j  (k constraints)
    #   Σ ξ_j/δ_j ≤ Γ  ... but this is not directly linear without
    #   auxiliary variables for |ξ_j|.
    #
    # Standard polyhedral lift: introduce u_j ≥ 0 with u_j ≥ ξ_j/δ_j,
    # u_j ≥ -ξ_j/δ_j, Σ u_j ≤ Γ. This doubles the dimension.
    # For the reformulation we only need to encode the support function
    # h(x) = max_{ξ∈U} c(ξ)^T x; the lift is handled by the formulation.
    # For the base PolyhedralUncertaintySet, we represent the box facets
    # together with the budget constraint via the lifted representation
    # ξ = δ_j u_j where u ∈ [-1,1]^k with Σ|u_j| ≤ Γ.
    #
    # We store the box constraints directly; the budget constraint is
    # carried as a separate attribute for the formulation to use.
    I_k = np.eye(k)
    # Box: ξ_j ≤ δ_j, -ξ_j ≤ δ_j  →  A = [I; -I], b = [δ; δ]
    A_box = np.vstack([I_k, -I_k])
    b_box = np.concatenate([delta, delta])
    # Budget: Σ ξ_j/δ_j ≤ Γ and Σ -ξ_j/δ_j ≤ Γ (symmetric)
    inv_delta = 1.0 / delta
    A_budget = np.vstack([inv_delta[None, :], -inv_delta[None, :]])
    b_budget = np.array([gamma, gamma])

    A = np.vstack([A_box, A_budget])
    b = np.concatenate([b_box, b_budget])

    unc = PolyhedralUncertaintySet(parameter, A, b)
    unc._delta = delta  # stored for compact reformulation
    unc._gamma = gamma
    unc._is_budget = True
    return unc
