"""Parameter estimability ranking and subset selection.

Tools for the chemical-engineering estimability literature:

- ``estimability_rank`` : Yao et al. (2003) orthogonalization ranking via
  rank-revealing QR on the scaled sensitivity matrix.
- ``collinearity_index`` : Brun, Reichert & Kuensch (2001) collinearity
  index gamma_K for a user-specified parameter subset.
- ``d_optimal_subset`` : Chu & Hahn (2007, 2012) D-optimal subset
  selection. ``method="auto"`` dispatches to enumeration for small
  problems and to the greedy Yao ranking for larger ones. A MINLP
  variant is reserved for a future release; see the ``method`` argument.

Scaling convention
------------------
The scaled sensitivity matrix Z follows Brun's recipe,

    Z = Sigma^{-1/2} J diag(s_theta)

so each column j measures "observable change per meaningful parameter
perturbation" and each row is noise-weighted. ``s_theta_j`` defaults to
``|theta_j|`` (relative scaling) with a floor of ``eps``; callers can
override via ``parameter_scales``. Measurement errors come from
``ExperimentModel.measurement_error``.

Reparameterization warning
--------------------------
The Yao ranking and the Brun collinearity index are *not* invariant
under a reparameterization ``theta -> log theta`` or any other nonlinear
change of variable. The ranking is a statement about the user's
parameterization. Profile likelihood (:mod:`discopt.doe.profile`) *is*
reparameterization-invariant and is the tool of choice when the
parameterization itself is in question.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import scipy.linalg

from discopt.doe.fim import compute_fim
from discopt.estimate import Experiment


@dataclass
class EstimabilityResult:
    """Result of a Yao-style estimability analysis.

    Attributes
    ----------
    ranking : list[str]
        Parameters ordered from most to least estimable.
    projected_norms : numpy.ndarray
        ``|diag(R)|`` from the pivoted QR, in ``ranking`` order. Matches
        Yao's projected 2-norms numerically.
    recommended_subset : list[str]
        Parameters above the user-specified cutoff.
    collinearity_index : float
        Brun gamma_K of the recommended subset.
    parameter_names : list[str]
        Original (unranked) parameter order, for reference.
    """

    ranking: list[str]
    projected_norms: np.ndarray
    recommended_subset: list[str]
    collinearity_index: float
    parameter_names: list[str]


def _scaled_sensitivity(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float] | None,
    parameter_scales: dict[str, float] | None,
    noise_covariance: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """Return the Brun-scaled sensitivity matrix Z and parameter order.

    Uses :func:`compute_fim` for the Jacobian, then applies
    ``Sigma^{-1/2}`` (diagonal noise-weighting) and ``diag(s_theta)``.
    """
    fim_result = compute_fim(experiment, param_values, design_values)
    J = np.asarray(fim_result.jacobian, dtype=np.float64)
    names = list(fim_result.parameter_names)

    em = experiment.create_model(**param_values)
    if noise_covariance is None:
        sigma = np.array(
            [em.measurement_error[name] for name in fim_result.response_names],
            dtype=np.float64,
        )
        J_weighted = J / sigma[:, np.newaxis]
    else:
        Sigma = np.asarray(noise_covariance, dtype=np.float64)
        L = np.linalg.cholesky(Sigma)
        J_weighted = scipy.linalg.solve_triangular(L, J, lower=True)

    eps = np.finfo(np.float64).eps
    scales = np.array(
        [
            (abs(float((parameter_scales or {}).get(name, param_values[name]))) or eps)
            for name in names
        ],
        dtype=np.float64,
    )
    Z = J_weighted * scales[np.newaxis, :]
    return Z, names


def estimability_rank(
    experiment: Experiment,
    param_values: dict[str, float],
    design_values: dict[str, float] | None = None,
    *,
    cutoff: float = 0.04,
    parameter_scales: dict[str, float] | None = None,
    noise_covariance: np.ndarray | None = None,
    _cache: tuple[np.ndarray, list[str]] | None = None,
) -> EstimabilityResult:
    """Rank parameters by estimability (Yao et al. 2003).

    Uses rank-revealing QR on the Brun-scaled sensitivity matrix Z.
    The permutation returned by ``scipy.linalg.qr(..., pivoting=True)``
    coincides with the Yao orthogonalization order, and ``|diag(R_kk)|``
    equals Yao's projected 2-norm at step k exactly.

    Parameters
    ----------
    experiment : Experiment
        Experiment definition.
    param_values : dict[str, float]
        Nominal parameter values.
    design_values : dict[str, float], optional
        Fixed design conditions.
    cutoff : float, default 0.04
        Relative cutoff for the recommended subset. A parameter is
        included if ``|R_kk| / |R_11| >= cutoff``. The default 0.04 is
        Yao's rule of thumb.
    parameter_scales : dict[str, float], optional
        Override parameter-axis scales ``s_theta``. Defaults to
        ``|param_values|``.
    noise_covariance : numpy.ndarray, optional
        Full noise covariance. Defaults to the diagonal
        ``ExperimentModel.measurement_error`` from ``experiment``.

    Returns
    -------
    EstimabilityResult
        Ranking, projected norms, recommended subset, collinearity index.
    """
    if _cache is not None:
        Z, names = _cache
    else:
        Z, names = _scaled_sensitivity(
            experiment, param_values, design_values, parameter_scales, noise_covariance
        )
    n_params = len(names)
    if n_params == 0:
        return EstimabilityResult([], np.zeros(0), [], 1.0, [])

    _, R, piv = scipy.linalg.qr(Z, pivoting=True, mode="economic")
    k = min(R.shape)
    projected = np.zeros(n_params)
    projected[:k] = np.abs(np.diag(R)[:k])

    ranking = [names[i] for i in piv]
    top = projected[0] if projected[0] > 0 else 1.0
    recommended = [ranking[i] for i in range(n_params) if projected[i] / top >= cutoff]

    coll_idx = (
        collinearity_index(
            experiment,
            param_values,
            recommended,
            design_values,
            parameter_scales=parameter_scales,
            noise_covariance=noise_covariance,
            _cache=(Z, names),
        )
        if recommended
        else float("inf")
    )

    return EstimabilityResult(
        ranking=ranking,
        projected_norms=projected,
        recommended_subset=recommended,
        collinearity_index=coll_idx,
        parameter_names=names,
    )


def collinearity_index(
    experiment: Experiment,
    param_values: dict[str, float],
    subset: list[str],
    design_values: dict[str, float] | None = None,
    *,
    parameter_scales: dict[str, float] | None = None,
    noise_covariance: np.ndarray | None = None,
    _cache: tuple[np.ndarray, list[str]] | None = None,
) -> float:
    """Brun-Reichert-Kuensch collinearity index for a parameter subset.

    Defined as ``gamma_K = 1 / sqrt(lambda_min(Z_K^T Z_K))`` where Z_K
    has columns of Z restricted to ``subset`` and further rescaled to
    unit column length (Brun's choice). ``gamma_K`` above ~10 indicates
    collinearity so severe that the subset cannot be jointly estimated.

    Parameters
    ----------
    subset : list[str]
        Parameter names to include. Duplicates and unknown names raise
        ``ValueError``.
    Other parameters
        See :func:`estimability_rank`.

    Returns
    -------
    float
        The collinearity index. ``inf`` if Z_K is rank-deficient.

    Raises
    ------
    ValueError
        If ``subset`` contains duplicate or unrecognized parameter names.
    """
    if _cache is not None:
        Z, names = _cache
    else:
        Z, names = _scaled_sensitivity(
            experiment, param_values, design_values, parameter_scales, noise_covariance
        )
    duplicates = sorted({n for n in subset if subset.count(n) > 1})
    if duplicates:
        raise ValueError(f"subset contains duplicate parameter names: {duplicates}")
    unknown = [n for n in subset if n not in names]
    if unknown:
        raise ValueError(f"subset contains names not in parameter_names {names}: {unknown}")
    idx = [names.index(name) for name in subset]
    Z_K = Z[:, idx]
    if Z_K.size == 0:
        return float("inf")
    col_norms = np.linalg.norm(Z_K, axis=0)
    if np.any(col_norms == 0):
        return float("inf")
    Z_K = Z_K / col_norms
    lam = np.linalg.eigvalsh(Z_K.T @ Z_K)
    lam_min = float(lam[0])
    if lam_min <= 0:
        return float("inf")
    return float(1.0 / np.sqrt(lam_min))


def d_optimal_subset(
    experiment: Experiment,
    param_values: dict[str, float],
    k: int,
    design_values: dict[str, float] | None = None,
    *,
    method: Literal["auto", "enumerate", "greedy", "minlp"] = "auto",
    parameter_scales: dict[str, float] | None = None,
    noise_covariance: np.ndarray | None = None,
) -> list[str]:
    """D-optimal subset of size ``k`` (Chu & Hahn 2007, 2012).

    Picks the size-``k`` subset S of parameters that maximizes
    ``log det(Z_S^T Z_S)``. Available methods:

    - ``"enumerate"``: exact, iterates all C(p, k) subsets. Uses
      ``numpy.linalg.slogdet``. Practical for p up to about 20.
    - ``"greedy"``: top-``k`` parameters from :func:`estimability_rank`.
      Cheap and typically close to optimal.
    - ``"auto"`` (default): enumerate for ``p <= 20``, greedy otherwise.
    - ``"minlp"``: reserved. Writing ``log det`` of a binary-masked
      matrix as an algebraic MINLP in discopt is non-trivial and
      outside the scope of the initial release. Raises
      :class:`NotImplementedError`; the Chu-Hahn paper uses
      combinatorial branch-and-bound with rank-one determinant
      updates, which is a better fit for a dedicated implementation.

    Parameters
    ----------
    k : int
        Subset size. Must satisfy ``0 < k <= n_parameters``.
    method : {"auto", "enumerate", "greedy", "minlp"}
        Solver.
    Other parameters
        See :func:`estimability_rank`.

    Returns
    -------
    list[str]
        Selected parameter names.

    Raises
    ------
    ValueError
        If ``k <= 0``, ``k > n_parameters``, or ``method`` is unknown.
    RuntimeError
        If ``method="enumerate"`` and no subset has positive determinant
        (Z is rank-deficient at rank < k).
    NotImplementedError
        If ``method="minlp"`` — see the method list above.
    """
    # Fail fast on invalid method/k before computing the Jacobian.
    if method not in ("auto", "enumerate", "greedy", "minlp"):
        raise ValueError(
            f"Unknown method {method!r}. Use 'auto', 'enumerate', 'greedy', or 'minlp'."
        )
    if method == "minlp":
        raise NotImplementedError(
            "d_optimal_subset(method='minlp') is reserved for a future release. "
            "The algebraic log-det of a binary-masked Gram matrix is not a "
            "clean MINLP nonlinearity; Chu & Hahn's combinatorial B&B is a "
            "better fit and will be added as a separate implementation. "
            "Use method='enumerate' (exact, p<=20) or method='greedy' (approx)."
        )
    if k <= 0:
        raise ValueError(f"k must be in (0, n_parameters], got {k}")

    Z, names = _scaled_sensitivity(
        experiment, param_values, design_values, parameter_scales, noise_covariance
    )
    p = len(names)
    if k > p:
        raise ValueError(f"k must be in (0, {p}], got {k}")

    chosen_method = method
    if chosen_method == "auto":
        chosen_method = "enumerate" if p <= 20 else "greedy"

    if chosen_method == "enumerate":
        return _dopt_enumerate(Z, names, k)
    # greedy
    res = estimability_rank(
        experiment,
        param_values,
        design_values,
        parameter_scales=parameter_scales,
        noise_covariance=noise_covariance,
        _cache=(Z, names),
    )
    return res.ranking[:k]


def _dopt_enumerate(Z: np.ndarray, names: list[str], k: int) -> list[str]:
    """Exact D-optimal subset by enumeration.

    Raises
    ------
    RuntimeError
        If no size-``k`` subset has a positive-determinant Gram matrix,
        i.e. the Brun-scaled sensitivity matrix Z has rank strictly
        less than ``k`` so no joint estimate is possible.
    """
    p = len(names)
    best_logdet = -np.inf
    best_subset: tuple[int, ...] | None = None
    for S in combinations(range(p), k):
        Z_S = Z[:, list(S)]
        sign, logabsdet = np.linalg.slogdet(Z_S.T @ Z_S)
        if sign > 0 and logabsdet > best_logdet:
            best_logdet = logabsdet
            best_subset = S
    if best_subset is None:
        raise RuntimeError(
            f"No size-{k} subset has a positive-determinant Gram matrix; "
            f"the scaled sensitivity matrix has rank < {k}. Try a smaller k "
            "or run diagnose_identifiability to locate the null directions."
        )
    return [names[i] for i in best_subset]
